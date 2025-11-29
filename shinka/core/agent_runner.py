"""
AgentEvolutionRunner: a derived runner that encapsulates the
multi-attempt, stateful full-rewrite pipeline in one place.

This keeps the base EvolutionRunner untouched for the default flow,
while making it easy to review agent-specific behavior.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Any
from datetime import datetime

from .runner import (
    EvolutionRunner,
    FOLDER_PREFIX,
    logger,
    RunningJob,
)
from shinka.prompts import (
    FULL_SYS_FORMAT_DEFAULT,
    FULL_ITER_MSG,
    AGENT_ITER_MSG,
    format_text_feedback_section,
    perf_str,
    ENV_TO_AGENT_SYS_MSG,
)

from shinka.agent_env import AgentEnv


class AgentEvolutionRunner(EvolutionRunner):
    """Agent-first runner that overrides the submission flow to use
    the multi-attempt, stateful full-rewrite pipeline."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_env = AgentEnv(
            db=self.db,
            scheduler=self.scheduler,
            results_dir=self.results_dir,
            language=self.evo_config.language,
            evo_config=self.evo_config,
            get_code_embedding_cb=self.get_code_embedding,
        )
        self.num_agents = self.evo_config.agent_num_agents
        self.agent_msg_histories: List[List[dict[str, Any]]] = [
            [] for _ in range(self.num_agents)
        ]
        self.agent_names = [
            "Atlas",
            "Nova",
            "Orion",
            "Lyra",
            "Zephyr",
            "Echo",
            "Kite",
            "Ridge",
            "Quill",
            "Sable",
        ][: self.num_agents]
        assert (
            len(self.agent_names) >= self.num_agents
        ), f"Not enough agent names for {self.num_agents} agents."
        self.action_counts = [
            {
                "modify_full": 0,
                "modify_diff": 0,
                "retrieve": 0,
                "reflect": 0,
                "summarize": 0,
            }
            for _ in range(self.num_agents)
        ]
        self.last_summarize_gen = [-1 for _ in range(self.num_agents)]

    def _update_action_counts(self, action_log: List[str], agent_idx: int) -> None:
        """Increment counters based on the action log entries."""
        counts = self.action_counts[agent_idx]
        for entry in action_log:
            if entry.startswith("modify"):
                if "mode=full" in entry:
                    counts["modify_full"] += 1
                elif "mode=diff" in entry:
                    counts["modify_diff"] += 1
            elif entry.startswith("retrieve"):
                counts["retrieve"] += 1
            elif entry.startswith("reflect"):
                counts["reflect"] += 1
            elif entry.startswith("summarize"):
                counts["summarize"] += 1

    def _append_action_counts(
        self, msg: str, allow_push: bool, current_gen: int, agent_idx: int
    ) -> str:
        """Attach a short action-usage summary to the agent message (preprended)."""
        counts = self.action_counts[agent_idx]
        summary = (
            "\n\nAction usage so far (this conversation): "
            f"modify_full={counts['modify_full']}, "
            f"modify_diff={counts['modify_diff']}, "
            f"retrieve={counts['retrieve']}, "
            f"reflect={counts['reflect']}, "
            f"summarize={counts['summarize']}."
        )
        reminder = ""
        if allow_push:
            last_sum = self.last_summarize_gen[agent_idx]
            agent_turn = current_gen // self.num_agents
            gens_since_summary = (
                agent_turn - last_sum if last_sum >= 0 else agent_turn + 1
            )
            # Push only one missing action at a time in priority order
            missing_order: List[str] = []
            if counts["modify_full"] + counts["modify_diff"] == 0:
                missing_order.append("modify")
            if counts["modify_full"] == 0:
                missing_order.append("modify_full")
            if counts["modify_diff"] == 0:
                missing_order.append("modify_diff")
            if counts["reflect"] == 0:
                missing_order.append("reflect")
            if counts["retrieve"] == 0:
                missing_order.append("retrieve")
            if counts["summarize"] == 0:
                missing_order.append("summarize")

            # If summarize is due, prefer that reminder
            if gens_since_summary >= 10:
                reminder = (
                    "It has been many generations since the last summarize. Try a /execute_action{summarize} to replace context with a detailed summary to carry forward"
                )
            elif missing_order:
                next_action = missing_order[0]
                if next_action == "modify":
                    reminder = "Start with exactly one /execute_action{modify_full|modify_diff}."
                elif next_action == "modify_full":
                    reminder = "Try a /execute_action{modify_full} with TARGET_PROGRAM set to a program id."
                elif next_action == "modify_diff":
                    reminder = "Try a /execute_action{modify_diff} with SEARCH/REPLACE blocks for a focused edit."
                elif next_action == "reflect":
                    reminder = (
                        "Add a /execute_action{reflect} line before your modify to note checks/risks."
                    )
                elif next_action == "retrieve":
                    reminder = (
                        "If you have not retrieved yet, include a /execute_action{retrieve} now to inspect past programs."
                    )
                else:  # summarize
                    reminder = (
                        "Try a /execute_action{summarize} to replace context with a detailed summary to carry forward."
                    )

        logger.info("action counts summary: " + summary)
        logger.info("Appending next action reminder: " + reminder)
        return msg + "\n\n" + reminder

    def _submit_new_job(self):
        """Submit a new job to the queue."""
        current_gen = self.next_generation_to_submit

        if current_gen >= self.evo_config.num_generations:
            return

        exec_fname = (
            f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/main.{self.lang_ext}"
        )
        results_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        archive_insp_ids: List[str] = []
        top_k_insp_ids: List[str] = []
        code_diff: Optional[str] = None
        meta_patch_data: dict = {}
        novelty_cost = 0.0

        agent_idx = current_gen % self.num_agents
        history = self.agent_msg_histories[agent_idx]
        allow_push = current_gen >= 5 * self.num_agents

        if not history:
            logger.info(f"Configuring the first obs for LLM for {agent_idx} agent.")
            parent_program = self.db.get_programs_by_generation(0)[0]
            code_content = parent_program.code
            performance_metrics = perf_str(
                parent_program.combined_score, parent_program.public_metrics
            )
            agent_patch_msg = (
                "Use the prior attempt's feedback to propose a better program.\n\n"
                + FULL_ITER_MSG.format(
                    language=self.evo_config.language,
                    code_content=code_content,
                    performance_metrics=performance_metrics,
                    text_feedback_section="",
                )
            )
            agent_patch_msg = self._append_action_counts(
                agent_patch_msg,
                allow_push=allow_push,
                current_gen=current_gen,
                agent_idx=agent_idx,
            )
        else:
            logger.info(f"Augmenting LLM obs with the most recent evaluation for agent {self.agent_names[agent_idx]} (idx {agent_idx}).")
            assert current_gen - self.num_agents >= 0, "Smt is wrong with generation counting"
            parent_program = self.db.get_programs_by_generation(current_gen - self.num_agents)[0]
            eval_feedback_lines = [
                f"Last iteration results (gen {parent_program.generation}, "
                f"name={parent_program.metadata.get('patch_name', 'N/A') if parent_program.metadata else 'N/A'}, "
                f"id={parent_program.id}, agent={parent_program.metadata.get('agent_name', 'unknown') if parent_program.metadata else 'unknown'}):",
                f"- combined_score: {parent_program.combined_score}",
                f"- correct: {parent_program.correct}",
            ]
            prog_agent_name = parent_program.metadata.get("agent_name", "N/A")
            assert prog_agent_name == 'N/A' or self.agent_names[agent_idx] == prog_agent_name, f"Agent name mismatch! {self.agent_names[agent_idx]} vs {prog_agent_name}"
            logger.info(eval_feedback_lines)
            if parent_program.public_metrics:
                eval_feedback_lines.append("- public metrics:")
                for k, v in parent_program.public_metrics.items():
                    eval_feedback_lines.append(f"  - {k}: {v}")
            eval_feedback = "\n".join(eval_feedback_lines)

            agent_patch_msg = AGENT_ITER_MSG.format(
                language=self.evo_config.language,
                eval_feedback=eval_feedback,
                text_feedback_section="",
            )
            agent_patch_msg = self._append_action_counts(
                agent_patch_msg,
                allow_push=allow_push,
                current_gen=current_gen,
                agent_idx=agent_idx,
            )

        # Build system prompt (includes action schema and leaderboard)
        sys_msg = self._build_system_prompt()

        llm_kwargs = self.llm.get_kwargs()
        response = None
        total_api_cost = 0.0
        running_job = None
        meta_patch_data = {}
        code_diff = None
        env_job = None

        logger.info("perform inner loop of agent attempts until a job is submitted")
        inner_steps = 0
        while env_job is None:
            inner_steps += 1
            history = self.agent_msg_histories[agent_idx]

            response = self.llm.query(
                msg=agent_patch_msg,
                system_msg=sys_msg,
                msg_history=history,
                llm_kwargs=llm_kwargs,
            )
            if response is None or response.content is None:
                logger.warning(
                    "LLM returned empty response in agent mode "
                    f"(inner_steps {inner_steps + 1}"
                    f"Retrying..."
                )
                agent_patch_msg = (
                    "The previous attempt was empty. Please try again."
                )
                continue

            total_api_cost += response.cost or 0.0

            if response.new_msg_history:
                self.agent_msg_histories[agent_idx] = response.new_msg_history
                self._write_agent_dialog(self.agent_msg_histories[agent_idx], current_gen, agent_idx)

            obs_text, env_job, meta_patch_data, code_diff, extra_cost, action_log = self.agent_env.step(
                response_content=response.content,
                current_gen=current_gen,
                parent_program=parent_program,
                exec_fname=exec_fname,
                results_dir=results_dir,
                llm_kwargs=llm_kwargs,
                agent_id=agent_idx,
                agent_name=self.agent_names[agent_idx],
            )
            self._update_action_counts(action_log, agent_idx)
            total_api_cost += extra_cost
            agent_name = self.agent_names[agent_idx]
            logger.info(
                f"Agent {agent_name} (idx {agent_idx}) generation {current_gen} executed actions: "
                + "; ".join(action_log)
            )
            self._write_agent_actions(agent_idx, current_gen, action_log)

            # If summary was provided, reset the message history to the summary only
            if any(a.startswith("summarize") for a in action_log):
                self.agent_msg_histories[agent_idx] = [
                    {"role": "user", "content": obs_text}
                ]
                self.last_summarize_gen[agent_idx] = current_gen
                logger.info("Agent history replaced with summarize content.")
                logger.info(f"New agent {agent_idx} context: {obs_text}")

            if env_job is None:
                logger.info("No job submitted; obs_text is the next prompt to the LLM")
                agent_patch_msg = self._append_action_counts(
                    obs_text,
                    allow_push=allow_push,
                    current_gen=current_gen,
                    agent_idx=agent_idx,
                )
                continue

            # env_job carries: (job_id, code_embedding, embed_cost, code_diff, meta_patch_data, parent_id)
            (
                job_id,
                code_embedding,
                embed_cost,
                code_diff,
                meta_patch_data_env,
                parent_id_override,
            ) = env_job
            meta_patch_data = meta_patch_data_env
            meta_patch_data["agent_name"] = self.agent_names[agent_idx]
            meta_patch_data["api_costs"] = meta_patch_data.get("api_costs", 0.0) + total_api_cost
            parent_program = self.db.get(parent_id_override) or parent_program
            break

        # Get code embedding for novelty downstream (optional)
        logger.info(f"Submitting a new job, Generation {current_gen} after {inner_steps} inner steps (agent {agent_idx})")
        code_embedding, embed_cost = self.get_code_embedding(exec_fname)
        job_id = self.scheduler.submit_async(exec_fname, results_dir)
        running_job = RunningJob(
            job_id=job_id,
            exec_fname=exec_fname,
            results_dir=results_dir,
            start_time=time.time(),
            generation=current_gen,
            parent_id=parent_program.id,
            archive_insp_ids=archive_insp_ids,
            top_k_insp_ids=top_k_insp_ids,
            code_diff=code_diff,
            meta_patch_data=meta_patch_data,
            code_embedding=code_embedding,
            embed_cost=embed_cost,
            novelty_cost=novelty_cost,
        )
        self.running_jobs.append(running_job)
        if self.verbose:
            logger.info(
                f"Submitted agent-mode job for generation {current_gen}, "
                f"queue size: {len(self.running_jobs)}"
            )
        # Advance generation counter only after successfully queuing a job
        self.next_generation_to_submit += 1

    def _write_agent_dialog(self, history: List[dict], generation: int, agent_idx: int) -> None:
        """Persist the full agent dialogue to disk for inspection."""
        agent_name = self.agent_names[agent_idx]
        dialog_path = Path(self.results_dir) / f"agent_dialog_{agent_name}.txt"
        dialog_path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append(
            f"Agent {agent_name} (idx {agent_idx}) Generation {generation} (timestamp: {datetime.now().isoformat()})"
        )
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"\n[{role.upper()}]")
            content_lines = str(content).splitlines()
            if len(content_lines) > 200:
                truncated_notice = f"... [truncated {len(content_lines) - 200} lines] ..."
                content_lines = content_lines[:100] + [truncated_notice] + content_lines[-100:]
            lines.extend([line.strip() for line in content_lines])
        lines.append("=" * 80)
        lines.append("")  # trailing newline
        with dialog_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _write_agent_actions(
        self, agent_idx: int, generation: int, action_log: List[str]
    ) -> None:
        """Append executed actions to a per-agent log file."""
        agent_name = self.agent_names[agent_idx]
        actions_path = Path(self.results_dir) / f"agent_actions_{agent_name}.txt"
        actions_path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = [
            "=" * 80,
            f"Agent {agent_name} (idx {agent_idx}) Generation {generation} (timestamp: {datetime.now().isoformat()})",
        ]
        lines.extend(action_log)
        lines.append("")  # trailing newline
        with actions_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _build_system_prompt(self,) -> str:
        """Construct a system prompt that reminds the agent of the task,
        action schema, and current leaderboard."""
        current_gen = self.next_generation_to_submit
        agent_idx = current_gen % self.num_agents
        agent_name = self.agent_names[agent_idx]
        peer_note = ""

        # Build leaderboard block
        leaderboard = self.db.get_top_programs(n=10_000, correct_only=False) # get all programs
        # For early steps, show local leaderboard for this agent only
        local_rounds = max(
            0, getattr(self.evo_config, "agent_local_leaderboard_rounds", 5)
        )
        local_window = local_rounds * self.num_agents
        if current_gen < local_window and self.num_agents > 1:
            leaderboard = [
                prog
                for prog in leaderboard
                if prog.metadata and prog.metadata.get("agent_name") == agent_name
            ] or leaderboard
            # Always include gen0 programs
            gen0_progs = self.db.get_programs_by_generation(0)
            if gen0_progs:
                existing_ids = {p.id for p in leaderboard}
                leaderboard.extend([p for p in gen0_progs if p.id not in existing_ids])
        else:
            if self.num_agents > 1:
                peer_note = (
                    f"\nThere are {self.num_agents} agents working in parallel; Now the leaderboard shows all agents' submissions."
                    "\nUse /execute_action{retrieve} to inspect other agents' programs you have not read yet and borrow ideas when useful."
                    "\nKeep your own plan, but if another agent's program is inspiring, you can set TARGET_PROGRAM to that id in your next modify_full or modify_diff."
                )
                logger.info("Showing peer_note!")
        if leaderboard and getattr(self.evo_config, "agent_leaderboard_correct_first", False):
            def sort_key(prog):
                score = prog.combined_score if prog.combined_score is not None else -float("inf")
                return (not prog.correct, -score)
            leaderboard = sorted(leaderboard, key=sort_key)
        if leaderboard:
            lb_lines = []
            for prog in leaderboard:
                patch_name = prog.metadata.get("patch_name") if prog.metadata else None
                display_name = patch_name or prog.id
                prog_agent = None
                if prog.metadata:
                    prog_agent = prog.metadata.get("agent_name")
                if not prog_agent:
                    prog_agent = "initial" if prog.generation == 0 else "unknown"
                lb_lines.append(
                    f"- name={display_name} (id={prog.id}) gen={prog.generation} "
                    f"agent={prog_agent} "
                    f"score={prog.combined_score:.4f} correct={prog.correct}"
                )
        else:
            lb_lines = ["- (no programs yet)"]
        leaderboard_block = "\n".join(lb_lines)

        logger.info('leaderboard_block: \n' + leaderboard_block)

        return ENV_TO_AGENT_SYS_MSG.format(
            task_sys_msg=(self.evo_config.task_sys_msg or "") + f"\nYou are agent '{agent_name}' (idx {agent_idx})." + peer_note,
            language=self.evo_config.language,
            leaderboard_block=leaderboard_block,
        )
