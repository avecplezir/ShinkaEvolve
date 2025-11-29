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
from shinka.edit import apply_full_patch
from shinka.llm import extract_between
from shinka.prompts import (
    FULL_SYS_FORMAT_DEFAULT,
    FULL_ITER_MSG,
    format_text_feedback_section,
    perf_str,
)


class AgentMultiTurnInitialEvolutionRunner(EvolutionRunner):
    """Agent-first runner that overrides the submission flow to use
    the multi-attempt, stateful full-rewrite pipeline."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _submit_new_job(self):
        """Submit a new job to the queue."""
        current_gen = self.next_generation_to_submit

        if current_gen >= self.evo_config.num_generations:
            return

        self.next_generation_to_submit += 1

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

        # Build full-rewrite prompt (optionally omit parent code/metrics)

        sys_msg = (
            (self.evo_config.task_sys_msg or "")
            + "\nYou are in a multi-attempt pipeline. After each evaluation you receive feedback; use it to improve the next full program."
            + FULL_SYS_FORMAT_DEFAULT.format(language=self.evo_config.language)
        )
        text_feedback_section = ""

        if self.evo_config.agent_include_parent_context or not self.agent_msg_history:
            if not self.agent_msg_history:
                print("Initializing new agent message history.")
                # handle first generation case, first message with the init program
                parent_program = self.db.get_programs_by_generation(0)[0]
            else:
                print('Including last program context in agent prompt.', current_gen - 1)
                # normal case, get last generation program
                parent_program = self.db.get_programs_by_generation(current_gen - 1)[0]
            code_content = parent_program.code
            performance_metrics = perf_str(
                parent_program.combined_score, parent_program.public_metrics
            )
            eval_feedback = "" # no extra eval feedback needed since we have eval in performance_metrics
        else:
            print('Omitting last program context from agent prompt; relying on history.')
            # Omit parent code/metrics from prompt, agent should rely on history
            code_content = "Code omitted from prompt; rely on conversation history."
            performance_metrics = "Metrics omitted from prompt; rely on conversation history."

            # create evaluation feedback from last attempt
            parent_program = self.db.get_programs_by_generation(current_gen - 1)[0]
            eval_feedback_lines = [
                f"Last iteration results (gen {parent_program.generation}):",
                f"- combined_score: {parent_program.combined_score}",
                f"- correct: {parent_program.correct}",
            ]
            if parent_program.public_metrics:
                eval_feedback_lines.append("- public metrics:")
                for k, v in parent_program.public_metrics.items():
                    eval_feedback_lines.append(f"  - {k}: {v}")
            eval_feedback = "\n".join(eval_feedback_lines)

        agent_patch_msg = (
            "Use the prior attempt's feedback below to propose a better full program.\n\n"
            f"{eval_feedback}\n\n"
            + FULL_ITER_MSG.format(
                language=self.evo_config.language,
                code_content=code_content,
                performance_metrics=performance_metrics,
                text_feedback_section=text_feedback_section,
            )
        )

        llm_kwargs = self.llm.get_kwargs()
        response = None
        patch_txt_attempt = None
        total_api_cost = 0.0
        for agent_attempt in range(self.evo_config.max_patch_attempts):
            response = self.llm.query(
                msg=agent_patch_msg,
                system_msg=sys_msg,
                msg_history=self.agent_msg_history,
                llm_kwargs=llm_kwargs,
            )
            if response is None or response.content is None:
                logger.warning(
                    "LLM returned empty response in agent mode "
                    f"(attempt {agent_attempt + 1}/"
                    f"{self.evo_config.max_patch_attempts}). Retrying..."
                )
                agent_patch_msg = (
                    "The previous attempt was empty. Please respond with "
                    "<NAME>, <DESCRIPTION>, and the full code fenced as "
                    f"```{self.evo_config.language} ...```."
                )
                continue

            total_api_cost += response.cost or 0.0

            if response.new_msg_history:
                self.agent_msg_history = response.new_msg_history
                self._write_agent_dialog(self.agent_msg_history, current_gen)

            patch_name = extract_between(response.content, "<NAME>", "</NAME>", False)
            if not patch_name:
                patch_name = f"agent_gen_{current_gen}"
            patch_description = extract_between(
                response.content, "<DESCRIPTION>", "</DESCRIPTION>", False
            )

            (
                _updated,
                num_applied_attempt,
                output_path_attempt,
                error_attempt,
                patch_txt_attempt,
                patch_path,
            ) = apply_full_patch(
                original_str=parent_program.code,
                patch_str=response.content,
                patch_dir=f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}",
                language=self.evo_config.language,
                verbose=True,
            )

            if error_attempt is None and num_applied_attempt > 0:
                code_diff = patch_txt_attempt
                meta_patch_data = {
                    "patch_type": "full",
                    "api_costs": total_api_cost,
                    "num_applied": num_applied_attempt,
                    "patch_name": patch_name,
                    "patch_description": patch_description,
                    "error_attempt": error_attempt,
                    **llm_kwargs,
                    "llm_result": response.to_dict() if response else None,
                }
                break

            error_str = str(error_attempt) if error_attempt else "No changes applied."
            logger.warning(
                "Full rewrite failed in agent mode "
                f"(attempt {agent_attempt + 1}/"
                f"{self.evo_config.max_patch_attempts}): {error_str}"
            )
            agent_patch_msg = (
                "The previous edit was not successful. This was the error message:\n\n"
                f"{error_str}\n\nTry again. Respond with <NAME>, <DESCRIPTION>, and "
                f"a full program fenced as ```{self.evo_config.language} ...```."
            )
            if response.new_msg_history:
                self.agent_msg_history = response.new_msg_history
                self._write_agent_dialog(self.agent_msg_history, current_gen)

        else:
            logger.error("Exhausted retries in agent mode without a usable response.")
            self.next_generation_to_submit -= 1
            return

        # Get code embedding for novelty downstream (optional)
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

    def _write_agent_dialog(self, history: List[dict], generation: int) -> None:
        """Persist the full agent dialogue to disk for inspection."""
        dialog_path = Path(self.results_dir) / "agent_dialog.txt"
        print('Writing agent dialog to', dialog_path)
        dialog_path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append(f"Generation {generation} (timestamp: {datetime.now().isoformat()})")
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
