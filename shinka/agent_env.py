import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from shinka.edit import apply_full_patch, apply_diff_patch
from shinka.llm import extract_between

logger = logging.getLogger(__name__)


class ActionEngine:
    """
    Minimal action parser for agent LLM outputs using /execute_action blocks.
    If parsing fails, no fallback is performed; the caller should surface
    the error back to the LLM for correction.
    """

    modify_modes = {"modify_full": "full", "modify_diff": "diff"}
    allowed_types = set(modify_modes) | {"retrieve", "reflect", "summarize"}
    action_attempts: Dict[str, int] = {
        "modify_full": 0,
        "modify_diff": 0,
        "retrieve": 0,
        "reflect": 0,
        "summarize": 0,
    }

    def parse_actions(self, content: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Parse actions from an LLM response.

        Returns:
            (actions, error_message)
            - actions: list of validated action dicts (may include partial actions)
            - error_message: None if at least one valid action exists; otherwise a short
              string describing the parse/validation error
        """
        actions: List[Dict[str, Any]] = []
        errors: List[str] = []

        pattern = re.compile(r"/execute_action\{([a-zA-Z_]+)\}", re.IGNORECASE)
        # logger.info("Parsing actions with pattern %s from content: %s", pattern.pattern, content)
        matches = list(pattern.finditer(content))
        logger.info("Found %d action blocks", len(matches))
        for idx, match in enumerate(matches):
            act_type = match.group(1).lower()
            logger.info("Found action type: %s", act_type)
            if act_type not in self.allowed_types:
                errors.append(f"Unknown action type '{act_type}'.")
                continue
            # Track attempts by raw type label
            if act_type in self.action_attempts:
                self.action_attempts[act_type] += 1

            # Slice the content for this action block
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            block = content[start:end]

            if act_type in self.modify_modes:
                target = self._extract_field(block, r"TARGET_PROGRAM\s*:\s*(.+)")
                if not target:
                    errors.append("Modify action missing TARGET_PROGRAM.")
                    continue
                patch = block.strip()
                if not patch:
                    errors.append("Modify action missing patch content.")
                    continue
                actions.append(
                    {
                        "type": "modify",
                        "mode": self.modify_modes[act_type],
                        "target": target.strip(),
                        "patch": patch,
                    }
                )
            elif act_type == "retrieve":
                targets_str = self._extract_field(block, r"TARGET_PROGRAM\s*:\s*(.+)")
                if not targets_str:
                    errors.append("Retrieve action missing TARGET_PROGRAM list.")
                    continue
                targets = [t.strip() for t in targets_str.split(",") if t.strip()]
                if not targets:
                    errors.append("Retrieve action has empty TARGET_PROGRAM list.")
                    continue
                actions.append({"type": "retrieve", "targets": targets})
            elif act_type == "reflect":
                actions.append({"type": "reflect"})
            elif act_type == "summarize":
                summary_text = block.strip()
                actions.append({"type": "summarize", "summary": summary_text})

        logger.info(
            "Action attempt counts: %s",
            {k: v for k, v in self.action_attempts.items()},
        )

        if actions:
            if errors:
                return actions, "; ".join(errors)
            return actions, None
        return [], "; ".join(errors) if errors else "No valid actions were found."

    def _extract_field(self, text: str, pattern: str) -> Optional[str]:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    def get_first_modify(self, actions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for act in actions:
            if act.get("type") == "modify":
                return act
        return None

    def get_retrieves(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [act for act in actions if act.get("type") == "retrieve"]

    def get_reflects(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [act for act in actions if act.get("type") == "reflect"]


class AgentEnv:
    """
    Minimal environment that executes parsed actions and returns the next
    observation (as text) plus an optional RunningJob to be tracked by
    the caller.
    """

    def __init__(
        self,
        db,
        scheduler,
        results_dir: str,
        language: str,
        evo_config,
        get_code_embedding_cb,
    ):
        self.db = db
        self.scheduler = scheduler
        self.results_dir = results_dir
        self.language = language
        self.evo_config = evo_config
        self.get_code_embedding = get_code_embedding_cb
        self.action_engine = ActionEngine()

    def _hamming_distance(self, a: str, b: str) -> int:
        if len(a) != len(b):
            return max(len(a), len(b))
        return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

    def _suggest_program_ids(self, target_id: str, max_suggestions: int = 1):
        try:
            programs = self.db.get_all_programs()
        except Exception as exc:
            logger.warning("Could not load programs for suggestions: %s", exc)
            return []

        scored = []
        for prog in programs:
            dist = self._hamming_distance(target_id, prog.id)
            display_name = (
                prog.metadata.get("patch_name", prog.id)
                if prog.metadata is not None
                else prog.id
            )
            prog_agent = "unknown"
            if prog.metadata:
                prog_agent = prog.metadata.get("agent_name", "unknown") or "unknown"
            if prog_agent is None or prog_agent == "":
                prog_agent = "unknown"
            scored.append(
                (
                    dist,
                    prog.id,
                    display_name,
                    prog_agent,
                    prog.combined_score,
                )
            )
        scored.sort(key=lambda x: x[0])
        return scored[:max_suggestions]

    def step(
        self,
        response_content: str,
        current_gen: int,
        parent_program,
        exec_fname: str,
        results_dir: str,
        llm_kwargs: dict,
    ):
        """
        Execute actions parsed from the LLM response.

        Returns:
            obs_text (str): message to feed back to the LLM
            running_job (Optional[RunningJob]): job handle if a modify was applied
            meta_patch_data (dict): metadata for the applied patch
            code_diff (Optional[str]): diff text of the applied patch
            total_api_cost (float): sum of LLM costs consumed up to this step
            action_log (List[str]): human-readable log of actions processed
        """
        actions, err = self.action_engine.parse_actions(response_content)
        if err:
            return (
                (
                    f"Action parsing failed: {err} Please return actions starting with "
                    "/execute_action{modify_full|modify_diff|retrieve|reflect|summarize} and "
                    "include required fields."
                ),
                None,
                {},
                None,
                0.0,
                [],
            )

        if not actions:
            return (
                (
                    "No valid actions found. Please send "
                    "/execute_action{modify_full|modify_diff|summarize|retrieve} with required fields."
                ),
                None,
                {},
                None,
                0.0,
                [],
            )

        obs_blocks: List[str] = []
        running_job = None
        meta_patch_data = {}
        code_diff = None
        extra_cost = 0.0
        action_log: List[str] = []

        # Summarize overrides all other actions if present
        summarize_actions = [a for a in actions if a.get("type") == "summarize"]
        if summarize_actions:
            summary_txt = summarize_actions[0].get("summary", "").strip()
            if not summary_txt:
                summary_txt = "Summary requested but no content provided."

            # Include the initial program for context preservation
            initial_prog = None
            try:
                gen0 = self.db.get_programs_by_generation(0)
                if gen0:
                    initial_prog = gen0[0]
            except Exception:
                initial_prog = None

            if initial_prog:
                initial_name = (
                    initial_prog.metadata.get("patch_name", initial_prog.id)
                    if initial_prog.metadata is not None
                    else initial_prog.id
                )
                initial_block = (
                    f"\n\nInitial program (gen 0, name={initial_name}, id={initial_prog.id}):\n"
                    f"```{self.language}\n{initial_prog.code}\n```"
                )
            else:
                initial_block = ""

            obs_blocks.append(
                initial_block +
                "Context replaced by summary (only summarize executed this turn):\n"
                + summary_txt
            )
            action_log.append("summarize status=recorded")
            return (
                "\n\n".join(obs_blocks),
                running_job,
                meta_patch_data,
                code_diff,
                extra_cost,
                action_log,
            )

        # Handle at most one modify
        modify_act = self.action_engine.get_first_modify(actions)
        if modify_act:
            target_id = modify_act["target"]
            target_program = self.db.get(target_id)
            if target_program is None:
                obs_blocks.append(
                    (
                        f"Modify action target '{target_id}' not found in database. "
                        "Please choose an existing program id."
                    )
                )
            else:
                mode = modify_act["mode"]
                patch_text = modify_act["patch"]
                apply_fn = apply_full_patch if mode == "full" else apply_diff_patch
                (
                    _updated,
                    num_applied_attempt,
                    output_path_attempt,
                    error_attempt,
                    patch_txt_attempt,
                    patch_path,
                ) = apply_fn(
                    original_str=target_program.code,
                    patch_str=patch_text,
                    patch_dir=Path(exec_fname).parent,
                    language=self.language,
                    verbose=True,
                )
                if error_attempt is None and num_applied_attempt > 0:
                    code_diff = patch_txt_attempt
                    job_id = self.scheduler.submit_async(exec_fname, results_dir)
                    code_embedding, embed_cost = self.get_code_embedding(exec_fname)
                    meta_patch_data = {
                        "patch_type": mode,
                        "api_costs": 0.0,
                        "num_applied": num_applied_attempt,
                        "patch_name": extract_between(patch_text, "<NAME>", "</NAME>", False),
                        "patch_description": extract_between(
                            patch_text, "<DESCRIPTION>", "</DESCRIPTION>", False
                        ),
                        "error_attempt": error_attempt,
                        **llm_kwargs,
                    }
                    running_job = (
                        job_id,
                        code_embedding,
                        embed_cost,
                        code_diff,
                        meta_patch_data,
                        target_program.id,
                    )
                    obs_blocks.append("Applied modify action and submitted for evaluation.")
                    action_log.append(f"modify target={target_id} mode={mode} status=success")
                else:
                    err_str = str(error_attempt) if error_attempt else "No changes applied."
                    obs_blocks.append(
                        (
                            f"Apply patch failed: {err_str}. Please resend a valid "
                            "modify_full or modify_diff action."
                        )
                    )
                    action_log.append(f"modify target={target_id} status=error msg={err_str}")

        # Handle retrieve actions (if any)
        retrieves = self.action_engine.get_retrieves(actions)
        if retrieves:
            retrieved_blocks = []
            for r in retrieves:
                for target_id in r.get("targets", []):
                    prog = self.db.get(target_id)
                    if prog:
                        display_name = (
                            prog.metadata.get("patch_name", prog.id)
                            if prog.metadata is not None
                            else prog.id
                        )
                        prog_agent = "unknown"
                        if prog.metadata:
                            prog_agent = prog.metadata.get("agent_name", "unknown")
                        if not prog_agent:
                            prog_agent = "unknown"
                        prog_score = prog.combined_score
                        metrics_summary = (
                            f"combined_score: {prog.combined_score}, "
                            f"correct: {prog.correct}, "
                            f"public: {prog.public_metrics}"
                        )
                        retrieved_blocks.append(
                            f"Program {display_name} (id {prog.id}, gen {prog.generation}):\n"
                            f"{metrics_summary}\n"
                            f"```{self.language}\n{prog.code}\n```"
                        )
                        action_log.append(
                            f"retrieve target={target_id} status=found name={display_name} score={prog_score} agent={prog_agent}"
                        )
                    else:
                        suggestions = self._suggest_program_ids(target_id)
                        if suggestions:
                            suggestion_text = "; ".join(
                                f"{sid} (name={name}, agent={agent}, score={score})"
                                for _, sid, name, agent, score in suggestions
                            )
                            retrieved_blocks.append(
                                f"Program {target_id} not found. Did you mean: {suggestion_text}."
                            )
                            action_log.append(
                                f"retrieve target={target_id} status=not_found suggestions={[sid for _, sid, _, _, _ in suggestions]}"
                            )
                        else:
                            retrieved_blocks.append(f"Program {target_id} not found.")
                            action_log.append(f"retrieve target={target_id} status=not_found")
            obs_blocks.append("Retrieved programs:\n\n" + "\n\n".join(retrieved_blocks))

        # Handle reflect actions (if any) without augmenting the observation
        reflects = self.action_engine.get_reflects(actions)
        if reflects:
            action_log.append("reflect status=recorded")

        if not obs_blocks:
            obs_blocks.append(
                (
                    "No modify, summarize, or retrieve action provided. Please return a modify_full "
                    "or modify_diff action with code."
                )
            )

        return "\n\n".join(obs_blocks), running_job, meta_patch_data, code_diff, extra_cost, action_log
