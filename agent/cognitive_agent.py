"""Cognitive Agent loop implementation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List

from agent.prompt_templates import (
    action_generation_prefix,
    action_logit_candidates,
    build_action_content_prompt,
    build_step_prompt,
)
from agent.slm_inference import ModelAdapter, softmax_from_logps


@dataclass
class CAConfig:
    max_steps: int = 20
    log_prompts: bool = False
    fail_on_nonfinite: bool = True
    fail_on_noop: bool = True
    noop_retry_attempts: int = 3
    force_recall_each_step: bool = True
    debug_prompt_char_limit: int = 12000


def _truncate_words(text: str, max_words: int) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _sanitize_action_content(text: str) -> str:
    t = " ".join((text or "").split())
    t = t.replace('\\"', '"')
    t = t.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    t = t.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Keep only output field before optional reasoning block.
    t = re.split(r"<reasoning>", t, maxsplit=1, flags=re.IGNORECASE)[0]
    # Drop obvious prompt-field leakage if it appears in generated content.
    t = re.split(
        r"\b(?:WorkingMemory|Reasoning|Goal|RecalledTop3|Recalled|Question|OPTIONS|INSTRUCTION):",
        t,
        maxsplit=1,
    )[0]
    # Remove leaked JSON key/value tails when continuation parsing fell back.
    t = re.split(
        r'"\s*,\s*"(?:reasoning|action|next_goal|next_thought|working_memory|answer|memory|cues|transformed subgoal)\b',
        t,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    # Drop leaked instruction blocks echoed by the model.
    t = re.split(r"\b(?:Required keys|Optional key|Output JSON only)\s*:", t, maxsplit=1, flags=re.IGNORECASE)[0]
    # Drop common tag echoes from prefix format.
    t = re.sub(r"\[(?:SUBGOAL|WM|REASONING|ANSWER)\]", " ", t)
    t = re.sub(r"<(?:G|R|A)>", "", t)
    t = re.sub(
        r"invalid response\.?\s*you must update new goal,\s*memory,\s*or answer\.?",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"^\s*Next Goal of Agent:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*Next thought:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*Answer to goal:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r'["}\]]+\s*$', "", t)
    return " ".join(t.split()).strip()


def _extract_json_string_value(text: str, start_idx: int) -> str:
    out: List[str] = []
    escaped = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            break
        out.append(ch)
    return "".join(out).strip()


def _extract_named_json_string(text: str, key: str) -> str:
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"', str(text or ""))
    if not m:
        return ""
    return _extract_json_string_value(str(text or ""), m.end())


def _normalize_next_thought(next_thought: str, reasoning: str = "") -> str:
    t = _sanitize_action_content(next_thought)
    if not t:
        return ""

    # Convert common meta wrappers into concrete thought text.
    # Example: "Working memory now includes 'X' and 'Y'" -> "X; Y"
    quoted = re.findall(r"'([^']+)'", t)
    if quoted and any(k in t.lower() for k in ("working memory now includes", "current reasoning")):
        t = "; ".join(q.strip() for q in quoted if q.strip())

    t = re.sub(r"^\s*working memory now includes\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*current reasoning(?: identifies| shows| states)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*goal remains unchanged\s*[;:,\-]*\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*applying recalled knowledge (?:confirms|shows|indicates)\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*the next step is to\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*i will\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*(includes?|states?)\s*", "", t, flags=re.IGNORECASE)
    t = _sanitize_action_content(t)

    if t:
        return t
    # Last resort: use a concise factual fragment from reasoning.
    return _sanitize_action_content(reasoning).split(".")[0].strip()


def _extract_r_fields(payload: Dict) -> tuple[str, str]:
    reasoning = _sanitize_action_content(str(payload.get("reasoning", "")))
    next_thought = _normalize_next_thought(str(payload.get("next_thought", "")), reasoning)
    return reasoning, next_thought


def _is_meta_next_thought(text: str) -> bool:
    low = " ".join(str(text or "").lower().split())
    blocked_phrases = (
        "goal remains",
        "current reasoning",
        "working memory",
        "recalledtop3",
        "need to recall",
        "applying recalled knowledge",
        "the next step is to recall",
        "i will select",
    )
    return any(p in low for p in blocked_phrases)


def _is_valid_r_payload(payload: Dict) -> bool:
    reasoning, next_thought = _extract_r_fields(payload)
    if not reasoning or not next_thought:
        return False
    if _is_meta_next_thought(next_thought):
        return False
    return True


def _extract_first_json_object(text: str) -> Dict:
    s = str(text or "").strip()
    if not s:
        raise json.JSONDecodeError("empty content", s, 0)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1))
    start = s.find("{")
    if start == -1:
        raise json.JSONDecodeError("no json object start", s, 0)
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start : i + 1])
    raise json.JSONDecodeError("no balanced json object", s, 0)


def _extract_action_text_from_payload(action: str, payload: Dict) -> str:
    if action == "<G>":
        return str(payload.get("next_goal", "")).strip()
    if action == "<R>":
        return str(payload.get("next_thought", "")).strip()
    return str(payload.get("answer", "")).strip()


def _extract_value_from_prefix_continuation(raw_generated: str) -> str:
    s = str(raw_generated or "")
    if not s:
        return ""
    out: List[str] = []
    escaped = False
    for ch in s:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            break
        out.append(ch)
    return "".join(out).strip()


def _parse_generated_action_output(action: str, forced_prefix: str, raw_generated: str) -> tuple[str, Dict]:
    raw = str(raw_generated or "").strip()
    candidates = []
    if raw:
        candidates.append(raw)
    candidates.append(f"{forced_prefix}{raw}")
    last_err = None
    for cand in candidates:
        try:
            payload = _extract_first_json_object(cand)
            content = _sanitize_action_content(_extract_action_text_from_payload(action, payload))
            return content, payload
        except Exception as exc:
            last_err = exc
    # Fallback for malformed continuations where the model echoes prompt schema text.
    fallback_raw = str(raw_generated or "")
    fallback_raw = re.split(r'\\",\\\"', fallback_raw, maxsplit=1)[0]
    fallback_raw = re.split(r'"\s*,\s*"', fallback_raw, maxsplit=1)[0]
    fallback_content = _sanitize_action_content(fallback_raw)
    if fallback_content:
        if action == "<G>":
            return fallback_content, {"action": "<G>", "next_goal": fallback_content, "_parse_fallback": True}
        if action == "<R>":
            merged = f"{forced_prefix}{str(raw_generated or '')}"
            fallback_reasoning = _sanitize_action_content(_extract_value_from_prefix_continuation(str(raw_generated or "")))
            fallback_next = _sanitize_action_content(_extract_named_json_string(merged, "next_thought"))
            if fallback_reasoning or fallback_next:
                return fallback_next, {
                    "action": "<R>",
                    "reasoning": fallback_reasoning,
                    "next_thought": fallback_next,
                    "_parse_fallback": True,
                }
            return "", {"action": "<R>", "reasoning": "", "next_thought": "", "_parse_fallback": True}
        return fallback_content, {"action": "<A>", "answer": fallback_content, "_parse_fallback": True}
    if last_err is not None:
        # Strict JSON mode with conservative continuation fallback above.
        raise last_err
    return "", {}


def _extract_content_line(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # First line is the operative content; trailing text is ignored.
    return t.split("\n")[0].strip()


def _normalized_state_text(text: str) -> str:
    return " ".join(_sanitize_action_content(text).lower().split())


def _format_recalled_dm_text(items: List[Dict]) -> str:
    if not items:
        return ""
    lines: List[str] = []
    for idx, it in enumerate(items, start=1):
        txt = _sanitize_action_content(str(it.get("dm_text", "")))
        if not txt:
            continue
        lines.append(f"{idx}. {txt}")
    return "\n".join(lines).strip()


def _retry_with_invalid_hint(
    model: ModelAdapter,
    action: str,
    forced_prefix: str,
    content_prompt: str,
    attempts: int,
    retry_hint: str,
    is_valid: Callable[[str, Dict], bool] | None = None,
    max_new_tokens: int = 96,
) -> tuple[str, str, Dict, List[Dict], bool]:
    retry_logs: List[Dict] = []
    last_raw = ""
    last_content = ""
    last_payload: Dict = {}
    requested_attempts = max(0, int(attempts))
    total_attempts = max(10, requested_attempts)
    first_phase = min(5, total_attempts)
    temperatures = [0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
    while len(temperatures) < total_attempts:
        temperatures.append(min(1.40, temperatures[-1] + 0.05))

    for retry_idx in range(total_attempts):
        is_phase2 = retry_idx >= first_phase
        phase_hint = retry_hint
        if is_phase2:
            if action == "<R>":
                phase_hint = (
                    retry_hint
                    + "\nCRITICAL: you must generate `next_thought` field."
                    + "\nCRITICAL: `next_thought` must be non-empty and concrete."
                    + '\nSchema reminder: {"action":"<R>","reasoning":"...","next_thought":"..."}'
                    + "\nKeep reasoning very short so JSON can include next_thought."
                )
            elif action == "<G>":
                phase_hint = (
                    retry_hint
                    + '\nCRITICAL: you must generate `next_goal` field in JSON.'
                )
            else:
                phase_hint = (
                    retry_hint
                    + '\nCRITICAL: you must generate `answer` field in JSON.'
                )

        retry_prompt = content_prompt + "\n[RETRY]\n" + phase_hint
        retry_temperature = temperatures[retry_idx]
        retry_raw = model.generate_after_prefix(
            retry_prompt,
            forced_prefix,
            max_new_tokens=max_new_tokens,
            temperature=retry_temperature,
        )
        try:
            retry_content, retry_payload = _parse_generated_action_output(action, forced_prefix, retry_raw)
        except Exception:
            retry_content, retry_payload = "", {}
        retry_logs.append(
            {
                "retry_index": retry_idx + 1,
                "retry_mode": "standard" if not is_phase2 else "enforced_schema",
                "retry_phase": 1 if not is_phase2 else 2,
                "retry_temperature": retry_temperature,
                "retry_raw_content": retry_raw,
                "retry_content": retry_content,
                "retry_payload": retry_payload,
            }
        )
        last_raw = retry_raw
        last_content = retry_content
        last_payload = retry_payload
        valid = bool(is_valid(retry_content, retry_payload)) if is_valid else True
        if retry_content and valid:
            return last_raw, last_content, last_payload, retry_logs, True
    return last_raw, last_content, last_payload, retry_logs, False


def _try_cross_action_recovery(
    model: ModelAdapter,
    *,
    original_action: str,
    question_text: str,
    options_labeled: Dict[str, str],
    current_goal: str,
    working_memory: str,
    recalled_dm_text: str,
    attempts: int = 2,
    max_new_tokens: int = 96,
) -> tuple[bool, Dict]:
    opposite_action = "<R>" if original_action == "<G>" else "<G>"
    content_prompt = build_action_content_prompt(
        question_text=question_text,
        options_labeled=options_labeled,
        current_goal=current_goal,
        working_memory=working_memory,
        recalled_dm_text=recalled_dm_text,
        action=opposite_action,
    )
    forced_prefix = action_generation_prefix(opposite_action)

    retry_logs: List[Dict] = []
    for retry_idx in range(max(0, int(attempts))):
        raw = model.generate_after_prefix(content_prompt, forced_prefix, max_new_tokens=max_new_tokens)
        try:
            content, payload = _parse_generated_action_output(opposite_action, forced_prefix, raw)
        except Exception:
            content, payload = "", {}

        if opposite_action == "<R>":
            valid = _is_valid_r_payload(payload)
            reasoning_text, next_thought_text = _extract_r_fields(payload)
            wm_candidate = _truncate_words(next_thought_text, 60)
            unchanged = _normalized_state_text(wm_candidate) == _normalized_state_text(working_memory)
            retry_logs.append(
                {
                    "retry_index": retry_idx + 1,
                    "retry_raw_content": raw,
                    "retry_content": content,
                    "retry_payload": payload,
                    "retry_working_memory": wm_candidate,
                    "retry_valid": bool(valid and wm_candidate and not unchanged),
                }
            )
            if valid and wm_candidate and not unchanged:
                return True, {
                    "recovered_action": "<R>",
                    "raw_content": raw,
                    "content": next_thought_text,
                    "parsed_payload": {"action": "<R>", "reasoning": reasoning_text, "next_thought": next_thought_text},
                    "working_memory_after": wm_candidate,
                    "retry_logs": retry_logs,
                    "forced_prefix": forced_prefix,
                    "content_prompt": content_prompt,
                }
            continue

        # opposite_action == "<G>"
        subgoal = _sanitize_action_content(_extract_content_line(content))
        subgoal = _truncate_words(subgoal, 20)
        if subgoal and not subgoal.endswith("."):
            subgoal = f"{subgoal}."
        unchanged = _normalized_state_text(subgoal) == _normalized_state_text(current_goal)
        retry_logs.append(
            {
                "retry_index": retry_idx + 1,
                "retry_raw_content": raw,
                "retry_content": content,
                "retry_payload": payload,
                "retry_subgoal": subgoal,
                "retry_valid": bool(subgoal and not unchanged),
            }
        )
        if subgoal and not unchanged:
            return True, {
                "recovered_action": "<G>",
                "raw_content": raw,
                "content": subgoal,
                "parsed_payload": {"action": "<G>", "next_goal": subgoal},
                "goal_after": subgoal,
                "retry_logs": retry_logs,
                "forced_prefix": forced_prefix,
                "content_prompt": content_prompt,
            }

    return False, {
        "recovered_action": opposite_action,
        "retry_logs": retry_logs,
        "forced_prefix": forced_prefix,
        "content_prompt": content_prompt,
    }


def _extract_answer_text(entry: Dict) -> str:
    if not isinstance(entry, dict):
        return ""
    parsed = entry.get("parsed_payload")
    if isinstance(parsed, dict):
        parsed_answer = _sanitize_action_content(str(parsed.get("answer", "")))
        if parsed_answer:
            return parsed_answer
    content_answer = _sanitize_action_content(str(entry.get("content", "")))
    if content_answer:
        return content_answer
    return ""


def _label_text_bias(answer_text: str, options_labeled: Dict[str, str]) -> Dict[str, float]:
    bias = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    normalized_answer = _normalized_state_text(answer_text)
    if not normalized_answer:
        return bias

    answer_padded = f" {normalized_answer} "
    option_matches: List[str] = []
    for label in ("A", "B", "C", "D"):
        opt_text = _normalized_state_text(str(options_labeled.get(label, "")))
        if opt_text and f" {opt_text} " in answer_padded:
            option_matches.append(label)

    option_letter_matches = [m.upper() for m in re.findall(r"\boption\s*([a-d])\b", str(answer_text or ""), flags=re.IGNORECASE)]
    matched = set(option_matches + option_letter_matches)
    if len(matched) == 1:
        only = next(iter(matched))
        bias[only] += 1.25
    elif len(matched) > 1:
        for label in matched:
            bias[label] += 0.25
    return bias


class CognitiveAgent:
    def __init__(self, model: ModelAdapter, kb_server, config: CAConfig) -> None:
        self.model = model
        self.kb_server = kb_server
        self.config = config

    def run(self, problem: Dict) -> Dict:
        goal_stack: List[str] = ["Solve the problem"]
        goal_stack_history: List[Dict] = [
            {
                "step": -1,
                "event": "init",
                "goal_stack": list(goal_stack),
                "goal": goal_stack[-1],
            }
        ]
        question_text = problem["question_text"]
        working_memory = question_text
        solution_steps_log: List[Dict] = []
        retrieved_dm_text = ""

        options_labeled = problem["options_labeled"]
        correct_label = problem["correct_label"]

        termination = "max_steps"
        pred_label = "A"
        probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        final_answer_scoring: Dict = {}

        def _record_goal_event(step_idx: int, event: str, **extra: object) -> None:
            rec = {
                "step": int(step_idx),
                "event": event,
                "goal_stack": list(goal_stack),
                "goal": goal_stack[-1],
            }
            rec.update(extra)
            goal_stack_history.append(rec)

        def _push_goal(step_idx: int, new_goal: str, source: str) -> None:
            goal_stack.append(new_goal)
            _record_goal_event(step_idx, "push", source=source, pushed_goal=new_goal)

        def _replace_top_goal(step_idx: int, new_goal: str, source: str) -> None:
            prev_goal = goal_stack[-1]
            goal_stack[-1] = new_goal
            _record_goal_event(step_idx, "replace_top", source=source, previous_goal=prev_goal, new_goal=new_goal)

        def _pop_goal(step_idx: int, source: str) -> str:
            if len(goal_stack) <= 1:
                _record_goal_event(step_idx, "pop_skipped", source=source, reason="at_root")
                return goal_stack[-1]
            popped = goal_stack.pop()
            _record_goal_event(step_idx, "pop", source=source, popped_goal=popped)
            return popped

        def _finalize_entry(entry: Dict) -> None:
            entry.setdefault("goal_after", goal_stack[-1])
            entry.setdefault("goal_stack_after", list(goal_stack))

        def _score_final_answer(step_idx: int, current_goal_text: str, final_entry: Dict | None = None) -> tuple[str, Dict[str, float], Dict]:
            steps_for_prompt = solution_steps_log + ([final_entry] if final_entry is not None else [])
            final_answer_text = _extract_answer_text(final_entry or {})
            final_prompt = build_step_prompt(
                question_text=question_text,
                options_labeled=options_labeled,
                solution_steps_log=steps_for_prompt,
                current_goal=current_goal_text,
                working_memory=working_memory,
                recalled_dm_text=retrieved_dm_text,
                instruction_text="Final decision mode.",
            )
            if final_answer_text:
                final_prompt = final_prompt + f"\nFinal response: {final_answer_text}"
            final_prompt = final_prompt + "\nFinal Answer is :"

            base_label_logps = {
                "A": self.model.sequence_logprob(final_prompt, " A"),
                "B": self.model.sequence_logprob(final_prompt, " B"),
                "C": self.model.sequence_logprob(final_prompt, " C"),
                "D": self.model.sequence_logprob(final_prompt, " D"),
            }
            text_bias = _label_text_bias(final_answer_text, options_labeled)
            label_logps = {k: base_label_logps[k] + text_bias[k] for k in base_label_logps}

            if self.config.fail_on_nonfinite and not all(math.isfinite(v) for v in label_logps.values()):
                diag = {
                    "error": "non_finite_label_logprob",
                    "model_name": self.model.model_name,
                    "step": step_idx,
                    "goal": current_goal_text,
                    "working_memory": working_memory,
                    "label_logps": label_logps,
                    "base_label_logps": base_label_logps,
                    "text_bias": text_bias,
                    "prompt": final_prompt[: self.config.debug_prompt_char_limit],
                }
                raise RuntimeError(json.dumps(diag, ensure_ascii=True))

            final_probs = softmax_from_logps(label_logps)
            final_pred_label = max(label_logps, key=label_logps.get)
            diagnostics = {
                "final_answer_text": final_answer_text,
                "final_label_prompt": final_prompt,
                "final_label_base_logps": base_label_logps,
                "final_label_text_bias": text_bias,
                "final_label_logps": label_logps,
                "final_label_probs": final_probs,
            }
            return final_pred_label, final_probs, diagnostics

        for step in range(self.config.max_steps):
            current_goal = goal_stack[-1]
            recall_res = self.kb_server.query_topk(goal_text=current_goal, thinking_text=working_memory, k=3)
            top_items = list(recall_res.get("items") or [])
            top_item = top_items[0] if top_items else None
            recall_passed_threshold = (
                bool(top_item) and float(top_item.get("score", -1e9)) >= float(getattr(self.kb_server, "retrieval_threshold", 0.0))
            )
            recalled_items: List[Dict] = []
            if top_item is not None and (self.config.force_recall_each_step or recall_passed_threshold):
                recalled_items = top_items[:3]
            retrieved_dm_text = _format_recalled_dm_text(recalled_items)

            step_prompt = build_step_prompt(
                question_text=question_text,
                options_labeled=options_labeled,
                solution_steps_log=solution_steps_log,
                current_goal=current_goal,
                working_memory=working_memory,
                recalled_dm_text=retrieved_dm_text,
            )

            action_candidates = action_logit_candidates()
            raw_action_logps = {
                "<G>": self.model.sequence_logprob(step_prompt, action_candidates["<G>"]),
                "<R>": self.model.sequence_logprob(step_prompt, action_candidates["<R>"]),
                "<A>": self.model.sequence_logprob(step_prompt, action_candidates["<A>"]),
            }
            if self.config.fail_on_nonfinite and not all(math.isfinite(v) for v in raw_action_logps.values()):
                diag = {
                    "error": "non_finite_action_logprob",
                    "model_name": self.model.model_name,
                    "step": step,
                    "goal": current_goal,
                    "working_memory": working_memory,
                    "action_logps": raw_action_logps,
                    "prompt": step_prompt[: self.config.debug_prompt_char_limit],
                }
                raise RuntimeError(json.dumps(diag, ensure_ascii=True))
            action_bias = {"<G>": 0.0, "<R>": 0.0, "<A>": 0.0}
            if current_goal == "Solve the problem" and not solution_steps_log:
                # At the top goal, encourage explicit decomposition first.
                action_bias["<G>"] += 0.35
                action_bias["<R>"] -= 0.15
            if (
                current_goal == "Solve the problem"
                and len(solution_steps_log) >= 2
                and solution_steps_log[-1].get("action") == "<R>"
                and solution_steps_log[-2].get("action") == "<R>"
            ):
                # Prevent long top-goal reasoning loops; push toward decomposition.
                action_bias["<G>"] += 0.25
                action_bias["<R>"] -= 0.10
            if current_goal != "Solve the problem":
                # Once a concrete goal exists, prefer refining/answering over spawning another goal.
                action_bias["<G>"] -= 0.35
                action_bias["<R>"] += 0.15
            if solution_steps_log and solution_steps_log[-1].get("action") == "<G>":
                # Discourage repeated goal-generation streaks.
                action_bias["<G>"] -= 0.25
                action_bias["<R>"] += 0.10
                action_bias["<A>"] += 0.10
            action_logps = {k: raw_action_logps[k] + action_bias[k] for k in raw_action_logps}
            action_probs = softmax_from_logps(action_logps)
            action = max(action_logps, key=action_logps.get)

            content_prompt = build_action_content_prompt(
                question_text=question_text,
                options_labeled=options_labeled,
                current_goal=current_goal,
                working_memory=working_memory,
                recalled_dm_text=retrieved_dm_text,
                action=action,
            )
            forced_prefix = action_generation_prefix(action)
            raw_content = self.model.generate_after_prefix(content_prompt, forced_prefix, max_new_tokens=96)
            parsed_payload = {}
            try:
                content, parsed_payload = _parse_generated_action_output(action, forced_prefix, raw_content)
            except Exception:
                content = ""
                parsed_payload = {}

            entry = {
                "step": step,
                "action": action,
                "action_probs": action_probs,
                "raw_action_logps": raw_action_logps,
                "action_logps": action_logps,
                "action_bias": action_bias,
                "goal_before": current_goal,
                "goal_stack_before": list(goal_stack),
                "working_memory_before": working_memory,
                "recalled_dm_text": retrieved_dm_text,
                "recalled_dm_items": [
                    {
                        "dm_id": it.get("dm_id"),
                        "score": float(it.get("score", 0.0)),
                        "dm_text": it.get("dm_text", ""),
                    }
                    for it in recalled_items
                ],
                "recalled_score": float(top_item["score"]) if top_item is not None else None,
                "recalled_threshold_passed": bool(recall_passed_threshold),
                "recall_force_enabled": bool(self.config.force_recall_each_step),
                "chat_template_enabled": bool(getattr(self.model, "use_chat_template", False)),
                "chat_template_available": bool(getattr(self.model, "chat_template_available", False)),
                "content": content,
                "raw_content": raw_content,
                "parsed_payload": parsed_payload,
            }
            if self.config.log_prompts:
                entry["step_prompt"] = step_prompt

            if action == "<G>":
                if not content:
                    retry_hint = (
                        "invalid response. return valid JSON for action <G>.\n"
                        "Do NOT repeat Current Goal verbatim in next_goal.\n"
                        f"FORBIDDEN next_goal: {current_goal}\n"
                        "You must output a transformed NEW next_goal."
                    )
                    retry_raw, retry_content, retry_payload, retry_logs, resolved = _retry_with_invalid_hint(
                        model=self.model,
                        action=action,
                        forced_prefix=forced_prefix,
                        content_prompt=content_prompt,
                        attempts=self.config.noop_retry_attempts,
                        retry_hint=retry_hint,
                        max_new_tokens=96,
                    )
                    if not resolved:
                        cross_ok, cross_res = _try_cross_action_recovery(
                            self.model,
                            original_action=action,
                            question_text=question_text,
                            options_labeled=options_labeled,
                            current_goal=current_goal,
                            working_memory=working_memory,
                            recalled_dm_text=retrieved_dm_text,
                            attempts=2,
                            max_new_tokens=96,
                        )
                        entry["cross_action_retry_generation"] = cross_res.get("retry_logs", [])
                        if cross_ok and cross_res.get("recovered_action") == "<R>":
                            working_memory = cross_res["working_memory_after"]
                            entry["action_recovered_to"] = "<R>"
                            entry["content"] = cross_res["content"]
                            entry["raw_content"] = cross_res["raw_content"]
                            entry["parsed_payload"] = cross_res["parsed_payload"]
                            entry["working_memory_after"] = working_memory
                            _finalize_entry(entry)
                            solution_steps_log.append(entry)
                            continue
                        diag = {
                            "error": "invalid_subgoal_format",
                            "reason": "empty_subgoal_content_after_retries",
                            "model_name": self.model.model_name,
                            "step": step,
                            "diagnostic_entry": {
                                **entry,
                                "retry_generation": retry_logs,
                                "retry_exhausted": True,
                                "forced_prefix": forced_prefix,
                                "content_prompt": content_prompt[: self.config.debug_prompt_char_limit],
                            },
                        }
                        raise RuntimeError(json.dumps(diag, ensure_ascii=True))
                    raw_content = retry_raw
                    content = retry_content
                    parsed_payload = retry_payload
                    entry["content"] = content
                    entry["raw_content"] = raw_content
                    entry["parsed_payload"] = parsed_payload
                    entry["retry_generation"] = retry_logs
                subgoal = _sanitize_action_content(_extract_content_line(content))
                subgoal = _truncate_words(subgoal, 20)
                if subgoal and not subgoal.endswith("."):
                    subgoal = f"{subgoal}."
                if not subgoal:
                    cross_ok, cross_res = _try_cross_action_recovery(
                        self.model,
                        original_action=action,
                        question_text=question_text,
                        options_labeled=options_labeled,
                        current_goal=current_goal,
                        working_memory=working_memory,
                        recalled_dm_text=retrieved_dm_text,
                        attempts=2,
                        max_new_tokens=96,
                    )
                    entry["cross_action_retry_generation"] = cross_res.get("retry_logs", [])
                    if cross_ok and cross_res.get("recovered_action") == "<R>":
                        working_memory = cross_res["working_memory_after"]
                        entry["action_recovered_to"] = "<R>"
                        entry["content"] = cross_res["content"]
                        entry["raw_content"] = cross_res["raw_content"]
                        entry["parsed_payload"] = cross_res["parsed_payload"]
                        entry["working_memory_after"] = working_memory
                        _finalize_entry(entry)
                        solution_steps_log.append(entry)
                        continue
                    diag = {
                        "error": "invalid_subgoal_format",
                        "reason": "empty_subgoal_after_sanitize",
                        "model_name": self.model.model_name,
                        "step": step,
                        "diagnostic_entry": {
                            **entry,
                            "forced_prefix": forced_prefix,
                            "content_prompt": content_prompt[: self.config.debug_prompt_char_limit],
                        },
                    }
                    raise RuntimeError(json.dumps(diag, ensure_ascii=True))

                _push_goal(step, subgoal, "action_<G>")
                entry["goal_after"] = goal_stack[-1]
                goal_before_norm = _normalized_state_text(current_goal)
                goal_after_norm = _normalized_state_text(goal_stack[-1])
                if goal_before_norm == goal_after_norm:
                    retry_logs = []
                    resolved = False
                    retry_hint = (
                        "invalid response. you repeated the current goal and this is not allowed.\n"
                        "next_goal must be DIFFERENT from Current Goal with different wording.\n"
                        f"FORBIDDEN next_goal: {current_goal}\n"
                        "Generate a narrower transformed subgoal using a specific cue from Recalled text."
                    )
                    last_retry_raw = ""
                    for retry_idx in range(self.config.noop_retry_attempts):
                        retry_prompt = (
                            content_prompt
                            + "\n[RETRY]\n"
                            + retry_hint
                        )
                        retry_raw = self.model.generate_after_prefix(retry_prompt, forced_prefix, max_new_tokens=96)
                        last_retry_raw = retry_raw
                        try:
                            retry_content, retry_payload = _parse_generated_action_output(action, forced_prefix, retry_raw)
                        except Exception:
                            retry_content, retry_payload = "", {}
                        retry_subgoal = _truncate_words(retry_content, 20)
                        if retry_subgoal and not retry_subgoal.endswith("."):
                            retry_subgoal = f"{retry_subgoal}."
                        retry_logs.append(
                            {
                                "retry_index": retry_idx + 1,
                                "retry_raw_content": retry_raw,
                                "retry_content": retry_content,
                                "retry_payload": retry_payload,
                                "retry_subgoal": retry_subgoal,
                            }
                        )
                        if retry_subgoal and _normalized_state_text(retry_subgoal) != _normalized_state_text(current_goal):
                            _replace_top_goal(step, retry_subgoal, "retry_<G>")
                            entry["goal_after"] = retry_subgoal
                            resolved = True
                            break

                    if not resolved:
                        for chat_retry_idx in range(3):
                            chat_retry_prompt = (
                                content_prompt
                                + "\n[CHAT-TURN RETRY]\n"
                                + "Previous assistant response:\n"
                                + (last_retry_raw or "<empty>")
                                + "\nFeedback:\n"
                                + retry_hint
                                + "\nGoal must change from the forbidden goal.\n"
                                + "Return JSON only."
                            )
                            chat_retry_raw = self.model.generate_after_prefix(
                                chat_retry_prompt, forced_prefix, max_new_tokens=96
                            )
                            last_retry_raw = chat_retry_raw
                            try:
                                chat_retry_content, chat_retry_payload = _parse_generated_action_output(
                                    action, forced_prefix, chat_retry_raw
                                )
                            except Exception:
                                chat_retry_content, chat_retry_payload = "", {}
                            chat_retry_subgoal = _truncate_words(chat_retry_content, 20)
                            if chat_retry_subgoal and not chat_retry_subgoal.endswith("."):
                                chat_retry_subgoal = f"{chat_retry_subgoal}."
                            retry_logs.append(
                                {
                                    "retry_index": self.config.noop_retry_attempts + chat_retry_idx + 1,
                                    "retry_mode": "chat_feedback",
                                    "retry_raw_content": chat_retry_raw,
                                    "retry_content": chat_retry_content,
                                    "retry_payload": chat_retry_payload,
                                    "retry_subgoal": chat_retry_subgoal,
                                }
                            )
                            if chat_retry_subgoal and _normalized_state_text(chat_retry_subgoal) != _normalized_state_text(
                                current_goal
                            ):
                                _replace_top_goal(step, chat_retry_subgoal, "chat_retry_<G>")
                                entry["goal_after"] = chat_retry_subgoal
                                resolved = True
                                break

                    if not resolved:
                        # Goal no-op: after final retry, keep the same goal and continue.
                        entry["retry_generation"] = retry_logs
                        entry["retry_exhausted"] = True
                        entry["no_op_detected"] = True
                        entry["no_op_reason"] = "action_<G>_goal_unchanged"
                        _pop_goal(step, "noop_revert_goal_duplicate")
                        entry["goal_after"] = goal_stack[-1]
                    else:
                        entry["retry_generation"] = retry_logs
            elif action == "<R>":
                parsed_payload_valid = _is_valid_r_payload(parsed_payload)
                if not parsed_payload_valid:
                    content = ""
                    entry["content"] = ""

                if not content:
                    retry_hint = (
                        "invalid response. return valid JSON for action <R>.\n"
                        'Required key order: {"action":"<R>","reasoning":"...","next_thought":"..."}\n'
                        "Both reasoning and next_thought are mandatory.\n"
                        "next_thought must be non-empty, concrete, and not meta narration."
                    )
                    retry_raw, retry_content, retry_payload, retry_logs, resolved = _retry_with_invalid_hint(
                        model=self.model,
                        action=action,
                        forced_prefix=forced_prefix,
                        content_prompt=content_prompt,
                        attempts=self.config.noop_retry_attempts,
                        retry_hint=retry_hint,
                        is_valid=lambda _c, p: _is_valid_r_payload(p),
                        max_new_tokens=96,
                    )
                    if not resolved:
                        cross_ok, cross_res = _try_cross_action_recovery(
                            self.model,
                            original_action=action,
                            question_text=question_text,
                            options_labeled=options_labeled,
                            current_goal=current_goal,
                            working_memory=working_memory,
                            recalled_dm_text=retrieved_dm_text,
                            attempts=2,
                            max_new_tokens=96,
                        )
                        entry["cross_action_retry_generation"] = cross_res.get("retry_logs", [])
                        if cross_ok and cross_res.get("recovered_action") == "<G>":
                            _push_goal(step, cross_res["goal_after"], "cross_recovery_from_<R>_invalid")
                            entry["action_recovered_to"] = "<G>"
                            entry["content"] = cross_res["content"]
                            entry["raw_content"] = cross_res["raw_content"]
                            entry["parsed_payload"] = cross_res["parsed_payload"]
                            entry["goal_after"] = goal_stack[-1]
                            _finalize_entry(entry)
                            solution_steps_log.append(entry)
                            continue
                        diag = {
                            "error": "invalid_reasoning_format",
                            "reason": "empty_reasoning_content_after_retries",
                            "model_name": self.model.model_name,
                            "step": step,
                            "diagnostic_entry": {
                                **entry,
                                "retry_generation": retry_logs,
                                "retry_exhausted": True,
                                "forced_prefix": forced_prefix,
                                "content_prompt": content_prompt[: self.config.debug_prompt_char_limit],
                            },
                        }
                        raise RuntimeError(json.dumps(diag, ensure_ascii=True))
                    raw_content = retry_raw
                    content = retry_content
                    parsed_payload = retry_payload
                    parsed_payload_valid = _is_valid_r_payload(parsed_payload)
                    entry["content"] = content
                    entry["raw_content"] = raw_content
                    entry["parsed_payload"] = parsed_payload
                    entry["retry_generation"] = retry_logs

                if not parsed_payload_valid:
                    cross_ok, cross_res = _try_cross_action_recovery(
                        self.model,
                        original_action=action,
                        question_text=question_text,
                        options_labeled=options_labeled,
                        current_goal=current_goal,
                        working_memory=working_memory,
                        recalled_dm_text=retrieved_dm_text,
                        attempts=2,
                        max_new_tokens=96,
                    )
                    entry["cross_action_retry_generation"] = cross_res.get("retry_logs", [])
                    if cross_ok and cross_res.get("recovered_action") == "<G>":
                        _push_goal(step, cross_res["goal_after"], "cross_recovery_from_<R>_invalid_post_retry")
                        entry["action_recovered_to"] = "<G>"
                        entry["content"] = cross_res["content"]
                        entry["raw_content"] = cross_res["raw_content"]
                        entry["parsed_payload"] = cross_res["parsed_payload"]
                        entry["goal_after"] = goal_stack[-1]
                        _finalize_entry(entry)
                        solution_steps_log.append(entry)
                        continue
                    diag = {
                        "error": "invalid_reasoning_format",
                        "reason": "missing_required_reasoning_or_next_thought",
                        "model_name": self.model.model_name,
                        "step": step,
                        "diagnostic_entry": {
                            **entry,
                            "forced_prefix": forced_prefix,
                            "content_prompt": content_prompt[: self.config.debug_prompt_char_limit],
                        },
                    }
                    raise RuntimeError(json.dumps(diag, ensure_ascii=True))

                reasoning_text, next_thought_text = _extract_r_fields(parsed_payload)
                content = next_thought_text
                entry["content"] = content
                entry["parsed_payload"] = {"action": "<R>", "reasoning": reasoning_text, "next_thought": next_thought_text}
                working_memory_before = working_memory
                working_memory = _truncate_words(content, 60)

                entry["working_memory_after"] = working_memory
                wm_before_norm = _normalized_state_text(working_memory_before)
                wm_after_norm = _normalized_state_text(working_memory)
                if wm_before_norm == wm_after_norm:
                    retry_logs = []
                    resolved = False
                    retry_hint = (
                        "invalid response for <R>. next_thought is unchanged and this is not allowed.\n"
                        'Required key order: {"action":"<R>","reasoning":"...","next_thought":"..."}\n'
                        f"FORBIDDEN next_thought: {working_memory_before}\n"
                        "Provide a NEW concrete next_thought line different from the forbidden text."
                    )
                    last_retry_raw = ""
                    for retry_idx in range(self.config.noop_retry_attempts):
                        retry_prompt = (
                            content_prompt
                            + "\n[RETRY]\n"
                            + retry_hint
                        )
                        retry_raw = self.model.generate_after_prefix(retry_prompt, forced_prefix, max_new_tokens=96)
                        last_retry_raw = retry_raw
                        try:
                            retry_content, retry_payload = _parse_generated_action_output(action, forced_prefix, retry_raw)
                        except Exception:
                            retry_content, retry_payload = "", {}
                        retry_valid = _is_valid_r_payload(retry_payload)
                        retry_reasoning, retry_next = _extract_r_fields(retry_payload)
                        retry_wm = _truncate_words(retry_next, 60)
                        retry_logs.append(
                            {
                                "retry_index": retry_idx + 1,
                                "retry_raw_content": retry_raw,
                                "retry_content": retry_content,
                                "retry_payload": retry_payload,
                                "retry_working_memory": retry_wm,
                            }
                        )
                        if (
                            retry_valid
                            and retry_wm
                            and _normalized_state_text(retry_wm) != _normalized_state_text(working_memory_before)
                        ):
                            working_memory = retry_wm
                            entry["content"] = retry_next
                            entry["parsed_payload"] = {
                                "action": "<R>",
                                "reasoning": retry_reasoning,
                                "next_thought": retry_next,
                            }
                            entry["working_memory_after"] = working_memory
                            resolved = True
                            break

                    if not resolved:
                        for chat_retry_idx in range(3):
                            chat_retry_prompt = (
                                content_prompt
                                + "\n[CHAT-TURN RETRY]\n"
                                + "Previous assistant response:\n"
                                + (last_retry_raw or "<empty>")
                                + "\nFeedback:\n"
                                + retry_hint
                                + "\nReasoning must change from the forbidden text.\n"
                                + "Return JSON only."
                            )
                            chat_retry_raw = self.model.generate_after_prefix(
                                chat_retry_prompt, forced_prefix, max_new_tokens=96
                            )
                            last_retry_raw = chat_retry_raw
                            try:
                                chat_retry_content, chat_retry_payload = _parse_generated_action_output(
                                    action, forced_prefix, chat_retry_raw
                                )
                            except Exception:
                                chat_retry_content, chat_retry_payload = "", {}
                            chat_retry_valid = _is_valid_r_payload(chat_retry_payload)
                            chat_retry_reasoning, chat_retry_next = _extract_r_fields(chat_retry_payload)
                            chat_retry_wm = _truncate_words(chat_retry_next, 60)
                            retry_logs.append(
                                {
                                    "retry_index": self.config.noop_retry_attempts + chat_retry_idx + 1,
                                    "retry_mode": "chat_feedback",
                                    "retry_raw_content": chat_retry_raw,
                                    "retry_content": chat_retry_content,
                                    "retry_payload": chat_retry_payload,
                                    "retry_working_memory": chat_retry_wm,
                                }
                            )
                            if (
                                chat_retry_valid
                                and chat_retry_wm
                                and _normalized_state_text(chat_retry_wm) != _normalized_state_text(working_memory_before)
                            ):
                                working_memory = chat_retry_wm
                                entry["content"] = chat_retry_next
                                entry["parsed_payload"] = {
                                    "action": "<R>",
                                    "reasoning": chat_retry_reasoning,
                                    "next_thought": chat_retry_next,
                                }
                                entry["working_memory_after"] = working_memory
                                resolved = True
                                break

                    if not resolved:
                        cross_ok, cross_res = _try_cross_action_recovery(
                            self.model,
                            original_action=action,
                            question_text=question_text,
                            options_labeled=options_labeled,
                            current_goal=current_goal,
                            working_memory=working_memory,
                            recalled_dm_text=retrieved_dm_text,
                            attempts=2,
                            max_new_tokens=96,
                        )
                        entry["cross_action_retry_generation"] = cross_res.get("retry_logs", [])
                        if cross_ok and cross_res.get("recovered_action") == "<G>":
                            _push_goal(step, cross_res["goal_after"], "cross_recovery_from_<R>_noop")
                            entry["action_recovered_to"] = "<G>"
                            entry["content"] = cross_res["content"]
                            entry["raw_content"] = cross_res["raw_content"]
                            entry["parsed_payload"] = cross_res["parsed_payload"]
                            entry["goal_after"] = goal_stack[-1]
                            _finalize_entry(entry)
                            solution_steps_log.append(entry)
                            continue
                        # No-op on <R>: keep current working memory and continue.
                        working_memory = working_memory_before
                        entry["working_memory_after"] = working_memory
                        entry["retry_generation"] = retry_logs
                        entry["retry_exhausted"] = True
                        entry["no_op_detected"] = True
                        entry["no_op_reason"] = "action_<R>_reasoning_unchanged"
                    else:
                        entry["retry_generation"] = retry_logs
            else:
                if current_goal != "Solve the problem":
                    popped_goal = _pop_goal(step, "action_<A>_subgoal_complete")
                    working_memory = f"Subgoal completed: {popped_goal}"
                    entry["subgoal_answer"] = _truncate_words(content, 30)
                    entry["working_memory_after"] = working_memory
                else:
                    pred_label, probs, final_diag = _score_final_answer(step, current_goal, entry)
                    final_answer_scoring = final_diag
                    entry.update(final_diag)
                    termination = "answered_top_goal"
                    _finalize_entry(entry)
                    solution_steps_log.append(entry)
                    break

            _finalize_entry(entry)
            solution_steps_log.append(entry)

        # No implicit uniform fallback: force final answer scoring at max_steps.
        if termination == "max_steps":
            current_goal = goal_stack[-1]
            pred_label, probs, final_diag = _score_final_answer(self.config.max_steps, current_goal)
            final_answer_scoring = final_diag
            termination = "forced_answer_at_max_steps"

        is_correct = pred_label == correct_label

        return {
            "model_name": self.model.model_name,
            "pred_label": pred_label,
            "probs": probs,
            "p_correct": probs.get(correct_label, 0.0),
            "is_correct": is_correct,
            "solution_steps_log": solution_steps_log,
            "termination": termination,
            "final_goal_stack": list(goal_stack),
            "final_goal_stack_history": goal_stack_history,
            "final_answer_scoring": final_answer_scoring,
            "final_working_memory": working_memory,
        }
