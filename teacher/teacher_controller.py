"""Teacher controller for DM distillation."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from urllib import error, request

from teacher.teacher_prompt import (
    TEACHER_MCP_PLANNER_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    build_teacher_tool_planner_prompt,
    build_teacher_user_prompt,
)


@dataclass
class TeacherConfig:
    endpoint: str = "http://127.0.0.1:8080/v1/chat/completions"
    model: str = "teacher"
    temperature: float = 0.0
    max_new_dm_per_teacher_call: int = 20
    max_mcp_query_topk_calls: int = 8
    max_mcp_score_calls: int = 60
    allow_heuristic_fallback: bool = False
    max_teacher_input_chars: int = 50000
    max_teacher_input_chars_retry: int = 15000
    max_steps_per_ca_run_for_teacher: int = 8
    request_timeout_s: int = 300
    max_tokens: int = 512
    request_retries: int = 2
    retry_backoff_s: float = 2.0
    enable_tool_planner: bool = False
    debug_log_dir: str = ""
    bootstrap_max_new_dm_per_teacher_call: int = 8


class TeacherController:
    def __init__(self, kb_server, config: TeacherConfig) -> None:
        self.kb_server = kb_server
        self.config = config

    @staticmethod
    def _recover_candidates_from_truncated_output(text: str) -> Dict | None:
        s = str(text or "")
        key_idx = s.find('"candidates"')
        if key_idx < 0:
            return None
        arr_start = s.find("[", key_idx)
        if arr_start < 0:
            return None

        candidates: List[Dict] = []
        depth = 0
        obj_start = -1
        in_str = False
        esc = False
        for i in range(arr_start, len(s)):
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
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and obj_start >= 0:
                        chunk = s[obj_start : i + 1]
                        try:
                            obj = json.loads(chunk)
                            if isinstance(obj, dict):
                                candidates.append(obj)
                        except Exception:
                            pass

        if not candidates:
            return None
        return {"candidates": candidates}

    @staticmethod
    def _extract_first_json_object(text: str) -> Dict:
        s = (text or "").strip()
        if not s:
            raise json.JSONDecodeError("empty content", s, 0)

        # Normalize common formatting noise.
        s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        s = re.sub(r"^\s*json\s*", "", s, flags=re.IGNORECASE)

        # Direct parse first.
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # Strip fenced blocks like ```json ... ```
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            return json.loads(fence.group(1))

        # Try first explicit JSON object region (non-greedy).
        obj_match = re.search(r"\{.*?\}", s, flags=re.DOTALL)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except Exception:
                pass

        # Find first balanced {...} region.
        start = s.find("{")
        if start == -1:
            raise json.JSONDecodeError("no json object start", s, 0)
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[start : i + 1]
                    return json.loads(chunk)
        recovered = TeacherController._recover_candidates_from_truncated_output(s)
        if recovered is not None:
            return recovered
        raise json.JSONDecodeError("no balanced json object", s, 0)

    def _write_debug_failure_log(self, tag: str, payload: Dict) -> None:
        if not self.config.debug_log_dir:
            return
        try:
            os.makedirs(self.config.debug_log_dir, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            out_path = os.path.join(self.config.debug_log_dir, f"teacher_{tag}_{ts}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
                f.write("\n")
        except Exception:
            # Never mask primary teacher failure with debug logging errors.
            pass

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _compact_teacher_input(self, teacher_input: Dict, aggressive: bool = False) -> Dict:
        out = dict(teacher_input)
        ca_runs = out.get("ca_runs", []) or []
        compact_runs = []
        per_run_steps = 3 if aggressive else self.config.max_steps_per_ca_run_for_teacher
        for run in ca_runs:
            if not isinstance(run, dict):
                continue
            steps = run.get("solution_steps_log", []) or []
            steps = steps[-per_run_steps:]
            compact_steps = []
            for s in steps:
                if not isinstance(s, dict):
                    continue
                action = str(s.get("action", ""))
                action_probs = s.get("action_probs", {}) if isinstance(s.get("action_probs", {}), dict) else {}
                action_conf = action_probs.get(action) if action in action_probs else None
                compact_steps.append(
                    {
                        "step": s.get("step"),
                        "action": action,
                        "action_confidence": action_conf,
                        "goal_before": self._truncate_text(str(s.get("goal_before", "")), 120),
                        "working_memory_before": self._truncate_text(str(s.get("working_memory_before", "")), 240),
                        "retrieved_dm_text": self._truncate_text(str(s.get("recalled_dm_text", "")), 320),
                        "content": self._truncate_text(str(s.get("content", "")), 240),
                        "working_memory_after": self._truncate_text(str(s.get("working_memory_after", "")), 240),
                        "goal_after": self._truncate_text(str(s.get("goal_after", "")), 120),
                    }
                )
            compact_runs.append(
                {
                    "model_name": run.get("model_name", ""),
                    "pred_label": run.get("pred_label", ""),
                    "is_correct": bool(run.get("is_correct", False)),
                    "p_correct": run.get("p_correct", 0.0),
                    "termination": run.get("termination", ""),
                    "solution_steps_log": compact_steps,
                }
            )
        out["ca_runs"] = compact_runs
        out["question_text"] = self._truncate_text(str(out.get("question_text", "")), 1200)
        out["gold_explanation"] = self._truncate_text(str(out.get("gold_explanation", "")), 1600)
        out["cognitive_agent_mechanics"] = self._truncate_text(str(out.get("cognitive_agent_mechanics", "")), 1600)
        out["previous_new_dm_candidates_in_this_problem"] = (out.get("previous_new_dm_candidates_in_this_problem", []) or [])[
            -20:
        ]
        budget = self.config.max_teacher_input_chars_retry if aggressive else self.config.max_teacher_input_chars
        serialized = json.dumps(out, ensure_ascii=True)
        if len(serialized) > budget:
            # Keep only incorrect runs first under tight budget.
            incorrect = [r for r in compact_runs if not r.get("is_correct", False)]
            out["ca_runs"] = incorrect[:8]
            serialized = json.dumps(out, ensure_ascii=True)
            if len(serialized) > budget:
                out["ca_runs"] = out["ca_runs"][:4]
        return out

    def _chat_json(self, system_prompt: str, user_prompt: str, timeout_s: int) -> Dict:
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                err_body = "<no-body>"
            raise RuntimeError(f"Teacher HTTP {exc.code} {exc.reason}: {err_body}") from exc
        obj = json.loads(body)
        choice0 = obj["choices"][0]
        finish_reason = str(choice0.get("finish_reason", ""))
        msg = choice0["message"]
        content = msg.get("content", "")
        if isinstance(content, list):
            # OpenAI-style content parts: [{"type":"text","text":"..."}]
            parts = []
            for p in content:
                if isinstance(p, dict):
                    parts.append(str(p.get("text", "")))
            content = "\n".join(parts)
        content_text = str(content)
        try:
            parsed = self._extract_first_json_object(content_text)
        except Exception as exc:
            snippet = content_text[:1200]
            raise RuntimeError(
                "teacher_json_parse_failed: "
                f"{exc}; finish_reason={finish_reason or 'unknown'}; "
                f"content_len={len(content_text)}; content_snippet={snippet}"
            ) from exc
        return parsed

    def _call_teacher_server(self, teacher_input: Dict, max_new_dm_per_call_override: int | None = None) -> Dict:
        max_new = (
            int(max_new_dm_per_call_override)
            if max_new_dm_per_call_override is not None
            else int(self.config.max_new_dm_per_teacher_call)
        )
        return self._chat_json(
            system_prompt=TEACHER_SYSTEM_PROMPT,
            user_prompt=build_teacher_user_prompt(
                teacher_input=teacher_input,
                max_new_dm_per_call=max_new,
            ),
            timeout_s=self.config.request_timeout_s,
        )

    def _call_teacher_tool_planner(self, teacher_input: Dict) -> Dict:
        return self._chat_json(
            system_prompt=TEACHER_MCP_PLANNER_PROMPT,
            user_prompt=build_teacher_tool_planner_prompt(
                teacher_input=teacher_input,
                max_query_topk_calls=self.config.max_mcp_query_topk_calls,
                max_score_calls=self.config.max_mcp_score_calls,
            ),
            timeout_s=max(20, int(self.config.request_timeout_s // 2)),
        )

    def _execute_tool_requests(self, tool_plan: Dict) -> Dict:
        requests = tool_plan.get("tool_requests", [])
        if not isinstance(requests, list):
            requests = []

        query_calls = 0
        score_calls = 0
        executed = []
        for req in requests:
            if not isinstance(req, dict):
                continue
            tool = req.get("tool")
            args = req.get("args", {}) if isinstance(req.get("args", {}), dict) else {}
            try:
                if tool == "kb.query_topk":
                    if query_calls >= self.config.max_mcp_query_topk_calls:
                        continue
                    out = self.kb_server.query_topk(
                        goal_text=str(args.get("goal_text", "")),
                        thinking_text=str(args.get("thinking_text", "")),
                        k=int(args.get("k", 5)),
                    )
                    query_calls += 1
                    executed.append({"tool": tool, "args": args, "output": out})
                elif tool == "kb.score":
                    if score_calls >= self.config.max_mcp_score_calls:
                        continue
                    out = self.kb_server.score(
                        goal_text=str(args.get("goal_text", "")),
                        thinking_text=str(args.get("thinking_text", "")),
                        candidate_dm_goal_text=str(args.get("candidate_dm_goal_text", "")),
                        candidate_dm_condition_text=str(args.get("candidate_dm_condition_text", "")),
                    )
                    score_calls += 1
                    executed.append({"tool": tool, "args": args, "output": out})
            except Exception as exc:
                executed.append({"tool": tool, "args": args, "error": str(exc)})

        return {
            "tool_calls": {
                "kb.query_topk": query_calls,
                "kb.score": score_calls,
            },
            "executed_requests": executed,
        }

    def _extract_failure_queries(self, teacher_input: Dict, max_queries: int = 6) -> List[Dict]:
        queries: List[Dict] = []
        for run in teacher_input.get("ca_runs", []):
            if run.get("is_correct", False):
                continue
            steps = run.get("solution_steps_log", []) or []
            if steps:
                last = steps[-1]
                goal_text = (last.get("goal_before") or "Solve the problem").strip()
                thinking_text = (last.get("working_memory_before") or "").strip()
            else:
                goal_text = "Solve the problem"
                thinking_text = ""
            queries.append({"goal_text": goal_text, "thinking_text": thinking_text})
            if len(queries) >= max_queries:
                break
        if not queries:
            queries.append({"goal_text": "Solve the problem", "thinking_text": ""})
        return queries

    def _collect_kb_tool_observations(self, teacher_input: Dict) -> Dict:
        queries = self._extract_failure_queries(teacher_input)
        query_topk_results = []
        for q in queries:
            topk = self.kb_server.query_topk(
                goal_text=q["goal_text"],
                thinking_text=q["thinking_text"],
                k=3,
            )
            query_topk_results.append({"query": q, "topk": topk.get("items", [])})
        return {
            "tool_calls": {
                "kb.query_topk": len(query_topk_results),
                "kb.score": 0,
            },
            "failure_queries": queries,
            "query_topk_results": query_topk_results,
        }

    def _heuristic_fallback(self, teacher_input: Dict) -> Dict:
        explanation = (teacher_input.get("gold_explanation") or "").strip()
        if not explanation:
            explanation = "Use elimination by checking defining biological properties in each option."
        pieces = [p.strip() for p in re.split(r"[.;]", explanation) if p.strip()]
        candidates: List[Dict] = [
            {
                "dm_text": "Before selecting an answer, explicitly compare every option using the same biological criterion.",
                "dm_goal_text": "Apply a consistent comparison strategy across all options.",
                "dm_condition_text": "The working memory is uncertain or only partially compares options.",
            },
            {
                "dm_text": "Update working memory with intermediate elimination results so the final choice reflects all checked options.",
                "dm_goal_text": "Keep a running elimination summary while solving the problem.",
                "dm_condition_text": "The trace shows missing or unstable intermediate conclusions.",
            },
        ]
        for p in pieces:
            candidates.append(
                {
                    "dm_text": p,
                    "dm_goal_text": "Recall a compact biology fact required for option discrimination.",
                    "dm_condition_text": "A specific biology fact is needed to separate similar options.",
                }
            )
            if len(candidates) >= self.config.max_new_dm_per_teacher_call:
                break
        return {"candidates": candidates[: self.config.max_new_dm_per_teacher_call]}

    def _postprocess_candidates(self, teacher_input: Dict, proposed: Dict) -> List[Dict]:
        raw = proposed.get("candidates", [])
        if not isinstance(raw, list):
            raw = []
        cleaned: List[Dict] = []
        for cand in raw:
            if not isinstance(cand, dict):
                continue
            dm_text = (cand.get("dm_text") or "").strip()
            dm_goal_text = (cand.get("dm_goal_text") or "").strip()
            dm_condition_text = (cand.get("dm_condition_text") or "").strip()
            if not dm_text or not dm_goal_text or not dm_condition_text:
                continue
            cleaned.append(
                {
                    "dm_text": dm_text,
                    "dm_goal_text": dm_goal_text,
                    "dm_condition_text": dm_condition_text,
                }
            )
            if len(cleaned) >= self.config.max_new_dm_per_teacher_call:
                break

        # If failures keep repeating and teacher produced too few DMs, supplement with fallback decomposition.
        if self.config.allow_heuristic_fallback and int(teacher_input.get("attempt_iter", 0)) >= 2 and len(cleaned) < 3:
            fallback = self._heuristic_fallback(teacher_input).get("candidates", [])
            for cand in fallback:
                if len(cleaned) >= self.config.max_new_dm_per_teacher_call:
                    break
                cleaned.append(cand)
        return cleaned[: self.config.max_new_dm_per_teacher_call]

    def _score_candidates_with_kb(self, teacher_input: Dict, candidates: List[Dict]) -> Tuple[List[Dict], int]:
        queries = self._extract_failure_queries(teacher_input)
        scored: List[Dict] = []
        score_calls = 0
        for cand in candidates:
            best_score = -1e9
            best_breakdown = None
            for q in queries:
                out = self.kb_server.score(
                    goal_text=q["goal_text"],
                    thinking_text=q["thinking_text"],
                    candidate_dm_goal_text=cand["dm_goal_text"],
                    candidate_dm_condition_text=cand["dm_condition_text"],
                )
                score_calls += 1
                if out["score"] > best_score:
                    best_score = out["score"]
                    best_breakdown = {
                        "query": q,
                        "score": out["score"],
                        "x_goal": out["x_goal"],
                        "y_condition": out["y_condition"],
                    }
            item = dict(cand)
            item["_retrieval_score"] = float(best_score)
            item["_retrieval_debug"] = best_breakdown
            scored.append(item)

        scored.sort(key=lambda x: x["_retrieval_score"], reverse=True)
        min_keep_score = max(0.0, float(getattr(self.kb_server, "retrieval_threshold", 0.30)) - 0.05)
        filtered = [c for c in scored if c["_retrieval_score"] >= min_keep_score]
        if not filtered:
            filtered = scored[: max(1, min(3, len(scored)))]
        trimmed = filtered[: self.config.max_new_dm_per_teacher_call]
        for c in trimmed:
            c.pop("_retrieval_debug", None)
        return trimmed, score_calls

    def distill_candidates(self, teacher_input: Dict) -> Dict:
        teacher_input_augmented = self._compact_teacher_input(teacher_input, aggressive=False)
        kb_obs = self._collect_kb_tool_observations(teacher_input_augmented)
        teacher_input_augmented["kb_tool_observations"] = kb_obs

        planned_obs = {"tool_calls": {"kb.query_topk": 0, "kb.score": 0}, "executed_requests": []}
        if self.config.enable_tool_planner:
            try:
                tool_plan = self._call_teacher_tool_planner(teacher_input_augmented)
                planned_obs = self._execute_tool_requests(tool_plan)
                teacher_input_augmented["kb_tool_observations"]["planned_tool_execution"] = planned_obs
            except Exception:
                pass

        proposed = None
        errors: List[str] = []
        requested_max_candidates_by_attempt: List[int] = []
        attempts = max(1, int(self.config.request_retries))
        bootstrap_stage = str(teacher_input.get("bootstrap_stage", "")).strip().lower()
        is_pre_ca_bootstrap = bootstrap_stage == "pre_ca_seed_kb"
        for i in range(attempts):
            requested_max = int(self.config.max_new_dm_per_teacher_call)
            if is_pre_ca_bootstrap:
                requested_max = min(requested_max, int(self.config.bootstrap_max_new_dm_per_teacher_call))
            if i > 0:
                requested_max = max(3, requested_max // (2**i))
            requested_max_candidates_by_attempt.append(requested_max)
            try:
                if i == 0:
                    proposed = self._call_teacher_server(
                        teacher_input_augmented,
                        max_new_dm_per_call_override=requested_max,
                    )
                else:
                    teacher_input_retry = self._compact_teacher_input(teacher_input, aggressive=True)
                    kb_obs_retry = self._collect_kb_tool_observations(teacher_input_retry)
                    teacher_input_retry["kb_tool_observations"] = kb_obs_retry
                    proposed = self._call_teacher_server(
                        teacher_input_retry,
                        max_new_dm_per_call_override=requested_max,
                    )
                    teacher_input_augmented = teacher_input_retry
                    kb_obs = kb_obs_retry
                    planned_obs = {"tool_calls": {"kb.query_topk": 0, "kb.score": 0}, "executed_requests": []}
                break
            except Exception as exc:
                errors.append(str(exc))
                if i < attempts - 1:
                    time.sleep(self.config.retry_backoff_s * (i + 1))

        if proposed is None:
            self._write_debug_failure_log(
                tag="distill_failure",
                payload={
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "errors": errors,
                    "attempts": attempts,
                    "requested_max_candidates_by_attempt": requested_max_candidates_by_attempt,
                    "teacher_input": teacher_input_augmented,
                    "config": {
                        "model": self.config.model,
                        "endpoint": self.config.endpoint,
                        "request_timeout_s": self.config.request_timeout_s,
                        "request_retries": self.config.request_retries,
                        "retry_backoff_s": self.config.retry_backoff_s,
                        "max_tokens": self.config.max_tokens,
                        "max_new_dm_per_teacher_call": self.config.max_new_dm_per_teacher_call,
                        "bootstrap_max_new_dm_per_teacher_call": self.config.bootstrap_max_new_dm_per_teacher_call,
                        "enable_tool_planner": self.config.enable_tool_planner,
                    },
                },
            )
            if not self.config.allow_heuristic_fallback:
                raise RuntimeError(
                    "Teacher LLM request failed and heuristic fallback is disabled. "
                    f"Errors: {' | '.join(errors)}"
                )
            proposed = self._heuristic_fallback(teacher_input_augmented)

        candidates = self._postprocess_candidates(teacher_input_augmented, proposed)
        candidates, score_calls = self._score_candidates_with_kb(teacher_input_augmented, candidates)
        return {
            "proposed": {"candidates": candidates},
            "mcp_usage": {
                "kb.query_topk_calls": (
                    kb_obs["tool_calls"]["kb.query_topk"] + planned_obs["tool_calls"]["kb.query_topk"]
                ),
                "kb.score_calls": score_calls + planned_obs["tool_calls"]["kb.score"],
            },
            "kb_tool_observations": teacher_input_augmented["kb_tool_observations"],
        }

    def distill_and_commit(self, teacher_input: Dict) -> Dict:
        distilled = self.distill_candidates(teacher_input)
        candidates = distilled["proposed"]["candidates"]
        commit_payload = []
        for cand in candidates:
            commit_payload.append(
                {
                    "dm_text": cand.get("dm_text", ""),
                    "dm_goal_text": cand.get("dm_goal_text", ""),
                    "dm_condition_text": cand.get("dm_condition_text", ""),
                    "metadata": {
                        "source_problem_id": teacher_input.get("problem_id", ""),
                        "epoch": teacher_input.get("epoch", 0),
                        "problem_iteration": teacher_input.get("attempt_iter", 0),
                        "created_by": "teacher_llm",
                    },
                }
            )

        commit_result = self.kb_server.commit_dm_candidates(commit_payload)
        out = dict(distilled)
        out["commit_result"] = commit_result
        return out
