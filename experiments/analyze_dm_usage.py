"""Analyze DM retrieval usage from attempt_logs.jsonl.

Outputs per-DM statistics based on:
- retrieval counts in final-pass problem traces
- first definition problem index
- post-definition usage probability per problem
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple


def _normalize_text(text: str) -> str:
    t = str(text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _loose_text_key(text: str) -> str:
    t = _normalize_text(text)
    # Keep only letters/digits to absorb minor punctuation/quote corruption in logs.
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t


def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                obj["_line_no"] = line_no
                yield obj


def _pick_output_path(attempt_log_path: str, output_path: str) -> str:
    if output_path:
        return output_path
    base_dir = os.path.dirname(os.path.abspath(attempt_log_path))
    return os.path.join(base_dir, "dm_usage_summary.json")


def _extract_defined_dm_texts(record: Dict) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for source_key in ("teacher_preseed_dm_candidates_added", "teacher_dm_candidates_added"):
        cands = record.get(source_key, [])
        if not isinstance(cands, list):
            continue
        for c in cands:
            if not isinstance(c, dict):
                continue
            dm_text = str(c.get("dm_text", "")).strip()
            if not dm_text:
                continue
            out.append((source_key, dm_text))
    return out


def _extract_recalled_dm_texts_from_step(step: Dict) -> List[str]:
    items = step.get("recalled_dm_items", [])
    texts: List[str] = []
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            txt = str(it.get("dm_text", "")).strip()
            if txt:
                texts.append(txt)
    if texts:
        return texts

    # Fallback: parse numbered recalled text block when recalled_dm_items is missing/broken.
    block = str(step.get("recalled_dm_text", "")).strip()
    if not block:
        return texts
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+\.\s*", "", line)
        line = line.strip()
        if line:
            texts.append(line)
    return texts


def analyze_attempt_log(attempt_log_path: str) -> Dict:
    records = list(_iter_jsonl(attempt_log_path))
    if not records:
        return {
            "input_attempt_log": os.path.abspath(attempt_log_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "num_records": 0,
            "message": "No records found.",
            "dm_stats": [],
        }

    # Track latest per-problem pass attempt (final pass version).
    final_pass_by_problem: Dict[Tuple[int, int], Dict] = {}
    latest_problem_idx = -1
    for rec in records:
        if bool(rec.get("skipped", False)):
            continue
        pidx = int(rec.get("problem_idx", -1))
        if pidx > latest_problem_idx:
            latest_problem_idx = pidx
        if not bool(rec.get("pass", False)):
            continue
        key = (int(rec.get("epoch", 0)), pidx)
        prev = final_pass_by_problem.get(key)
        if prev is None or int(rec.get("attempt_iter", 0)) > int(prev.get("attempt_iter", 0)):
            final_pass_by_problem[key] = rec

    # Definition metadata (string-mapped by normalized dm_text).
    dm_def: Dict[str, Dict] = {}
    for rec in records:
        epoch = int(rec.get("epoch", 0))
        pidx = int(rec.get("problem_idx", -1))
        aiter = int(rec.get("attempt_iter", 0))
        for source, dm_text in _extract_defined_dm_texts(rec):
            key = _normalize_text(dm_text)
            if not key:
                continue
            ent = dm_def.setdefault(
                key,
                {
                    "dm_text": dm_text,
                    "dm_key": key,
                    "first_defined_epoch": epoch,
                    "first_defined_problem_idx": pidx,
                    "first_defined_attempt_iter": aiter,
                    "definition_events": [],
                },
            )
            # Keep earliest definition point.
            old_marker = (
                int(ent["first_defined_epoch"]),
                int(ent["first_defined_problem_idx"]),
                int(ent["first_defined_attempt_iter"]),
            )
            new_marker = (epoch, pidx, aiter)
            if new_marker < old_marker:
                ent["first_defined_epoch"] = epoch
                ent["first_defined_problem_idx"] = pidx
                ent["first_defined_attempt_iter"] = aiter
                ent["dm_text"] = dm_text
            ent["definition_events"].append(
                {
                    "epoch": epoch,
                    "problem_idx": pidx,
                    "attempt_iter": aiter,
                    "source": source,
                    "line_no": int(rec.get("_line_no", -1)),
                }
            )

    # String-mapping fallback index for retrieval text recovery.
    loose_to_defined_keys: Dict[str, set] = defaultdict(set)
    for strict_key in dm_def:
        loose = _loose_text_key(strict_key)
        if loose:
            loose_to_defined_keys[loose].add(strict_key)

    def _resolve_retrieval_key(text: str) -> str:
        strict = _normalize_text(text)
        if not strict:
            return ""
        if strict in dm_def:
            return strict
        loose = _loose_text_key(strict)
        if loose:
            cands = loose_to_defined_keys.get(loose, set())
            if len(cands) == 1:
                return next(iter(cands))
        return strict

    # Retrieval stats from final-pass traces.
    retrieval_count_final = defaultdict(int)
    retrieval_by_problem: Dict[str, set] = defaultdict(set)  # dm_key -> set[(epoch,problem_idx)]
    remapped_retrieval_events = 0
    for (epoch, pidx), rec in final_pass_by_problem.items():
        ca_runs = rec.get("ca_outputs", [])
        if not isinstance(ca_runs, list):
            continue
        for run in ca_runs:
            if not isinstance(run, dict):
                continue
            steps = run.get("solution_steps_log", [])
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                for recalled_text in _extract_recalled_dm_texts_from_step(step):
                    key = _resolve_retrieval_key(recalled_text)
                    if not key:
                        continue
                    if key != _normalize_text(recalled_text):
                        remapped_retrieval_events += 1
                    retrieval_count_final[key] += 1
                    retrieval_by_problem[key].add((epoch, pidx))

    all_dm_keys = set(dm_def.keys()) | set(retrieval_count_final.keys())
    dm_stats: List[Dict] = []
    for key in all_dm_keys:
        def_meta = dm_def.get(key, {})
        dm_text = def_meta.get("dm_text", key)
        first_defined_problem_idx = def_meta.get("first_defined_problem_idx", None)
        total_final_retrievals = int(retrieval_count_final.get(key, 0))
        used_problem_pairs = retrieval_by_problem.get(key, set())

        used_problem_count_final = len(used_problem_pairs)
        retrieval_after_def = None
        used_problem_count_after_def = None
        solved_problem_count_after_def = None
        per_problem_call_probability = None
        avg_calls_per_problem_after_def = None

        if first_defined_problem_idx is not None and latest_problem_idx >= int(first_defined_problem_idx):
            solved_problem_count_after_def = max(0, int(latest_problem_idx) - int(first_defined_problem_idx))
            retrieval_after_def = 0
            used_after_def = set()
            for (epoch, pidx) in used_problem_pairs:
                if int(pidx) > int(first_defined_problem_idx):
                    used_after_def.add((epoch, pidx))
            used_problem_count_after_def = len(used_after_def)

            # Count events after definition using final-pass events only.
            for (epoch, pidx), rec in final_pass_by_problem.items():
                if int(pidx) <= int(first_defined_problem_idx):
                    continue
                ca_runs = rec.get("ca_outputs", [])
                if not isinstance(ca_runs, list):
                    continue
                for run in ca_runs:
                    if not isinstance(run, dict):
                        continue
                    steps = run.get("solution_steps_log", [])
                    if not isinstance(steps, list):
                        continue
                    for step in steps:
                        if not isinstance(step, dict):
                            continue
                        for recalled_text in _extract_recalled_dm_texts_from_step(step):
                            if _resolve_retrieval_key(recalled_text) == key:
                                retrieval_after_def += 1

            if solved_problem_count_after_def > 0:
                per_problem_call_probability = used_problem_count_after_def / solved_problem_count_after_def
                avg_calls_per_problem_after_def = retrieval_after_def / solved_problem_count_after_def

        dm_stats.append(
            {
                "dm_key": key,
                "dm_text": dm_text,
                "first_defined_epoch": def_meta.get("first_defined_epoch", None),
                "first_defined_problem_idx": first_defined_problem_idx,
                "first_defined_attempt_iter": def_meta.get("first_defined_attempt_iter", None),
                "definition_event_count": len(def_meta.get("definition_events", [])),
                "retrieval_count_in_final_pass_versions": total_final_retrievals,
                "retrieved_problem_count_in_final_pass_versions": used_problem_count_final,
                "latest_problem_idx": latest_problem_idx,
                "solved_problem_count_after_definition": solved_problem_count_after_def,
                "retrieval_count_after_definition_in_final_pass_versions": retrieval_after_def,
                "retrieved_problem_count_after_definition_in_final_pass_versions": used_problem_count_after_def,
                "per_problem_call_probability_after_definition": per_problem_call_probability,
                "avg_calls_per_problem_after_definition": avg_calls_per_problem_after_def,
            }
        )

    dm_stats.sort(
        key=lambda x: (
            int(x.get("retrieval_count_in_final_pass_versions", 0)),
            int(x.get("retrieved_problem_count_in_final_pass_versions", 0)),
        ),
        reverse=True,
    )

    return {
        "input_attempt_log": os.path.abspath(attempt_log_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_records": len(records),
        "num_final_pass_problem_instances": len(final_pass_by_problem),
        "latest_problem_idx": latest_problem_idx,
        "num_distinct_dm_keys": len(dm_stats),
        "num_retrieval_events_remapped_by_loose_string_key": remapped_retrieval_events,
        "dm_stats": dm_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DM retrieval usage from attempt_logs.jsonl")
    parser.add_argument(
        "--attempt-log",
        type=str,
        required=True,
        help="Path to attempt_logs.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSON path (default: <attempt-log-dir>/dm_usage_summary.json)",
    )
    args = parser.parse_args()

    attempt_log_path = os.path.abspath(args.attempt_log)
    if not os.path.exists(attempt_log_path):
        raise FileNotFoundError(f"attempt log not found: {attempt_log_path}")

    out_path = _pick_output_path(attempt_log_path, args.output)
    result = analyze_attempt_log(attempt_log_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True, indent=2)
        f.write("\n")

    print(json.dumps({"output": out_path, "num_dm": result.get("num_distinct_dm_keys", 0)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
