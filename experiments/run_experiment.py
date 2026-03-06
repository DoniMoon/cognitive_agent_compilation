"""Experiment driver for cognitive-agent distillation POC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cognitive_agent import CAConfig, CognitiveAgent
from agent.slm_inference import build_model_adapters
from kb.server import KBServer, OverlayKBServer
from teacher.teacher_controller import TeacherConfig, TeacherController


@dataclass
class ProblemRecord:
    problem_id: str
    question_text: str
    options: List[str]
    answer_index: int
    explanation: str


def load_dataset(path: str) -> List[ProblemRecord]:
    records: List[ProblemRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not obj.get("validity", False):
                continue
            options = obj.get("options", [])
            if not isinstance(options, list):
                options = []
            answer_index = int(obj.get("answer_index", -1))
            records.append(
                ProblemRecord(
                    problem_id=obj["problem_id"],
                    question_text=obj["question_text"],
                    options=options,
                    answer_index=answer_index,
                    explanation=obj.get("explanation", ""),
                )
            )
    return records


def make_attempt_seed(global_seed: int, epoch: int, problem_idx: int, attempt_iter: int) -> int:
    raw = f"{global_seed}:{epoch}:{problem_idx}:{attempt_iter}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def shuffled_options(
    options: List[str], answer_index: int, seed: int
) -> Tuple[Dict[str, str], Dict[str, int], Dict[int, str], str]:
    if len(options) != 4:
        raise ValueError(f"shuffled_options expects 4 options, got {len(options)}")
    rng = random.Random(seed)
    indices = list(range(4))
    rng.shuffle(indices)
    labels = ["A", "B", "C", "D"]

    label_to_original_index = {label: indices[pos] for pos, label in enumerate(labels)}
    original_index_to_label = {orig: label for label, orig in label_to_original_index.items()}
    options_labeled = {label: options[label_to_original_index[label]] for label in labels}
    correct_label = original_index_to_label[answer_index]
    return options_labeled, label_to_original_index, original_index_to_label, correct_label


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _agent_mechanics_text(max_steps: int) -> str:
    return (
        "Cognitive Agent state = goal_stack + working_memory + solution_steps_log. "
        "Initial working_memory is the raw question text. "
        "At each step, retrieval is embedding-similarity based over dm_goal_text and dm_condition_text "
        "against (current_goal, working_memory), returning top items with score/x_goal/y_condition; agent injects top-3 DM texts into prompt. "
        "Action is selected by log-prob comparison among <G>, <R>, <A> with JSON action-prefix candidates. "
        "<G> writes a transformed next_goal and pushes it to goal_stack. "
        "<R> keeps current goal and overwrites working_memory with a new concise reasoning line derived from recalled knowledge, "
        "including conditions for applying that knowledge. "
        "<A> answers current goal; for subgoals it pops stack, for top-level it triggers final option scoring over ' A'/' B'/' C'/' D'. "
        "If output is invalid, agent retries generation with an explicit 'invalid response' hint. "
        "If repeated <G>/<R> no-op persists after retries, the agent logs no-op and continues with unchanged goal/working memory. "
        f"Each attempt has hard max {max_steps} steps."
    )


def _progress(enabled: bool, message: str) -> None:
    if not enabled:
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {message}", flush=True)


def _make_run_dir(base_logs_dir: str) -> str:
    ensure_dir(base_logs_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_logs_dir, stamp)
    suffix = 1
    while os.path.exists(run_dir):
        suffix += 1
        run_dir = os.path.join(base_logs_dir, f"{stamp}_{suffix:02d}")
    ensure_dir(run_dir)
    return run_dir


def _read_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _resolve_resume_cursor(
    logs_dir: str,
    *,
    max_epochs: int,
    dataset_len: int,
    require_pass_before_next_problem: bool,
) -> Tuple[int, int, Dict[int, int]]:
    attempt_logs = _read_jsonl(os.path.join(logs_dir, "attempt_logs.jsonl"))
    epoch_summaries = _read_jsonl(os.path.join(logs_dir, "epoch_summary.jsonl"))

    first_try_pass_by_epoch: Dict[int, int] = {}
    for rec in attempt_logs:
        epoch = int(rec.get("epoch", 0))
        if int(rec.get("attempt_iter", -1)) == 0 and bool(rec.get("pass", False)):
            first_try_pass_by_epoch[epoch] = first_try_pass_by_epoch.get(epoch, 0) + 1

    if not attempt_logs:
        return 0, 0, first_try_pass_by_epoch

    completed_epochs = {int(rec.get("epoch", -1)) for rec in epoch_summaries}
    max_logged_epoch = max(int(rec.get("epoch", 0)) for rec in attempt_logs)

    if max_logged_epoch in completed_epochs:
        start_epoch = max_logged_epoch + 1
        start_problem_idx = 0
    else:
        start_epoch = max_logged_epoch
        epoch_records = [rec for rec in attempt_logs if int(rec.get("epoch", 0)) == start_epoch]
        if not epoch_records:
            start_problem_idx = 0
        else:
            max_problem_idx = max(int(rec.get("problem_idx", 0)) for rec in epoch_records)
            if require_pass_before_next_problem:
                last_problem_records = [rec for rec in epoch_records if int(rec.get("problem_idx", -1)) == max_problem_idx]
                last_problem_records.sort(key=lambda x: int(x.get("attempt_iter", 0)))
                last_problem_pass = bool(last_problem_records[-1].get("pass", False)) or bool(
                    last_problem_records[-1].get("skipped", False)
                )
                start_problem_idx = max_problem_idx + 1 if last_problem_pass else max_problem_idx
            else:
                start_problem_idx = max_problem_idx + 1

    while start_epoch < max_epochs and start_problem_idx >= dataset_len:
        start_epoch += 1
        start_problem_idx = 0

    return start_epoch, start_problem_idx, first_try_pass_by_epoch


def _resolve_problem_resume_state(
    attempt_logs: List[Dict],
    *,
    epoch: int,
    problem_idx: int,
) -> Tuple[int, List[Dict]]:
    records = [
        rec
        for rec in attempt_logs
        if int(rec.get("epoch", -1)) == int(epoch)
        and int(rec.get("problem_idx", -1)) == int(problem_idx)
    ]
    if not records:
        return 0, []
    records.sort(key=lambda x: int(x.get("attempt_iter", 0)))
    last_attempt_iter = int(records[-1].get("attempt_iter", 0))

    staged_pool: List[Dict] = []
    for rec in reversed(records):
        cand = rec.get("teacher_dm_candidates_added", [])
        if isinstance(cand, list) and cand:
            staged_pool = cand
            break
    if not staged_pool:
        for rec in reversed(records):
            cand = rec.get("teacher_preseed_dm_candidates_added", [])
            if isinstance(cand, list) and cand:
                staged_pool = cand
                break
    return last_attempt_iter + 1, staged_pool


def _gpu_memory_line(device: str) -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "GPU status: CUDA unavailable"
        idx = 0
        if ":" in device:
            idx = int(device.split(":")[1])
        allocated = torch.cuda.memory_allocated(idx) / (1024**3)
        reserved = torch.cuda.memory_reserved(idx) / (1024**3)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        return f"GPU status ({device}): allocated={allocated:.2f}GB reserved={reserved:.2f}GB total={total:.2f}GB"
    except Exception as exc:
        return f"GPU status unavailable: {exc}"


def _validate_runtime_config(config: Dict) -> None:
    backend = str(config.get("inference_backend", ""))
    if backend != "transformers":
        raise ValueError("inference_backend must be 'transformers'. Mock backend is disabled.")
    slm_model = str(config.get("slm_model", "")).strip()
    if not slm_model:
        models = config.get("models", [])
        if isinstance(models, list) and len(models) == 1 and str(models[0]).strip():
            slm_model = str(models[0]).strip()
            config["slm_model"] = slm_model
        else:
            raise ValueError("Set exactly one SLM via 'slm_model'. Multi-model 'models' is no longer supported.")


def run_experiment(
    config: Dict,
    progress: bool = True,
    progress_every: int = 1,
    resume_experiment_id: str = "",
) -> None:
    _validate_runtime_config(config)
    global_seed = int(config["global_seed"])
    random.seed(global_seed)

    base_logs_dir = config["logs_dir"]
    resume_id = str(resume_experiment_id or "").strip()
    is_resume = bool(resume_id)
    if is_resume:
        logs_dir = os.path.join(base_logs_dir, resume_id)
        if not os.path.isdir(logs_dir):
            raise FileNotFoundError(f"Resume run directory not found: {logs_dir}")
    else:
        logs_dir = _make_run_dir(base_logs_dir)
    config["run_logs_dir"] = logs_dir
    kb_isolate_per_run = bool(config.get("kb", {}).get("isolate_per_run", True))
    if kb_isolate_per_run:
        kb_dir = os.path.join(logs_dir, "kb")
    else:
        kb_dir = config["kb"]["dir"]
    ensure_dir(kb_dir)
    config["resolved_kb_dir"] = kb_dir

    dataset = load_dataset(config["dataset_path"])
    max_problems = int(config.get("max_problems", 0) or 0)
    if max_problems > 0:
        dataset = dataset[:max_problems]

    kb = KBServer(
        kb_dir=kb_dir,
        embedding_model=config["kb"]["embedding_model"],
        device=config["device"],
        beta=float(config["kb"]["beta"]),
        retrieval_threshold=float(config["kb"]["retrieval_threshold"]),
        enable_near_duplicate=bool(config["kb"].get("enable_near_duplicate", False)),
        allow_hash_fallback=bool(config["kb"].get("allow_hash_fallback", False)),
    )

    adapters = build_model_adapters(
        model_names=[config["slm_model"]],
        backend=config["inference_backend"],
        device=config["device"],
        seed=global_seed,
        torch_dtype=str(config.get("runtime", {}).get("torch_dtype", "float16")),
        low_cpu_mem_usage=bool(config.get("runtime", {}).get("low_cpu_mem_usage", True)),
        model_torch_dtype_overrides=dict(config.get("runtime", {}).get("model_torch_dtype_overrides", {})),
        use_chat_template=bool(config.get("runtime", {}).get("use_chat_template", True)),
    )
    _progress(progress, f"Loaded SLM adapter ({config['slm_model']}) on {config['device']} using backend={config['inference_backend']}")
    _progress(progress, _gpu_memory_line(str(config["device"])))
    ca_cfg = CAConfig(
        max_steps=config["max_steps_per_problem_per_agent"],
        log_prompts=bool(config.get("runtime", {}).get("log_prompts", False)),
        fail_on_nonfinite=bool(config.get("runtime", {}).get("fail_on_nonfinite", True)),
        fail_on_noop=bool(config.get("runtime", {}).get("fail_on_noop", True)),
        noop_retry_attempts=int(config.get("runtime", {}).get("noop_retry_attempts", 3)),
        force_recall_each_step=bool(config.get("runtime", {}).get("force_recall_each_step", True)),
        debug_prompt_char_limit=int(config.get("runtime", {}).get("debug_prompt_char_limit", 12000)),
    )
    agent = CognitiveAgent(model=adapters[0], kb_server=kb, config=ca_cfg)

    teacher = TeacherController(
        kb_server=kb,
        config=TeacherConfig(
            endpoint=config["teacher"]["endpoint"],
            model=config["teacher"]["model"],
            temperature=float(config["teacher"]["temperature"]),
            max_new_dm_per_teacher_call=int(config["teacher"]["max_new_dm_per_teacher_call"]),
            max_mcp_query_topk_calls=int(config["teacher"].get("max_mcp_query_topk_calls", 8)),
            max_mcp_score_calls=int(config["teacher"].get("max_mcp_score_calls", 60)),
            allow_heuristic_fallback=bool(config["teacher"].get("allow_heuristic_fallback", False)),
            max_teacher_input_chars=int(config["teacher"].get("max_teacher_input_chars", 50000)),
            max_teacher_input_chars_retry=int(config["teacher"].get("max_teacher_input_chars_retry", 15000)),
            max_steps_per_ca_run_for_teacher=int(config["teacher"].get("max_steps_per_ca_run_for_teacher", 8)),
            request_timeout_s=int(config["teacher"].get("request_timeout_s", 300)),
            max_tokens=int(config["teacher"].get("max_tokens", 512)),
            request_retries=int(config["teacher"].get("request_retries", 2)),
            retry_backoff_s=float(config["teacher"].get("retry_backoff_s", 2.0)),
            enable_tool_planner=bool(config["teacher"].get("enable_tool_planner", False)),
            debug_log_dir=os.path.join(logs_dir, "teacher_debug"),
            bootstrap_max_new_dm_per_teacher_call=int(
                config["teacher"].get("bootstrap_max_new_dm_per_teacher_call", 8)
            ),
        ),
    )

    log_path = os.path.join(logs_dir, "attempt_logs.jsonl")
    epoch_summary_path = os.path.join(logs_dir, "epoch_summary.jsonl")
    resume_attempt_logs: List[Dict] = _read_jsonl(log_path) if is_resume else []

    require_pass_before_next_problem = bool(config.get("require_pass_before_next_problem", True))
    hard_max_attempts_per_problem = int(config.get("hard_max_attempts_per_problem", 0) or 0)
    max_teacher_iterations_per_problem = int(config.get("max_teacher_iterations_per_problem", 10))
    threshold = float(config["threshold"])
    max_steps = int(config["max_steps_per_problem_per_agent"])
    start_epoch = 0
    start_problem_idx = 0
    first_try_pass_seed: Dict[int, int] = {}
    if is_resume:
        start_epoch, start_problem_idx, first_try_pass_seed = _resolve_resume_cursor(
            logs_dir,
            max_epochs=int(config["max_epochs"]),
            dataset_len=len(dataset),
            require_pass_before_next_problem=require_pass_before_next_problem,
        )

    _progress(
        progress,
        (
            f"Experiment start | dataset={len(dataset)} problems | epochs={config['max_epochs']} | "
            f"threshold={threshold:.3f} | require_pass_before_next_problem={require_pass_before_next_problem} | "
            f"run_logs_dir={logs_dir}"
        ),
    )
    if is_resume:
        _progress(
            progress,
            f"Resume mode | experiment_id={resume_id} | start_epoch={start_epoch} | start_problem_idx={start_problem_idx}",
        )

    if not is_resume:
        with open(os.path.join(logs_dir, "config.effective.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        input_cfg = config.get("_input_config_path", "")
        if input_cfg and os.path.exists(input_cfg):
            with open(input_cfg, "r", encoding="utf-8") as src, open(
                os.path.join(logs_dir, "config.input.json"), "w", encoding="utf-8"
            ) as dst:
                dst.write(src.read())

    for epoch in range(start_epoch, config["max_epochs"]):
        epoch_start = time.time()
        _progress(progress, f"Epoch {epoch} start")
        first_try_pass_count = int(first_try_pass_seed.get(epoch, 0))
        problem_start_idx = start_problem_idx if (is_resume and epoch == start_epoch) else 0
        for problem_idx, rec in enumerate(dataset[problem_start_idx:], start=problem_start_idx):
            problem_passed = False
            problem_dm_pool: List[Dict] = []
            attempt_iter = 0
            if is_resume and epoch == start_epoch and problem_idx == start_problem_idx:
                attempt_iter, problem_dm_pool = _resolve_problem_resume_state(
                    resume_attempt_logs,
                    epoch=epoch,
                    problem_idx=problem_idx,
                )
                _progress(
                    progress,
                    (
                        f"Resume problem state | epoch={epoch} problem={problem_idx} "
                        f"next_attempt_iter={attempt_iter} staged_pool_size={len(problem_dm_pool)}"
                    ),
                )

            _progress(progress, f"Problem {problem_idx}/{len(dataset)-1} ({rec.problem_id}) start")
            if len(rec.options) != 4 or rec.answer_index < 0 or rec.answer_index >= len(rec.options):
                _progress(
                    progress,
                    (
                        f"Problem {problem_idx} SKIP | unsupported option shape "
                        f"(n_options={len(rec.options)} answer_index={rec.answer_index})"
                    ),
                )
                skip_record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "epoch": epoch,
                    "problem_idx": problem_idx,
                    "attempt_iter": 0,
                    "problem_id": rec.problem_id,
                    "skipped": True,
                    "skip_reason": "unsupported_option_shape_for_abcd_pipeline",
                    "n_options": len(rec.options),
                    "answer_index": rec.answer_index,
                    "pass": False,
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(skip_record, ensure_ascii=True) + "\n")
                continue

            while True:
                if hard_max_attempts_per_problem > 0 and attempt_iter >= hard_max_attempts_per_problem:
                    raise RuntimeError(
                        f"Problem {rec.problem_id} exceeded hard_max_attempts_per_problem={hard_max_attempts_per_problem} "
                        "without reaching pass threshold."
                    )
                if (not require_pass_before_next_problem) and attempt_iter >= max_teacher_iterations_per_problem:
                    break

                seed = make_attempt_seed(global_seed, epoch, problem_idx, attempt_iter)
                options_labeled, label_to_original_index, original_index_to_label, correct_label = shuffled_options(
                    rec.options, rec.answer_index, seed
                )

                problem_payload = {
                    "problem_id": rec.problem_id,
                    "question_text": rec.question_text,
                    "options_labeled": options_labeled,
                    "correct_label": correct_label,
                }

                ca_outputs = []
                global_kb_size_before_attempt = len(kb.dm_items)
                temp_kb_size_before_attempt = len(problem_dm_pool)
                active_kb = OverlayKBServer(base_kb=kb, staged_candidates=problem_dm_pool)
                overlay_kb_size_before_agent = len(active_kb.dm_items)
                preseed_teacher_added = []
                preseed_teacher_mcp_usage = {}
                preseed_teacher_kb_tool_observations = {}
                if len(active_kb.dm_items) == 0:
                    teacher.kb_server = active_kb
                    teacher_seed_input = {
                        "epoch": epoch,
                        "attempt_iter": attempt_iter,
                        "problem_id": rec.problem_id,
                        "question_text": rec.question_text,
                        "options_labeled": options_labeled,
                        "correct_label": correct_label,
                        "gold_explanation": rec.explanation,
                        "previous_new_dm_candidates_in_this_problem": problem_dm_pool,
                        "cognitive_agent_mechanics": _agent_mechanics_text(max_steps=max_steps),
                        "ca_runs": [],
                        "bootstrap_stage": "pre_ca_seed_kb",
                    }
                    teacher_seed_result = teacher.distill_candidates(teacher_seed_input)
                    preseed_teacher_added = teacher_seed_result["proposed"].get("candidates", [])
                    preseed_teacher_mcp_usage = teacher_seed_result.get("mcp_usage", {})
                    preseed_teacher_kb_tool_observations = teacher_seed_result.get("kb_tool_observations", {})
                    if not preseed_teacher_added:
                        raise RuntimeError(
                            "Teacher pre-CA KB bootstrap produced zero DM candidates while KB is empty."
                        )
                    problem_dm_pool = preseed_teacher_added
                    active_kb = OverlayKBServer(base_kb=kb, staged_candidates=problem_dm_pool)
                    temp_kb_size_before_attempt = len(problem_dm_pool)
                    overlay_kb_size_before_agent = len(active_kb.dm_items)
                    _progress(
                        progress,
                        (
                            f"Teacher pre-seed | Epoch {epoch} Problem {problem_idx} Iter {attempt_iter} | "
                            f"seed_pool_size={len(problem_dm_pool)} "
                            f"mcp_query_topk={preseed_teacher_mcp_usage.get('kb.query_topk_calls', 0)} "
                            f"mcp_score={preseed_teacher_mcp_usage.get('kb.score_calls', 0)}"
                        ),
                    )
                try:
                    agent.kb_server = active_kb
                    ca_outputs.append(agent.run(problem_payload))
                except Exception as exc:
                    fail_record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "epoch": epoch,
                        "problem_idx": problem_idx,
                        "attempt_iter": attempt_iter,
                        "problem_id": rec.problem_id,
                        "shuffle_seed": seed,
                        "options_labeled": options_labeled,
                        "label_to_original_index": label_to_original_index,
                        "original_index_to_label": {str(k): v for k, v in original_index_to_label.items()},
                        "correct_label": correct_label,
                        "ca_outputs": ca_outputs,
                        "mean_correct_prob": None,
                        "teacher_preseed_dm_candidates_added": preseed_teacher_added,
                        "teacher_preseed_mcp_usage": preseed_teacher_mcp_usage,
                        "teacher_preseed_kb_tool_observations": preseed_teacher_kb_tool_observations,
                        "teacher_dm_candidates_added": [],
                        "teacher_mcp_usage": {},
                        "teacher_kb_tool_observations": {},
                        "global_kb_size_before_attempt": global_kb_size_before_attempt,
                        "overlay_kb_size_before_agent": overlay_kb_size_before_agent,
                        "temp_kb_size_before_attempt": temp_kb_size_before_attempt,
                        "temp_kb_size_after_teacher": len(problem_dm_pool),
                        "global_kb_size_after_attempt": len(kb.dm_items),
                        "global_kb_commit_added": 0,
                        "global_kb_commit_skipped_duplicates": 0,
                        "problem_dm_pool_size": len(problem_dm_pool),
                        "pass": False,
                        "agent_failure": {
                            "model_name": getattr(agent.model, "model_name", "unknown"),
                            "error": str(exc),
                        },
                    }
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(fail_record, ensure_ascii=True) + "\n")
                    err_obj = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "epoch": epoch,
                        "problem_idx": problem_idx,
                        "attempt_iter": attempt_iter,
                        "problem_id": rec.problem_id,
                        "model_name": getattr(agent.model, "model_name", "unknown"),
                        "error": str(exc),
                    }
                    err_path = os.path.join(
                        logs_dir,
                        f"agent_error_epoch{epoch:02d}_problem{problem_idx:04d}_iter{attempt_iter:03d}_{getattr(agent.model, 'model_name', 'unknown').replace('/', '__')}.json",
                    )
                    with open(err_path, "w", encoding="utf-8") as ef:
                        json.dump(err_obj, ef, ensure_ascii=True, indent=2)
                        ef.write("\n")
                    raise RuntimeError(f"Agent failure captured at {err_path}") from exc

                mean_correct_prob = ca_outputs[0]["p_correct"]

                if progress and (attempt_iter % max(1, progress_every) == 0):
                    n_correct = 1 if ca_outputs[0]["is_correct"] else 0
                    n_agents = 1
                    _progress(
                        progress,
                        (
                            f"Epoch {epoch} | Problem {problem_idx} | Iter {attempt_iter} | "
                            f"mean_correct_prob={mean_correct_prob:.4f} | correct_agents={n_correct}/{n_agents} | "
                            f"kb_size={len(kb.dm_items)}"
                        ),
                    )

                if attempt_iter == 0 and mean_correct_prob >= threshold:
                    first_try_pass_count += 1

                teacher_added = []
                teacher_mcp_usage = {}
                teacher_kb_tool_observations = {}
                temp_kb_size_after_teacher = len(problem_dm_pool)
                if mean_correct_prob >= threshold:
                    problem_passed = True
                else:
                    teacher.kb_server = active_kb
                    slm_not_concluded = any(
                        str(run.get("termination", "")) == "forced_answer_at_max_steps" for run in ca_outputs
                    )
                    teacher_input = {
                        "epoch": epoch,
                        "attempt_iter": attempt_iter,
                        "problem_id": rec.problem_id,
                        "question_text": rec.question_text,
                        "options_labeled": options_labeled,
                        "correct_label": correct_label,
                        "gold_explanation": rec.explanation,
                        "previous_new_dm_candidates_in_this_problem": problem_dm_pool,
                        "cognitive_agent_mechanics": _agent_mechanics_text(max_steps=max_steps),
                        "ca_runs": ca_outputs,
                        "slm_not_concluded": bool(slm_not_concluded),
                        "slm_conclusion_notice": (
                            "SLM did not derive a conclusion before max steps; final label was selected by forced scoring."
                            if slm_not_concluded
                            else ""
                        ),
                    }
                    teacher_result = teacher.distill_candidates(teacher_input)
                    teacher_added = teacher_result["proposed"].get("candidates", [])
                    teacher_mcp_usage = teacher_result.get("mcp_usage", {})
                    teacher_kb_tool_observations = teacher_result.get("kb_tool_observations", {})
                    # Replace per-problem DM pool each iteration with full set proposed by Teacher.
                    problem_dm_pool = teacher_added
                    temp_kb_size_after_teacher = len(problem_dm_pool)
                    _progress(
                        progress,
                        (
                            f"Teacher stage update | Epoch {epoch} Problem {problem_idx} Iter {attempt_iter} | "
                            f"temp_pool_size={len(problem_dm_pool)} "
                            f"global_kb_size={len(kb.dm_items)} "
                            f"mcp_query_topk={teacher_mcp_usage.get('kb.query_topk_calls', 0)} "
                            f"mcp_score={teacher_mcp_usage.get('kb.score_calls', 0)}"
                        ),
                    )

                    # Snapshot current per-problem staged pool.
                    snap_name = f"kb_after_commit_epoch{epoch:02d}_problem{problem_idx:04d}_iter{attempt_iter:02d}.jsonl"
                    with open(os.path.join(logs_dir, snap_name), "w", encoding="utf-8") as sf:
                        for idx, cand in enumerate(problem_dm_pool, start=1):
                            obj = {
                                "dm_id": f"stage_{idx:05d}",
                                "dm_text": cand.get("dm_text", ""),
                                "dm_goal_text": cand.get("dm_goal_text", ""),
                                "dm_condition_text": cand.get("dm_condition_text", ""),
                            }
                            sf.write(json.dumps(obj, ensure_ascii=True) + "\n")

                commit_res = {"added": [], "skipped_duplicates": [], "kb_size": len(kb.dm_items)}
                if problem_passed and problem_dm_pool:
                    commit_payload = []
                    for cand in problem_dm_pool:
                        commit_payload.append(
                            {
                                "dm_text": cand.get("dm_text", ""),
                                "dm_goal_text": cand.get("dm_goal_text", ""),
                                "dm_condition_text": cand.get("dm_condition_text", ""),
                                "metadata": {
                                    "source_problem_id": rec.problem_id,
                                    "epoch": epoch,
                                    "problem_iteration": attempt_iter,
                                    "created_by": "teacher_llm",
                                },
                            }
                        )
                    commit_res = kb.commit_dm_candidates(commit_payload)
                    _progress(
                        progress,
                        (
                            f"Global KB commit on PASS | added={len(commit_res.get('added', []))} "
                            f"skipped={len(commit_res.get('skipped_duplicates', []))} "
                            f"kb_size={commit_res.get('kb_size', len(kb.dm_items))}"
                        ),
                    )

                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "epoch": epoch,
                    "problem_idx": problem_idx,
                    "attempt_iter": attempt_iter,
                    "problem_id": rec.problem_id,
                    "shuffle_seed": seed,
                    "options_labeled": options_labeled,
                    "label_to_original_index": label_to_original_index,
                    "original_index_to_label": {str(k): v for k, v in original_index_to_label.items()},
                    "correct_label": correct_label,
                    "ca_outputs": ca_outputs,
                    "mean_correct_prob": mean_correct_prob,
                    "teacher_preseed_dm_candidates_added": preseed_teacher_added,
                    "teacher_preseed_mcp_usage": preseed_teacher_mcp_usage,
                    "teacher_preseed_kb_tool_observations": preseed_teacher_kb_tool_observations,
                    "teacher_dm_candidates_added": teacher_added,
                    "teacher_mcp_usage": teacher_mcp_usage,
                    "teacher_kb_tool_observations": teacher_kb_tool_observations,
                    "global_kb_size_before_attempt": global_kb_size_before_attempt,
                    "overlay_kb_size_before_agent": overlay_kb_size_before_agent,
                    "temp_kb_size_before_attempt": temp_kb_size_before_attempt,
                    "temp_kb_size_after_teacher": temp_kb_size_after_teacher,
                    "global_kb_size_after_attempt": len(kb.dm_items),
                    "global_kb_commit_added": len(commit_res.get("added", [])),
                    "global_kb_commit_skipped_duplicates": len(commit_res.get("skipped_duplicates", [])),
                    "problem_dm_pool_size": len(problem_dm_pool),
                    "pass": mean_correct_prob >= threshold,
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")

                if problem_passed:
                    _progress(progress, f"Problem {problem_idx} PASS at iter {attempt_iter} (mean_correct_prob={mean_correct_prob:.4f})")
                    break
                attempt_iter += 1

            if not problem_passed:
                _progress(progress, f"Problem {problem_idx} FAIL after {attempt_iter} iterations (moving on due to config)")

        first_try_pass_rate = first_try_pass_count / max(1, len(dataset))
        epoch_summary = {
            "epoch": epoch,
            "first_try_pass_rate": first_try_pass_rate,
            "target_pass_rate": config["target_pass_rate"],
            "threshold": config["threshold"],
        }
        with open(epoch_summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_summary, ensure_ascii=True) + "\n")

        kb.snapshot(os.path.join(logs_dir, f"kb_end_epoch_{epoch:02d}.jsonl"))
        _progress(
            progress,
            (
                f"Epoch {epoch} end | first_try_pass_rate={first_try_pass_rate:.4f} | "
                f"elapsed_sec={time.time()-epoch_start:.1f}"
            ),
        )

        if first_try_pass_rate >= config["target_pass_rate"]:
            _progress(progress, f"Early stop: first_try_pass_rate >= target_pass_rate ({config['target_pass_rate']})")
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/poc_config.json",
        help="Path to JSON config",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable command-line progress logs",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print attempt-level progress every N iterations",
    )
    parser.add_argument(
        "--resume-experiment-id",
        type=str,
        default="",
        help="Resume from an existing logs_dir experiment id (e.g., 20260306_014542)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Preserve original input config under each run directory for reproducibility.
    config["_input_config_path"] = os.path.abspath(args.config)

    run_experiment(
        config,
        progress=not args.no_progress,
        progress_every=max(1, args.progress_every),
        resume_experiment_id=args.resume_experiment_id,
    )


if __name__ == "__main__":
    main()
