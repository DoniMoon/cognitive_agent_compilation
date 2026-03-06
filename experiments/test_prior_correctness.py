"""Evaluate direct MCQ prior correctness for the configured single SLM."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.slm_inference import build_model_adapters, softmax_from_logps


def load_dataset(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("validity", False):
                rows.append(obj)
    return rows


def build_direct_mcq_prompt(question: str, options: List[str]) -> str:
    return (
        "You are solving a multiple-choice biology question.\n"
        "Choose exactly one best answer.\n"
        "Respond with only one letter: A, B, C, or D.\n\n"
        f"Question: {question}\n\n"
        "Options:\n"
        f"A: {options[0]}\n"
        f"B: {options[1]}\n"
        f"C: {options[2]}\n"
        f"D: {options[3]}\n\n"
        "Answer:"
    )


def evaluate_problem(config: Dict, problem_idx: int) -> Dict:
    dataset = load_dataset(config["dataset_path"])
    if problem_idx < 0 or problem_idx >= len(dataset):
        raise IndexError(f"--problem out of range: {problem_idx} (dataset size={len(dataset)})")

    row = dataset[problem_idx]
    question = row["question_text"]
    options = row["options"]
    answer_index = int(row["answer_index"])
    correct_label = ["A", "B", "C", "D"][answer_index]

    prompt = build_direct_mcq_prompt(question=question, options=options)

    slm_model = str(config.get("slm_model", "")).strip()
    if not slm_model:
        models = config.get("models", [])
        if isinstance(models, list) and len(models) == 1 and str(models[0]).strip():
            slm_model = str(models[0]).strip()
        else:
            raise ValueError("Set exactly one SLM via 'slm_model'.")

    adapters = build_model_adapters(
        model_names=[slm_model],
        backend=config["inference_backend"],
        device=config["device"],
        seed=int(config.get("global_seed", 1234)),
        torch_dtype=str(config.get("runtime", {}).get("torch_dtype", "float16")),
        low_cpu_mem_usage=bool(config.get("runtime", {}).get("low_cpu_mem_usage", True)),
        model_torch_dtype_overrides=dict(config.get("runtime", {}).get("model_torch_dtype_overrides", {})),
        use_chat_template=bool(config.get("runtime", {}).get("use_chat_template", True)),
    )

    per_model = []
    for model in adapters:
        label_logps = {
            "A": model.sequence_logprob(prompt, " A"),
            "B": model.sequence_logprob(prompt, " B"),
            "C": model.sequence_logprob(prompt, " C"),
            "D": model.sequence_logprob(prompt, " D"),
        }
        probs = softmax_from_logps(label_logps)
        pred = max(label_logps, key=label_logps.get)
        per_model.append(
            {
                "model_name": model.model_name,
                "pred_label": pred,
                "probs": probs,
                "p_correct": probs[correct_label],
                "is_correct_argmax": pred == correct_label,
            }
        )

    mean_correct_prob = sum(m["p_correct"] for m in per_model) / len(per_model)
    return {
        "problem_idx": problem_idx,
        "problem_id": row.get("problem_id", ""),
        "question_text": question,
        "correct_label": correct_label,
        "mean_correct_prob": mean_correct_prob,
        "per_model": per_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/configs/poc_config.json")
    parser.add_argument("--problem", type=int, required=True, help="0-based problem index")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    out = evaluate_problem(config=config, problem_idx=args.problem)
    print(json.dumps(out, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
