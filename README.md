# Distill LLM: Cognitive Agent + Teacher Distillation

This repository contains an experiment pipeline where an SLM-based Cognitive Agent solves multiple-choice questions, and a Teacher model proposes new DM (Declarative Memory) candidates on failure to progressively improve the KB.

## What This Repository Does

- Shuffles MCQ options into `A/B/C/D` per attempt
- Runs an explicit-state agent with `goal_stack + working_memory + recalled DM`
- Calls the Teacher to generate DM candidates when an attempt fails
- Commits staged DM candidates to the global KB on pass
- Logs every attempt in detail to `attempt_logs.jsonl`

## Key Components

- `experiments/run_experiment.py`: main experiment driver (epoch/problem/attempt loop)
- `agent/cognitive_agent.py`: agent loop, step trace, and final answer scoring
- `agent/prompt_templates.py`: SLM prompt templates and JSON output schema
- `agent/slm_inference.py`: HF model loading, generation, and log-prob scoring
- `teacher/teacher_controller.py`: teacher distillation and candidate scoring/filtering
- `kb/server.py`: DM retrieval/commit and overlay KB logic

## Setup

1. Create and activate a Python environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start/prepare your Teacher endpoint.  
Default endpoint: `http://127.0.0.1:8080/v1/chat/completions`

3. Check config file(s).  
Default example: `experiments/configs/poc_config.json`

## Run

```bash
python experiments/run_experiment.py --config experiments/configs/poc_config.json
```

Optional flags:
```bash
python experiments/run_experiment.py --config experiments/configs/poc_config.json --progress-every 1
python experiments/run_experiment.py --config experiments/configs/poc_config.json --no-progress
```

## Resume an Experiment

Resume from an existing experiment id:
```bash
python experiments/run_experiment.py \
  --config experiments/configs/poc_config.json \
  --resume-experiment-id 20260306_014542
```

Resume behavior:
- Restores the last epoch/problem cursor
- Restores the latest staged DM pool for that problem
- Continues from the next `attempt_iter`

## Logs and Artifacts

Run outputs are stored in `experiments/logs/{experiment_id}/`.

- `attempt_logs.jsonl`: main attempt-level log
- `epoch_summary.jsonl`: epoch-level summary
- `agent_error_*.json`: detailed agent failure diagnostics
- `kb_after_commit_*.jsonl`: per-attempt staged DM snapshots
- `kb/`: run-local KB (`dm_items.jsonl`, `emb_goal.npy`, `emb_condition.npy`)
- `config.input.json`, `config.effective.json`

For the complete `attempt_logs.jsonl` schema and semantics, see:

- `data_explanation.md`

## Analyze DM Usage

To compute DM retrieval counts (final-pass traces) and post-definition usage probability:

```bash
python experiments/analyze_dm_usage.py \
  --attempt-log experiments/logs/20260306_014542/attempt_logs.jsonl
```

Output:
- `experiments/logs/20260306_014542/dm_usage_summary.json`

Example metrics:
- `retrieval_count_in_final_pass_versions`
- `first_defined_problem_idx`
- `solved_problem_count_after_definition`
- `per_problem_call_probability_after_definition`

## Notes

- The current pipeline is built around 4-choice MCQs (`A/B/C/D`).
- Problems that do not match this shape are logged with `skipped=true` in `attempt_logs.jsonl`.
- Final label selection is not based on free-form text alone; it is decided via log-prob comparison after injecting the `"Final Answer is :"` prompt.
