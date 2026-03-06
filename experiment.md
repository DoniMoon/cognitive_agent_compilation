<!-- Experiment.md -->

# Experiment Specification (Dataset, Loop, Metrics, Stopping Criteria)

## 1. Dataset

### 1.1 File Format

Input dataset is JSONL with one problem per line.

Each line must have:

- `validity: bool`
- `custom_id: str`
- `problem_id: str`
- `question_text: str`
- `options: List[str]` length 4
- `answer_index: int` in 0,1,2,3 referencing the original `options` list
- `explanation: str` (gold explanation)

Example:

```json
{
  "validity": true,
  "custom_id": "request-0-0",
  "problem_id": "biochem_carbs_assemble_DIGT",
  "question_text": "Which of the following carbohydrates contains a monosaccharide different from the others?",
  "options": ["glycogen","cellulose","lactose","starch"],
  "answer_index": 2,
  "explanation": "Glycogen, cellulose, and starch are all polymers of glucose only, whereas lactose is a disaccharide composed of glucose and galactose..."
}
```

Total problems: 1642 biology MCQs.

---

## 2. Option Shuffling and Labeling

### 2.1 Requirement

Every problem attempt must present options in a shuffled order with labels:

- `A: ...`
- `B: ...`
- `C: ...`
- `D: ...`

### 2.2 Deterministic Shuffle Policy (POC Default)

To keep labels consistent across the 8 models in the same attempt:

- For each `problem_attempt_id` (one group run of the 8 CAs), shuffle ONCE.
- Use a deterministic seed derived from:
  - global_seed
  - epoch index
  - problem index
  - problem_attempt_iteration index (within-problem distillation loop)

Example seed:
- `seed = hash(global_seed, epoch, problem_idx, attempt_iter)`

Then apply the same shuffled labeled options to all 8 models.

### 2.3 Correct Label After Shuffle

Compute:
- `correct_label` as the label position of the original `answer_index` after shuffle.

Store mapping:

- `label_to_original_index`
- `original_index_to_label`

This is needed to compute correctness and to interpret CA probabilities.

---

## 3. Models (8 SLM Agents)

Run the following SLMs as independent cognitive agents:

1. Qwen/Qwen2.5-0.5B-Instruct
2. google/gemma-3-270m-it
3. meta-llama/Llama-3.2-1B-Instruct
4. LiquidAI/LFM2-350M
5. teapotai/tinyteapot
6. NoesisLab/Kai-0.35B-Instruct
7. state-spaces/mamba2-370m
8. state-spaces/mamba2-130m

All are inference-only (no fine-tuning in this POC).

---

## 4. Pass Metric

Each CA outputs:
- probability distribution over {A,B,C,D} using log-prob comparison
- `p_correct_i = probs_i[correct_label]`

Define:

- `mean_correct_prob = mean_i(p_correct_i)` across the 8 models

POC pass threshold:
- `threshold = 0.75`

Interpretation:
- Equivalent to “about 6 out of 8” if outputs were near-deterministic, but it is probability-based and continuous.

---

## 5. Per-Problem Distillation Loop

For each problem, run the following loop until pass:

Parameters (explicit defaults):
- `max_teacher_iterations_per_problem = 10`
- `max_steps_per_problem_per_agent = 20` (from Cognitive_Agent.md)
- `threshold = 0.75`

Algorithm:

1. Shuffle and label options for this attempt iteration.
2. Run 8 CAs with the current KB.
3. Compute `mean_correct_prob`.
4. If `mean_correct_prob >= threshold`:
   - mark problem as PASS
   - proceed to next problem
5. Else:
   - call Teacher LLM with:
     - problem statement, labeled options, correct label
     - CA traces for failures (include all 8 for simplicity in POC)
     - list of DM candidates already added in this problem loop
   - Teacher returns new DM candidates
   - commit them to KB via `kb.commit_dm_candidates`
   - increment attempt iteration and repeat

Stop condition if exceeded iteration limit:
- If attempts reach `max_teacher_iterations_per_problem`:
  - mark as FAIL (for analysis)
  - proceed to next problem (do not stall the full epoch)

---

## 6. Epoch Loop and Global Stopping Criteria

Define:
- 1 epoch = iterating all problems once in dataset order.

The KB is NOT reset between problems or epochs.

Maximum epochs:
- `max_epochs = 10`

Additional stopping rule:
- If in the last completed epoch, the **first-try** pass rate is above a target, stop early.

### 6.1 First-Try Pass Rate Definition

For each problem in an epoch:
- run exactly one attempt with current KB (no teacher updates yet for that problem)
- compute `mean_correct_prob_first_try`
- if `>= threshold`, it counts as first-try pass

Compute:
- `first_try_pass_rate = (# first-try pass problems) / (total problems)`

Early stop:
- If `first_try_pass_rate >= target_pass_rate` in an epoch, stop.
POC default:
- `target_pass_rate = 0.90`

---

## 7. Logging and Artifacts

### 7.1 Required Logs (Per Problem Attempt)

Write a JSON record containing:
- epoch, problem_idx, attempt_iter
- shuffle seed and labeled options
- correct label
- per-model CA outputs (pred, probs, is_correct, solution steps, termination)
- mean_correct_prob
- teacher DM candidates added (if any)
- KB size after commit

### 7.2 KB Snapshots

Persist at least:
- after each committed DM batch
- at the end of each epoch

---

## 8. Reproducibility Requirements

- `global_seed` fixed and stored in experiment config.
- Option shuffling deterministic per attempt iteration.
- Record HF model commit hashes if available (optional but recommended).

---

## 9. Minimal Config File (Recommended)

Store all hyperparameters in one config:

```json
{
  "global_seed": 1234,
  "max_epochs": 10,
  "threshold": 0.75,
  "target_pass_rate": 0.90,
  "max_teacher_iterations_per_problem": 10,
  "max_steps_per_problem_per_agent": 20,
  "kb": {
    "embedding_model": "google/embeddinggemma-300m",
    "beta": 5.0,
    "retrieval_threshold": 0.30
  },
  "teacher": {
    "max_new_dm_per_call": 5
  }
}
```
