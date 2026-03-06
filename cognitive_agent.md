<!-- Cognitive_Agent.md -->

# Cognitive Agent Specification (Soar + ACT-R Inspired, SLM-Executed)

## 1. Overview

The Cognitive Agent (CA) is an explicit-state problem solver executed by a small instruction-tuned language model (SLM). It mixes:
- Soar-style goal stack and subgoals
- ACT-R-style declarative memory retrieval (DM)

At each step, the CA:
1. retrieves at most one DM (`Recalled`) from the KB server using current goal + working memory
2. asks the SLM to choose one of three actions:
   - set a new subgoal
   - update working memory
   - answer and complete a goal (possibly the final MCQ answer)
3. logs the decision into Solution Steps and updates state

This repeats until the top-level goal (solve the problem) is answered or a step limit is reached.

---

## 2. State Representation

### 2.1 Minimal CA State

- `goal_stack: List[str]`
  - Top of stack is the current goal.
  - Stack represents subgoal nesting.
- `working_memory: str`
  - A short textual workspace (“Thinking” field in prompt).
- `solution_steps_log: List[StepEntry]`
  - Append-only trace used as prompt context and passed to Teacher.
- `retrieved_dm_text: str | ""`
  - The recalled DM payload for this step (DM_text only).
- `question_context: QuestionContext`
  - includes question, labeled options, and mapping to original indices.

### 2.2 Initialization

At the start of each problem attempt:
- `goal_stack = ["Solve the problem"]`
- `working_memory = ""`
- `solution_steps_log = []`

---

## 3. Actions and Transition Semantics

SLMs infer using the transformers library, which must be loaded on cuda:0.

At each step, the CA chooses exactly one of:

1. `<1>` Set New Goal (push subgoal)
2. `<2>` Renew Working Memory (overwrite working_memory)
3. `<3>` Get Answer (complete current goal; if top-level, produce final MCQ answer)

### 3.1 Action `<1>`: Set New Goal

- SLM generates: a short subgoal description
- CA pushes it onto `goal_stack`
- CA logs it in Solution Steps

Constraints for parsing:
- Subgoal must be a single line.
- Hard max length: 20 words (enforced by truncation if needed).

### 3.2 Action `<2>`: Renew Working Memory

- SLM generates updated working memory content
- CA overwrites `working_memory`
- CA logs it in Solution Steps

Constraints:
- Must be short (max 60 words).
- Should contain derived facts, intermediate conclusions, or structured notes.

### 3.3 Action `<3>`: Get Answer

This action has two modes, determined by current goal.

#### Mode A: Current goal is NOT top-level

If `goal_stack[-1] != "Solve the problem"`:
- SLM generates a short answer for the current subgoal
- CA logs it
- CA pops the goal (goal completed)
- CA sets `working_memory = "Subgoal completed: <popped_goal>"`

Then the loop continues with the next goal on the stack.

#### Mode B: Current goal IS top-level

If `goal_stack[-1] == "Solve the problem"`:
- CA must produce a final multiple-choice answer label (A/B/C/D).
- This must be decided deterministically by comparing the SLM log-probabilities for A/B/C/D.

CA then terminates the episode and outputs:
- predicted label
- probability distribution over A/B/C/D
- full solution trace

---

## 4. Deterministic Decision via Log-Probability Comparison

SLM generation templates are not perfectly controllable, so the CA uses **log-prob comparisons** instead of trusting free-form output.

### 4.1 Decision Token Selection (<1>, <2>, <3>)

Given the step prompt (see Section 5), compute:

- `logp("<1>")`, `logp("<2>")`, `logp("<3>")` as next output
- choose the argmax token as action
- compute `p_action = softmax([logp1, logp2, logp3])`

Important tokenizer detail:
- `<1>` may be multiple tokens for some tokenizers.
- Therefore compute sequence log-prob for the full string using teacher forcing:
  - tokenize the candidate string
  - sum token log-probs sequentially conditioned on the prompt + previous candidate tokens

This is required to be stable across different HF tokenizers.

### 4.2 Final Answer Label Selection (A/B/C/D)

When top-level goal is ready to be answered:
- Build a final-answer prompt that ends with:

`<3> final answer is:`

Then compute sequence log-prob for each candidate label string:

- `" A"`, `" B"`, `" C"`, `" D"` (note leading space)

Pick argmax and compute:

- `p_label = softmax([logpA, logpB, logpC, logpD])`

Return:
- `pred_label`
- `p_label` distribution

---

## 5. SLM Prompt Specification

### 5.1 Prompt Fields

Each CA step prompt must contain:

- System Prompt (fixed)
- Question (with labeled options)
- Solution Steps (trace so far)
- Current Goal
- Working Memory
- Recalled (retrieved DM_text or empty)

No other hidden state is allowed.

### 5.2 System Prompt (SLM)

This exact text is the required system prompt for all SLMs:

```text
You are a Cognitive Agent that solves a multiple-choice question using explicit goals and a working memory.

You must output exactly one action token as the first thing in your response:
<1> to set a new subgoal,
<2> to update working memory,
<3> to answer the current goal.

After the action token, output content that matches the action:
- After <1>: output ONE line describing the new subgoal. Keep it short and actionable.
- After <2>: output ONE short paragraph updating working memory. Include only problem-relevant facts or intermediate conclusions.
- After <3>:
  - If the current goal is not "Solve the problem": output ONE short sentence answering the current goal.
  - If the prompt ends with "<3> final answer is:": output ONLY one letter: A, B, C, or D. Output nothing else.

Never write anything before the action token.
Never output multiple action tokens.
Use plain English. Be concise.
```

### 5.3 Step Prompt Template (SLM)

The CA should format the step prompt as a plain text document:

```text
[QUESTION]
{question_text}

[OPTIONS]
A: {option_A}
B: {option_B}
C: {option_C}
D: {option_D}

[SOLUTION STEPS SO FAR]
{solution_steps_log_text_or_EMPTY}

[CURRENT STATE]
Goal: {current_goal}
WorkingMemory: {working_memory_or_EMPTY}
Recalled: {retrieved_dm_text_or_EMPTY}

[INSTRUCTION]
Choose the next action now.
```

### 5.4 Final Answer Prompt Template

When answering the top-level goal, use the same prompt structure but replace the last instruction line with:

```text
[INSTRUCTION]
<3> final answer is:
```

Then do A/B/C/D log-prob comparison.

---

## 6. Solution Steps Logging Format

The log must be human-readable and machine-parsable.

### 6.1 Step Entry Format (Required)

Each step entry is appended as:

```text
Step {t}
GoalStack: {goal_stack_as_list}
Action: <1|2|3>
RecalledDM: {dm_id_or_NONE}
Generated: {generated_text_or_label}
```

If Action is final answer:
- `Generated` must include predicted label and probabilities, for example:

```text
Generated: pred=C; probs={A:0.02,B:0.03,C:0.92,D:0.03}
```

### 6.2 Goal Completion Recording

When `<3>` completes a subgoal (non-top-level), record:

```text
CompletedGoal: {popped_goal_text}
```

This matters for the Teacher LLM to diagnose missing knowledge.

---

## 7. Failure and Safety Stops

To avoid infinite loops:

- `max_steps_per_problem = 20` (required POC default)
- If exceeded:
  - terminate with status `timeout`
  - pick final answer anyway using A/B/C/D log-prob method
  - log `Termination: timeout`

Also stop if:
- goal stack becomes empty (should not happen). Treat as error.

---

## 8. Supported SLM Models

All must be runnable via Hugging Face Transformers (or HF-compatible wrapper):

- Qwen/Qwen2.5-0.5B-Instruct
- google/gemma-3-270m-it
- meta-llama/Llama-3.2-1B-Instruct
- LiquidAI/LFM2-350M
- teapotai/tinyteapot
- NoesisLab/Kai-0.35B-Instruct
- state-spaces/mamba2-370m
- state-spaces/mamba2-130m

Implementation requirement:
- Use a unified inference wrapper that exposes:
  - `tokenize(text)`
  - `logprob_of_sequence(prompt_text, candidate_text)`
  - `generate_text(prompt_text, max_new_tokens, stop_rules)`

---

## 9. CA Output Contract (Per Problem, Per Model)

Return a JSON object like:

```json
{
  "model_name": "google/gemma-3-270m-it",
  "problem_id": "biochem_carbs_assemble_DIGT",
  "pred_label": "C",
  "probs": {"A": 0.01, "B": 0.02, "C": 0.95, "D": 0.02},
  "is_correct": true,
  "solution_steps_log": "string",
  "num_steps": 7,
  "termination": "solved"
}
```

This object is passed to the Teacher LLM when `is_correct == false`.
