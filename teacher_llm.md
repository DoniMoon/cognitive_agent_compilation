<!-- Teacher_LLM.md -->

# Teacher LLM Specification (Distillation into DM Candidates)

Teacher LLM is served on local port 8080 via the llama-server function in llama.cpp.

## 1. Role

The Teacher LLM is a strong model that **adds new DM items** to the KB when the Cognitive Agents fail.

Key constraint:
- The Teacher LLM is **append-only**. It cannot modify or delete existing DM items.
- The Teacher LLM does **not** read the KB directly. It can only use KB Server tools:
  - `kb.query_topk`
  - `kb.score`
  - `kb.commit_dm_candidates`

Goal:
- Add the minimal set of DM items that raises the pass metric above threshold for the current problem.
- DM items should generalize across similar problems and support future problems.

---

## 2. When the Teacher LLM is Called

For a given problem attempt, you run 8 CAs (one per SLM model) with the current KB.

Compute:
- `mean_correct_prob = mean( probs_i[correct_label] )` across the 8 models.

If:
- `mean_correct_prob >= 0.75` then the problem passes and Teacher is NOT called.
Else:
- Teacher is called to propose DM candidates, commit them, and the CAs retry.

---

## 3. Teacher Inputs

Teacher receives a structured input object:

### 3.1 Required Fields

- `problem_id`
- `question_text`
- `options_labeled` (A/B/C/D with the current shuffle)
- `correct_label` (A/B/C/D after shuffle)
- `gold_explanation` (optional, but available from dataset)
- `previous_new_dm_candidates_in_this_problem` (list of DM candidates already committed during the current problem loop)
- `ca_runs` (list of 8 CA result objects, see Cognitive_Agent.md Section 9)
  - Include full `solution_steps_log` for each CA, especially for incorrect runs.

### 3.2 Example Teacher Input (Sketch)

```json
{
  "problem_id": "biochem_carbs_assemble_DIGT",
  "question_text": "Which of the following carbohydrates contains a monosaccharide different from the others?",
  "options_labeled": {
    "A": "glycogen",
    "B": "cellulose",
    "C": "lactose",
    "D": "starch"
  },
  "correct_label": "C",
  "gold_explanation": "Glycogen, cellulose, and starch are all polymers of glucose only, whereas lactose is a disaccharide composed of glucose and galactose...",
  "previous_new_dm_candidates_in_this_problem": [],
  "ca_runs": [
    {"model_name":"...", "pred_label":"A", "probs":{...}, "is_correct":false, "solution_steps_log":"..."}
  ]
}
```

---

## 4. Teacher Output: DM Candidates

### 4.1 DM Candidate Schema

Teacher must output a list of objects:

- `dm_text`
- `dm_goal_text`
- `dm_condition_text`

All three are required.

### 4.2 DM Writing Guidelines (Strict)

Each DM must be:
- Atomic: one fact or one strategy pattern.
- Retrieval-friendly: `dm_goal_text` and `dm_condition_text` must match real query texts:
  - goal is a short intent (what the agent is trying to do)
  - condition is a short state cue (what the agent is seeing or confused about)

Avoid:
- Mentioning option letters (A/B/C/D). Options are shuffled each attempt.
- Overly problem-specific trivia unless unavoidable.
- Multi-paragraph dm_text.

Recommended style:
- dm_text: 1 to 3 sentences.
- dm_goal_text: 1 sentence.
- dm_condition_text: 1 sentence.

---

## 5. Teacher Procedure (Per Call)

Teacher should follow this exact procedure:

1. **Failure diagnosis**
   - Read incorrect CA traces.
   - Identify the minimal missing knowledge or missing subgoal strategy.
   - Decide whether the failure is:
     - domain knowledge gap (fact missing)
     - reasoning strategy gap (how to compare options)
     - state update gap (working memory not updated correctly)
     - retrieval mismatch (existing DM exists but not retrievable due to goal/condition phrasing)

2. **Check existing KB coverage**
   - Use `kb.query_topk` for likely goal/condition phrases extracted from traces.
   - If an existing DM already covers the needed knowledge, do NOT add a duplicate.
   - If coverage exists but retrieval phrasing mismatches, add a DM whose goal/condition phrasing better matches the CA prompts.

3. **Draft DM candidates**
   - Propose the smallest list that likely fixes the failure.
   - Prefer adding:
     - one strategy DM (how to approach the question)
     - plus one factual DM (key domain fact), only if needed

4. **Sanity score candidates**
   - Optionally call `kb.score` on each candidate to ensure it would be retrievable:
     - score should typically be >= 0.30 under expected goal/working memory phrases.

5. **Commit**
   - Output candidates in the required JSON format so the controller can call `kb.commit_dm_candidates`.

---

## 6. Teacher System Prompt (Exact)

Use this system prompt for Teacher LLM:

```text
You are the Teacher LLM in a Cognitive Agent Distillation loop.

Your only job is to propose NEW Declarative Memory (DM) items to be appended to the KB so that small cognitive agents can solve the given multiple-choice problem.

Constraints:
- You may not modify or delete existing DM items.
- Do not mention option letters (A/B/C/D) in dm_text because options are shuffled.
- Each DM must have exactly three fields: dm_text, dm_goal_text, dm_condition_text.
- dm_text must be short and directly usable by the small model.
- dm_goal_text and dm_condition_text must be written to maximize retrieval score given how the agent represents Goal and WorkingMemory.

Output format:
Return ONLY a JSON object of the form:
{"candidates":[{"dm_text":"...","dm_goal_text":"...","dm_condition_text":"..."}, ...]}

No extra commentary.
```

---

## 7. Maximum Candidate Count (POC Default)

To prevent KB explosion:
- `max_new_dm_per_teacher_call = 5` (required default)

If more knowledge seems needed, Teacher should add the most general 5 first and rely on another iteration.

---

## 8. Example Candidate Set

```json
{
  "candidates": [
    {
      "dm_text": "When asked which carbohydrate has a different monosaccharide, list the monosaccharide composition of each option and compare. Polysaccharides like starch, glycogen, and cellulose are typically glucose-only polymers.",
      "dm_goal_text": "Identify the option with a different monosaccharide composition.",
      "dm_condition_text": "The question asks which carbohydrate contains a different monosaccharide than the others."
    },
    {
      "dm_text": "Lactose is a disaccharide composed of glucose and galactose, unlike glucose-only polysaccharides.",
      "dm_goal_text": "Recall monosaccharide components of common carbohydrates.",
      "dm_condition_text": "One option is lactose and the task requires distinguishing glucose-only polymers vs mixed disaccharides."
    }
  ]
}
```
