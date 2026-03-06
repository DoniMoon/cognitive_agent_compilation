<!-- General_description.md -->

# Cognitive Agent Distillation POC: General Description

## 1. Project Goal

Build a proof-of-concept pipeline that:
1. uses a strong Teacher LLM to extract problem-solving knowledge,
2. compiles that knowledge into an explicit **Declarative Memory (DM) knowledge base**,
3. enables multiple small instruction-tuned models (SLMs) to solve biology MCQs using an explicit-state **Cognitive Agent (CA)**,
4. accumulates DM items across many problems,
5. treats the accumulated DM library as a candidate set of **Knowledge Components (KCs)** for analysis.

This POC does not attempt cognitive plausibility beyond the core mechanics:
- explicit goal stack
- explicit working memory text
- retrieval of explicit memory items (DM) using embeddings
- deterministic action choice via log-prob comparison

Reference background paper (provided in this workspace): CAD.pdf.

---

## 2. Key Concepts and Terminology

- SLM (Small Language Model): one of the 8 HF models used as the CA “engine”.
- CA (Cognitive Agent): the explicit-state solver that interacts with SLM.
- Goal stack: stack of goals (top is current); supports subgoals.
- Working Memory (WM) / Workspace: short string updated across steps.
- DM (Declarative Memory): explicit knowledge chunk with retrieval metadata.
- KB (Knowledge Base): persistent store of DM items (append-only).
- Recalled: the DM_text injected into the SLM prompt for the current step.
- Solution Steps: append-only trace of the CA’s decisions and generated content.

---

## 3. System Architecture (Modules)

### 3.1 Modules

1. **Dataset Loader**
   - reads JSONL problems
   - shuffles options and produces labeled (A/B/C/D) prompt material

2. **KB Server**
   - stores DM items
   - retrieves top 1 DM per CA step
   - exposes MCP-style tools for Teacher LLM

3. **Cognitive Agent Runner**
   - runs the CA loop for one SLM on one problem attempt
   - logs full trace and returns final answer probabilities

4. **Teacher Distiller**
   - invoked when the 8-model group performance is below threshold
   - proposes new DM candidates
   - commits them to KB

5. **Experiment Driver**
   - iterates problems and epochs
   - controls distillation loops
   - writes logs and KB snapshots

---

## 4. End-to-End Flow (One Problem)

Given a problem:
1. Shuffle options once and label A/B/C/D.
2. For each of the 8 SLMs:
   - Initialize CA state: goal stack, working memory, empty solution steps.
   - Repeat for up to 20 steps:
     - Retrieve a DM from KB using (current goal, working memory).
     - Prompt SLM with state and recalled DM.
     - Choose action by comparing log-probs for <1>/<2>/<3>.
     - Update goal stack or working memory, or answer subgoal/final.
   - At final top-level answer, choose A/B/C/D by log-prob comparison.
3. Compute `mean_correct_prob` across the 8 models.
4. If below threshold:
   - Provide all traces to Teacher LLM.
   - Teacher proposes DM candidates.
   - Commit DM candidates to KB.
   - Retry the 8 models on the same problem (new attempt iteration).
5. When passing:
   - move to next problem (KB persists).

---

## 5. Prompting Strategy Summary

The CA uses two kinds of prompts:
1. Step action prompt: SLM outputs <1>, <2>, or <3> followed by content.
2. Final answer prompt: prompt ends with `<3> final answer is:` and SLM outputs one letter A/B/C/D.

All prompt templates and system prompts are specified in Cognitive_Agent.md and Teacher_LLM.md.

---

## 6. Why This Supports KC Identification

The KB grows by adding DM items that are required to solve problems reliably across multiple SLMs.

You can later treat DM items as KC candidates by analyzing:
- which problems first required which DM items
- retrieval frequency patterns across problems
- clustering DM texts by embedding similarity
- mapping DM usage to error patterns and subgoal structures

This POC’s output is an explicit, inspectable, editable knowledge inventory.

---

## 7. Repository Layout (Recommended)

Suggested files/folders:

- `data/`
  - `biology_mcq.jsonl`
- `kb/`
  - `dm_items.jsonl`
  - `emb_goal.npy`
  - `emb_condition.npy`
  - `meta.json`
- `agent/`
  - `cognitive_agent.py`
  - `slm_inference.py`
  - `prompt_templates.py`
- `teacher/`
  - `teacher_controller.py`
  - `teacher_prompt.py`
- `experiments/`
  - `run_experiment.py`
  - `configs/`
  - `logs/`

---

## 8. POC Assumptions and Non-Goals

Assumptions:
- No forgetting model.
- DM retrieval is top 1 only.
- Teacher is append-only.
- Determinism is approximated by log-prob comparisons.

Non-goals in this POC:
- fine-tuning SLMs
- modeling human-like learning curves
- automatic KC clustering as part of the main loop (can be post-hoc)

---

## 9. Implementation Checklist (Concrete)

- [ ] Parse dataset JSONL
- [ ] Implement deterministic option shuffling and correct label mapping
- [ ] Implement KB server with embeddinggemma, LSE scoring, threshold gating
- [ ] Implement SLM wrapper with sequence log-prob scoring
- [ ] Implement CA loop with goal stack, WM, step logging, max step stop
- [ ] Implement Teacher controller: input packaging, JSON parsing, commit calls
- [ ] Implement experiment driver: per-problem loops, epoch loops, stopping
- [ ] Persist logs and KB snapshots
