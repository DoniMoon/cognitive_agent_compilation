<!-- KB_Server.md -->

# KB Server Specification (Declarative Memory Store)

## 1. Purpose

The KB Server stores and serves **Declarative Memory (DM)** items for a Soar + ACT-R inspired **Cognitive Agent (CA)** that is executed by small instruction-tuned language models (SLMs).

The CA retrieves at most one DM item per step (top 1), based on the current:

- `goal_text` (current goal at top of the goal stack)
- `thinking_text` (current working memory content, also called workspace)

The Teacher LLM can only interact with the KB through the KB Server APIs (MCP-style tools) to:
- query top-k DM candidates for a query state
- compute the similarity score between a query state and a DM candidate
- commit new DM candidates (append-only)

The KB is cumulative across problems and epochs.

---

## 2. Core Data Model

### 2.1 DM Item Schema

Each DM item has three author-controlled text fields:

1. `dm_text`  
   - The knowledge payload that is injected into the SLM prompt as `Recalled`.
2. `dm_goal_text`  
   - A short description of the goal contexts where this DM is useful.
3. `dm_condition_text`  
   - A short description of the state conditions (workspace cues, question cues, error cues) where this DM is useful.

Recommended: keep each field short and atomic.

### 2.2 Canonical JSON Object

Store DM items as JSONL, one object per line:

```json
{
  "dm_id": "dm_00001234",
  "dm_text": "Glycogen, cellulose, and starch are polymers of glucose. Lactose is a disaccharide made of glucose and galactose.",
  "dm_goal_text": "Choose the option with a different monosaccharide composition.",
  "dm_condition_text": "The question compares carbohydrates by which monosaccharides they contain (polymer vs disaccharide).",
  "metadata": {
    "created_at": "2026-03-05T00:00:00Z",
    "created_by": "teacher_llm",
    "source_problem_id": "biochem_carbs_assemble_DIGT",
    "epoch": 1,
    "problem_iteration": 2
  }
}
```

---

## 3. Embedding and Retrieval

### 3.1 Embedding Model

- Use Hugging Face model: `google/embeddinggemma-300m`
- The KB server must embed:
  - query texts: `(goal_text, thinking_text)`
  - DM texts: `(dm_goal_text, dm_condition_text)`

Implementation requirement:
- Output vector must be L2-normalized to unit length for cosine similarity.

### 3.2 Similarity Terms

For each DM item `i`:

- `x_i = cosine( embed(goal_text), embed(dm_goal_text_i) )`
- `y_i = cosine( embed(thinking_text), embed(dm_condition_text_i) )`

### 3.3 Score Aggregation (LSE)

Use Log-Sum-Exp (LSE) with beta = 5:

- `beta = 5.0`
- `score_i = (1/beta) * log( exp(beta*x_i) + exp(beta*y_i) )`

Intuition:
- If either `goal_text` strongly matches `dm_goal_text` OR `thinking_text` strongly matches `dm_condition_text`, the score becomes high.
- This is a soft-max over two channels (goal-match and condition-match).

### 3.4 Retrieval Policy

Parameters:
- `top_k`: requested number of items (Teacher query)
- `top_1`: always used by CA
- `retrieval_threshold`: if `max(score_i) < retrieval_threshold`, return no DM (empty recall)

Defaults (explicit, so implementation is unambiguous):
- `beta = 5.0`
- `retrieval_threshold = 0.30`  
  Rationale: cosine similarities often cluster around 0.0 to 0.4 for short texts; 0.30 is a conservative initial gate.
  You can tune later, but this is the required default in the POC.

Return:
- For CA: return at most one DM item (top 1) or none.
- For Teacher: return top-k list with scores.

---

## 4. Deduplication and Commit Rules

The KB is append-only, but the server must prevent trivial duplicates.

### 4.1 Exact-Match Dedup

When committing a DM candidate, reject it if **any existing DM** has identical:
- `dm_text` after whitespace normalization (strip, collapse multiple spaces)
OR identical tuple:
- `(dm_goal_text, dm_condition_text)` after normalization

### 4.2 Near-Duplicate Optional Rule (POC Default: OFF)

Near-duplicate detection by cosine similarity between embeddings can be implemented, but in the POC it is OFF by default for simplicity.

Defaults:
- `enable_near_duplicate = false`

---

## 5. APIs (MCP Tool Contracts)

All APIs are pure JSON in/out. The exact method transport (HTTP, local function, etc.) is a code decision, but input/output schemas must match.

### 5.1 Tool: `kb.query_topk`

**Purpose:** For Teacher LLM use. Retrieve top-k matching DM items.

**Input**
```json
{
  "goal_text": "string",
  "thinking_text": "string",
  "k": 5
}
```

**Output**
```json
{
  "items": [
    {
      "dm_id": "dm_00001234",
      "score": 0.4123,
      "x_goal": 0.3901,
      "y_condition": 0.4207,
      "dm_text": "string",
      "dm_goal_text": "string",
      "dm_condition_text": "string"
    }
  ]
}
```

Notes:
- Always return `x_goal` and `y_condition` for debugging.
- Items sorted by `score` descending.

### 5.2 Tool: `kb.score`

**Purpose:** For Teacher LLM use. Compute score between a query and an arbitrary candidate goal/condition without committing it.

**Input**
```json
{
  "goal_text": "string",
  "thinking_text": "string",
  "candidate_dm_goal_text": "string",
  "candidate_dm_condition_text": "string"
}
```

**Output**
```json
{
  "score": 0.4012,
  "x_goal": 0.3777,
  "y_condition": 0.4111,
  "beta": 5.0
}
```

### 5.3 Tool: `kb.commit_dm_candidates`

**Purpose:** Append new DM items to KB (Teacher output finalization).

**Input**
```json
{
  "candidates": [
    {
      "dm_text": "string",
      "dm_goal_text": "string",
      "dm_condition_text": "string",
      "metadata": {
        "source_problem_id": "string",
        "epoch": 1,
        "problem_iteration": 2
      }
    }
  ]
}
```

**Output**
```json
{
  "added": [
    {"dm_id": "dm_00001235"}
  ],
  "skipped_duplicates": [
    {"reason": "exact_match_dm_text", "candidate_index": 0}
  ],
  "kb_size": 1235
}
```

---

## 6. Persistence Format

Minimum persistence requirements:
- `kb/dm_items.jsonl` (append-only)
- `kb/emb_goal.npy` (float32 matrix: N x D) for `dm_goal_text`
- `kb/emb_condition.npy` (float32 matrix: N x D) for `dm_condition_text`
- `kb/meta.json` including embedding model name and dimension

On commit:
- append to JSONL
- compute embeddings for new items
- append to embedding matrices

---

## 7. Performance Notes (POC Constraints)

- N is expected to grow across 1642 problems and multiple iterations. Use a vector index if needed later.
- POC can do brute-force cosine similarity with normalized embeddings if N is not too large (few tens of thousands).
- Cache query embeddings per CA step to avoid re-embedding repeated text.

---

## 8. Required Unit Tests

1. Score formula correctness:
   - compare server score with a direct Python implementation for fixed vectors.
2. Retrieval threshold gating:
   - ensure empty return when max score below threshold.
3. Commit dedup:
   - ensure exact duplicate dm_text is rejected.
4. Deterministic ordering:
   - ensure stable sort by score, then dm_id for tie-break.
