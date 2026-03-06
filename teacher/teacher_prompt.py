"""Teacher prompt templates."""

from __future__ import annotations

import json
from typing import Dict

MCP_TOOL_CONTRACTS = {
    "kb.query_topk": {
        "input": {"goal_text": "string", "thinking_text": "string", "k": 5},
        "output": {
            "items": [
                {
                    "dm_id": "string",
                    "score": 0.0,
                    "x_goal": 0.0,
                    "y_condition": 0.0,
                    "dm_text": "string",
                    "dm_goal_text": "string",
                    "dm_condition_text": "string",
                }
            ]
        },
    },
    "kb.score": {
        "input": {
            "goal_text": "string",
            "thinking_text": "string",
            "candidate_dm_goal_text": "string",
            "candidate_dm_condition_text": "string",
        },
        "output": {"score": 0.0, "x_goal": 0.0, "y_condition": 0.0, "beta": 5.0},
    },
}

TEACHER_MCP_PLANNER_PROMPT = """You are the Teacher LLM in a Cognitive Agent Distillation loop.

Before proposing final DM candidates, you can request MCP tool calls.

Available tools:
- kb.query_topk(goal_text, thinking_text, k)
- kb.score(goal_text, thinking_text, candidate_dm_goal_text, candidate_dm_condition_text)

Rules:
- Return ONLY JSON with this exact schema:
{"tool_requests":[{"tool":"kb.query_topk","args":{...}}, {"tool":"kb.score","args":{...}}]}
- Use at most the limit given by the user prompt.
- If no tool calls are needed, return {"tool_requests":[]}
- Do not output commentary.
"""

TEACHER_SYSTEM_PROMPT = """You are the Teacher LLM in a Cognitive Agent Distillation loop.

Your only job is to propose NEW Declarative Memory (DM) items to be appended to the KB so that small cognitive agents can solve the given multiple-choice problem.

Constraints:
- You may not modify or delete existing DM items.
- Do not mention option letters (A/B/C/D) in dm_text because options are shuffled.
- Each DM must have exactly three fields: dm_text, dm_goal_text, dm_condition_text.
- dm_text must be short and directly usable by the small model.
- dm_goal_text and dm_condition_text must be written to maximize retrieval score given how the agent represents Goal and WorkingMemory (reasoning state).
- If agents keep failing across iterations, decompose knowledge into smaller atomic DMs instead of one broad DM.
- It is acceptable to output many candidates when needed (up to the maximum allowed by the caller).
- Candidate count is NOT a target; output only the set you want to keep in the per-problem temporary KB.
- Soft target: when budget allows and failure is non-trivial, propose at least 10 candidates.
- Prefer multiple retrieval-friendly DMs that each map to a specific failure pattern in traces.
- Include DM candidates for subgoal-generation failures, not only factual recall failures.
- When traces show malformed/repeated goals, add "goal-construction" DMs that teach how to create transformed, specific next subgoals.
- For subgoal-construction DMs, dm_goal_text should target goal-writing intents and dm_condition_text should target repeated/no-op goal patterns.
- Interpret action semantics as: <G>=new transformed goal, <R>=reasoning update with knowledge-application condition, <A>=answer.
- Do NOT add always-on, task-specific reset rules such as "If stuck, reset goal to X" without tight conditions.
- If a DM is specific, dm_condition_text must be equally specific so it activates only in the intended failure context.
- Avoid DMs that can hijack unrelated problems; prioritize transfer-safe procedural guidance.
- Write as if teaching an elementary student: use explicit, concrete, step-by-step DM wording.
- If SLM fails to update goal (repeated/no-op <G>), add at least one "goal reset" DM.
- For each goal reset DM, set dm_goal_text to the exact previous failing goal text from trace (goal_before at failure step).
- In dm_condition_text, describe the failure trigger (e.g., repeated <G>, unchanged goal, malformed JSON tail).
- For loop failures, prefer DMs that encourage switching from <G> to <R>/<A> once a specific goal already exists.
- Assume retrieval is embedding-based on (goal_text, thinking_text) versus (dm_goal_text, dm_condition_text); write fields to maximize that match.

Output format:
Return ONLY a JSON object of the form:
{"candidates":[{"dm_text":"...","dm_goal_text":"...","dm_condition_text":"..."}, ...]}

No extra commentary."""


def build_teacher_user_prompt(teacher_input: Dict, max_new_dm_per_call: int) -> str:
    recommended_min = min(10, int(max_new_dm_per_call))
    return (
        "Generate NEW DM candidates for this failure case.\n"
        "If repeated failures are present, split the missing knowledge into finer-grained DMs.\n"
        "Also define DMs for new subgoal construction behavior when action/goal loops are observed.\n"
        "If goal update fails, include goal-reset DM(s) and use previous failing goal text as dm_goal_text.\n"
        f"Maximum candidates: {max_new_dm_per_call}.\n"
        f"Recommended minimum candidates: {recommended_min} (unless failure is trivial).\n"
        "Return UP TO this number, not necessarily exactly this number.\n\n"
        "Write as if teaching an elementary student: be concrete, explicit, and step-by-step.\n\n"
        "Important: The JSON input includes `cognitive_agent_mechanics` that explains how the Cognitive Agent operates.\n"
        "Use that mechanism to write retrieval-friendly dm_goal_text/dm_condition_text.\n\n"
        "If `slm_conclusion_notice` is non-empty, treat it as a priority failure signal: SLM could not conclude before max steps.\n"
        "In that case, include procedural DMs that help the agent terminate with a concrete <A> answer earlier.\n\n"
        "Focus on per-step traces in ca_runs[].solution_steps_log with fields:\n"
        "- action (selected action)\n"
        "- content (model output for that action)\n"
        "- retrieved_dm_text (DM recalled at that step)\n"
        "- goal_before / working_memory_before (+ after-fields if present; working_memory stores reasoning state)\n"
        "Do not expect full SLM step prompts in this input.\n\n"
        "Required coverage:\n"
        "- Factual DMs for domain knowledge gaps.\n"
        "- Procedural DMs for subgoal generation under action <G> (how to transform current goal into a new specific subgoal).\n"
        "- Procedural DMs for reasoning updates under action <R> (how to add NEW reasoning with explicit knowledge-application conditions instead of repeating text).\n\n"
        "Guardrails for procedural DMs:\n"
        "- Avoid task-locked phrasing like fixed goal text resets unless dm_condition_text narrowly pins when it should fire.\n"
        "- Prefer anti-loop guidance that shifts behavior from repeated <G> to <R>/<A> after a specific goal exists.\n"
        "- When writing specific DMs, include specific trigger context so unrelated problems are not affected.\n\n"
        "The JSON may include `kb_tool_observations` from kb.query_topk and kb.score-style diagnostics.\n"
        "Use those observations to avoid duplicates and improve retrievability.\n\n"
        "Retrieval targeting guidance:\n"
        "- The agent queries with goal_text=current goal and thinking_text=current working memory.\n"
        "- Similarity score combines goal-side and condition-side embedding similarities.\n"
        "- Therefore, write dm_goal_text to closely mirror likely failing goal strings, and dm_condition_text to mirror failure context in working memory/trace.\n\n"
        "MCP tool contracts:\n"
        f"{json.dumps(MCP_TOOL_CONTRACTS, ensure_ascii=True)}\n\n"
        "Teacher input JSON:\n"
        f"{json.dumps(teacher_input, ensure_ascii=True)}"
    )


def build_teacher_tool_planner_prompt(
    teacher_input: Dict,
    max_query_topk_calls: int,
    max_score_calls: int,
) -> str:
    planner_payload = {
        "call_limits": {
            "kb.query_topk": max_query_topk_calls,
            "kb.score": max_score_calls,
        },
        "mcp_tool_contracts": MCP_TOOL_CONTRACTS,
        "teacher_input": teacher_input,
    }
    return (
        "Plan MCP tool calls before final DM generation.\n"
        "Request only calls that help diagnose failures or check retrievability.\n"
        "Return strict JSON only.\n\n"
        f"{json.dumps(planner_payload, ensure_ascii=True)}"
    )
