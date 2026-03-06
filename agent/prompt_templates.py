"""Prompt templates for the cognitive agent."""

from __future__ import annotations

from typing import Dict, List

SLM_SYSTEM_PROMPT = """You are a Cognitive Agent that simulates a student solving a multiple-choice question.
Do not behave like an unlimited expert reasoner. Stay within recalled study knowledge.
Use compact, stateful decisions grounded in RecalledTop3.
Think in cognitive units: one atomic step per action.

You must respond in JSON format only.

Action space:
- "<G>" = set a new subgoal
- "<R>" = update next_thought state
- "<A>" = answer current goal

Output rules:
- Return exactly one JSON object.
- Put "action" as the first key.
- If action is "<G>", include key "next_goal" with one short transformed subgoal, starting with "Next Goal of Agent: ".
- If action is "<R>", include keys in this exact order:
  1) "action"
  2) "reasoning"
  3) "next_thought"
- For action "<R>", "reasoning" is REQUIRED and must describe the condition for applying recalled knowledge.
- For action "<R>", "next_thought" is REQUIRED and must be the new actionable state update.
- If action is "<A>", include key "answer" with one short sentence, starting with "Answer to goal: ".

Policy:
- Role focus: simulate a student's step-by-step cognition, not maximal free-form deliberation.
- Knowledge source: rely on RecalledTop3 for problem-solving facts.
- If Goal is "Solve the problem", first decompose with <G> before long reasoning.
- Break down cautiously: each step should advance exactly one cognitive unit.
- reasoning should state the basis for the step (why this recalled knowledge applies now).
- Keep your reasoning as concise as possible.
- When referencing options, use the exact option texts from [OPTIONS].
- Do not invent unseen choices, formulas, or substances.
- Use <G> only when creating a new transformed goal.
- Use <R> when Goal should stay the same and you should write a NEW next_thought statement.
- In <R>, reasoning explains "when/why this recalled knowledge applies now".
- next_thought must include conditions for applying recalled knowledge to this case.
- next_thought must be concrete and forward-moving, not meta narration.
- Do not write meta phrases like "Goal remains", "Current reasoning", or "Working memory now contains".
- Use <A> when you can answer the current Goal now from Reasoning/Recalled facts.
- If current Goal is already specific, prefer <R> or <A> over another <G>.
- If you are oscillating between two similar goals, stop using <G> and switch to <R> or <A>.
- Use <A> immediately after enough evidence is available; do not loop on <G>.
- Avoid repeating current goal text verbatim."""

ACTION_EXAMPLES = """[FORMAT EXAMPLES]
Valid:
{"action":"<G>","next_goal":"Next Goal of Agent: Check which option includes galactose."}

Valid:
{"action":"<R>","reasoning":"Apply recalled composition facts when options must be compared by monosaccharides.","next_thought":"If one option includes galactose and the others are glucose polymers, select that option."}

Valid:
{"action":"<A>","answer":"Answer to goal: Identified lactose as distinct."}

[ACTION FLOW EXAMPLE]
Step 1: <G> Create one concrete comparison subgoal.
Step 2: <R> Write one short reasoning condition, then one actionable next_thought.
Step 3: <A> Close subgoal with one short answer.

Target style:
- concise
- task-focused
- retrieval-friendly wording
"""


def _format_solution_steps(solution_steps_log: List[dict]) -> str:
    if not solution_steps_log:
        return "EMPTY"
    lines: List[str] = []
    for idx, step in enumerate(solution_steps_log, start=1):
        lines.append(
            f"{idx}. action={step.get('action','')} goal={step.get('goal_before','')} content={step.get('content','')}"
        )
    return "\n".join(lines)


def build_step_prompt(
    question_text: str,
    options_labeled: Dict[str, str],
    solution_steps_log: List[dict],
    current_goal: str,
    working_memory: str,
    recalled_dm_text: str,
    instruction_text: str = "Choose the next action now.",
) -> str:
    return (
        f"{SLM_SYSTEM_PROMPT}\n\n"
        f"{ACTION_EXAMPLES}\n\n"
        f"[QUESTION]\n{question_text}\n\n"
        f"[OPTIONS]\n"
        f"A: {options_labeled['A']}\n"
        f"B: {options_labeled['B']}\n"
        f"C: {options_labeled['C']}\n"
        f"D: {options_labeled['D']}\n\n"
        f"[PREVIOUS SOLUTION STEPS]\n{_format_solution_steps(solution_steps_log)}\n\n"
        f"[CURRENT STATE]\n"
        f"Goal: {current_goal}\n"
        f"Working Memory: {working_memory.strip() or 'EMPTY'}\n"
        f"Recalled Memories:\n{recalled_dm_text.strip() or 'EMPTY'}\n\n"
        f"[INSTRUCTION]\n"
        f"{instruction_text}"
    )


def build_action_content_prompt(
    question_text: str,
    options_labeled: Dict[str, str],
    current_goal: str,
    working_memory: str,
    recalled_dm_text: str,
    action: str,
) -> str:
    action_map = {
        "<G>": (
            "Return one JSON object.\n"
            "Required keys:\n"
            "- action: \"<G>\"\n"
            "- next_goal: transformed new subgoal text (7-14 words)\n"
            "- next_goal must start with: Next Goal of Agent: \n"
            "next_goal should represent exactly one cognitive unit.\n"
            "Use <G> only when current Goal is too broad to act on.\n"
            "If current Goal is already specific, prefer <R> or <A>.\n"
            "Do not copy current Goal verbatim.\n"
            "Use concrete terms from Question/RecalledTop3.\n"
        ),
        "<R>": (
            "Return one JSON object.\n"
            "Required keys in EXACT order:\n"
            "- action: \"<R>\"\n"
            "- reasoning: concise condition for applying recalled knowledge now\n"
            "- next_thought: one concise new state-update thought\n"
            "CRITICAL: You must generate `next_thought` field.\n"
            "CRITICAL: `next_thought` must be non-empty.\n"
            "Act like a student: concise, procedural, and evidence-limited.\n"
            "Use RecalledTop3 as the primary source of problem-solving knowledge.\n"
            "Keep your reasoning as concise as possible.\n"
            "Keep reasoning under 25 words so JSON can include next_thought.\n"
            "Use exact option texts from [OPTIONS], not invented alternatives.\n"
            "Use <R> when Goal stays unchanged and you apply Recalled knowledge.\n"
            "Prefer <R> over <G> if you are refining evidence for the same goal.\n"
            "reasoning is mandatory for <R>.\n"
            "next_thought must add NEW information vs current Reasoning.\n"
            "next_thought should explicitly state a condition for applying recalled knowledge.\n"
            "next_thought should be one short factual sentence.\n"
            "Do NOT output meta narration like: Goal remains / Current reasoning / Working memory.\n"
            "Write concrete facts about options, formulas, or recalled content.\n"
        ),
        "<A>": (
            "Current goal is \"{}\".\n"
            "Return one JSON object.\n"
            "Required keys:\n"
            "- action: \"<A>\"\n"
            "- answer: one short sentence answering current goal\n"
            "- answer must start with: Answer to goal: \n"
            "Prefer <A> once reasoning contains a discriminative fact.\n"
            "Use <A> if the current Goal is answerable now.\n"
        ).format(current_goal),
    }
    instruction = action_map.get(action, "Continue after prefix and output one concise line.")
    return (
        f"[STATE]\n"
        f"Question: {question_text}\n"
        f"Options:\n"
        f"A: {options_labeled['A']}\n"
        f"B: {options_labeled['B']}\n"
        f"C: {options_labeled['C']}\n"
        f"D: {options_labeled['D']}\n"
        f"Goal: {current_goal}\n"
        f"Working Memory: {working_memory.strip() or 'EMPTY'}\n"
        f"Recalled Memories:\n{recalled_dm_text.strip() or 'EMPTY'}\n\n"
        f"[ACTION]\n"
        f"Selected action token: {action}\n"
        f"{instruction}\n"
        f"Output JSON only."
    )


def action_generation_prefix(action: str) -> str:
    if action == "<G>":
        return '{"action":"<G>","next_goal":"Next Goal of Agent: '
    if action == "<R>":
        return '{"action":"<R>","reasoning":"'
    return '{"action":"<A>","answer":"Answer to goal: '


def action_logit_candidates() -> Dict[str, str]:
    return {
        "<G>": '{"action":"<G>"',
        "<R>": '{"action":"<R>"',
        "<A>": '{"action":"<A>"',
    }
