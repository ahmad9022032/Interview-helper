"""Builds the (short) chat message list sent to the LLM."""
from __future__ import annotations

from .system_prompt import BASE_SYSTEM_PROMPT, BRIEF_REASONING, depth_addendum, mode_addendum


def build_messages(
    question: str,
    mode: str,
    candidate_context: str,
    history: list[tuple[str, str]],
    *,
    max_context_chars: int = 6000,
    max_history_answer_chars: int = 400,
    brief_reasoning: bool = False,
    depth: str = "brief",
) -> list[dict]:
    context = candidate_context.strip()
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n...[truncated]"

    system = BASE_SYSTEM_PROMPT + "\n" + mode_addendum(mode) + depth_addendum(depth)
    if brief_reasoning:
        system += "\n" + BRIEF_REASONING
    if context:
        system += "\n\nCANDIDATE CONTEXT (resume, skills, projects, experience):\n" + context
    else:
        system += "\n\nCANDIDATE CONTEXT: none provided. Do not claim any specific experience."

    messages: list[dict] = [{"role": "system", "content": system}]
    for q, a in history:
        a = a.strip()
        if len(a) > max_history_answer_chars:
            a = a[:max_history_answer_chars] + "..."
        messages.append({"role": "user", "content": q.strip()})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question.strip()})
    return messages
