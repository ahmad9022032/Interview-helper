"""Small rolling conversation memory (last N question/answer pairs)."""
from __future__ import annotations

from collections import deque


class ConversationHistory:
    def __init__(self, max_turns: int = 3) -> None:
        self._turns: deque[tuple[str, str]] = deque(maxlen=max(0, max_turns))

    def add(self, question: str, answer: str) -> None:
        if self._turns.maxlen and question.strip() and answer.strip():
            self._turns.append((question.strip(), answer.strip()))

    def clear(self) -> None:
        self._turns.clear()

    def turns(self) -> list[tuple[str, str]]:
        return list(self._turns)

    def __len__(self) -> int:
        return len(self._turns)
