"""Decides when a finalized utterance is a complete question worth answering.

Short fragments are held and merged with the next utterance (within a time
window) so that "So..." + "what is RAG?" becomes one question instead of two
LLM calls. Identical repeats of the last question are ignored."""
from __future__ import annotations

import re
import time

_QUESTION_STARTS = (
    "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
    "can", "could", "would", "should", "will", "do", "does", "did", "is", "are",
    "was", "were", "have", "has", "had", "tell me", "explain", "describe",
    "walk me through", "walk us through", "talk about", "talk me through",
    "give me", "give an example", "share", "let's talk", "so ", "and ", "okay", "ok",
    "define", "compare", "differentiate", "elaborate", "any", "in your",
    "suppose", "imagine", "say you", "if you",
)

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


class QuestionDetector:
    def __init__(self, min_words: int = 4, merge_window_s: float = 8.0) -> None:
        self.min_words = min_words
        self.merge_window_s = merge_window_s
        self._pending: str = ""
        self._pending_time: float = 0.0
        self._last_question: str = ""

    def reset(self) -> None:
        self._pending = ""
        self._pending_time = 0.0
        self._last_question = ""

    @property
    def pending(self) -> str:
        return self._pending

    def looks_complete(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        lower = stripped.lower()
        n = _words(stripped)
        if stripped.endswith("?"):
            return True
        if lower.startswith(_QUESTION_STARTS) and n >= 3:
            return True
        return n >= self.min_words

    def feed(self, text: str, now: float | None = None) -> str | None:
        """Feed a finalized utterance. Returns a question to answer, or None."""
        now = time.monotonic() if now is None else now
        text = text.strip()
        if not text:
            return None
        if self._pending and now - self._pending_time <= self.merge_window_s:
            combined = f"{self._pending} {text}".strip()
        else:
            combined = text

        if self.looks_complete(combined):
            self._pending = ""
            return self._accept(combined)

        # too short to be a full question: hold it and wait for more
        self._pending = combined
        self._pending_time = now
        return None

    def flush(self, min_words: int = 2) -> str | None:
        """Force out whatever is pending (called after a hold timeout)."""
        pending, self._pending = self._pending, ""
        if pending and _words(pending) >= min_words:
            return self._accept(pending)
        return None

    def _accept(self, question: str) -> str | None:
        if _normalize(question) == _normalize(self._last_question):
            return None
        self._last_question = question
        return question
