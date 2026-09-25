"""Unit tests for the pure logic: chain-of-thought stripping, question
detection and rolling history. No server, no models needed.

    python tests/test_units.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.history import ConversationHistory  # noqa: E402
from app.llm.ollama_client import ThinkStripper  # noqa: E402
from app.question_detector import QuestionDetector  # noqa: E402

failures: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {name}")


def run(stripper: ThinkStripper, chunks: list[str]) -> tuple[str, int]:
    """Feed chunks and replay the resets, returning the final visible text."""
    text = ""
    resets = 0
    for chunk in chunks:
        for d in stripper.feed(chunk):
            if d.reset:
                text = ""
                resets += 1
            text += d.text
    for d in stripper.flush():
        if d.reset:
            text = ""
            resets += 1
        text += d.text
    return text, resets


print("ThinkStripper")
# Plain answer from a non-thinking model streams untouched, with no reset.
check("plain", run(ThinkStripper(), ["RAG combines ", "retrieval ", "with generation."]),
      ("RAG combines retrieval with generation.", 0))
# Balanced think block.
check("balanced", run(ThinkStripper(), ["<think>let me reason</think>The answer."]),
      ("The answer.", 0))
# Tags split across chunk boundaries.
check("split tags", run(ThinkStripper(), ["<thi", "nk>hidden</thi", "nk>Visible."]),
      ("Visible.", 0))
# Qwen3 under Ollama think:false - reasoning first, unmatched closing tag.
check("unmatched close", run(ThinkStripper(), ["We are asked about RAG.\n", "It retrieves.\n</think>\n\n", "RAG combines retrieval with generation."]),
      ("RAG combines retrieval with generation.", 1))
# Unmatched close split across chunks.
check("unmatched split", run(ThinkStripper(), ["reasoning </th", "ink> Real answer"]),
      ("Real answer", 1))
# A lone "<" or a partial tag must not be swallowed forever.
check("angle bracket", run(ThinkStripper(), ["a < b and c > d"]), ("a < b and c > d", 0))
# Multiple think blocks.
check("two blocks", run(ThinkStripper(), ["<think>a</think>One. <think>b</think>Two."]),
      ("One. Two.", 0))
# Text before an opening tag is kept.
check("text then think", run(ThinkStripper(), ["Answer. <think>why</think> More."]),
      ("Answer.  More.", 0))

print("QuestionDetector")
d = QuestionDetector(min_words=4)
check("question mark", d.feed("What is RAG?"), "What is RAG?")
check("dedupe repeat", d.feed("What is RAG?"), None)
check("interrogative start", d.feed("Why would you use it instead of fine-tuning"),
      "Why would you use it instead of fine-tuning")
check("imperative", d.feed("Tell me about yourself"), "Tell me about yourself")
d2 = QuestionDetector(min_words=4)
check("fragment held", d2.feed("So,"), None)
check("fragment merged", d2.feed("what is attention?"), "So, what is attention?")
d3 = QuestionDetector(min_words=4)
check("long statement accepted", d3.feed("Explain the difference between a list and a tuple"),
      "Explain the difference between a list and a tuple")
d4 = QuestionDetector(min_words=4)
d4.feed("Okay so")
check("flush pending", d4.flush(), "Okay so")
check("empty ignored", QuestionDetector().feed("   "), None)

print("ConversationHistory")
h = ConversationHistory(max_turns=2)
h.add("q1", "a1")
h.add("q2", "a2")
h.add("q3", "a3")
check("rolling window", h.turns(), [("q2", "a2"), ("q3", "a3")])
h.add("q4", "")
check("empty answer ignored", len(h), 2)
h.clear()
check("clear", len(h), 0)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    sys.exit(1)
print("all unit tests passed")
