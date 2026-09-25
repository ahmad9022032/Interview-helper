"""The answer-length selector: does each setting actually produce that much answer?

Short is the default you read straight out. The other three are opt-in from the
toolbar for questions that need room, and each gets its own token budget so a
longer answer is not simply cut off.

Backend must be running.

    python tests/test_answer_depth.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import websockets

QUESTION = "Explain the architecture of a RAG system."

# (depth, min lines, max lines, label) - generous bounds; we are checking the
# setting has a real, monotonic effect, not grading prose.
EXPECT = [
    ("brief", 1, 6, "Short answer, 2-5 lines"),
    ("detailed", 3, 12, "Detailed, 6-8 lines"),
    ("steps", 5, 18, "Step by step, 10 lines"),
    ("architecture", 5, 24, "Architecture explanation"),
]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def line_count(text: str) -> int:
    """Sentences or bullet lines, whichever the model used."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) > 1:
        return len(lines)
    return max(1, text.count(". ") + text.count(".\n") + (1 if text.strip().endswith(".") else 0))


async def ask(ws, depth: str, text: str, timeout: float = 180.0) -> tuple[str, dict]:
    await ws.send(json.dumps({"type": "depth", "depth": depth}))
    await ws.send(json.dumps({"type": "ask", "text": text}))
    answer, latency = "", {}
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        if isinstance(raw, bytes):
            continue
        ev = json.loads(raw)
        if ev["type"] == "answer_delta":
            answer += ev["text"]
        elif ev["type"] == "answer_reset":
            answer = ""
        elif ev["type"] == "answer_done":
            return (ev.get("text") or answer), ev
        elif ev["type"] == "error":
            return f"[ERROR] {ev['message']}", {}
    return "[TIMEOUT]", {}


async def main(url: str) -> int:
    results = []
    async with websockets.connect(url, max_size=None) as ws:
        await asyncio.sleep(0.3)
        while True:  # drain hello
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.3)
            except asyncio.TimeoutError:
                break

        for depth, lo, hi, label in EXPECT:
            print(f"\n{label}")
            text, ev = await ask(ws, depth, QUESTION)
            words, lines = len(text.split()), line_count(text)
            print(f"   {text.strip()[:220]}{'...' if len(text) > 220 else ''}")
            print(f"   -> {lines} lines / {words} words")
            results.append((depth, words, lines, text, ev))
            check(f"{depth}: produced an answer", bool(text.strip()) and not text.startswith("["))
            check(f"{depth}: within {lo}-{hi} lines", lo <= lines <= hi, f"{lines} lines")
            check(f"{depth}: not cut off at the token cap", not ev.get("truncated", False))

        # The whole point is that the setting changes the length.
        brief = next(r for r in results if r[0] == "brief")
        for depth in ("detailed", "steps", "architecture"):
            longer = next(r for r in results if r[0] == depth)
            check(f"{depth} is longer than short", longer[1] > brief[1],
                  f"{longer[1]} vs {brief[1]} words")

        steps = next(r for r in results if r[0] == "steps")[3]
        numbered = sum(1 for ln in steps.splitlines() if ln.strip()[:2].rstrip(".").isdigit())
        check("step-by-step is actually stepped", numbered >= 3, f"{numbered} numbered lines")

        arch = next(r for r in results if r[0] == "architecture")[3].lower()
        parts = [w for w in ("chunk", "embed", "vector", "retriev", "index", "generat", "prompt") if w in arch]
        check("architecture names real components", len(parts) >= 4, ", ".join(parts))

        # Back to the default so the session is left as it started.
        await ws.send(json.dumps({"type": "depth", "depth": "brief"}))
        await asyncio.sleep(0.2)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        return 1
    print("all answer-depth checks passed")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000/ws")
    sys.exit(asyncio.run(main(ap.parse_args().url)))
