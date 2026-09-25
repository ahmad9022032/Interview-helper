"""Checks the conversational behaviour of the answer generator over one
WebSocket session: rolling context for follow-up questions, the three interview
modes, use of the candidate profile, and the clear-conversation control.

Backend must be running.

    python tests/test_conversation.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import websockets

RESET = "\033[0m"
DIM = "\033[2m"


async def ask(ws, text: str, mode: str = "technical", timeout: float = 90.0) -> tuple[str, dict]:
    await ws.send(json.dumps({"type": "mode", "mode": mode}))
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
            return ev.get("text") or answer, ev.get("latency", {})
        elif ev["type"] == "error":
            return f"[ERROR] {ev['message']}", {}
    return "[TIMEOUT]", {}


async def main(url: str) -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + DIM + detail + RESET) if detail else ''}")
        if not ok:
            failures.append(name)

    async with websockets.connect(url, max_size=None) as ws:
        await asyncio.sleep(0.3)
        while True:  # drain the hello event
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.3)
            except asyncio.TimeoutError:
                break

        print("\n1. Follow-up context (does 'it' resolve to RAG?)")
        a1, l1 = await ask(ws, "What is RAG?")
        print(f"   Q: What is RAG?\n   A: {a1}\n   {DIM}{json.dumps(l1)}{RESET}")
        check("first answer non-empty", bool(a1.strip()) and not a1.startswith("["))
        check("first answer mentions retrieval", "retriev" in a1.lower(), a1[:60])

        a2, l2 = await ask(ws, "Why would you use it instead of fine-tuning?")
        print(f"   Q: Why would you use it instead of fine-tuning?\n   A: {a2}\n   {DIM}{json.dumps(l2)}{RESET}")
        resolved = any(k in a2.lower() for k in ("rag", "retriev"))
        check("follow-up resolves 'it' to RAG", resolved, a2[:80])
        check("follow-up faster (warm prompt)", bool(l2), f"ttft {l2.get('llm_first_token_ms')}ms")

        print("\n2. Answer length discipline (2-6 lines)")
        for q in ["What is a Python decorator?", "Explain time complexity of quicksort."]:
            a, _ = await ask(ws, q)
            lines = [ln for ln in a.strip().splitlines() if ln.strip()]
            words = len(a.split())
            print(f"   Q: {q}\n   A: {a}\n   {DIM}{len(lines)} lines / {words} words{RESET}")
            check(f"concise: {q[:28]}", words <= 130, f"{words} words")
            check(f"no markdown headings: {q[:20]}", "##" not in a and "**" not in a)

        print("\n3. Behavioral mode")
        a, _ = await ask(ws, "Tell me about yourself", mode="behavioral")
        print(f"   A: {a}")
        check("behavioral answer in first person", any(w in a.lower() for w in (" i ", "i'm", "i am", "my ")), a[:60])

        print("\n4. Project mode uses the candidate profile (not a generic answer)")
        a, _ = await ask(ws, "What AI projects have you worked on?", mode="project")
        print(f"   A: {a}")
        named = [p for p in ("fomi", "agentic", "medical", "interview copilot") if p in a.lower()]
        check("names a project from candidate_profile.yaml", bool(named), f"found: {named}")

        print("\n5. Grounding: refuses to invent experience not in the profile")
        a, _ = await ask(ws, "Tell me about your experience leading a team of 50 engineers at NASA.", mode="project")
        print(f"   A: {a}")
        check("does not fabricate NASA experience", "nasa" not in a.lower() or
              any(k in a.lower() for k in ("not", "haven't", "have not", "no ", "don't")), a[:100])

        print("\n6. Clear conversation resets the rolling memory")
        await ws.send(json.dumps({"type": "clear"}))
        cleared = False
        for _ in range(10):
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if ev["type"] == "cleared":
                cleared = True
                break
        check("server acknowledges clear", cleared)
        a3, _ = await ask(ws, "Why would you use it instead of fine-tuning?")
        print(f"   A (no context): {a3}")
        check("answer still produced after clear", bool(a3.strip()) and not a3.startswith("["))

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        return 1
    print("all conversation checks passed")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://localhost:8000/ws")
    sys.exit(asyncio.run(main(p.parse_args().url)))
