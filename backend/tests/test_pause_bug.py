"""Reproduces the mid-question pause bug.

The interviewer pauses in the middle of a question. If the pause is longer than
VAD_SILENCE_MS the utterance ends, the first half is treated as a whole question
and answered, and the second half then arrives as a separate "question" that
cancels the first answer. The candidate ends up reading an answer to a fragment.

    python tests/test_pause_bug.py                 # 1.2s pause, should FAIL today
    python tests/test_pause_bug.py --pause 0.3     # short pause, should pass
    python tests/test_pause_bug.py --pause 2.5

Exit code 0 means the whole question was answered as one; 1 means it was split.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import wave

import websockets

SR = 16000
CHUNK = 1024
PART1 = "Can you explain what a transformer is"
PART2 = "and why we use attention"
FULL_WORDS = {"transformer", "attention"}


def synth(text: str, name: str) -> bytes:
    path = os.path.join(tempfile.gettempdir(), f"ic_pause_{name}.wav")
    if not os.path.exists(path):
        subprocess.run(
            ["say", "-o", path, "--file-format=WAVE", "--data-format=LEI16@16000", text],
            check=True,
        )
    with wave.open(path, "rb") as w:
        return w.readframes(w.getnframes())


async def run(args: argparse.Namespace) -> int:
    silence = b"\x00\x00" * SR
    stream = (
        silence
        + synth(PART1, "p1")
        + silence[: int(SR * 2 * args.pause) // 2 * 2]   # the mid-question pause
        + synth(PART2, "p2")
        + silence * 2
    )

    questions: list[str] = []
    answers: list[dict] = []
    cancels: list[str] = []
    finals: list[str] = []

    async with websockets.connect(args.url, max_size=None) as ws:
        async def reader() -> None:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                ev = json.loads(raw)
                t = ev.get("type")
                if t == "question":
                    questions.append(ev["text"])
                    print(f"  [question] {ev['text']!r}")
                elif t == "transcript_final":
                    finals.append(ev["text"])
                    print(f"  [final]    {ev['text']!r}")
                elif t == "answer_cancelled":
                    cancels.append(ev.get("question", ""))
                    print(f"  [CANCELLED] answer to {ev.get('question')!r}")
                elif t == "answer_done":
                    answers.append(ev)
                    print(f"  [answer]   {(ev.get('text') or '')[:90]}...")
                elif t == "error":
                    print(f"  [error]    {ev['message']}")

        task = asyncio.create_task(reader())
        await asyncio.sleep(0.3)
        await ws.send(json.dumps({"type": "start", "mode": "technical"}))

        print(f"streaming: {PART1!r} + {args.pause}s pause + {PART2!r}")
        t0 = time.perf_counter()
        step = CHUNK * 2
        for i in range(0, len(stream), step):
            await ws.send(stream[i : i + step])
            delay = t0 + ((i + step) / 2) / SR - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

        await asyncio.sleep(args.settle)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.3)
        task.cancel()

    print()
    print(f"transcripts : {finals}")
    print(f"questions   : {questions}")
    print(f"cancelled   : {cancels}")

    ok = True
    if len(questions) != 1:
        print(f"FAIL  the question was split into {len(questions)} separate questions")
        ok = False
    if cancels:
        print(f"FAIL  an in-flight answer was discarded ({len(cancels)}x)")
        ok = False
    final_q = questions[-1] if questions else ""
    missing = [w for w in FULL_WORDS if w not in final_q.lower()]
    if missing:
        print(f"FAIL  the answered question is missing {missing}: {final_q!r}")
        ok = False
    if answers:
        text = (answers[-1].get("text") or "").lower()
        if "attention" not in text:
            print("WARN  the final answer never mentions attention")

    print("\nRESULT:", "PASS - answered as one complete question" if ok else "FAIL - question was split")
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://localhost:8000/ws")
    p.add_argument("--pause", type=float, default=1.2, help="seconds of silence mid-question")
    p.add_argument("--settle", type=float, default=25.0, help="seconds to wait for answers after the audio")
    sys.exit(asyncio.run(run(p.parse_args())))


if __name__ == "__main__":
    main()
