"""Push-to-ask: the candidate holds a key for exactly the question they want
answered, so nothing has to guess where a question ended.

Checks:
  1. A question with a long mid-sentence pause stays ONE question.
  2. Audio outside the hold is discarded entirely.
  3. A tap with no speech in it does not reach the model.

Backend must be running.

    python tests/test_push_to_ask.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import wave

import websockets

SR = 16000
CHUNK = 1024
STEP = CHUNK * 2


def synth(text: str, name: str) -> bytes:
    path = os.path.join(tempfile.gettempdir(), f"ic_push_{name}.wav")
    if not os.path.exists(path):
        subprocess.run(
            ["say", "-o", path, "--file-format=WAVE", "--data-format=LEI16@16000", text],
            check=True,
        )
    with wave.open(path, "rb") as w:
        return w.readframes(w.getnframes())


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(SR * seconds)


class Client:
    """Streams audio at real-time pace and records what the server decided."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.questions: list[str] = []
        self.finals: list[str] = []
        self.answers: list[dict] = []
        self.cancels: list[str] = []
        self.errors: list[str] = []

    async def reader(self) -> None:
        async for raw in self.ws:
            if isinstance(raw, bytes):
                continue
            ev = json.loads(raw)
            t = ev.get("type")
            if t == "question":
                self.questions.append(ev["text"])
                print(f"    [question] {ev['text']!r}")
            elif t == "transcript_final":
                self.finals.append(ev["text"])
                print(f"    [final]    {ev['text']!r}")
            elif t == "answer_cancelled":
                self.cancels.append(ev.get("question", ""))
                print(f"    [CANCELLED] {ev.get('question')!r}")
            elif t == "answer_done":
                self.answers.append(ev)
                print(f"    [answer]   {(ev.get('text') or '')[:80]}...")
            elif t == "error":
                self.errors.append(ev["message"])
                print(f"    [error]    {ev['message']}")

    async def send_audio(self, pcm: bytes) -> None:
        for i in range(0, len(pcm), STEP):
            await self.ws.send(pcm[i : i + STEP])
            await asyncio.sleep((STEP / 2) / SR)

    async def hold(self, pcm: bytes) -> None:
        await self.ws.send(json.dumps({"type": "capture_start"}))
        await self.send_audio(pcm)
        await self.ws.send(json.dumps({"type": "capture_stop"}))


failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


async def main(url: str, settle: float) -> int:
    p1 = synth("Can you explain what a transformer is", "p1")
    p2 = synth("and why we use attention", "p2")
    junk = synth("Right, let me just find your resume, one moment", "junk")

    # 1. one question, held across a long pause
    print("\n1. A 2s pause in the middle of the question")
    async with websockets.connect(url, max_size=None) as ws:
        c = Client(ws)
        task = asyncio.create_task(c.reader())
        await asyncio.sleep(0.3)
        await ws.send(json.dumps({"type": "start", "mode": "technical", "capture_mode": "push"}))
        await asyncio.sleep(0.2)
        await c.hold(p1 + silence(2.0) + p2)
        await asyncio.sleep(settle)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.2)
        task.cancel()

    check("exactly one question", len(c.questions) == 1, f"{len(c.questions)} questions")
    check("no answer was discarded", not c.cancels)
    q = (c.questions[-1] if c.questions else "").lower()
    check("question kept the first half", "transformer" in q, q[:70])
    check("question kept the second half", "attention" in q, q[:70])
    if c.answers:
        a = (c.answers[-1].get("text") or "").lower()
        check("answer covers both halves", "transformer" in a and "attention" in a)

    # 2. audio outside the hold is discarded
    print("\n2. Talking while the key is up")
    async with websockets.connect(url, max_size=None) as ws:
        c2 = Client(ws)
        task = asyncio.create_task(c2.reader())
        await asyncio.sleep(0.3)
        await ws.send(json.dumps({"type": "start", "mode": "technical", "capture_mode": "push"}))
        await asyncio.sleep(0.2)
        await c2.send_audio(junk + silence(1.5))          # key UP - must be ignored
        await c2.hold(synth("What is RAG", "rag"))        # key DOWN - the real question
        await asyncio.sleep(settle)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.2)
        task.cancel()

    check("only the held audio became a question", len(c2.questions) == 1, f"{len(c2.questions)} questions")
    joined = " ".join(c2.finals).lower()
    check("un-held speech never transcribed", "resume" not in joined and "moment" not in joined, joined[:70])
    check("held question transcribed", any("rag" in f.lower() for f in c2.finals), str(c2.finals))

    # 3. an accidental tap
    print("\n3. An accidental tap with no speech")
    async with websockets.connect(url, max_size=None) as ws:
        c3 = Client(ws)
        task = asyncio.create_task(c3.reader())
        await asyncio.sleep(0.3)
        await ws.send(json.dumps({"type": "start", "mode": "technical", "capture_mode": "push"}))
        await asyncio.sleep(0.2)
        await c3.hold(silence(0.15))
        await asyncio.sleep(3.0)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.2)
        task.cancel()

    check("a tap asks the model nothing", not c3.questions, str(c3.questions))

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        return 1
    print("all push-to-ask checks passed")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000/ws")
    ap.add_argument("--settle", type=float, default=22.0)
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.url, a.settle)))
