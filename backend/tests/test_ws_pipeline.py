"""End-to-end test client for the real-time pipeline.

Streams a spoken question (synthesized with macOS `say`, or any 16 kHz mono
16-bit WAV you pass) to the running backend over the WebSocket at real-time
pace, exactly like the browser does, and prints every event plus latencies.

Usage (backend must be running on :8000):
    python tests/test_ws_pipeline.py                      # synthesize + stream default question
    python tests/test_ws_pipeline.py --wav my_question.wav
    python tests/test_ws_pipeline.py --ask "What is RAG?"  # skip audio, test LLM only
    python tests/test_ws_pipeline.py --text "Why use attention?" --mode technical
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

try:
    import websockets
except ImportError:  # pragma: no cover
    print("pip install websockets")
    sys.exit(1)

DEFAULT_QUESTION = "Can you explain what a transformer is and why we use attention?"
CHUNK = 1024  # samples per binary frame (~64 ms), same as the browser worklet
SR = 16000


def synthesize(text: str) -> str:
    """Use macOS `say` to create a 16 kHz mono 16-bit WAV. Returns the path."""
    path = os.path.join(tempfile.gettempdir(), "interview_copilot_test_q.wav")
    subprocess.run(
        ["say", "-o", path, "--file-format=WAVE", "--data-format=LEI16@16000", text],
        check=True,
    )
    return path


def read_wav(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1, "need mono"
        assert w.getsampwidth() == 2, "need 16-bit"
        assert w.getframerate() == SR, f"need {SR} Hz"
        return w.readframes(w.getnframes())


async def run(args: argparse.Namespace) -> int:
    uri = args.url
    events: list[dict] = []
    answer_done = asyncio.Event()
    got_question = asyncio.Event()

    async with websockets.connect(uri, max_size=None) as ws:
        async def reader() -> None:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                ev = json.loads(raw)
                events.append(ev)
                t = ev.get("type")
                if t == "answer_delta":
                    print(ev["text"], end="", flush=True)
                    continue
                if t == "question":
                    got_question.set()
                    print(f"\n[question] {ev['text']}")
                    print("[answer] ", end="", flush=True)
                elif t == "answer_done":
                    print("\n[latency]", json.dumps(ev["latency"]))
                    answer_done.set()
                elif t == "transcript_partial":
                    print(f"[partial] {ev['text']}")
                elif t == "transcript_final":
                    print(f"[final] {ev['text']}  (stt {ev['stt_ms']} ms)")
                elif t == "error":
                    print(f"[error] {ev['message']}")
                    if ev.get("stage") == "llm":
                        answer_done.set()
                else:
                    print(f"[{t}] {json.dumps({k: v for k, v in ev.items() if k != 'type'})}")

        reader_task = asyncio.create_task(reader())
        await asyncio.sleep(0.2)

        if args.ask:
            await ws.send(json.dumps({"type": "mode", "mode": args.mode}))
            await ws.send(json.dumps({"type": "ask", "text": args.ask}))
        else:
            wav = args.wav or synthesize(args.text)
            pcm = read_wav(wav)
            print(f"streaming {len(pcm) / 2 / SR:.1f}s of audio from {wav}")
            await ws.send(json.dumps({"type": "start", "mode": args.mode}))
            t0 = time.perf_counter()
            # 1 s of leading silence, the speech, then 1.5 s trailing silence
            silence = b"\x00\x00" * SR
            stream = silence + pcm + silence + silence[: SR]
            step = CHUNK * 2
            sent = 0
            for i in range(0, len(stream), step):
                await ws.send(stream[i : i + step])
                sent += step
                # pace at real time
                target = t0 + (sent / 2) / SR
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            print(f"audio sent in {time.perf_counter() - t0:.1f}s; waiting for answer...")

        try:
            await asyncio.wait_for(answer_done.wait(), timeout=args.timeout)
        except asyncio.TimeoutError:
            print("\nTIMEOUT waiting for answer_done")
        if not args.ask:
            await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.2)
        reader_task.cancel()

    types = [e.get("type") for e in events]
    ok_question = "question" in types
    ok_answer = any(e.get("type") == "answer_done" and e.get("text") for e in events)
    print("\nRESULT:", "PASS" if ok_question and ok_answer else "FAIL", f"(events: {types})")
    return 0 if ok_question and ok_answer else 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://localhost:8000/ws")
    p.add_argument("--wav", help="16 kHz mono 16-bit WAV to stream")
    p.add_argument("--text", default=DEFAULT_QUESTION, help="question to synthesize with `say`")
    p.add_argument("--ask", help="send a typed question instead of audio")
    p.add_argument("--mode", default="technical", choices=["technical", "behavioral", "project"])
    p.add_argument("--timeout", type=float, default=90.0)
    args = p.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
