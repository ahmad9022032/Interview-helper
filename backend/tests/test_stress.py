"""Stress and soak test: hammer the running backend the way a messy real session
would, and check nothing leaks, crashes, or wedges.

Covers the awkward sequences a careful unit test never reaches - interrupting
yourself, changing settings mid-answer, two tabs at once, malformed frames - and
watches process memory and asyncio task counts for growth.

Backend must be running.

    python tests/test_stress.py
    python tests/test_stress.py --rounds 25
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

SR, STEP = 16000, 2048
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def synth(text: str, name: str) -> bytes:
    path = os.path.join(tempfile.gettempdir(), f"ic_stress_{name}.wav")
    if not os.path.exists(path):
        subprocess.run(
            ["say", "-o", path, "--file-format=WAVE", "--data-format=LEI16@16000", text],
            check=True,
        )
    with wave.open(path, "rb") as w:
        return w.readframes(w.getnframes())


def silence(sec: float) -> bytes:
    return b"\x00\x00" * int(SR * sec)


def backend_pid() -> str | None:
    out = subprocess.run(["pgrep", "-f", "uvicorn app.main:app"], capture_output=True, text=True)
    for pid in out.stdout.split():
        cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
        if "uvicorn" in cmd and "pgrep" not in cmd:
            return pid
    return None


def rss_mb(pid: str | None) -> float | None:
    """Resident memory of the backend, or None if we cannot see it."""
    if not pid:
        return None
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True).stdout.strip()
        return int(out) / 1024 if out else None
    except Exception:  # noqa: BLE001
        return None


class Session:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.events: list[dict] = []
        self.errors: list[str] = []

    async def reader(self) -> None:
        try:
            async for raw in self.ws:
                if isinstance(raw, bytes):
                    continue
                ev = json.loads(raw)
                self.events.append(ev)
                if ev.get("type") == "error":
                    self.errors.append(ev["message"])
        except Exception:  # noqa: BLE001 - closing mid-read is expected
            pass

    async def ctl(self, **msg) -> None:
        await self.ws.send(json.dumps(msg))

    async def audio(self, pcm: bytes, realtime: bool = False) -> None:
        for i in range(0, len(pcm), STEP):
            await self.ws.send(pcm[i : i + STEP])
            if realtime:
                await asyncio.sleep((STEP / 2) / SR)
            elif i % (STEP * 8) == 0:
                await asyncio.sleep(0)

    def count(self, kind: str) -> int:
        return sum(1 for e in self.events if e.get("type") == kind)


async def soak(url: str, rounds: int) -> None:
    """Many captures back to back, watching for growth and stray errors."""
    print(f"\n1. Soak: {rounds} captures back to back")
    q = synth("What is retrieval augmented generation", "q")
    pid = backend_pid()
    before = None

    async with websockets.connect(url, max_size=None) as ws:
        s = Session(ws)
        t = asyncio.create_task(s.reader())
        await asyncio.sleep(0.3)
        await s.ctl(type="start", mode="technical", capture_mode="push")
        for i in range(rounds):
            # Measure from a WARM baseline. Whisper and the model allocate their
            # working set on first real use, so comparing against a cold process
            # just measures start-up, not a leak.
            if i == 2:
                await asyncio.sleep(2.0)
                before = rss_mb(pid)
            await s.ctl(type="capture_start")
            await s.audio(silence(0.2) + q + silence(0.2))
            await s.ctl(type="capture_stop")
            await asyncio.sleep(0.9)          # do not wait for the full answer
            if i % 5 == 4:
                await s.ctl(type="clear")
        await asyncio.sleep(6.0)
        await s.ctl(type="stop")
        await asyncio.sleep(0.4)
        t.cancel()

    after = rss_mb(pid)
    qs = s.count("question")
    check("every capture produced a question", qs >= rounds - 1, f"{qs}/{rounds}")
    check("no errors during the soak", not s.errors, "; ".join(s.errors[:2]))
    if before and after:
        growth = after - before
        check("memory plateaus rather than leaking", growth < 150,
              f"warm {before:.0f}MB -> {after:.0f}MB ({growth:+.0f}MB over {rounds - 2} captures)")
    else:
        print("  skip memory check (backend process not visible)")
    check("backend still healthy", (await health(url)) == "ok")


async def health(url: str) -> str:
    import urllib.request

    base = url.replace("ws://", "http://").replace("/ws", "")
    try:
        with urllib.request.urlopen(f"{base}/api/health", timeout=10) as r:
            return json.load(r)["status"]
    except Exception as exc:  # noqa: BLE001
        return f"unreachable: {exc}"


async def interruptions(url: str) -> None:
    """Contradicting yourself mid-flight must never wedge the session."""
    print("\n2. Interrupting yourself")
    q = synth("Explain the transformer architecture", "q2")

    async with websockets.connect(url, max_size=None) as ws:
        s = Session(ws)
        t = asyncio.create_task(s.reader())
        await asyncio.sleep(0.3)
        await s.ctl(type="start", mode="technical", capture_mode="push")

        # capture_stop with no start, and two starts in a row
        await s.ctl(type="capture_stop")
        await s.ctl(type="capture_start")
        await s.ctl(type="capture_start")
        await s.audio(q)
        await s.ctl(type="capture_stop")
        await asyncio.sleep(2.5)

        # new question while the previous answer is still streaming
        await s.ctl(type="capture_start")
        await s.audio(q)
        await s.ctl(type="capture_stop")
        await asyncio.sleep(1.5)

        # settings churn mid-answer
        for d in ("architecture", "steps", "brief"):
            await s.ctl(type="depth", depth=d)
        await s.ctl(type="mode", mode="behavioral")
        await s.ctl(type="mode", mode="technical")
        await s.ctl(type="capture_mode", mode="auto")
        await s.ctl(type="capture_mode", mode="push")
        await s.ctl(type="clear")

        # stop while holding
        await s.ctl(type="capture_start")
        await s.audio(q[: len(q) // 2])
        await s.ctl(type="stop")
        await asyncio.sleep(3.0)

        # and the session still works afterwards
        await s.ctl(type="start", mode="technical", capture_mode="push")
        await s.ctl(type="capture_start")
        await s.audio(silence(0.2) + q + silence(0.2))
        await s.ctl(type="capture_stop")
        for _ in range(200):
            if s.count("answer_done"):
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.3)
        t.cancel()

    check("survived contradictory controls", s.count("answer_done") >= 1, f"{s.count('answer_done')} answers")
    fatal = [e for e in s.errors if "Could not understand" not in e and "Too short" not in e]
    check("no unexpected errors", not fatal, "; ".join(fatal[:2]))
    check("backend still healthy", (await health(url)) == "ok")


async def garbage(url: str) -> None:
    """Malformed and hostile frames must not kill the connection."""
    print("\n3. Malformed input")
    async with websockets.connect(url, max_size=None) as ws:
        s = Session(ws)
        t = asyncio.create_task(s.reader())
        await asyncio.sleep(0.3)
        for payload in [
            "not json at all",
            "[]",
            "null",
            '{"type": 12345}',
            '{"no_type": true}',
            '{"type": "capture_start", "extra": ' + json.dumps("x" * 50000) + "}",
            '{"type": "depth", "depth": "../../etc/passwd"}',
            '{"type": "mode", "mode": {"nested": true}}',
            '{"type": "capture_mode", "mode": 999}',
            '{"type": "ask", "text": ""}',
            '{"type": "ask", "text": ' + json.dumps("y" * 20000) + "}",
        ]:
            await ws.send(payload)
        await ws.send(b"\x01\x02\x03")            # odd-length PCM
        await ws.send(b"")                        # empty binary
        await ws.send(b"\x00" * 300000)           # a big chunk with no capture open
        await asyncio.sleep(1.5)
        await s.ctl(type="ping")
        for _ in range(40):
            if s.count("pong"):
                break
            await asyncio.sleep(0.1)
        alive = s.count("pong") > 0
        t.cancel()

    check("connection survived malformed frames", alive)
    check("backend still healthy", (await health(url)) == "ok")


async def concurrent(url: str) -> None:
    """Two tabs open at once, as happens when someone reopens the page."""
    print("\n4. Two connections at once")
    q = synth("What is a vector database", "q3")

    async def one(tag: str) -> Session:
        async with websockets.connect(url, max_size=None) as ws:
            s = Session(ws)
            t = asyncio.create_task(s.reader())
            await asyncio.sleep(0.2)
            await s.ctl(type="start", mode="technical", capture_mode="push")
            await s.ctl(type="capture_start")
            await s.audio(silence(0.2) + q + silence(0.2))
            await s.ctl(type="capture_stop")
            for _ in range(300):
                if s.count("answer_done"):
                    break
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.2)
            t.cancel()
            return s

    a, b = await asyncio.gather(one("A"), one("B"))
    check("both connections got an answer", a.count("answer_done") and b.count("answer_done"),
          f"A={a.count('answer_done')} B={b.count('answer_done')}")
    check("neither connection errored", not a.errors and not b.errors,
          "; ".join((a.errors + b.errors)[:2]))
    check("backend still healthy", (await health(url)) == "ok")


async def main(url: str, rounds: int) -> int:
    print(f"backend health before: {await health(url)}")
    await soak(url, rounds)
    await interruptions(url)
    await garbage(url)
    await concurrent(url)
    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        return 1
    print("all stress checks passed")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000/ws")
    ap.add_argument("--rounds", type=int, default=12)
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.url, a.rounds)))
