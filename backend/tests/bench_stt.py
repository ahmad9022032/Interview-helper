"""Measures faster-whisper accuracy and latency per model size on this machine,
so you can pick WHISPER_MODEL with real numbers instead of guessing.

    python tests/bench_stt.py                      # tiny, base, small
    python tests/bench_stt.py --models base small medium
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings  # noqa: E402
from app.stt.whisper import WhisperEngine  # noqa: E402

QUESTIONS = [
    "Can you explain what a transformer is and why we use attention?",
    "What is the difference between a list and a tuple in Python?",
    "Tell me about a time you had to debug a production issue.",
]


def synth(text: str, idx: int) -> np.ndarray:
    path = os.path.join(tempfile.gettempdir(), f"ic_bench_{idx}.wav")
    if not os.path.exists(path):
        subprocess.run(
            ["say", "-o", path, "--file-format=WAVE", "--data-format=LEI16@16000", text],
            check=True,
        )
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def wer(ref: str, hyp: str) -> float:
    """Word error rate, ignoring case and punctuation."""
    strip = str.maketrans("", "", ".,?!;:'\"")
    r = ref.lower().translate(strip).split()
    h = hyp.lower().translate(strip).split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / max(1, len(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tiny", "base", "small"])
    args = ap.parse_args()

    clips = [(q, synth(q, i)) for i, q in enumerate(QUESTIONS)]
    total_audio = sum(a.size for _, a in clips) / 16000
    print(f"{len(clips)} clips, {total_audio:.1f}s of audio\n")
    print(f"{'model':<12}{'load':>8}{'stt/clip':>11}{'realtime':>10}{'WER':>8}")
    print("-" * 49)

    for name in args.models:
        t0 = time.perf_counter()
        try:
            eng = WhisperEngine(
                name,
                settings.whisper_device,
                settings.whisper_compute_type,
                settings.whisper_language,
                settings.whisper_beam_size,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<12}  failed to load: {exc}")
            continue
        load = time.perf_counter() - t0
        eng.warmup()

        times, errors = [], []
        for ref, audio in clips:
            t = time.perf_counter()
            hyp = eng.transcribe(audio)
            times.append(time.perf_counter() - t)
            errors.append(wer(ref, hyp))
            if wer(ref, hyp) > 0:
                print(f"{'':<12}  {name} heard: {hyp!r}")
        avg = sum(times) / len(times)
        rtf = sum(times) / total_audio
        print(f"{name:<12}{load:>7.1f}s{avg * 1000:>10.0f}ms{rtf:>9.2f}x{sum(errors) / len(errors):>7.1%}")
        del eng

    print("\nrealtime = STT seconds per second of audio (lower is better).")
    print("Set the winner as WHISPER_MODEL in .env.")


if __name__ == "__main__":
    main()
