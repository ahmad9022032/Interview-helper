"""Compares faster-whisper variants on accented interview speech.

Speech-to-text dominates the time between the interviewer finishing and the first
word of the answer, so this is where latency is won or lost. English-only (`.en`)
and distilled (`distil-*`) builds are often both faster AND more accurate than the
multilingual model of the same size, which is why the default is worth revisiting.

    python tests/bench_stt_variants.py
    python tests/bench_stt_variants.py --models small small.en distil-small.en
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
sys.path.insert(0, os.path.dirname(__file__))

from app.config import settings  # noqa: E402
from app.stt.whisper import WhisperEngine  # noqa: E402
from bench_stt import wer  # noqa: E402

# Four accents, and sentences full of the jargon an interviewer actually uses.
VOICES = ["Samantha", "Daniel", "Aman", "Karen"]
SENTENCES = [
    "Can you explain what a transformer is and why we use attention?",
    "How would you design a retrieval augmented generation pipeline?",
    "What is the difference between a process and a thread?",
    "Tell me about a time you had to debug a production issue.",
    "Why would you use RAG instead of fine tuning for a knowledge base?",
]
# Getting these wrong changes the question, so they are scored separately.
KEY_TERMS = ["transformer", "attention", "retrieval", "augmented", "generation",
             "process", "thread", "production", "rag", "fine", "tuning", "knowledge"]


def load_clips() -> list[tuple[str, str, np.ndarray]]:
    clips = []
    for voice in VOICES:
        for i, sentence in enumerate(SENTENCES):
            path = os.path.join(tempfile.gettempdir(), f"ic_var_{voice}_{i}.wav")
            if not os.path.exists(path):
                r = subprocess.run(
                    ["say", "-v", voice, "-o", path, "--file-format=WAVE",
                     "--data-format=LEI16@16000", sentence],
                    capture_output=True,
                )
                if r.returncode != 0:
                    print(f"  (voice {voice} unavailable, skipping)")
                    break
            with wave.open(path, "rb") as w:
                audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
            clips.append((voice, sentence, audio))
    return clips


def key_term_recall(ref: str, hyp: str) -> tuple[int, int]:
    low_ref, low_hyp = ref.lower(), hyp.lower()
    present = [t for t in KEY_TERMS if t in low_ref]
    kept = [t for t in present if t in low_hyp]
    return len(kept), len(present)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["base.en", "small", "small.en", "distil-small.en"])
    args = ap.parse_args()

    clips = load_clips()
    total_audio = sum(a.size for _, _, a in clips) / 16000
    print(f"{len(clips)} clips, {len(set(c[0] for c in clips))} accents, {total_audio:.1f}s of audio\n")
    print(f"{'model':<20}{'load':>7}{'ms/clip':>9}{'WER':>8}{'key terms':>11}   worst")
    print("-" * 86)

    rows = []
    for name in args.models:
        t0 = time.perf_counter()
        try:
            eng = WhisperEngine(name, settings.whisper_device, settings.whisper_compute_type,
                                "en", settings.whisper_beam_size)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<20}  failed to load: {str(exc)[:50]}")
            continue
        load = time.perf_counter() - t0
        eng.warmup()

        times, errs, kept, total_terms, worst = [], [], 0, 0, ("", 0.0)
        for voice, ref, audio in clips:
            t = time.perf_counter()
            hyp = eng.transcribe(audio)
            times.append((time.perf_counter() - t) * 1000)
            e = wer(ref, hyp)
            errs.append(e)
            k, n = key_term_recall(ref, hyp)
            kept += k
            total_terms += n
            if e > worst[1]:
                worst = (f"[{voice}] {hyp[:44]}", e)
        avg_ms = sum(times) / len(times)
        avg_wer = sum(errs) / len(errs)
        recall = kept / max(1, total_terms)
        rows.append((name, avg_ms, avg_wer, recall))
        print(f"{name:<20}{load:>6.1f}s{avg_ms:>8.0f}{avg_wer:>7.1%}{recall:>10.1%}   {worst[0]}")
        del eng

    if rows:
        base = next((r for r in rows if r[0] == "small"), None)
        print()
        if base:
            print(f"vs the current default (small, {base[1]:.0f}ms, {base[2]:.1%} WER):")
            for name, ms, w, rc in rows:
                if name == "small":
                    continue
                faster = base[1] - ms
                print(f"  {name:<18} {faster:+.0f}ms   WER {w - base[2]:+.1%}   key terms {rc - base[3]:+.1%}")
        print("\nkey terms = share of technical words kept; getting one wrong changes the question.")


if __name__ == "__main__":
    main()
