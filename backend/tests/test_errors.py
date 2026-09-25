"""Error-handling tests: the app must degrade gracefully and never crash when
Ollama is down, the model is missing, or the audio is unintelligible.

Only needs the Ollama-independent paths; run it with the backend stopped or
running, it does not matter.

    python tests/test_errors.py
"""
from __future__ import annotations

import asyncio
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings  # noqa: E402
from app.context_store import ContextStore  # noqa: E402
from app.llm.ollama_client import LLM_UNAVAILABLE_MSG, LLMUnavailable, OllamaClient  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


async def main() -> int:
    print("Ollama unreachable")
    dead = OllamaClient("http://127.0.0.1:59999", "whatever", timeout_s=3.0)
    health = await dead.health()
    check("health reports unreachable", health["reachable"] is False)
    try:
        async for _ in dead.stream_chat([{"role": "user", "content": "hi"}]):
            pass
        check("stream raises LLMUnavailable", False, "no exception")
    except LLMUnavailable as exc:
        check("stream raises LLMUnavailable", True, str(exc)[:40])
    except Exception as exc:  # noqa: BLE001
        check("stream raises LLMUnavailable", False, f"got {type(exc).__name__}")
    check("user-facing message is the documented one", LLM_UNAVAILABLE_MSG == "AI model unavailable. Check Ollama.")
    caps = await dead.detect_capabilities()
    check("capability probe survives a dead server", caps == [])
    await dead.warmup()  # must not raise
    check("warmup survives a dead server", True)
    await dead.aclose()

    print("\nMissing model on a live server")
    live = OllamaClient(settings.ollama_host, "definitely-not-a-real-model:1b", timeout_s=5.0)
    h = await live.health()
    if h["reachable"]:
        check("health reports model missing", h["model_available"] is False)
        try:
            async for _ in live.stream_chat([{"role": "user", "content": "hi"}]):
                pass
            check("missing model raises LLMUnavailable", False, "no exception")
        except LLMUnavailable:
            check("missing model raises LLMUnavailable", True)
        except Exception as exc:  # noqa: BLE001
            check("missing model raises LLMUnavailable", False, f"got {type(exc).__name__}")
    else:
        print("  skip (Ollama not running)")
    await live.aclose()

    print("\nSpeech-to-text edge cases")
    try:
        from app.stt.whisper import WhisperEngine

        eng = WhisperEngine(settings.whisper_model, settings.whisper_device,
                            settings.whisper_compute_type, settings.whisper_language, 1)
        check("silence produces no text", eng.transcribe(np.zeros(16000, np.float32)) == "")
        check("a too-short buffer produces no text", eng.transcribe(np.zeros(100, np.float32)) == "")
        loud = (np.random.default_rng(1).standard_normal(16000 * 2) * 0.3).astype(np.float32)
        noise_text = eng.transcribe(loud)
        check("white noise does not crash", isinstance(noise_text, str), repr(noise_text)[:40])
    except Exception as exc:  # noqa: BLE001
        check("whisper edge cases", False, str(exc)[:60])

    print("\nVoice activity detection edge cases")
    from app.stt.vad import FRAME_SAMPLES, StreamingVAD

    vad = StreamingVAD()
    check("silence scores low", vad(np.zeros(FRAME_SAMPLES, np.float32)) < 0.5)
    try:
        vad(np.zeros(100, np.float32))
        check("wrong frame size is rejected", False)
    except ValueError:
        check("wrong frame size is rejected", True)

    print("\nCandidate context")
    store = ContextStore(settings.profile_path, settings.resume_path)
    check("profile renders", len(store.render()) > 100, f"{len(store.render())} chars")
    missing = ContextStore(settings.profile_path.parent / "nope.yaml",
                           settings.resume_path.parent / "nope.txt")
    check("missing files do not crash", missing.render() == "")
    try:
        store.set_resume_pdf(b"not a pdf at all")
        check("invalid PDF raises, does not crash", False)
    except Exception:  # noqa: BLE001
        check("invalid PDF raises, does not crash", True)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        return 1
    print("all error-handling checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
