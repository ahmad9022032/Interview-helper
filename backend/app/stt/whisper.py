"""faster-whisper engine. The model is loaded once at startup and reused for
every transcription. Audio is passed as an in-memory float32 numpy array, so
there is no disk I/O and no ffmpeg dependency."""
from __future__ import annotations

import logging
import re
import time

import numpy as np
from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

# Whisper is known to hallucinate these on silence / noise.
_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks.", "thanks", "you", "you.", "bye", "bye.",
    "thanks for watching", "thanks for watching.", "thank you for watching.",
    "subtitles by the amara.org community", "please subscribe", ".", "..", "...",
}


class TranscriptionError(Exception):
    pass


class WhisperEngine:
    def __init__(
        self,
        model_size: str,
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = "en",
        beam_size: int = 1,
        cpu_threads: int = 0,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        t0 = time.perf_counter()
        log.info("Loading faster-whisper model '%s' (device=%s, compute=%s)...", model_size, device, compute_type)
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        log.info("Whisper loaded in %.1fs", time.perf_counter() - t0)

    def warmup(self) -> None:
        """Run one short transcription so the first real request is fast."""
        try:
            noise = (np.random.default_rng(0).standard_normal(16000) * 0.001).astype(np.float32)
            self.transcribe(noise)
        except Exception:  # noqa: BLE001 - warmup must never crash startup
            log.exception("Whisper warmup failed (ignored)")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a mono float32 16 kHz array. Returns cleaned text ('' if nothing)."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.size < 1600:  # < 0.1 s
            return ""
        try:
            segments, _info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=self.beam_size,
                best_of=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
            )
            parts: list[str] = []
            for seg in segments:
                if seg.no_speech_prob is not None and seg.no_speech_prob > 0.8:
                    continue
                text = seg.text.strip()
                if text:
                    parts.append(text)
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(str(exc)) from exc

        text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        if text.lower() in _HALLUCINATIONS:
            return ""
        return text
