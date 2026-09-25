"""Stateful streaming wrapper around the Silero VAD ONNX model that ships with
faster-whisper. The bundled SileroVADModel only supports batch inference (it
resets the LSTM state on every call), so we drive the same onnxruntime session
frame-by-frame and keep h/c/context between calls. Each frame is 512 samples
(32 ms at 16 kHz)."""
from __future__ import annotations

import numpy as np
from faster_whisper.vad import get_vad_model

FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64
SAMPLE_RATE = 16000


class StreamingVAD:
    def __init__(self) -> None:
        self._session = get_vad_model().session
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def reset(self) -> None:
        self._h[:] = 0
        self._c[:] = 0
        self._context[:] = 0

    def __call__(self, frame: np.ndarray) -> float:
        """Return speech probability (0..1) for one 512-sample float32 frame."""
        if frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(f"VAD frame must be {FRAME_SAMPLES} samples, got {frame.shape[0]}")
        x = np.concatenate([self._context, frame.reshape(1, -1)], axis=1).astype(np.float32)
        out, self._h, self._c = self._session.run(None, {"input": x, "h": self._h, "c": self._c})
        self._context = frame[-CONTEXT_SAMPLES:].reshape(1, -1).astype(np.float32)
        return float(np.asarray(out).reshape(-1)[0])
