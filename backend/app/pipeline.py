"""Per-WebSocket-connection real-time pipeline:

    PCM chunks -> Silero VAD (32 ms frames) -> utterance buffer
      -> partial transcripts while speaking (every PARTIAL_INTERVAL_S)
      -> final transcript on end of speech
      -> question detection -> streaming LLM answer -> events to the client

Every stage is guarded so one failure never kills the connection."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Awaitable, Callable

import numpy as np

from .config import Settings
from .context_store import ContextStore
from .history import ConversationHistory
from .llm.ollama_client import LLM_UNAVAILABLE_MSG, LLMUnavailable, OllamaClient
from .prompts.builder import build_messages
from .prompts.system_prompt import (
    DEFAULT_DEPTH,
    DEFAULT_MODE,
    DEPTH_ADDENDA,
    MODE_ADDENDA,
    depth_tokens,
)
from .question_detector import QuestionDetector
from .stt.vad import FRAME_SAMPLES, StreamingVAD
from .stt.whisper import TranscriptionError, WhisperEngine

log = logging.getLogger(__name__)

STT_FAILED_MSG = "Could not understand audio. Please repeat."
STT_UNAVAILABLE_MSG = "Speech recognition unavailable. Check backend logs."
PUSH_TOO_SHORT_MSG = "Too short. Hold the key for the whole question."

SendFn = Callable[[dict], Awaitable[None]]


class Session:
    def __init__(
        self,
        send: SendFn,
        *,
        whisper: WhisperEngine | None,
        llm: OllamaClient,
        context_store: ContextStore,
        settings: Settings,
        stt_lock: asyncio.Lock,
    ) -> None:
        self._send = send
        self.whisper = whisper
        self.llm = llm
        self.context_store = context_store
        self.s = settings
        self._stt_lock = stt_lock

        self.mode = DEFAULT_MODE
        self.depth = DEFAULT_DEPTH
        self.capture_mode = settings.capture_mode if settings.capture_mode in {"push", "auto"} else "push"
        self.listening = False
        self.history = ConversationHistory(settings.history_turns)
        self.detector = QuestionDetector(settings.min_question_words)
        self.vad = StreamingVAD()

        self._frame_ms = FRAME_SAMPLES * 1000 / settings.sample_rate  # 32 ms
        preroll_frames = max(1, int(settings.vad_preroll_ms / self._frame_ms))
        self._preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
        self._remainder = np.zeros(0, dtype=np.float32)
        self._utterance: list[np.ndarray] = []
        self._utt_id = 0
        self._speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._last_partial_at = 0.0
        self._last_stt_ms = 0.0
        self._last_speech_end = 0.0

        # Push-to-ask: the candidate holds a key while the interviewer asks, so the
        # question boundary is stated rather than guessed. Everything outside a
        # capture is discarded without being transcribed.
        self._capturing = False
        self._capture_chunks: list[np.ndarray] = []
        # Live transcript state. Each pass covers only the audio that arrived since
        # the last one, so a long hold never makes an individual pass expensive.
        self._push_partial_offset = 0
        self._push_partial_text = ""

        self._partial_task: asyncio.Task | None = None
        self._final_tasks: set[asyncio.Task] = set()
        self._llm_task: asyncio.Task | None = None
        self._hold_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ utils
    async def send(self, payload: dict) -> None:
        try:
            await self._send(payload)
        except Exception:  # noqa: BLE001 - client may have gone away
            log.debug("send failed (client gone?)", exc_info=True)

    async def status(self, stage: str) -> None:
        await self.send({"type": "status", "stage": stage, "listening": self.listening})

    async def error(self, message: str, *, stage: str | None = None) -> None:
        await self.send({"type": "error", "message": message, "stage": stage})

    async def hello(self) -> None:
        reasoning = getattr(self.llm, "think_enabled", False)
        await self.send(
            {
                "type": "hello",
                "model": self.llm.model,
                "whisper_model": self.whisper.model_size if self.whisper else None,
                "whisper_ready": self.whisper is not None,
                "modes": list(MODE_ADDENDA.keys()),
                "mode": self.mode,
                "capture_mode": self.capture_mode,
                "depth": self.depth,
                "depths": list(DEPTH_ADDENDA.keys()),
                "sample_rate": self.s.sample_rate,
                "reasoning_model": reasoning,
                "notice": (
                    f"{self.llm.model} is a reasoning model: it thinks before answering, so expect "
                    "roughly 10-30s per answer. For a live interview set MODEL to a non-reasoning "
                    "model such as qwen2.5:3b-instruct."
                )
                if reasoning
                else None,
            }
        )

    # --------------------------------------------------------------- controls
    async def handle_control(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "start":
            self._set_mode(msg.get("mode"))
            self._set_capture_mode(msg.get("capture_mode"))
            if msg.get("depth") in DEPTH_ADDENDA:
                self.depth = msg["depth"]
            self._reset_audio_state()
            self.listening = True
            await self.status("listening")
        elif kind == "stop":
            self.listening = False
            if self._capturing:
                await self._finish_capture()
            elif self._speaking:
                await self._end_utterance()
            await self.status("idle")
        elif kind == "capture_start":
            # The key went down: everything from here until capture_stop is the question.
            self._capturing = True
            self._capture_chunks = []
            self._push_partial_offset = 0
            self._push_partial_text = ""
            self._last_partial_at = time.perf_counter()
            self._cancel(self._hold_task)
            await self.status("capturing")
        elif kind == "capture_stop":
            await self._finish_capture()
        elif kind == "capture_mode":
            self._set_capture_mode(msg.get("mode"))
            self._reset_audio_state()
            await self.send({"type": "capture_mode", "capture_mode": self.capture_mode})
            await self.status("listening" if self.listening else "idle")
        elif kind == "mode":
            self._set_mode(msg.get("mode"))
            await self.send({"type": "mode", "mode": self.mode})
        elif kind == "depth":
            wanted = msg.get("depth")
            if wanted in DEPTH_ADDENDA:
                self.depth = wanted
            await self.send({"type": "depth", "depth": self.depth})
        elif kind == "clear":
            self.history.clear()
            self.detector.reset()
            self._cancel(self._llm_task)
            self._cancel(self._hold_task)
            await self.send({"type": "cleared"})
            await self.status("listening" if self.listening else "idle")
        elif kind == "ask":
            # Manually typed question (also used by the test client).
            text = str(msg.get("text", "")).strip()
            if text:
                self._last_speech_end = time.perf_counter()
                self._last_stt_ms = 0.0
                self._start_answer(text)
        elif kind == "ping":
            await self.send({"type": "pong"})
        else:
            log.debug("unknown control message: %s", msg)

    def _set_mode(self, mode: str | None) -> None:
        if mode in MODE_ADDENDA:
            self.mode = mode

    def _set_capture_mode(self, mode: str | None) -> None:
        if mode in {"push", "auto"}:
            self.capture_mode = mode

    def _reset_audio_state(self) -> None:
        self._capturing = False
        self._capture_chunks = []
        self._push_partial_offset = 0
        self._push_partial_text = ""
        self.detector.reset()
        self._cancel(self._hold_task)
        self._remainder = np.zeros(0, dtype=np.float32)
        self._utterance = []
        self._preroll.clear()
        self._speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
        self.vad.reset()

    # ------------------------------------------------------------------ audio
    async def handle_audio(self, data: bytes) -> None:
        if not self.listening or not data:
            return
        try:
            samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        except ValueError:
            return

        if self.capture_mode == "push":
            await self._handle_push_audio(samples)
            return

        buf = np.concatenate([self._remainder, samples]) if self._remainder.size else samples
        n_frames = buf.shape[0] // FRAME_SAMPLES
        for i in range(n_frames):
            frame = buf[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            await self._process_frame(frame)
        self._remainder = buf[n_frames * FRAME_SAMPLES :].copy()

    async def _process_frame(self, frame: np.ndarray) -> None:
        try:
            prob = self.vad(frame)
        except Exception:  # noqa: BLE001
            log.exception("VAD failed on frame")
            return

        thr = self.s.vad_threshold
        if not self._speaking:
            if prob >= thr:
                self._speaking = True
                self._utt_id += 1
                self._utterance = [f.copy() for f in self._preroll] + [frame.copy()]
                self._preroll.clear()
                self._speech_frames = 1
                self._silence_frames = 0
                self._last_partial_at = time.perf_counter()
                await self.status("speech")
            else:
                self._preroll.append(frame.copy())
            return

        self._utterance.append(frame.copy())
        if prob >= max(0.0, thr - 0.15):
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        silence_ms = self._silence_frames * self._frame_ms
        duration_s = len(self._utterance) * self._frame_ms / 1000.0
        if silence_ms >= self.s.vad_silence_ms or duration_s >= self.s.max_utterance_s:
            await self._end_utterance()
            return

        now = time.perf_counter()
        if (
            self.whisper is not None
            # Only while actually speaking: a partial started during a pause would
            # still hold the STT lock (threads cannot be cancelled) and delay the
            # final transcription, which is the one the answer waits on.
            and self._silence_frames == 0
            and now - self._last_partial_at >= self.s.partial_interval_s
            and self._speech_frames * self._frame_ms >= self.s.vad_min_speech_ms
            and (self._partial_task is None or self._partial_task.done())
            and not self._stt_lock.locked()
        ):
            self._last_partial_at = now
            audio = np.concatenate(self._utterance)
            self._partial_task = asyncio.create_task(self._transcribe_partial(audio, self._utt_id))

    # ------------------------------------------------------- push-to-ask audio
    async def _handle_push_audio(self, samples: np.ndarray) -> None:
        """Audio only matters while the key is held; the rest is dropped on the floor."""
        if not self._capturing:
            return
        self._capture_chunks.append(samples)

        captured_s = sum(c.size for c in self._capture_chunks) / self.s.sample_rate
        if captured_s >= self.s.push_max_s:
            log.info("push capture hit the %.0fs cap, answering what we have", self.s.push_max_s)
            await self._finish_capture()
            return

        # Live transcript while the key is down, so the candidate can see it working.
        #
        # Each pass covers ONLY the audio that arrived since the last one. Passing the
        # whole buffer instead would get slower as the question went on, and because a
        # Whisper call runs in a worker thread that cannot be cancelled, the final
        # transcription on release would queue behind it. Measured, that roughly
        # doubled the worst-case wait after releasing the key.
        now = time.perf_counter()
        if (
            self.whisper is not None
            and now - self._last_partial_at >= self.s.partial_interval_s
            and (self._partial_task is None or self._partial_task.done())
            and not self._stt_lock.locked()
        ):
            total = sum(c.size for c in self._capture_chunks)
            fresh = total - self._push_partial_offset
            if fresh >= int(self.s.sample_rate * 0.6):
                self._last_partial_at = now
                audio = np.concatenate(self._capture_chunks)[self._push_partial_offset : total]
                self._push_partial_offset = total
                self._partial_task = asyncio.create_task(self._transcribe_push_partial(audio))

    async def _transcribe_push_partial(self, audio: np.ndarray) -> None:
        """Transcribe one slice and append it to the running live transcript.

        Only ever shown while the key is held; the final pass on release re-reads
        the whole question properly, so imperfect slice boundaries do not matter.
        """
        try:
            text = await self._run_stt(audio)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.debug("push partial transcription failed", exc_info=True)
            return
        if not text or not self._capturing:
            return
        self._push_partial_text = f"{self._push_partial_text} {text}".strip()
        await self.send({"type": "transcript_partial", "text": self._push_partial_text})

    def _trim_to_speech(self, audio: np.ndarray) -> np.ndarray:
        """Drop leading and trailing non-speech so Whisper only sees the question.

        The candidate will not release the key the instant the interviewer stops,
        and may press it early, so this keeps transcription time proportional to
        what was actually said.
        """
        n_frames = audio.shape[0] // FRAME_SAMPLES
        if n_frames < 2:
            return audio
        try:
            self.vad.reset()
            probs = [
                self.vad(audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES])
                for i in range(n_frames)
            ]
        except Exception:  # noqa: BLE001 - trimming is an optimisation, never fatal
            log.debug("VAD trim failed, using the whole buffer", exc_info=True)
            return audio

        speech = [i for i, p in enumerate(probs) if p >= self.s.vad_threshold]
        if not speech:
            return audio
        pad = max(1, int(self.s.vad_preroll_ms / self._frame_ms))
        first = max(0, speech[0] - pad)
        last = min(n_frames, speech[-1] + 1 + pad)
        return audio[first * FRAME_SAMPLES : last * FRAME_SAMPLES]

    async def _finish_capture(self) -> None:
        if not self._capturing:
            return
        self._capturing = False
        chunks, self._capture_chunks = self._capture_chunks, []
        self._cancel(self._partial_task)

        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        if audio.size / self.s.sample_rate < self.s.push_min_s:
            await self.send({"type": "transcript_partial", "text": ""})
            await self.error(PUSH_TOO_SHORT_MSG, stage="capture")
            await self.status("listening" if self.listening else "idle")
            return

        audio = self._trim_to_speech(audio)
        released_at = time.perf_counter()
        await self.status("transcribing")
        task = asyncio.create_task(self._transcribe_pushed(audio, released_at))
        self._final_tasks.add(task)
        task.add_done_callback(self._final_tasks.discard)

    async def _transcribe_pushed(self, audio: np.ndarray, released_at: float) -> None:
        """The captured audio IS the question. No end-of-question guessing, no
        merging with neighbours, no dedupe against the last one."""
        if self.whisper is None:
            await self.error(STT_UNAVAILABLE_MSG, stage="stt")
            await self.status("listening" if self.listening else "idle")
            return
        t0 = time.perf_counter()
        try:
            text = await self._run_stt(audio)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("push transcription failed")
            await self.error(STT_FAILED_MSG, stage="stt")
            await self.status("listening" if self.listening else "idle")
            return

        stt_ms = (time.perf_counter() - t0) * 1000.0
        if not text:
            await self.error(STT_FAILED_MSG, stage="stt")
            await self.status("listening" if self.listening else "idle")
            return

        self._last_stt_ms = stt_ms
        self._last_speech_end = released_at
        await self.send(
            {"type": "transcript_final", "text": text, "stt_ms": round(stt_ms), "utt_id": self._utt_id}
        )
        self._start_answer(text)

    async def _end_utterance(self) -> None:
        audio_frames = self._utterance
        speech_frames = self._speech_frames
        silence_frames = self._silence_frames
        utt_id = self._utt_id
        self._utterance = []
        self._speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._cancel(self._partial_task)

        if speech_frames * self._frame_ms < self.s.vad_min_speech_ms:
            await self.status("listening")
            return  # a click / cough, not speech

        # trim most of the trailing silence (keep ~200 ms) to shorten STT time
        keep_tail = int(200 / self._frame_ms)
        drop = max(0, silence_frames - keep_tail)
        if drop:
            audio_frames = audio_frames[: len(audio_frames) - drop]
        audio = np.concatenate(audio_frames) if audio_frames else np.zeros(0, dtype=np.float32)

        self._last_speech_end = time.perf_counter()
        await self.status("transcribing")
        task = asyncio.create_task(self._transcribe_final(audio, utt_id, self._last_speech_end))
        self._final_tasks.add(task)
        task.add_done_callback(self._final_tasks.discard)

    # -------------------------------------------------------------------- STT
    async def _run_stt(self, audio: np.ndarray) -> str:
        assert self.whisper is not None
        async with self._stt_lock:
            return await asyncio.to_thread(self.whisper.transcribe, audio)

    async def _transcribe_partial(self, audio: np.ndarray, utt_id: int) -> None:
        try:
            text = await self._run_stt(audio)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.debug("partial transcription failed", exc_info=True)
            return
        if text and self._speaking and utt_id == self._utt_id:
            await self.send({"type": "transcript_partial", "text": text})

    async def _transcribe_final(self, audio: np.ndarray, utt_id: int, speech_end: float) -> None:
        if self.whisper is None:
            await self.error(STT_UNAVAILABLE_MSG, stage="stt")
            await self.status("listening" if self.listening else "idle")
            return
        t0 = time.perf_counter()
        try:
            text = await self._run_stt(audio)
        except asyncio.CancelledError:
            raise
        except TranscriptionError:
            log.exception("transcription failed")
            await self.error(STT_FAILED_MSG, stage="stt")
            await self.status("listening" if self.listening else "idle")
            return
        except Exception:  # noqa: BLE001
            log.exception("unexpected STT failure")
            await self.error(STT_FAILED_MSG, stage="stt")
            await self.status("listening" if self.listening else "idle")
            return

        stt_ms = (time.perf_counter() - t0) * 1000.0
        duration_s = audio.shape[0] / self.s.sample_rate
        if not text:
            if duration_s >= 1.0:
                await self.error(STT_FAILED_MSG, stage="stt")
            await self.status("listening" if self.listening else "idle")
            return

        self._last_stt_ms = stt_ms
        self._last_speech_end = speech_end
        await self.send({"type": "transcript_final", "text": text, "stt_ms": round(stt_ms), "utt_id": utt_id})

        question = self.detector.feed(text)
        if question:
            self._cancel(self._hold_task)
            self._start_answer(question)
        elif self.detector.pending:
            # Fragment too short - wait briefly for a continuation, then answer anyway.
            self._cancel(self._hold_task)
            self._hold_task = asyncio.create_task(self._hold_then_flush(self.detector.pending))
            await self.status("listening" if self.listening else "idle")
        else:
            await self.status("listening" if self.listening else "idle")

    async def _hold_then_flush(self, pending_snapshot: str) -> None:
        try:
            await asyncio.sleep(self.s.question_hold_s)
        except asyncio.CancelledError:
            return
        if self.detector.pending == pending_snapshot:
            question = self.detector.flush()
            if question:
                self._start_answer(question)

    # -------------------------------------------------------------------- LLM
    def _start_answer(self, question: str) -> None:
        self._cancel(self._llm_task)
        self._llm_task = asyncio.create_task(self._answer(question, self._last_stt_ms, self._last_speech_end))

    async def _answer(self, question: str, stt_ms: float, speech_end: float) -> None:
        await self.send({"type": "question", "text": question, "mode": self.mode, "depth": self.depth})
        await self.status("thinking")
        messages = build_messages(
            question,
            self.mode,
            self.context_store.render(),
            self.history.turns(),
            max_context_chars=self.s.max_context_chars,
            brief_reasoning=getattr(self.llm, "think_enabled", False),
            depth=self.depth,
        )
        budget = depth_tokens(self.depth, self.s.max_answer_tokens)
        t0 = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []
        truncated = False
        try:
            async for delta in self.llm.stream_chat(messages, num_predict=budget):
                if delta.truncated:
                    truncated = True
                    continue
                if delta.reset:
                    # What we streamed turned out to be the model's reasoning.
                    chunks.clear()
                    first_token_at = None
                    await self.send({"type": "answer_reset"})
                    await self.status("thinking")
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    await self.status("answering")
                chunks.append(delta.text)
                await self.send({"type": "answer_delta", "text": delta.text})
            answer = "".join(chunks).strip()
            done = time.perf_counter()
            latency = {
                "stt_ms": round(stt_ms),
                "llm_first_token_ms": round(((first_token_at or done) - t0) * 1000.0),
                "llm_ms": round((done - t0) * 1000.0),
                "total_ms": round((done - speech_end) * 1000.0) if speech_end else round((done - t0) * 1000.0),
                "first_answer_ms": round(((first_token_at or done) - speech_end) * 1000.0) if speech_end else None,
            }
            if not answer:
                hint = (
                    " The model used its whole budget thinking - raise LLM_THINKING_BUDGET "
                    "or switch to a non-reasoning model."
                    if getattr(self.llm, "think_enabled", False)
                    else " Try again."
                )
                await self.error("The model returned an empty answer." + hint, stage="llm")
            else:
                self.history.add(question, answer)
            await self.send(
                {
                    "type": "answer_done",
                    "question": question,
                    "text": answer,
                    "latency": latency,
                    "truncated": truncated,
                }
            )
        except asyncio.CancelledError:
            await self.send({"type": "answer_cancelled", "question": question})
            raise
        except LLMUnavailable as exc:
            log.warning("LLM unavailable: %s", exc)
            await self.error(LLM_UNAVAILABLE_MSG, stage="llm")
        except Exception:  # noqa: BLE001
            log.exception("answer generation failed")
            await self.error("Answer generation failed. Please try again.", stage="llm")
        finally:
            await self.status("listening" if self.listening else "idle")

    # ---------------------------------------------------------------- cleanup
    @staticmethod
    def _cancel(task: asyncio.Task | None) -> None:
        if task is not None and not task.done():
            task.cancel()

    async def close(self) -> None:
        self.listening = False
        self._capturing = False
        self._capture_chunks = []
        for task in [self._partial_task, self._llm_task, self._hold_task, *self._final_tasks]:
            self._cancel(task)
