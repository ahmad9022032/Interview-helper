"""Application settings. Everything important is configurable through the .env
file at the project root (see .env.example). Nothing model-related is hard-coded."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# project root = <root>/backend/app/config.py -> parents[2]
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")
load_dotenv(ROOT_DIR / "backend" / ".env")


def _str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _str(name, "true" if default else "false").lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # LLM
    model: str
    ollama_host: str
    llm_temperature: float
    max_answer_tokens: int
    llm_num_ctx: int
    llm_timeout_s: float
    llm_keep_alive: str
    llm_think: str
    llm_thinking_budget: int
    llm_warmup: bool

    # STT
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_language: str | None
    whisper_beam_size: int
    whisper_cpu_threads: int

    # Capture
    capture_mode: str
    push_max_s: float
    push_min_s: float

    # VAD / segmentation
    sample_rate: int
    vad_threshold: float
    vad_silence_ms: int
    vad_min_speech_ms: int
    vad_preroll_ms: int
    max_utterance_s: float
    partial_interval_s: float

    # Question detection / memory
    min_question_words: int
    question_hold_s: float
    history_turns: int
    max_context_chars: int

    # Server
    backend_host: str
    backend_port: int
    cors_origins: list[str]

    # Files
    profile_path: Path
    resume_path: Path


def load_settings() -> Settings:
    language = _str("WHISPER_LANGUAGE", "en")
    return Settings(
        model=_str("MODEL", "qwen3:8b"),
        ollama_host=_str("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        llm_temperature=_float("LLM_TEMPERATURE", 0.3),
        max_answer_tokens=_int("MAX_ANSWER_TOKENS", 220),
        llm_num_ctx=_int("LLM_NUM_CTX", 4096),
        llm_timeout_s=_float("LLM_TIMEOUT_S", 60.0),
        llm_keep_alive=_str("LLM_KEEP_ALIVE", "30m"),
        llm_think=_str("LLM_THINK", "auto").lower(),
        llm_thinking_budget=_int("LLM_THINKING_BUDGET", 2000),
        llm_warmup=_bool("LLM_WARMUP", True),
        whisper_model=_str("WHISPER_MODEL", "small"),
        whisper_device=_str("WHISPER_DEVICE", "auto"),
        whisper_compute_type=_str("WHISPER_COMPUTE_TYPE", "auto"),
        whisper_language=None if language.lower() in {"auto", "none", ""} else language,
        whisper_beam_size=_int("WHISPER_BEAM_SIZE", 1),
        whisper_cpu_threads=_int("WHISPER_CPU_THREADS", 0),
        capture_mode=_str("CAPTURE_MODE", "push").lower(),
        push_max_s=_float("PUSH_MAX_S", 180.0),
        push_min_s=_float("PUSH_MIN_S", 0.4),
        sample_rate=16000,
        vad_threshold=_float("VAD_THRESHOLD", 0.5),
        vad_silence_ms=_int("VAD_SILENCE_MS", 700),
        vad_min_speech_ms=_int("VAD_MIN_SPEECH_MS", 300),
        vad_preroll_ms=_int("VAD_PREROLL_MS", 320),
        max_utterance_s=_float("MAX_UTTERANCE_S", 30.0),
        partial_interval_s=_float("PARTIAL_INTERVAL_S", 1.2),
        min_question_words=_int("MIN_QUESTION_WORDS", 4),
        question_hold_s=_float("QUESTION_HOLD_S", 1.5),
        history_turns=_int("HISTORY_TURNS", 3),
        max_context_chars=_int("MAX_CONTEXT_CHARS", 6000),
        backend_host=_str("BACKEND_HOST", "0.0.0.0"),
        backend_port=_int("BACKEND_PORT", 8000),
        cors_origins=[o.strip() for o in _str("CORS_ORIGINS", "*").split(",") if o.strip()],
        profile_path=ROOT_DIR / _str("PROFILE_PATH", "config/candidate_profile.yaml"),
        resume_path=ROOT_DIR / _str("RESUME_PATH", "data/resume.txt"),
    )


settings = load_settings()
