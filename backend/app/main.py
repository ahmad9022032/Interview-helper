"""Interview Copilot backend (FastAPI).

Models are loaded once in the lifespan handler and shared by all connections:
- faster-whisper (STT)
- Silero VAD (via onnxruntime, bundled with faster-whisper)
- one pooled HTTP client to the local Ollama server
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .context_store import ContextStore
from .llm.ollama_client import OllamaClient
from .pipeline import Session
from .routers.context import router as context_router
from .routers.models import load_selected_model, router as models_router
from .stt.whisper import WhisperEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("interview_copilot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.context_store = ContextStore(settings.profile_path, settings.resume_path)
    app.state.stt_lock = asyncio.Lock()
    app.state.whisper = None
    app.state.whisper_error = None

    try:
        app.state.whisper = await asyncio.to_thread(
            WhisperEngine,
            settings.whisper_model,
            settings.whisper_device,
            settings.whisper_compute_type,
            settings.whisper_language,
            settings.whisper_beam_size,
            settings.whisper_cpu_threads,
        )
        await asyncio.to_thread(app.state.whisper.warmup)
    except Exception as exc:  # noqa: BLE001 - keep serving; health reports the problem
        app.state.whisper_error = str(exc)
        log.exception("Failed to load Whisper model '%s'", settings.whisper_model)

    # A model picked in the UI outlives a restart; MODEL in .env is the default.
    chosen = load_selected_model(settings) or settings.model
    if chosen != settings.model:
        log.info("Using model '%s' selected in the UI (default is '%s')", chosen, settings.model)
    app.state.llm = OllamaClient(
        settings.ollama_host,
        chosen,
        temperature=settings.llm_temperature,
        num_predict=settings.max_answer_tokens,
        num_ctx=settings.llm_num_ctx,
        keep_alive=settings.llm_keep_alive,
        timeout_s=settings.llm_timeout_s,
        think=settings.llm_think,
        thinking_budget=settings.llm_thinking_budget,
    )
    health = await app.state.llm.health()
    if not health["reachable"]:
        log.warning("Ollama not reachable at %s - answers will fail until it is started", settings.ollama_host)
    elif not health["model_available"]:
        log.warning("Model '%s' not found in Ollama. Run: ollama pull %s", chosen, chosen)
    else:
        await app.state.llm.detect_capabilities()
        if settings.llm_warmup:
            asyncio.create_task(app.state.llm.warmup())

    log.info("Interview Copilot ready (MODEL=%s, WHISPER_MODEL=%s)", chosen, settings.whisper_model)
    try:
        yield
    finally:
        await app.state.llm.aclose()


app = FastAPI(title="Interview Copilot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(context_router)
app.include_router(models_router)


@app.get("/")
async def root() -> dict:
    return {"app": "Interview Copilot", "ws": "/ws", "health": "/api/health"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    state = ws.app.state

    async def send(payload: dict) -> None:
        await ws.send_json(payload)

    session = Session(
        send,
        whisper=state.whisper,
        llm=state.llm,
        context_store=state.context_store,
        settings=state.settings,
        stt_lock=state.stt_lock,
    )
    await session.hello()
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await session.handle_audio(message["bytes"])
            elif message.get("text"):
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    try:
                        await session.handle_control(payload)
                    except Exception:  # noqa: BLE001
                        log.exception("control message failed: %s", payload)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("websocket loop error")
    finally:
        await session.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.backend_host, port=settings.backend_port, log_level="info")
