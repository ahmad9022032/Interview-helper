"""Switch the answering model at runtime, without editing .env or restarting.

The choice is remembered in data/selected_model.txt so it survives a restart;
delete that file to fall back to MODEL from .env.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class ModelBody(BaseModel):
    model: str


def selection_path(settings) -> Path:
    return settings.resume_path.parent / "selected_model.txt"


def load_selected_model(settings) -> str | None:
    path = selection_path(settings)
    if not path.exists():
        return None
    try:
        name = path.read_text(encoding="utf-8").strip()
        return name or None
    except Exception:  # noqa: BLE001
        log.exception("Could not read %s", path)
        return None


def save_selected_model(settings, model: str) -> None:
    path = selection_path(settings)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model, encoding="utf-8")
    except Exception:  # noqa: BLE001
        log.exception("Could not persist model choice to %s", path)


@router.get("/models")
async def list_models(request: Request) -> dict:
    llm = request.app.state.llm
    return {
        "active": llm.model,
        "default": request.app.state.settings.model,
        "reasoning_model": bool(getattr(llm, "_supports_thinking", False)),
        "thinking_enabled": llm.think_enabled,
        "models": await llm.list_local(),
    }


@router.post("/models")
async def select_model(body: ModelBody, request: Request) -> dict:
    llm = request.app.state.llm
    settings = request.app.state.settings
    wanted = body.model.strip()
    if not wanted:
        raise HTTPException(status_code=400, detail="Model name is required.")

    available = {m["name"] for m in await llm.list_local()}
    if wanted not in available:
        raise HTTPException(
            status_code=404,
            detail=f"'{wanted}' is not pulled into Ollama. Run: ollama pull {wanted}",
        )

    previous = llm.model
    info = await llm.set_model(wanted)
    save_selected_model(settings, wanted)
    log.info("Answering model switched from '%s' to '%s'", previous, wanted)
    # Load it into memory so the first question after the switch is not slow.
    asyncio.create_task(llm.warmup())
    return {"ok": True, "previous": previous, **info}
