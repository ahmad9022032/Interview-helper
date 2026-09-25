"""HTTP routes: health check, and candidate context (resume) management."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from ..prompts.system_prompt import MODE_ADDENDA

router = APIRouter(prefix="/api")


class ResumeBody(BaseModel):
    resume_text: str


@router.get("/health")
async def health(request: Request) -> dict:
    state = request.app.state
    ollama = await state.llm.health()
    whisper_loaded = state.whisper is not None
    return {
        "status": "ok" if whisper_loaded and ollama["reachable"] and ollama["model_available"] else "degraded",
        "whisper": {
            "loaded": whisper_loaded,
            "model": state.settings.whisper_model,
            "error": state.whisper_error,
        },
        "ollama": {"host": state.settings.ollama_host, "model": state.settings.model, **ollama},
        "modes": list(MODE_ADDENDA.keys()),
    }


@router.get("/context")
async def get_context(request: Request) -> dict:
    store = request.app.state.context_store
    rendered = store.render()
    return {
        "profile": store.profile,
        "profile_path": str(store.profile_path),
        "resume_text": store.resume_text,
        "rendered_chars": len(rendered),
    }


@router.put("/context")
async def put_context(body: ResumeBody, request: Request) -> dict:
    store = request.app.state.context_store
    store.set_resume_text(body.resume_text)
    return {"ok": True, "resume_chars": len(store.resume_text), "rendered_chars": len(store.render())}


@router.post("/context/pdf")
async def upload_pdf(request: Request, file: UploadFile = File(...)) -> dict:
    store = request.app.state.context_store
    if not (file.filename or "").lower().endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF too large (max 10 MB).")
    try:
        text = store.set_resume_pdf(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {exc}") from exc
    return {"ok": True, "resume_text": text, "resume_chars": len(text)}


@router.post("/context/reload")
async def reload_context(request: Request) -> dict:
    store = request.app.state.context_store
    store.reload()
    return {"ok": True, "rendered_chars": len(store.render())}
