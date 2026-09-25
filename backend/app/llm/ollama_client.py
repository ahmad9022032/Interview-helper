"""Async streaming client for a local Ollama server. One httpx.AsyncClient is
created at startup and reused for every request (connection pooling)."""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

log = logging.getLogger(__name__)

LLM_UNAVAILABLE_MSG = "AI model unavailable. Check Ollama."


class LLMUnavailable(Exception):
    """Raised when Ollama cannot be reached or returns an error."""


class Delta:
    """One piece of visible answer text. `reset` means: discard everything shown
    so far and start again from this text."""

    __slots__ = ("text", "reset", "truncated")

    def __init__(self, text: str, reset: bool = False, truncated: bool = False) -> None:
        self.text = text
        self.reset = reset
        self.truncated = truncated


class ThinkStripper:
    """Removes chain-of-thought from a token stream, handling every shape that
    local models produce:

    * balanced ``<think>...</think>`` blocks (tags may be split across chunks),
    * an *unmatched* trailing ``</think>`` - what Qwen3 emits under Ollama's
      ``think: false``: the template pre-opens the block, so the stream is
      "reasoning... </think> real answer" with no opening tag in sight.

    The second case is only detectable once the closing tag arrives, so text is
    streamed optimistically and a ``reset`` is signalled if it turns out to have
    been reasoning. Models that do not think never trigger a reset, so the
    recommended setup streams with zero added latency.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False
        self._closed = False  # a closing tag has already been consumed

    def _hold_back(self) -> int:
        """Length of the trailing partial tag to keep buffered, if any."""
        for n in range(min(len(self.CLOSE) - 1, len(self._buf)), 0, -1):
            tail = self._buf[-n:]
            if self.OPEN.startswith(tail) or self.CLOSE.startswith(tail):
                return n
        return 0

    def feed(self, chunk: str) -> list[Delta]:
        self._buf += chunk
        out: list[Delta] = []
        while self._buf:
            if self._inside:
                idx = self._buf.find(self.CLOSE)
                if idx == -1:
                    keep = self._hold_back()
                    self._buf = self._buf[-keep:] if keep else ""
                    break
                self._buf = self._buf[idx + len(self.CLOSE) :]
                self._inside = False
                self._closed = True
                continue

            open_at = self._buf.find(self.OPEN)
            close_at = self._buf.find(self.CLOSE)

            if close_at != -1 and (open_at == -1 or close_at < open_at) and not self._closed:
                # Unmatched close: everything before it (already streamed) was reasoning.
                self._buf = self._buf[close_at + len(self.CLOSE) :].lstrip()
                self._closed = True
                out.append(Delta("", reset=True))
                continue
            if close_at != -1 and (open_at == -1 or close_at < open_at):
                # A stray close after we already handled one - just drop the tag.
                out.append(Delta(self._buf[:close_at]))
                self._buf = self._buf[close_at + len(self.CLOSE) :]
                continue
            if open_at == -1:
                keep = self._hold_back()
                visible = self._buf[: len(self._buf) - keep] if keep else self._buf
                if visible:
                    out.append(Delta(visible))
                self._buf = self._buf[len(self._buf) - keep :] if keep else ""
                break
            if open_at:
                out.append(Delta(self._buf[:open_at]))
            self._buf = self._buf[open_at + len(self.OPEN) :]
            self._inside = True
        return [d for d in out if d.text or d.reset]

    def flush(self) -> list[Delta]:
        out: list[Delta] = []
        if not self._inside and self._buf:
            out.append(Delta(self._buf))
        self._buf = ""
        return out


class OllamaClient:
    def __init__(
        self,
        host: str,
        model: str,
        *,
        temperature: float = 0.3,
        num_predict: int = 220,
        num_ctx: int = 4096,
        keep_alive: str = "30m",
        timeout_s: float = 60.0,
        think: str = "auto",
        thinking_budget: int = 700,
    ) -> None:
        self.host = host
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.thinking_budget = thinking_budget
        self._think_setting = str(think).lower()
        # None until probed; True if the model advertises the "thinking" capability.
        self._supports_thinking: bool | None = None
        # None until probed; True if `think: false` actually silences its reasoning.
        self._respects_think_off: bool | None = None
        # Probe results per model, so switching back and forth costs nothing.
        self._think_off_cache: dict[str, bool] = {}
        self._send_think = True  # disabled automatically if the server rejects the field
        self._client = httpx.AsyncClient(
            base_url=host,
            timeout=httpx.Timeout(timeout_s, connect=3.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- model capabilities -------------------------------------------------
    @property
    def think_enabled(self) -> bool:
        """Whether to ask the model to think.

        Thinking-capable models split into two camps, and the difference is worth
        several seconds per answer:

        * **Hybrids** (Qwen3.5 and later) genuinely switch reasoning off when
          asked, and then answer immediately like any instruct model.
        * **Always-on reasoners** (Qwen3, DeepSeek-R1) ignore the switch: they
          emit their chain-of-thought as ordinary content and burn the whole
          answer budget doing it, often leaving no answer at all. For these,
          turning thinking *on* is the fix, because Ollama then returns the
          reasoning in a separate field and the answer stays clean.

        `auto` tells the two apart by probing once at startup.
        """
        if not self._supports_thinking:
            return False
        if self._think_setting in {"true", "1", "yes", "on"}:
            return True
        if self._think_setting in {"false", "0", "no", "off"}:
            return False
        # auto: only force thinking on for models that cannot be talked out of it.
        return self._respects_think_off is False

    async def set_model(self, model: str) -> dict:
        """Point the client at a different Ollama model, at runtime.

        Only the model name changes; the pooled HTTP connection and every other
        setting stay as they are, so switching costs one capability lookup.
        """
        self.model = model
        self._supports_thinking = None
        self._respects_think_off = self._think_off_cache.get(model)
        self._send_think = True
        await self.detect_capabilities()
        return {
            "model": self.model,
            "reasoning_model": bool(self._supports_thinking),
            "thinking_enabled": self.think_enabled,
        }

    async def list_local(self) -> list[dict]:
        """Every model pulled into Ollama, with size and whether it reasons."""
        try:
            r = await self._client.get("/api/tags", timeout=5.0)
            r.raise_for_status()
            models = r.json().get("models", [])
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not list Ollama models: %s", exc)
            return []

        async def describe(entry: dict) -> dict:
            name = entry.get("name", "")
            caps: list[str] = []
            try:
                show = await self._client.post("/api/show", json={"model": name}, timeout=10.0)
                if show.status_code == 200:
                    caps = [str(c).lower() for c in (show.json().get("capabilities") or [])]
            except Exception:  # noqa: BLE001
                pass
            thinks = "thinking" in caps
            # Only call a model slow once a probe has actually caught it reasoning
            # regardless of the switch. Hybrids advertise "thinking" but are fast.
            return {
                "name": name,
                "size_gb": round(entry.get("size", 0) / 1e9, 1),
                "parameter_size": (entry.get("details") or {}).get("parameter_size"),
                "thinking_capable": thinks,
                "reasoning": thinks and self._think_off_cache.get(name) is False,
                "probed": name in self._think_off_cache,
            }

        import asyncio

        return sorted(
            await asyncio.gather(*(describe(m) for m in models)),
            key=lambda m: m["name"],
        )

    async def detect_capabilities(self) -> list[str]:
        """Ask Ollama what the model can do. Free (no generation) and cached."""
        try:
            r = await self._client.post("/api/show", json={"model": self.model}, timeout=10.0)
            r.raise_for_status()
            caps = [str(c).lower() for c in (r.json().get("capabilities") or [])]
        except Exception as exc:  # noqa: BLE001 - fall back to non-thinking behaviour
            log.debug("Could not read capabilities for '%s': %s", self.model, exc)
            self._supports_thinking = False
            return []
        self._supports_thinking = "thinking" in caps
        if self._supports_thinking and self._think_setting not in {"true", "1", "yes", "on", "false", "0", "no", "off"}:
            if self.model in self._think_off_cache:
                self._respects_think_off = self._think_off_cache[self.model]
            else:
                self._respects_think_off = await self._probe_think_off()
                self._think_off_cache[self.model] = self._respects_think_off
        if self.think_enabled:
            log.warning(
                "'%s' always reasons before answering, which adds several seconds of "
                "latency. For a live interview prefer a model that answers immediately, "
                "such as qwen3.5:4b or qwen2.5:7b-instruct.",
                self.model,
            )
        elif self._supports_thinking:
            log.info("'%s' supports thinking but honours think:false, so it answers immediately", self.model)
        return caps

    async def _probe_think_off(self) -> bool:
        """Does `think: false` actually stop this model reasoning out loud?

        One short generation at startup, on a question answerable in a sentence.
        A model that honours the switch answers it and stops after a few dozen
        tokens. One that ignores it monologues instead, so it either closes with
        a stray `</think>`, produces no answer, or simply runs to the token cap.
        Hitting the cap is the reliable tell, and needs no language heuristics.
        """
        probe_budget = 220
        try:
            r = await self._client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "think": False,
                    "stream": False,
                    "messages": [{"role": "user", "content": "In one short sentence, why is the sky blue?"}],
                    "options": {"temperature": 0, "num_predict": probe_budget},
                    "keep_alive": self.keep_alive,
                },
                timeout=90.0,
            )
            if r.status_code != 200:
                return False
            data = r.json()
            content = (data.get("message", {}).get("content") or "").strip()
            hit_cap = data.get("done_reason") == "length"
        except Exception as exc:  # noqa: BLE001 - assume the safe (slower) path
            log.debug("think:false probe failed for '%s': %s", self.model, exc)
            return False
        leaked = hit_cap or "</think>" in content or "<think>" in content or not content
        return not leaked

    def _effective_num_predict(self, override: int | None = None) -> int:
        """Reasoning tokens come out of the same budget as the answer."""
        base = override or self.num_predict
        return base + self.thinking_budget if self.think_enabled else base

    async def health(self) -> dict:
        """Return {"reachable": bool, "model_available": bool, "models": [...]}."""
        try:
            r = await self._client.get("/api/tags", timeout=3.0)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as exc:  # noqa: BLE001
            log.warning("Ollama health check failed: %s", exc)
            return {"reachable": False, "model_available": False, "models": []}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return {
            "reachable": True,
            "model_available": wanted in names or self.model in names,
            "models": names,
            "reasoning_model": bool(self._supports_thinking),
            "thinking_enabled": self.think_enabled,
        }

    async def warmup(self) -> None:
        """Ask Ollama to load the model into memory (empty prompt = load only)."""
        try:
            await self._client.post(
                "/api/generate",
                json={"model": self.model, "prompt": "", "keep_alive": self.keep_alive},
                timeout=120.0,
            )
            log.info("Ollama model '%s' warmed up", self.model)
        except Exception as exc:  # noqa: BLE001
            log.warning("Ollama warmup skipped: %s", exc)

    def _payload(self, messages: list[dict], num_predict: int | None = None) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": self._effective_num_predict(num_predict),
                "num_ctx": self.num_ctx,
            },
        }
        # Only send the field to models that advertise the capability: Ollama
        # rejects it outright for the rest ("does not support thinking").
        if self._send_think and self._supports_thinking:
            payload["think"] = self.think_enabled
        return payload

    async def stream_chat(
        self, messages: list[dict], num_predict: int | None = None
    ) -> AsyncIterator[Delta]:
        """Yield visible answer deltas. Raises LLMUnavailable on failure.

        Any `thinking` field Ollama reports separately is ignored, and inline
        chain-of-thought is stripped, so only the answer reaches the UI.
        """
        stripper = ThinkStripper()
        try:
            async with self._client.stream(
                "POST", "/api/chat", json=self._payload(messages, num_predict)
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "ignore")
                    if resp.status_code == 400 and "think" in body.lower() and self._send_think:
                        # Older Ollama / model without thinking support: retry without the field.
                        log.info("Ollama rejected the 'think' field, retrying without it")
                        self._send_think = False
                        async for delta in self.stream_chat(messages, num_predict):
                            yield delta
                        return
                    raise LLMUnavailable(f"Ollama returned {resp.status_code}: {body[:200]}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in data:
                        raise LLMUnavailable(str(data["error"]))
                    content = data.get("message", {}).get("content", "")
                    if content:
                        for delta in stripper.feed(content):
                            yield delta
                    if data.get("done"):
                        # "length" means the token budget ran out mid-sentence.
                        if data.get("done_reason") == "length":
                            yield Delta("", truncated=True)
                        break
            for delta in stripper.flush():
                yield delta
        except LLMUnavailable:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise LLMUnavailable(str(exc)) from exc
