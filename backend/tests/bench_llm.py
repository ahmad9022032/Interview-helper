"""Measures answer latency and length for each Ollama model on this machine,
using the app's real prompts and candidate context.

Ollama must be running and each model already pulled.

    python tests/bench_llm.py --models qwen2.5:3b-instruct qwen2.5:7b-instruct
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings  # noqa: E402
from app.context_store import ContextStore  # noqa: E402
from app.llm.ollama_client import LLMUnavailable, OllamaClient  # noqa: E402
from app.prompts.builder import build_messages  # noqa: E402

QUESTIONS = [
    ("technical", "Can you explain what a transformer is and why we use attention?"),
    ("technical", "What is the difference between a process and a thread?"),
    ("technical", "Why would you use RAG instead of fine-tuning?"),
    ("behavioral", "Tell me about yourself"),
    ("project", "What AI projects have you worked on?"),
]


async def bench(model: str, context: str, timeout: float) -> None:
    client = OllamaClient(
        settings.ollama_host,
        model,
        temperature=settings.llm_temperature,
        num_predict=settings.max_answer_tokens,
        num_ctx=settings.llm_num_ctx,
        keep_alive=settings.llm_keep_alive,
        timeout_s=timeout,
        think=settings.llm_think,
        thinking_budget=settings.llm_thinking_budget,
    )
    health = await client.health()
    if not health["model_available"]:
        print(f"{model:<24} not pulled - run: ollama pull {model}")
        await client.aclose()
        return
    await client.detect_capabilities()
    await client.warmup()

    ttfts, totals, words, fails = [], [], [], 0
    for mode, q in QUESTIONS:
        messages = build_messages(q, mode, context, [], max_context_chars=settings.max_context_chars,
                                  brief_reasoning=client.think_enabled)
        t0 = time.perf_counter()
        first: float | None = None
        text = ""
        try:
            async for d in client.stream_chat(messages):
                if d.truncated or d.reset:
                    if d.reset:
                        text, first = "", None
                    continue
                if first is None:
                    first = time.perf_counter()
                text += d.text
        except LLMUnavailable as exc:
            print(f"  {q[:40]!r}: {exc}")
            fails += 1
            continue
        if not text.strip():
            fails += 1
            continue
        ttfts.append(((first or time.perf_counter()) - t0) * 1000)
        totals.append((time.perf_counter() - t0) * 1000)
        words.append(len(text.split()))

    tag = " (reasoning)" if client.think_enabled else ""
    if ttfts:
        print(f"{model + tag:<30}{statistics.median(ttfts):>10.0f}ms{statistics.median(totals):>11.0f}ms"
              f"{statistics.median(words):>8.0f}{fails:>8}")
    else:
        print(f"{model + tag:<30}{'all failed':>10}{'':>11}{'':>8}{fails:>8}")
    await client.aclose()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen2.5:3b-instruct"])
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    context = ContextStore(settings.profile_path, settings.resume_path).render()
    print(f"{len(QUESTIONS)} questions, candidate context {len(context)} chars, "
          f"MAX_ANSWER_TOKENS={settings.max_answer_tokens}\n")
    print(f"{'model':<30}{'TTFT':>12}{'total':>11}{'words':>8}{'failed':>8}")
    print("-" * 69)
    for m in args.models:
        await bench(m, context, args.timeout)
    print("\nTTFT = time to first answer word (median). Both exclude speech-to-text.")


if __name__ == "__main__":
    asyncio.run(main())
