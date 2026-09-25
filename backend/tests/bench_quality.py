"""Scores how well each model explains modern AI architecture - RAG, LangChain,
agents, vector search - which is where small or older models tend to be vague.

For every question there is a list of concepts a genuinely good interview answer
would touch. The score is how many of them the model actually covers, so it
rewards substance rather than fluency. Latency is measured at the same time, so
you can see what extra quality costs you.

    python tests/bench_quality.py --models qwen2.5:3b-instruct qwen2.5:7b-instruct
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

# (question, mode, [concepts a good answer covers; nested list = any one of them])
QUESTIONS: list[tuple[str, str, list]] = [
    (
        "Explain the architecture of a RAG system.",
        "technical",
        [["chunk", "split"], ["embed", "vector"], ["vector store", "vector database", "index"],
         ["retriev", "search"], ["top-k", "top k", "similar", "relevant"], ["context", "prompt"], ["generat", "llm"]],
    ),
    (
        "How does LangChain work and what are its main components?",
        "technical",
        [["chain", "lcel", "runnable"], ["agent"], ["tool"], ["memory", "history"],
         ["prompt template", "prompt"], ["retriev", "vector"], ["model", "llm"]],
    ),
    (
        "What is an agentic AI system and how is it architected?",
        "technical",
        [["tool", "function call"], ["plan", "reason"], ["loop", "iterat", "step"],
         ["observ", "feedback", "result"], ["memory", "state"], ["goal", "task"]],
    ),
    (
        "How do vector databases work?",
        "technical",
        [["embed", "vector"], ["similar", "cosine", "distance", "dot product"],
         ["index", "hnsw", "ivf", "ann", "approximate"], ["nearest neighbor", "nearest neighbour", "top-k", "top k"],
         ["dimension", "high-dimensional", "semantic"]],
    ),
    (
        "Explain the transformer architecture and why attention matters.",
        "technical",
        [["attention"], ["self-attention", "self attention"], ["multi-head", "multi head", "heads"],
         ["positional", "position"], ["encoder", "decoder", "stack", "layer"],
         ["parallel", "long-range", "long range", "dependenc", "context"]],
    ),
    (
        "What is LangGraph and how does it differ from LangChain?",
        "technical",
        [["graph"], ["node"], ["edge", "transition"], ["state"], ["cycle", "loop", "branch"],
         ["control", "orchestrat", "workflow"]],
    ),
    (
        "How would you evaluate a RAG pipeline?",
        "technical",
        [["retriev"], ["recall", "precision", "hit rate", "ndcg", "mrr"],
         ["faithful", "grounded", "hallucin"], ["relevan"], ["answer", "generation"],
         ["dataset", "ground truth", "human", "judge", "llm-as"]],
    ),
    (
        "Why would you use RAG instead of fine-tuning?",
        "technical",
        [["update", "fresh", "current", "new data", "change"], ["cost", "expensive", "compute", "cheap"],
         ["hallucin", "ground", "cite", "source"], ["knowledge", "domain", "document"],
         ["retrain", "fine-tun"]],
    ),
]


def covered(answer: str, concepts: list) -> tuple[int, list[str]]:
    low = answer.lower()
    hits, missed = 0, []
    for concept in concepts:
        options = concept if isinstance(concept, list) else [concept]
        if any(o in low for o in options):
            hits += 1
        else:
            missed.append(options[0])
    return hits, missed


async def run_model(model: str, context: str, timeout: float, verbose: bool,
                    temperature: float, repeats: int) -> dict | None:
    client = OllamaClient(
        settings.ollama_host,
        model,
        temperature=temperature,
        num_predict=settings.max_answer_tokens,
        num_ctx=settings.llm_num_ctx,
        keep_alive=settings.llm_keep_alive,
        timeout_s=timeout,
        think=settings.llm_think,
        thinking_budget=settings.llm_thinking_budget,
    )
    if not (await client.health())["model_available"]:
        print(f"{model:<26} not pulled - run: ollama pull {model}")
        await client.aclose()
        return None
    await client.detect_capabilities()
    await client.warmup()

    scores, ttfts, totals, words = [], [], [], []
    for question, mode, concepts in QUESTIONS * repeats:
        messages = build_messages(question, mode, context, [],
                                  max_context_chars=settings.max_context_chars,
                                  brief_reasoning=client.think_enabled)
        t0 = time.perf_counter()
        first: float | None = None
        text = ""
        try:
            async for d in client.stream_chat(messages):
                if d.truncated:
                    continue
                if d.reset:
                    text, first = "", None
                    continue
                if first is None:
                    first = time.perf_counter()
                text += d.text
        except LLMUnavailable as exc:
            print(f"  {question[:44]}: {exc}")
            continue
        if not text.strip():
            scores.append(0.0)
            continue
        hits, missed = covered(text, concepts)
        scores.append(hits / len(concepts))
        ttfts.append(((first or time.perf_counter()) - t0) * 1000)
        totals.append((time.perf_counter() - t0) * 1000)
        words.append(len(text.split()))
        if verbose:
            print(f"\n  Q: {question}\n  A: {text.strip()}")
            print(f"  covered {hits}/{len(concepts)}" + (f", missed: {', '.join(missed)}" if missed else ""))

    await client.aclose()
    if not scores:
        return None
    return {
        "model": model + (" (reasoning)" if client.think_enabled else ""),
        "score": statistics.mean(scores) * 100,
        "ttft": statistics.median(ttfts) if ttfts else 0,
        "total": statistics.median(totals) if totals else 0,
        "words": statistics.median(words) if words else 0,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen2.5:3b-instruct"])
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("-v", "--verbose", action="store_true", help="print every answer")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 makes the comparison deterministic and repeatable (default)")
    ap.add_argument("--repeats", type=int, default=1, help="ask each question N times")
    args = ap.parse_args()

    context = ContextStore(settings.profile_path, settings.resume_path).render()
    print(f"{len(QUESTIONS)} architecture questions x{args.repeats}, "
          f"MAX_ANSWER_TOKENS={settings.max_answer_tokens}, temperature={args.temperature}\n")

    rows = []
    for m in args.models:
        row = await run_model(m, context, args.timeout, args.verbose, args.temperature, args.repeats)
        if row:
            rows.append(row)
            print(f"  ...{m} done: {row['score']:.0f}% concepts, {row['ttft']:.0f}ms to first word")

    print(f"\n{'model':<30}{'concepts':>10}{'1st word':>11}{'full':>10}{'words':>8}")
    print("-" * 69)
    for r in sorted(rows, key=lambda x: -x["score"]):
        print(f"{r['model']:<30}{r['score']:>9.0f}%{r['ttft']:>10.0f}ms{r['total'] / 1000:>9.1f}s{r['words']:>8.0f}")
    print("\nconcepts = share of the key ideas a good interview answer should mention.")


if __name__ == "__main__":
    asyncio.run(main())
