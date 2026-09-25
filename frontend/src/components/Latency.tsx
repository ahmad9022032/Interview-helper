"use client";

import type { Latency as LatencyT } from "@/lib/types";

const fmt = (ms: number | null | undefined) => (ms == null ? "–" : ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`);

export function Latency({ latency }: { latency: LatencyT | null }) {
  return (
    <section className="panel panel-latency">
      <h2>Latency</h2>
      <div className="latency">
        <span>
          STT <b>{fmt(latency?.stt_ms)}</b>
        </span>
        <span>
          LLM first token <b>{fmt(latency?.llm_first_token_ms)}</b>
        </span>
        <span>
          LLM total <b>{fmt(latency?.llm_ms)}</b>
        </span>
        <span title="end of speech → first answer text">
          To first word <b>{fmt(latency?.first_answer_ms)}</b>
        </span>
        <span className="total" title="end of speech → full answer">
          Total <b>{fmt(latency?.total_ms)}</b>
        </span>
      </div>
    </section>
  );
}
