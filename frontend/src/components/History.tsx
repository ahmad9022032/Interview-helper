"use client";

import type { QA } from "@/lib/types";

export function History({ items }: { items: QA[] }) {
  return (
    <details className="panel history drawer">
      <summary>
        Previous questions
        {items.length > 0 && <span className="peek">{items.length}</span>}
      </summary>
      {items.length === 0 ? (
        <div className="empty">No questions answered yet.</div>
      ) : (
        items.map((qa) => (
          <details key={qa.id}>
            <summary>
              {qa.question}
              {qa.latency && <span className="l">{(qa.latency.total_ms / 1000).toFixed(1)}s</span>}
            </summary>
            <p className="a">{qa.answer}</p>
          </details>
        ))
      )}
    </details>
  );
}
