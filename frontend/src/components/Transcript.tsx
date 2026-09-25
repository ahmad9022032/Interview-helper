"use client";

import { useEffect, useRef } from "react";

interface Props {
  lines: string[];
  partial: string;
}

export function Transcript({ lines, partial }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest" });
  }, [lines, partial]);

  const latest = partial || lines[lines.length - 1] || "";

  return (
    <details className="panel panel-transcript drawer">
      <summary>
        Live transcript
        {latest && <span className="peek">{latest.slice(0, 64)}</span>}
      </summary>
      <div className="transcript">
        {lines.length === 0 && !partial && <div className="empty">Nothing heard yet.</div>}
        {lines.map((line, i) => (
          <div key={i} className={`line ${i < lines.length - 1 ? "old" : ""}`}>
            &ldquo;{line}&rdquo;
          </div>
        ))}
        {partial && <div className="partial">{partial}…</div>}
        <div ref={endRef} />
      </div>
    </details>
  );
}
