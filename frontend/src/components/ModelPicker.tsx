"use client";

import { useCallback, useEffect, useState } from "react";
import { BACKEND_URL, type OllamaModel } from "@/lib/types";

interface Props {
  active: string | null;
  onSwitched: (model: string, reasoning: boolean) => void;
  onError: (message: string) => void;
}

export function ModelPicker({ active, onSwitched, onError }: Props) {
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [current, setCurrent] = useState<string>("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(() => {
    fetch(`${BACKEND_URL}/api/models`)
      .then((r) => r.json())
      .then((d) => {
        setModels(d.models ?? []);
        setCurrent(d.active ?? "");
      })
      .catch(() => undefined);
  }, []);

  useEffect(refresh, [refresh]);
  useEffect(() => {
    if (active) setCurrent(active);
  }, [active]);

  const switchTo = async (name: string) => {
    if (!name || name === current) return;
    const previous = current;
    setCurrent(name);
    setBusy(true);
    try {
      const r = await fetch(`${BACKEND_URL}/api/models`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: name }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? "switch failed");
      onSwitched(d.model, Boolean(d.thinking_enabled));
    } catch (e) {
      setCurrent(previous);
      onError(`Could not switch model: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  if (models.length === 0) return null;

  return (
    <label className="meta">
      Model{" "}
      <select value={current} onChange={(e) => switchTo(e.target.value)} disabled={busy} title="Switch the answering model">
        {models.map((m) => (
          <option key={m.name} value={m.name}>
            {m.name}
            {m.size_gb ? ` · ${m.size_gb}GB` : ""}
            {m.reasoning ? " · always reasons, slow" : ""}
          </option>
        ))}
      </select>
    </label>
  );
}
