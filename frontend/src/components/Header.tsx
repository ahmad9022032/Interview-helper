"use client";

import type { AudioSource, Mode } from "@/lib/types";
import { ModelPicker } from "./ModelPicker";

interface Props {
  connected: boolean;
  listening: boolean;
  source: AudioSource | null;
  model: string | null;
  whisperModel: string | null;
  mode: Mode;
  modes: Mode[];
  onModeChange: (mode: Mode) => void;
  onModelSwitched: (model: string, reasoning: boolean) => void;
  onModelError: (message: string) => void;
}

const MODE_LABELS: Record<Mode, string> = {
  technical: "Technical",
  behavioral: "Behavioral",
  project: "Project",
};

export function Header(props: Props) {
  const { connected, listening, source, model, whisperModel, mode, modes, onModeChange } = props;
  return (
    <header className="header">
      <div>
        <h1>INTERVIEW COPILOT</h1>
        <div className="meta">
          <span className={`dot ${connected ? "green" : "red"}`} />
          {connected ? "backend connected" : "backend offline - start the FastAPI server"}
          {whisperModel ? ` · whisper ${whisperModel}` : ""}
        </div>
      </div>
      <div className="right">
        <span className="meta">
          <span className={`dot ${listening ? "green pulse" : ""}`} />
          {listening ? `Listening to ${source === "mic" ? "microphone" : "interviewer"}` : "Not listening"}
        </span>
        <ModelPicker active={model} onSwitched={props.onModelSwitched} onError={props.onModelError} />
        <label className="meta">
          Mode{" "}
          <select value={mode} onChange={(e) => onModeChange(e.target.value as Mode)}>
            {modes.map((m) => (
              <option key={m} value={m}>
                {MODE_LABELS[m] ?? m}
              </option>
            ))}
          </select>
        </label>
      </div>
    </header>
  );
}
