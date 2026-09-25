"use client";

import { useState } from "react";
import { PUSH_KEY_LABEL, type AnswerDepth, type AudioSource, type CaptureMode, type MicStatus, type Stage } from "@/lib/types";
import { DepthPicker } from "./DepthPicker";

interface Props {
  connected: boolean;
  listening: boolean;
  micStatus: MicStatus;
  source: AudioSource | null;
  level: number;
  stage: Stage;
  whisperReady: boolean;
  captureMode: CaptureMode;
  keyHeld: boolean;
  depth: AnswerDepth;
  onDepthChange: (depth: AnswerDepth) => void;
  onCaptureModeChange: (mode: CaptureMode) => void;
  onStart: (source: AudioSource) => void;
  onStop: () => void;
  onClear: () => void;
  onAsk: (text: string) => void;
}

const STAGE_LABEL: Record<Stage, string> = {
  idle: "Idle",
  listening: "Ready",
  capturing: "Capturing the question",
  speech: "Interviewer speaking",
  transcribing: "Transcribing",
  thinking: "Generating answer",
  answering: "Answering",
};

export function Controls(props: Props) {
  const {
    connected, listening, micStatus, source, level, stage, whisperReady,
    captureMode, keyHeld, depth, onDepthChange, onCaptureModeChange,
    onStart, onStop, onClear, onAsk,
  } = props;
  const [typed, setTyped] = useState("");
  const active = micStatus === "active";
  const push = captureMode === "push";

  const submit = () => {
    const text = typed.trim();
    if (!text) return;
    onAsk(text);
    setTyped("");
  };

  const dot =
    keyHeld
      ? "red pulse"
      : stage === "speech"
        ? "amber pulse"
        : stage === "thinking" || stage === "answering" || stage === "transcribing"
          ? "blue pulse"
          : active
            ? "green"
            : "";

  return (
    <section className="panel panel-controls">
      {active ? (
        <>
          <div className="bar">
            <span className="live">
              <span className={`dot ${dot}`} />
              {STAGE_LABEL[stage]}
              <span className="src">{source === "system" ? "interviewer audio" : "microphone"}</span>
            </span>
            <span className="level" title="input level">
              <div style={{ width: `${Math.round(level * 100)}%` }} />
            </span>
            <DepthPicker depth={depth} onChange={onDepthChange} />
            <span className="spacer" />
            <button onClick={onStop}>■ Stop</button>
            <button className="danger" onClick={onClear} disabled={!connected}>
              Clear
            </button>
          </div>

          {push && (
            <div className={`pushbar ${keyHeld ? "held" : ""}`} aria-live="polite">
              {keyHeld ? (
                <>
                  <span className="rec" /> Recording the question. Release <kbd>{PUSH_KEY_LABEL}</kbd> to answer.
                </>
              ) : (
                <>
                  Hold <kbd>{PUSH_KEY_LABEL}</kbd> while the interviewer asks. Nothing else is used.
                </>
              )}
            </div>
          )}
        </>
      ) : (
        <div className="bar start">
          <button className="primary" onClick={() => onStart("system")} disabled={!connected || micStatus === "requesting"}>
            ● Listen to the interviewer
          </button>
          <button onClick={() => onStart("mic")} disabled={!connected || micStatus === "requesting"}>
            Use microphone instead
          </button>
          <DepthPicker depth={depth} onChange={onDepthChange} />
          <span className="spacer" />
          <button className="danger" onClick={onClear} disabled={!connected}>
            Clear
          </button>
        </div>
      )}

      <div className="bar modes">
        <span className="meta">Question capture</span>
        <label className={`pick ${push ? "on" : ""}`}>
          <input
            type="radio"
            name="capture-mode"
            checked={push}
            onChange={() => onCaptureModeChange("push")}
          />
          Hold <kbd>{PUSH_KEY_LABEL}</kbd> to ask
        </label>
        <label className={`pick ${!push ? "on" : ""}`}>
          <input
            type="radio"
            name="capture-mode"
            checked={!push}
            onChange={() => onCaptureModeChange("auto")}
          />
          Automatic
        </label>
        {!whisperReady && <span className="badge warn">Speech recognition not loaded</span>}
      </div>

      {!active && push && (
        <p className="hint">
          Start listening, then hold <kbd>{PUSH_KEY_LABEL}</kbd> for exactly the question you want answered. Pauses in the
          middle no longer split it, and anything said while the key is up is discarded.
        </p>
      )}

      <div className="ask">
        <input
          placeholder="Or type a question and press Enter"
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          disabled={!connected}
        />
        <button onClick={submit} disabled={!connected || !typed.trim()}>
          Ask
        </button>
      </div>
    </section>
  );
}
