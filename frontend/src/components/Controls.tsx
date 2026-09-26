"use client";

import { useState } from "react";
import type { AnswerDepth, AudioSource, CaptureMode, MicStatus, Stage } from "@/lib/types";
import { DepthPicker } from "./DepthPicker";
import { RecordButton } from "./RecordButton";

interface Props {
  connected: boolean;
  listening: boolean;
  micStatus: MicStatus;
  source: AudioSource | null;
  level: number;
  stage: Stage;
  whisperReady: boolean;
  captureMode: CaptureMode;
  recording: boolean;
  depth: AnswerDepth;
  onToggleRecording: () => void;
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
  capturing: "Recording the question",
  speech: "Interviewer speaking",
  transcribing: "Transcribing",
  thinking: "Generating answer",
  answering: "Answering",
};

export function Controls(props: Props) {
  const {
    connected, listening, micStatus, source, level, stage, whisperReady,
    captureMode, recording, depth, onToggleRecording, onDepthChange,
    onCaptureModeChange, onStart, onStop, onClear, onAsk,
  } = props;
  const [typed, setTyped] = useState("");
  const active = micStatus === "active";
  const manual = captureMode === "push";
  const busy = stage === "transcribing" || stage === "thinking" || stage === "answering";

  const submit = () => {
    const text = typed.trim();
    if (!text) return;
    onAsk(text);
    setTyped("");
  };

  const dot =
    recording
      ? "red pulse"
      : stage === "speech"
        ? "amber pulse"
        : busy
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

          {manual && (
            <div className={`recordbar ${recording ? "on" : ""}`}>
              <RecordButton
                recording={recording}
                busy={busy}
                disabled={!connected || !listening}
                onToggle={onToggleRecording}
              />
              <span className="recordhint" aria-live="polite">
                {recording
                  ? "Recording. Click again the moment they finish and the answer starts straight away."
                  : busy
                    ? `${STAGE_LABEL[stage]}…`
                    : "Click when the interviewer starts asking. Nothing else is sent."}
              </span>
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
        <label className={`pick ${manual ? "on" : ""}`}>
          <input
            type="radio"
            name="capture-mode"
            checked={manual}
            onChange={() => onCaptureModeChange("push")}
          />
          Record button
        </label>
        <label className={`pick ${!manual ? "on" : ""}`}>
          <input
            type="radio"
            name="capture-mode"
            checked={!manual}
            onChange={() => onCaptureModeChange("auto")}
          />
          Automatic
        </label>
        {!whisperReady && <span className="badge warn">Speech recognition not loaded</span>}
      </div>

      {!active && manual && (
        <p className="hint">
          Start listening, then click <b>Record question</b> when the interviewer begins and click it
          again when they finish. Only what you record is transcribed, and the answer starts the
          moment you stop.
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
