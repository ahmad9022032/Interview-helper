"use client";

import { useEffect, useState } from "react";

interface Props {
  recording: boolean;
  /** The last question is still being transcribed or answered. */
  busy: boolean;
  disabled?: boolean;
  onToggle: () => void;
}

/** One button for the whole question: click to start, click again to send.
 *
 *  This replaced hold-to-talk. Holding a key through a live interview meant
 *  keeping a hand parked on the keyboard and never looking away, and letting go
 *  a moment early clipped the end of the question. A toggle costs two clicks and
 *  cannot be released by accident.
 */
export function RecordButton({ recording, busy, disabled, onToggle }: Props) {
  const seconds = useElapsed(recording);
  const state = recording ? "rec" : busy ? "busy" : "idle";
  const label = recording ? "Stop and answer" : busy ? "Working" : "Record question";

  return (
    <button
      type="button"
      className={`record ${state}`}
      onClick={onToggle}
      disabled={disabled || (busy && !recording)}
      aria-pressed={recording}
      aria-label={label}
      title={recording ? "Click to stop recording and answer" : "Click to record the question"}
    >
      <span className="ricon" aria-hidden="true">
        {recording ? <StopIcon /> : busy ? <span className="spinner" /> : <MicIcon />}
      </span>
      <span className="rlabel">{label}</span>
      {recording && <span className="timer">{fmt(seconds)}</span>}
    </button>
  );
}

/** Seconds since recording started, so a long question is visibly still running. */
function useElapsed(running: boolean): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!running) {
      setSeconds(0);
      return;
    }
    const startedAt = Date.now();
    const id = setInterval(() => setSeconds(Math.floor((Date.now() - startedAt) / 1000)), 250);
    return () => clearInterval(id);
  }, [running]);
  return seconds;
}

function fmt(total: number): string {
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <rect x="9" y="2" width="6" height="11" rx="3" fill="currentColor" stroke="none" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <line x1="12" y1="18" x2="12" y2="22" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true">
      <rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor" />
    </svg>
  );
}
