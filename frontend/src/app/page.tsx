"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Answer } from "@/components/Answer";
import { ContextPanel } from "@/components/ContextPanel";
import { Controls } from "@/components/Controls";
import { Header } from "@/components/Header";
import { History } from "@/components/History";
import { Latency } from "@/components/Latency";
import { Transcript } from "@/components/Transcript";
import {
  CAPTURE_MODE_STORAGE_KEY,
  DEPTH_STORAGE_KEY,
  DEPTHS,
  PREROLL_S,
  SOURCE_STORAGE_KEY,
  type AnswerDepth,
  type AudioSource,
  type CaptureMode,
} from "@/lib/types";
import { micAlreadyGranted, useAudioCapture } from "@/lib/useAudioCapture";
import { useCopilotSocket } from "@/lib/useCopilotSocket";

// The worklet posts 1024 samples at 16 kHz, so 64 ms per chunk.
const CHUNK_MS = 64;
const PREROLL_CHUNKS = Math.ceil((PREROLL_S * 1000) / CHUNK_MS);

export default function Page() {
  const {
    state, sendControl, sendAudio, setMode, setError, dismissNotice, clearLocal,
    setModel, setCaptureMode, setDepth,
  } = useCopilotSocket();

  const [captureMode, setCaptureModeLocal] = useState<CaptureMode>("push");
  const captureModeRef = useRef<CaptureMode>("push");
  captureModeRef.current = captureMode;

  const [depth, setDepthLocal] = useState<AnswerDepth>("brief");
  const depthRef = useRef<AnswerDepth>("brief");
  depthRef.current = depth;

  // Audio heard while the key is up. Kept only so that pressing a moment late
  // does not clip the first words; never sent unless a capture starts.
  const preroll = useRef<ArrayBuffer[]>([]);
  const capturing = useRef(false);

  const onChunk = useCallback(
    (pcm: ArrayBuffer) => {
      if (captureModeRef.current === "auto" || capturing.current) {
        sendAudio(pcm);
        return;
      }
      preroll.current.push(pcm);
      if (preroll.current.length > PREROLL_CHUNKS) preroll.current.shift();
    },
    [sendAudio],
  );

  const onAutoStop = useCallback(() => sendControl({ type: "stop" }), [sendControl]);
  const { micStatus, micError, source, level, start, stop, setMicError } = useAudioCapture(onChunk, onAutoStop);

  const modeRef = useRef(state.mode);
  modeRef.current = state.mode;
  const autoStarted = useRef(false);

  // Restore the last capture mode before the first render that matters.
  useEffect(() => {
    try {
      const saved = localStorage.getItem(CAPTURE_MODE_STORAGE_KEY);
      if (saved === "auto" || saved === "push") {
        setCaptureModeLocal(saved);
        captureModeRef.current = saved;
      }
      const d = localStorage.getItem(DEPTH_STORAGE_KEY) as AnswerDepth | null;
      if (d && DEPTHS.includes(d)) {
        setDepthLocal(d);
        depthRef.current = d;
      }
    } catch {
      /* private browsing */
    }
  }, []);

  const onDepthChange = useCallback(
    (next: AnswerDepth) => {
      setDepthLocal(next);
      depthRef.current = next;
      setDepth(next);
      try {
        localStorage.setItem(DEPTH_STORAGE_KEY, next);
      } catch {
        /* private browsing */
      }
      sendControl({ type: "depth", depth: next });
    },
    [sendControl, setDepth],
  );

  const beginCapture = useCallback(() => {
    capturing.current = true;
    sendControl({ type: "capture_start" });
    // Hand over the moments just before the key went down, then go live.
    for (const chunk of preroll.current) sendAudio(chunk);
    preroll.current = [];
  }, [sendControl, sendAudio]);

  const endCapture = useCallback(() => {
    if (!capturing.current) return;
    capturing.current = false;
    sendControl({ type: "capture_stop" });
  }, [sendControl]);

  // Click to start, click to send. `recording` is optimistic so the button reacts
  // on the click rather than waiting for the server to echo the stage back.
  const [recording, setRecording] = useState(false);
  const recordingRef = useRef(false);

  const clearRecording = useCallback(() => {
    if (!recordingRef.current) return;
    recordingRef.current = false;
    setRecording(false);
  }, []);

  const canRecord = captureMode === "push" && state.listening && micStatus === "active" && state.connected;

  const toggleRecording = useCallback(() => {
    if (!canRecord) return;
    if (recordingRef.current) {
      recordingRef.current = false;
      setRecording(false);
      endCapture();
    } else {
      recordingRef.current = true;
      setRecording(true);
      beginCapture();
    }
  }, [canRecord, beginCapture, endCapture]);

  // The backend can end a capture on its own: it caps a single question at
  // PUSH_MAX_S, and it rejects one too short to transcribe. Follow it in both
  // cases, otherwise the button would sit there claiming to still be recording.
  useEffect(() => {
    if (!recording) return;
    if (state.stage === "transcribing" || state.stage === "thinking" || state.stage === "answering") {
      clearRecording();
    }
  }, [state.stage, recording, clearRecording]);

  useEffect(() => {
    if (state.error) clearRecording();
  }, [state.error, clearRecording]);

  useEffect(() => {
    if (recording && (!state.connected || micStatus !== "active")) clearRecording();
  }, [recording, state.connected, micStatus, clearRecording]);

  const onStart = useCallback(
    async (which: AudioSource) => {
      setError(null);
      setMicError(null);
      const ok = await start(which);
      if (!ok) return;
      try {
        localStorage.setItem(SOURCE_STORAGE_KEY, which);
      } catch {
        /* private browsing */
      }
      preroll.current = [];
      capturing.current = false;
      sendControl({
        type: "start",
        mode: modeRef.current,
        capture_mode: captureModeRef.current,
        depth: depthRef.current,
      });
    },
    [start, sendControl, setError, setMicError],
  );

  const onStop = useCallback(() => {
    capturing.current = false;
    preroll.current = [];
    clearRecording();
    stop();
    sendControl({ type: "stop" });
  }, [stop, sendControl, clearRecording]);

  const onClear = useCallback(() => {
    clearLocal();
    sendControl({ type: "clear" });
  }, [clearLocal, sendControl]);

  const onAsk = useCallback((text: string) => sendControl({ type: "ask", text }), [sendControl]);

  const onCaptureModeChange = useCallback(
    (next: CaptureMode) => {
      capturing.current = false;
      preroll.current = [];
      setCaptureModeLocal(next);
      captureModeRef.current = next;
      setCaptureMode(next);
      try {
        localStorage.setItem(CAPTURE_MODE_STORAGE_KEY, next);
      } catch {
        /* private browsing */
      }
      sendControl({ type: "capture_mode", mode: next });
    },
    [sendControl, setCaptureMode],
  );

  // Resume automatically where the browser allows it. Microphone permission is
  // remembered across reloads; speaker capture always needs a fresh gesture.
  useEffect(() => {
    if (!state.connected || autoStarted.current || micStatus !== "idle") return;
    let cancelled = false;
    (async () => {
      let saved: string | null = null;
      try {
        saved = localStorage.getItem(SOURCE_STORAGE_KEY);
      } catch {
        return;
      }
      if (saved !== "mic" || !(await micAlreadyGranted()) || cancelled) return;
      autoStarted.current = true;
      onStart("mic");
    })();
    return () => {
      cancelled = true;
    };
  }, [state.connected, micStatus, onStart]);

  // If the socket drops while capturing, stop so the UI stays truthful.
  useEffect(() => {
    if (!state.connected && micStatus === "active") stop();
  }, [state.connected, micStatus, stop]);

  const error = micError ?? state.error;
  const stage = useMemo(() => (recording ? "capturing" : state.stage), [recording, state.stage]);

  return (
    <main className="app">
      <Header
        connected={state.connected}
        listening={state.listening && micStatus === "active"}
        source={source}
        model={state.model}
        whisperModel={state.whisperModel}
        mode={state.mode}
        modes={state.modes}
        onModeChange={setMode}
        onModelSwitched={setModel}
        onModelError={setError}
      />

      {error && (
        <div className="banner" role="alert">
          <span>{error}</span>
          <button onClick={() => (micError ? setMicError(null) : setError(null))}>dismiss</button>
        </div>
      )}

      {state.notice && (
        <div className="banner notice-banner">
          <span>{state.notice}</span>
          <button onClick={dismissNotice}>dismiss</button>
        </div>
      )}

      <Controls
        connected={state.connected}
        listening={state.listening}
        micStatus={micStatus}
        source={source}
        level={level}
        stage={stage}
        whisperReady={state.whisperReady}
        captureMode={captureMode}
        recording={recording}
        onToggleRecording={toggleRecording}
        depth={depth}
        onDepthChange={onDepthChange}
        onCaptureModeChange={onCaptureModeChange}
        onStart={onStart}
        onStop={onStop}
        onClear={onClear}
        onAsk={onAsk}
      />

      <Answer question={state.question} answer={state.answer} stage={stage} truncated={state.truncated} />

      <Latency latency={state.latency} />

      <div className="secondary">
        <Transcript lines={state.transcript} partial={state.partial} />
        <History items={state.history} />
        <ContextPanel />
      </div>
    </main>
  );
}
