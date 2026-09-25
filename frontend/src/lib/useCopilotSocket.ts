"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";
import { wsUrl, type AnswerDepth, type CaptureMode, type Latency, type Mode, type QA, type Stage } from "./types";

export interface CopilotState {
  connected: boolean;
  stage: Stage;
  listening: boolean;
  model: string | null;
  whisperModel: string | null;
  whisperReady: boolean;
  modes: Mode[];
  mode: Mode;
  captureMode: CaptureMode;
  depth: AnswerDepth;
  partial: string;
  transcript: string[];
  question: string;
  answer: string;
  latency: Latency | null;
  truncated: boolean;
  history: QA[];
  error: string | null;
  notice: string | null;
}

const initialState: CopilotState = {
  connected: false,
  stage: "idle",
  listening: false,
  model: null,
  whisperModel: null,
  whisperReady: true,
  modes: ["technical", "behavioral", "project"],
  mode: "technical",
  captureMode: "push",
  depth: "brief",
  partial: "",
  transcript: [],
  question: "",
  answer: "",
  latency: null,
  truncated: false,
  history: [],
  error: null,
  notice: null,
};

type Action =
  | { type: "connected"; value: boolean }
  | { type: "server"; event: Record<string, unknown> }
  | { type: "set_mode"; mode: Mode }
  | { type: "set_error"; message: string | null }
  | { type: "dismiss_notice" }
  | { type: "set_model"; model: string; reasoning: boolean }
  | { type: "set_capture_mode"; mode: CaptureMode }
  | { type: "set_depth"; depth: AnswerDepth }
  | { type: "clear" };

let qaId = 0;

function reducer(state: CopilotState, action: Action): CopilotState {
  switch (action.type) {
    case "connected":
      return { ...state, connected: action.value, ...(action.value ? {} : { stage: "idle", listening: false }) };
    case "set_mode":
      return { ...state, mode: action.mode };
    case "set_error":
      return { ...state, error: action.message };
    case "dismiss_notice":
      return { ...state, notice: null };
    case "set_capture_mode":
      return { ...state, captureMode: action.mode };
    case "set_depth":
      return { ...state, depth: action.depth };
    case "set_model":
      return {
        ...state,
        model: action.model,
        notice: action.reasoning
          ? `${action.model} is a reasoning model: it thinks before answering, so expect roughly 10-30s per answer.`
          : null,
      };
    case "clear":
      return { ...state, partial: "", transcript: [], question: "", answer: "", latency: null, truncated: false, history: [], error: null };
    case "server": {
      const ev = action.event;
      switch (ev.type) {
        case "hello":
          return {
            ...state,
            model: (ev.model as string) ?? null,
            whisperModel: (ev.whisper_model as string) ?? null,
            whisperReady: Boolean(ev.whisper_ready),
            modes: (ev.modes as Mode[]) ?? state.modes,
            captureMode: (ev.capture_mode as CaptureMode) ?? state.captureMode,
            depth: (ev.depth as AnswerDepth) ?? state.depth,
            notice: (ev.notice as string) ?? null,
          };
        case "status":
          return { ...state, stage: ev.stage as Stage, listening: Boolean(ev.listening) };
        case "transcript_partial":
          return { ...state, partial: String(ev.text ?? "") };
        case "transcript_final":
          return {
            ...state,
            partial: "",
            transcript: [...state.transcript, String(ev.text ?? "")].slice(-12),
          };
        case "question":
          return { ...state, question: String(ev.text ?? ""), answer: "", latency: null, truncated: false, error: null };
        case "answer_delta":
          return { ...state, answer: state.answer + String(ev.text ?? "") };
        case "answer_reset":
          // What was streamed so far turned out to be the model's reasoning.
          return { ...state, answer: "" };
        case "answer_done": {
          const answer = String(ev.text ?? state.answer);
          const latency = (ev.latency as Latency) ?? null;
          const qa: QA = { id: ++qaId, question: String(ev.question ?? state.question), answer, latency };
          return {
            ...state,
            answer,
            latency,
            truncated: Boolean(ev.truncated),
            history: answer ? [qa, ...state.history].slice(0, 10) : state.history,
          };
        }
        case "answer_cancelled":
          return state;
        case "error":
          return { ...state, error: String(ev.message ?? "Unknown error") };
        case "cleared":
          return { ...state, partial: "", transcript: [], question: "", answer: "", latency: null, truncated: false, history: [] };
        case "mode":
          return { ...state, mode: ev.mode as Mode };
        case "capture_mode":
          return { ...state, captureMode: ev.capture_mode as CaptureMode };
        case "depth":
          return { ...state, depth: ev.depth as AnswerDepth };
        default:
          return state;
      }
    }
    default:
      return state;
  }
}

export function useCopilotSocket() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const closedRef = useRef(false);

  useEffect(() => {
    closedRef.current = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      if (closedRef.current) return;
      const ws = new WebSocket(wsUrl());
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      ws.onopen = () => {
        retryRef.current = 0;
        dispatch({ type: "connected", value: true });
      };
      ws.onmessage = (msg) => {
        if (typeof msg.data !== "string") return;
        try {
          dispatch({ type: "server", event: JSON.parse(msg.data) });
        } catch {
          /* ignore malformed frames */
        }
      };
      ws.onclose = () => {
        dispatch({ type: "connected", value: false });
        if (closedRef.current) return;
        const delay = Math.min(8000, 500 * 2 ** retryRef.current++);
        timer = setTimeout(connect, delay);
      };
      ws.onerror = () => ws.close();
    };
    connect();

    return () => {
      closedRef.current = true;
      if (timer) clearTimeout(timer);
      wsRef.current?.close();
    };
  }, []);

  const sendControl = useCallback((payload: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));
  }, []);

  const sendAudio = useCallback((pcm: ArrayBuffer) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN && ws.bufferedAmount < 256 * 1024) ws.send(pcm);
  }, []);

  const setMode = useCallback(
    (mode: Mode) => {
      dispatch({ type: "set_mode", mode });
      sendControl({ type: "mode", mode });
    },
    [sendControl],
  );

  const setError = useCallback((message: string | null) => dispatch({ type: "set_error", message }), []);
  const dismissNotice = useCallback(() => dispatch({ type: "dismiss_notice" }), []);
  const setCaptureMode = useCallback((mode: CaptureMode) => dispatch({ type: "set_capture_mode", mode }), []);
  const setDepth = useCallback((depth: AnswerDepth) => dispatch({ type: "set_depth", depth }), []);
  const setModel = useCallback(
    (model: string, reasoning: boolean) => dispatch({ type: "set_model", model, reasoning }),
    [],
  );
  const clearLocal = useCallback(() => dispatch({ type: "clear" }), []);

  return { state, sendControl, sendAudio, setMode, setError, dismissNotice, clearLocal, setModel, setCaptureMode, setDepth };
}
