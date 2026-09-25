export type Mode = "technical" | "behavioral" | "project";

export type Stage =
  | "idle"
  | "listening"
  | "capturing"
  | "speech"
  | "transcribing"
  | "thinking"
  | "answering";

export interface Latency {
  stt_ms: number;
  llm_first_token_ms: number;
  llm_ms: number;
  total_ms: number;
  first_answer_ms: number | null;
}

export interface QA {
  id: number;
  question: string;
  answer: string;
  latency: Latency | null;
}

export type MicStatus = "idle" | "requesting" | "active" | "denied" | "error";

/** Where the interviewer's voice comes from. "system" captures speaker output,
 *  so your own voice is never picked up and never answered. */
export type AudioSource = "system" | "mic";

export const SOURCE_STORAGE_KEY = "interview-copilot.source";
export const DEPTH_STORAGE_KEY = "interview-copilot.depth";

/** How much answer to generate. "brief" is the default you can read straight out. */
export type AnswerDepth = "brief" | "detailed" | "steps" | "architecture";

export const DEPTHS: AnswerDepth[] = ["brief", "detailed", "steps", "architecture"];

export const DEPTH_LABELS: Record<AnswerDepth, string> = {
  brief: "Short answer · 2-5 lines",
  detailed: "Detailed · 6-8 lines",
  steps: "Step by step · 10 lines",
  architecture: "Architecture explanation",
};

/** Short enough to sit on the toolbar button without pushing the row around. */
export const DEPTH_BUTTON_LABELS: Record<AnswerDepth, string> = {
  brief: "Short answer",
  detailed: "Detailed",
  steps: "Step by step",
  architecture: "Architecture",
};

/** The one-line "what you actually get" shown under each option in the menu. */
export const DEPTH_HINTS: Record<AnswerDepth, string> = {
  brief: "2-5 lines you can read straight out. The default.",
  detailed: "6-8 lines, including the detail a one-liner would drop.",
  steps: "About 10 short numbered steps, in the order you would do them.",
  architecture: "Every component in data-flow order, plus the main trade-off.",
};
export const CAPTURE_MODE_STORAGE_KEY = "interview-copilot.captureMode";

/** "push": hold a key while the interviewer asks, and only that audio becomes the
 *  question. "auto": the old behaviour, where silence is used to guess where a
 *  question ended. */
export type CaptureMode = "push" | "auto";

/** Held down to mark the question. The space bar is easy to find without looking
 *  and comfortable to hold for a long question. `KeyboardEvent.key` for it is a
 *  single space, so the label is kept separate for display. */
export const PUSH_KEY = " ";
export const PUSH_KEY_CODE = "Space";
export const PUSH_KEY_LABEL = "Space";

/** Seconds of audio kept before the key goes down, so pressing slightly late
 *  does not clip the first words off the question. */
export const PREROLL_S = 1.5;

export interface OllamaModel {
  name: string;
  size_gb: number;
  parameter_size: string | null;
  /** Confirmed by probe to always reason first, which costs many seconds. */
  reasoning: boolean;
  /** Advertises a thinking mode, which it may or may not honour turning off. */
  thinking_capable: boolean;
  probed: boolean;
}

export const BACKEND_URL =
  (typeof process !== "undefined" && process.env.NEXT_PUBLIC_BACKEND_URL) || "http://localhost:8000";

export function wsUrl(): string {
  return BACKEND_URL.replace(/^http/, "ws").replace(/\/$/, "") + "/ws";
}
