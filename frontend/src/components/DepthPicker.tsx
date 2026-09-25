"use client";

import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import {
  DEPTH_BUTTON_LABELS,
  DEPTH_HINTS,
  DEPTH_LABELS,
  DEPTHS,
  type AnswerDepth,
} from "@/lib/types";

interface Props {
  depth: AnswerDepth;
  onChange: (depth: AnswerDepth) => void;
  disabled?: boolean;
}

/** How much answer to generate. Short is the default you read straight out; the
 *  other three are for questions that genuinely need room.
 *
 *  A button rather than a <select> because the difference between the options is
 *  not obvious from a one-word name, and mid-interview is the wrong moment to
 *  guess: each row carries the line count it will actually produce.
 *
 *  `data-no-push` keeps the space bar working as a menu key in here. Everywhere
 *  else space is reserved for push-to-ask (see usePushToAsk).
 */
export function DepthPicker({ depth, onChange, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const items = useRef<(HTMLButtonElement | null)[]>([]);

  const close = useCallback((refocus: boolean) => {
    setOpen(false);
    if (refocus) button.current?.focus();
  }, []);

  // Clicking anywhere else, or tabbing away, puts the menu away again.
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false);
    };
    const onFocusIn = (e: FocusEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("focusin", onFocusIn);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("focusin", onFocusIn);
    };
  }, [open]);

  // Land on the current choice, so arrow keys move from where you are.
  useEffect(() => {
    if (open) items.current[DEPTHS.indexOf(depth)]?.focus();
  }, [open, depth]);

  const pick = (next: AnswerDepth) => {
    if (next !== depth) onChange(next);
    close(true);
  };

  const onMenuKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      close(true);
      return;
    }
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const here = items.current.findIndex((el) => el === document.activeElement);
    const step = e.key === "ArrowDown" ? 1 : -1;
    const next = (here + step + DEPTHS.length) % DEPTHS.length;
    items.current[next]?.focus();
  };

  return (
    <div className="depth" ref={root} data-no-push>
      <button
        ref={button}
        type="button"
        className={`depth-button ${open ? "open" : ""} ${depth !== "brief" ? "set" : ""}`}
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown" && !open) {
            e.preventDefault();
            setOpen(true);
          }
        }}
        disabled={disabled}
        aria-haspopup="menu"
        aria-expanded={open}
        title={`Answer length: ${DEPTH_LABELS[depth]}`}
      >
        <span className="depth-what">Answer</span>
        <span className="depth-now">{DEPTH_BUTTON_LABELS[depth]}</span>
        <span className="caret" aria-hidden="true" />
      </button>

      {open && (
        <div className="depth-menu" role="menu" aria-label="Answer length" onKeyDown={onMenuKeyDown}>
          {DEPTHS.map((d, i) => (
            <button
              key={d}
              type="button"
              role="menuitemradio"
              aria-checked={d === depth}
              className={`depth-item ${d === depth ? "on" : ""}`}
              ref={(el) => {
                items.current[i] = el;
              }}
              onClick={() => pick(d)}
            >
              <span className="tick" aria-hidden="true">
                {d === depth ? "✓" : ""}
              </span>
              <span className="depth-text">
                <span className="depth-name">{DEPTH_LABELS[d]}</span>
                <span className="depth-hint">{DEPTH_HINTS[d]}</span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
