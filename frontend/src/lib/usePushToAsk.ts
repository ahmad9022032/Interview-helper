"use client";

import { useEffect, useRef, useState } from "react";
import { PUSH_KEY, PUSH_KEY_CODE } from "./types";

interface Options {
  /** Only arm the key while the app is actually capturing audio. */
  enabled: boolean;
  onPress: () => void;
  onRelease: () => void;
}

/**
 * Hold-to-ask. While the space bar is down the interviewer's words are the
 * question; everything else is thrown away, so nothing has to guess where a
 * question ends.
 *
 * Space normally scrolls the page and presses whatever button has focus, so the
 * default is suppressed while this is armed. Typing in a text field is left
 * alone, otherwise you could not type a space in the Ask box.
 */
export function usePushToAsk({ enabled, onPress, onRelease }: Options) {
  const [held, setHeld] = useState(false);
  const heldRef = useRef(false);
  const cb = useRef({ onPress, onRelease });
  cb.current = { onPress, onRelease };

  useEffect(() => {
    if (!enabled) return;

    // Space belongs to push-to-ask everywhere except where it would break the
    // control under the cursor: text fields need to type one, and an open menu
    // needs it to choose a row rather than start a silent capture behind itself.
    const inTextField = (target: EventTarget | null) => {
      const el = target as HTMLElement | null;
      if (!el) return false;
      const tag = el.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable) return true;
      return typeof el.closest === "function" && el.closest("[data-no-push]") !== null;
    };

    const press = () => {
      if (heldRef.current) return;
      heldRef.current = true;
      setHeld(true);
      cb.current.onPress();
    };

    const release = () => {
      if (!heldRef.current) return;
      heldRef.current = false;
      setHeld(false);
      cb.current.onRelease();
    };

    const isPushKey = (e: KeyboardEvent) => e.code === PUSH_KEY_CODE || e.key === PUSH_KEY;

    const onKeyDown = (e: KeyboardEvent) => {
      if (!isPushKey(e)) return;
      if (inTextField(e.target)) return; // you still need to type spaces
      // Stops the page scrolling and stops space "clicking" a focused button.
      e.preventDefault();
      if (e.repeat) return; // holding a key fires keydown over and over
      press();
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (!isPushKey(e)) return;
      if (inTextField(e.target)) return;
      e.preventDefault();
      release();
    };

    // Switching window or tab while holding the key would never deliver keyup,
    // leaving the capture stuck open.
    const onBlur = () => release();
    const onVisibility = () => document.hidden && release();

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      document.removeEventListener("visibilitychange", onVisibility);
      release();
    };
  }, [enabled]);

  return held;
}
