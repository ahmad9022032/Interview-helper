"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { AudioSource, MicStatus } from "./types";

export const MIC_REQUIRED_MSG = "Microphone access required.";
export const SHARE_CANCELLED_MSG = "Audio capture cancelled.";
export const NO_AUDIO_MSG =
  "That source has no audio. Re-share and tick \"Also share tab audio\" (pick a browser tab, not a window).";

/**
 * Captures audio and hands 16 kHz Int16 PCM chunks to `onChunk`.
 *
 * Two sources:
 *  - "system": the interviewer's voice as it comes out of your speakers, via
 *    getDisplayMedia. This is the one you want: your own voice is never
 *    captured, so the app only ever answers the interviewer.
 *  - "mic": the microphone, as a fallback. Echo cancellation, noise suppression
 *    and auto gain are all disabled, otherwise the browser filters away the
 *    interviewer's voice coming from the speakers.
 */
export function useAudioCapture(onChunk: (pcm: ArrayBuffer) => void, onAutoStop?: () => void) {
  const [micStatus, setMicStatus] = useState<MicStatus>("idle");
  const [source, setSource] = useState<AudioSource | null>(null);
  const [level, setLevel] = useState(0);
  const [micError, setMicError] = useState<string | null>(null);

  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nodeRef = useRef<AudioWorkletNode | null>(null);
  const onChunkRef = useRef(onChunk);
  const onAutoStopRef = useRef(onAutoStop);
  onChunkRef.current = onChunk;
  onAutoStopRef.current = onAutoStop;

  const teardown = useCallback(() => {
    nodeRef.current?.disconnect();
    nodeRef.current = null;
    streamRef.current?.getTracks().forEach((t) => {
      t.onended = null;
      t.stop();
    });
    streamRef.current = null;
    ctxRef.current?.close().catch(() => undefined);
    ctxRef.current = null;
    setLevel(0);
  }, []);

  const stop = useCallback(() => {
    teardown();
    setSource(null);
    setMicStatus("idle");
  }, [teardown]);

  const start = useCallback(
    async (which: AudioSource) => {
      setMicError(null);
      if (!navigator.mediaDevices) {
        setMicStatus("error");
        setMicError("Audio capture needs https or localhost.");
        return false;
      }
      setMicStatus("requesting");
      let stream: MediaStream;
      try {
        if (which === "system") {
          // Chrome only offers the "share audio" checkbox when video is requested
          // too; we drop the video track immediately and keep just the audio.
          stream = await navigator.mediaDevices.getDisplayMedia({
            video: true,
            audio: {
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: false,
            },
          });
          stream.getVideoTracks().forEach((t) => t.stop());
          if (stream.getAudioTracks().length === 0) {
            stream.getTracks().forEach((t) => t.stop());
            setMicStatus("error");
            setMicError(NO_AUDIO_MSG);
            return false;
          }
        } else {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              channelCount: 1,
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: false,
            },
          });
        }
      } catch (err) {
        const e = err as DOMException;
        const denied = e?.name === "NotAllowedError" || e?.name === "SecurityError";
        setMicStatus(denied ? "denied" : "error");
        setMicError(
          which === "system"
            ? denied
              ? SHARE_CANCELLED_MSG
              : `Could not capture that source (${e?.message || String(err)})`
            : denied
              ? MIC_REQUIRED_MSG
              : `${MIC_REQUIRED_MSG} (${e?.message || String(err)})`,
        );
        return false;
      }

      try {
        streamRef.current = stream;
        // The user can end a screen share from the browser's own bar.
        stream.getAudioTracks().forEach((t) => {
          t.onended = () => {
            stop();
            onAutoStopRef.current?.();
          };
        });

        const ctx = new AudioContext();
        ctxRef.current = ctx;
        await ctx.audioWorklet.addModule("/audio-worklet.js");
        const src = ctx.createMediaStreamSource(stream);
        const node = new AudioWorkletNode(ctx, "pcm-processor", {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          channelCount: 1,
        });
        nodeRef.current = node;
        let lastLevelAt = 0;
        node.port.onmessage = (ev: MessageEvent<{ pcm: ArrayBuffer; rms: number }>) => {
          onChunkRef.current(ev.data.pcm);
          const now = performance.now();
          if (now - lastLevelAt > 80) {
            lastLevelAt = now;
            setLevel(Math.min(1, ev.data.rms * 6));
          }
        };
        // A muted gain node keeps the graph pulling without playing anything back
        // (which would otherwise echo the interviewer through your speakers).
        const silent = ctx.createGain();
        silent.gain.value = 0;
        src.connect(node);
        node.connect(silent);
        silent.connect(ctx.destination);
        if (ctx.state === "suspended") await ctx.resume();
        setSource(which);
        setMicStatus("active");
        return true;
      } catch (err) {
        teardown();
        setMicStatus("error");
        setMicError(`Could not start audio processing: ${(err as Error).message}`);
        return false;
      }
    },
    [stop, teardown],
  );

  useEffect(() => () => teardown(), [teardown]);

  return { micStatus, micError, source, level, start, stop, setMicError };
}

/** True when the microphone was already granted, so we can start without a click. */
export async function micAlreadyGranted(): Promise<boolean> {
  try {
    const status = await navigator.permissions.query({ name: "microphone" as PermissionName });
    return status.state === "granted";
  } catch {
    return false;
  }
}
