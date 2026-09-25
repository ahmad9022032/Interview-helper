"use client";

import { useEffect, useRef, useState } from "react";
import { BACKEND_URL } from "@/lib/types";

export function ContextPanel() {
  const [text, setText] = useState("");
  const [profilePath, setProfilePath] = useState("");
  const [renderedChars, setRenderedChars] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const loaded = useRef(false);

  const load = () => {
    if (loaded.current) return;
    loaded.current = true;
    fetch(`${BACKEND_URL}/api/context`)
      .then((r) => r.json())
      .then((d) => {
        setText(d.resume_text ?? "");
        setProfilePath(d.profile_path ?? "");
        setRenderedChars(d.rendered_chars ?? null);
      })
      .catch(() => setMsg("Could not load context from backend."));
  };

  useEffect(() => {
    // Show the context size in the summary without opening the drawer.
    fetch(`${BACKEND_URL}/api/context`)
      .then((r) => r.json())
      .then((d) => setRenderedChars(d.rendered_chars ?? null))
      .catch(() => undefined);
  }, []);

  const save = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await fetch(`${BACKEND_URL}/api/context`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ resume_text: text }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? "save failed");
      setRenderedChars(d.rendered_chars);
      setMsg(`Saved (${d.resume_chars} chars).`);
    } catch (e) {
      setMsg(`Save failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const upload = async (file: File) => {
    setBusy(true);
    setMsg(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${BACKEND_URL}/api/context/pdf`, { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? "upload failed");
      setText(d.resume_text ?? "");
      setMsg(`PDF imported (${d.resume_chars} chars). Review the text and adjust if needed.`);
    } catch (e) {
      setMsg(`Upload failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <details className="panel context drawer" onToggle={load}>
      <summary>
        Resume / candidate context
        {renderedChars != null && <span className="peek">{renderedChars} chars</span>}
      </summary>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste your resume here (plain text). It is stored locally in data/resume.txt and used as context for every answer."
      />
      <div className="row">
        <button className="primary" onClick={save} disabled={busy}>
          Save resume text
        </button>
        <label>
          <input
            type="file"
            accept="application/pdf,.pdf"
            disabled={busy}
            onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
          />
        </label>
        {msg && <span className="badge ok">{msg}</span>}
      </div>
      <div className="hint">
        Skills, projects, experience and education come from <code>{profilePath || "config/candidate_profile.yaml"}</code>.
        Edit that file and it is picked up on the next answer.
      </div>
    </details>
  );
}
