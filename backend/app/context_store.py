"""Candidate context: a structured YAML profile (config/candidate_profile.yaml)
plus free-form resume text (data/resume.txt) that can be pasted in the UI or
extracted from an uploaded PDF. Rendered into one compact text block for the
system prompt."""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


def _as_lines(value: Any, indent: str = "") -> list[str]:
    """Render nested YAML values as compact indented text."""
    lines: list[str] = []
    if value is None:
        return lines
    if isinstance(value, dict):
        for k, v in value.items():
            key = str(k).replace("_", " ").title()
            if isinstance(v, (dict, list)):
                lines.append(f"{indent}{key}:")
                lines.extend(_as_lines(v, indent + "  "))
            else:
                lines.append(f"{indent}{key}: {v}")
    elif isinstance(value, list):
        if all(not isinstance(i, (dict, list)) for i in value):
            lines.append(f"{indent}" + ", ".join(str(i) for i in value))
        else:
            for item in value:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("title") or item.get("role") or ""
                    lines.append(f"{indent}- {name}".rstrip())
                    rest = {k: v for k, v in item.items() if k not in {"name", "title", "role"}}
                    lines.extend(_as_lines(rest, indent + "  "))
                else:
                    lines.append(f"{indent}- {item}")
    else:
        lines.append(f"{indent}{value}")
    return lines


class ContextStore:
    def __init__(self, profile_path: Path, resume_path: Path) -> None:
        self.profile_path = profile_path
        self.resume_path = resume_path
        self._profile: dict = {}
        self._resume_text: str = ""
        self.reload()

    # ---- loading / saving -------------------------------------------------
    def reload(self) -> None:
        self._profile = {}
        if self.profile_path.exists():
            try:
                data = yaml.safe_load(self.profile_path.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict):
                    self._profile = data
            except Exception:  # noqa: BLE001
                log.exception("Failed to read profile %s", self.profile_path)
        self._resume_text = ""
        if self.resume_path.exists():
            try:
                self._resume_text = self.resume_path.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                log.exception("Failed to read resume %s", self.resume_path)

    def set_resume_text(self, text: str) -> None:
        self._resume_text = text.strip()
        self.resume_path.parent.mkdir(parents=True, exist_ok=True)
        self.resume_path.write_text(self._resume_text, encoding="utf-8")

    def set_resume_pdf(self, data: bytes) -> str:
        from pypdf import PdfReader  # imported lazily; only needed for uploads

        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(p.strip() for p in pages if p.strip())
        if not text.strip():
            raise ValueError("No extractable text found in the PDF (is it scanned?).")
        self.set_resume_text(text)
        return text

    # ---- accessors --------------------------------------------------------
    @property
    def resume_text(self) -> str:
        return self._resume_text

    @property
    def profile(self) -> dict:
        return self._profile

    def render(self) -> str:
        parts: list[str] = []
        if self._profile:
            parts.append("PROFILE:")
            parts.extend(_as_lines(self._profile))
        if self._resume_text:
            parts.append("")
            parts.append("RESUME:")
            parts.append(self._resume_text.strip())
        return "\n".join(parts).strip()
