"""Tests the candidate-context HTTP API: reading the profile, pasting resume
text, uploading a resume PDF, rejecting bad uploads, and reloading from disk.

Backend must be running.

    python tests/test_context_api.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

failures: list[str] = []
BOUNDARY = "----InterviewCopilotTest"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def req(method: str, url: str, data: bytes | None = None, ctype: str | None = None):
    r = urllib.request.Request(url, data=data, method=method)
    if ctype:
        r.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            import json

            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        import json

        body = e.read()
        try:
            return e.code, json.loads(body or b"{}")
        except Exception:  # noqa: BLE001
            return e.code, {"raw": body[:200].decode("utf-8", "ignore")}


def multipart(filename: str, content: bytes, ctype: str) -> bytes:
    return (
        f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: {ctype}\r\n\r\n".encode()
        + content
        + f"\r\n--{BOUNDARY}--\r\n".encode()
    )


def make_pdf(text: str) -> bytes | None:
    """Build a small real PDF using macOS cupsfilter, if available."""
    txt = os.path.join(tempfile.gettempdir(), f"ic_{uuid.uuid4().hex}.txt")
    pdf = txt.replace(".txt", ".pdf")
    with open(txt, "w") as fh:
        fh.write(text)
    try:
        with open(pdf, "wb") as out:
            subprocess.run(["cupsfilter", txt], stdout=out, stderr=subprocess.DEVNULL, check=True)
        return open(pdf, "rb").read()
    except Exception:  # noqa: BLE001
        return None
    finally:
        for p in (txt, pdf):
            if os.path.exists(p):
                os.unlink(p)


def main(base: str) -> int:
    print("GET /api/health")
    status, health = req("GET", f"{base}/api/health")
    check("health responds", status == 200, health.get("status", ""))
    check("reports whisper", bool(health.get("whisper", {}).get("model")))
    check("reports the configured model", bool(health.get("ollama", {}).get("model")))
    check("reports the three modes", health.get("modes") == ["technical", "behavioral", "project"])

    print("\nGET /api/context")
    status, ctx = req("GET", f"{base}/api/context")
    check("context responds", status == 200)
    check("profile loaded from YAML", bool(ctx.get("profile")), f"{len(ctx.get('profile') or {})} keys")
    check("profile has projects", "projects" in (ctx.get("profile") or {}))
    original = ctx.get("resume_text", "")

    print("\nPUT /api/context (paste resume text)")
    marker = f"Unique marker {uuid.uuid4().hex[:8]}"
    body = f'{{"resume_text": "{marker}"}}'.encode()
    status, out = req("PUT", f"{base}/api/context", body, "application/json")
    check("saves pasted resume", status == 200 and out.get("ok") is True)
    _, ctx2 = req("GET", f"{base}/api/context")
    check("pasted resume is read back", marker in ctx2.get("resume_text", ""))
    check("context grew", ctx2.get("rendered_chars", 0) > len(marker))

    print("\nPOST /api/context/pdf")
    pdf_marker = f"PDFMARKER{uuid.uuid4().hex[:6]}"
    pdf = make_pdf(f"Resume of a Candidate\n\n{pdf_marker}\n\nSkills: Python, PyTorch, FastAPI.")
    if pdf:
        status, out = req("POST", f"{base}/api/context/pdf", multipart("resume.pdf", pdf, "application/pdf"),
                          f"multipart/form-data; boundary={BOUNDARY}")
        check("PDF accepted", status == 200 and out.get("ok") is True, str(out.get("resume_chars", out))[:60])
        check("PDF text extracted", pdf_marker in out.get("resume_text", ""), out.get("resume_text", "")[:60])
    else:
        print("  skip (cupsfilter unavailable - cannot build a test PDF)")

    print("\nPOST /api/context/pdf with a non-PDF")
    status, out = req("POST", f"{base}/api/context/pdf", multipart("resume.txt", b"just text", "text/plain"),
                      f"multipart/form-data; boundary={BOUNDARY}")
    check("non-PDF rejected with 400", status == 400, str(out.get("detail"))[:50])

    status, out = req("POST", f"{base}/api/context/pdf", multipart("broken.pdf", b"%PDF-1.4 garbage", "application/pdf"),
                      f"multipart/form-data; boundary={BOUNDARY}")
    check("corrupt PDF rejected with 422, no crash", status == 422, str(out.get("detail"))[:50])

    print("\nGET /api/models")
    status, ml = req("GET", f"{base}/api/models")
    check("models endpoint responds", status == 200)
    names = [m["name"] for m in ml.get("models", [])]
    check("lists pulled models", len(names) > 0, f"{len(names)} models")
    check("reports the active model", ml.get("active") in names, str(ml.get("active")))
    check("flags reasoning models", all("reasoning" in m for m in ml.get("models", [])))

    print("\nPOST /api/models")
    active = ml.get("active")
    other = next((n for n in names if n != active), None)
    if other:
        status, out = req("POST", f"{base}/api/models", f'{{"model": "{other}"}}'.encode(), "application/json")
        check("switches model", status == 200 and out.get("model") == other, f"{active} -> {other}")
        _, after = req("GET", f"{base}/api/models")
        check("switch is visible afterwards", after.get("active") == other)
        status, _ = req("POST", f"{base}/api/models", f'{{"model": "{active}"}}'.encode(), "application/json")
        check("switches back", status == 200)
    else:
        print("  skip (only one model pulled)")
    status, out = req("POST", f"{base}/api/models", b'{"model": "no-such-model:1b"}', "application/json")
    check("unknown model rejected with 404", status == 404, str(out.get("detail"))[:52])
    status, out = req("POST", f"{base}/api/models", b'{"model": "   "}', "application/json")
    check("blank model rejected", status == 400, str(out.get("detail"))[:40])

    print("\nPOST /api/context/reload")
    status, out = req("POST", f"{base}/api/context/reload")
    check("reload works", status == 200 and out.get("ok") is True)

    # restore whatever was there before the test
    req("PUT", f"{base}/api/context", ("{\"resume_text\": " + repr(original).replace("'", '"') + "}").encode()
        if original else b'{"resume_text": ""}', "application/json")
    status, _ = req("GET", f"{base}/api/health")
    check("backend still healthy after all of it", status == 200)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        return 1
    print("all context API checks passed")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8000")
    sys.exit(main(p.parse_args().base))
