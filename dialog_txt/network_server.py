from __future__ import annotations

import argparse
import io
import json
import tempfile
import threading
import uuid
import zipfile
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import DESKTOP_FILE_NAME, MIC_FILE_NAME, MIX_FILE_NAME
from .models import TranscriptionOptions
from .network_whisper import PROTOCOL_VERSION, RESULT_FILES
from .transcription import WhisperTranscriber

_TRANSCRIBER = WhisperTranscriber()
_GPU_JOB_LOCK = threading.Lock()
_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}


def _snapshot(job: dict) -> dict:
    with job["lock"]:
        return {key: job[key] for key in ("state", "message", "progress", "error", "files")}


def _run_job(job_id: str, body: bytes, options: TranscriptionOptions, ui_language: str) -> None:
    with _JOBS_LOCK:
        job = _JOBS[job_id]
    try:
        with tempfile.TemporaryDirectory(prefix="dialog-txt-whisper-") as temp_name:
            session_dir = Path(temp_name)
            with zipfile.ZipFile(io.BytesIO(body), "r") as archive:
                names = set(archive.namelist())
                if not {MIC_FILE_NAME, DESKTOP_FILE_NAME}.issubset(names):
                    raise ValueError("archive must contain mic.ogg and desktop.ogg")
                for name in (MIC_FILE_NAME, DESKTOP_FILE_NAME, MIX_FILE_NAME):
                    if name in names:
                        (session_dir / name).write_bytes(archive.read(name))

            def progress(message: str, pct: float) -> None:
                files = {}
                for name in RESULT_FILES:
                    path = session_dir / name
                    if path.is_file():
                        files[name] = path.read_text(encoding="utf-8", errors="replace")
                with job["lock"]:
                    job.update(message=message, progress=float(pct), files=files)
                print(f"{pct:5.1f}% {message}", flush=True)

            with job["lock"]:
                job["state"] = "running"
            with _GPU_JOB_LOCK:
                _TRANSCRIBER.transcribe_session(
                    session_dir=session_dir,
                    progress_cb=progress,
                    cancel_event=job["cancel_event"],
                    options=options,
                    ui_language=ui_language,
                )
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name in RESULT_FILES:
                    path = session_dir / name
                    if path.is_file():
                        archive.write(path, name)
            with job["lock"]:
                job.update(result=output.getvalue(), state="done", progress=100.0)
    except Exception as exc:
        with job["lock"]:
            job["state"] = "cancelled" if job["cancel_event"].is_set() else "error"
            job["error"] = str(exc)


class Handler(BaseHTTPRequestHandler):
    server_version = "DialogTxtWhisper/2"

    def _send(self, status: int, body: bytes, content_type="text/plain; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict):
        self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json")

    @staticmethod
    def _parse_job_path(path: str) -> tuple[str | None, bool]:
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["v1", "jobs"]:
            return parts[2], False
        if len(parts) == 4 and parts[:2] == ["v1", "jobs"] and parts[3] == "result":
            return parts[2], True
        return None, False

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "protocol": PROTOCOL_VERSION})
            return
        job_id, result_requested = self._parse_job_path(self.path)
        with _JOBS_LOCK:
            job = _JOBS.get(job_id or "")
        if job is None:
            self._send(404, b"job not found")
            return
        if not result_requested:
            self._json(200, _snapshot(job))
            return
        with job["lock"]:
            result = job["result"] if job["state"] == "done" else None
        if result is None:
            self._send(409, b"job is not complete")
            return
        self._send(200, result, "application/zip")
        with _JOBS_LOCK:
            _JOBS.pop(job_id, None)

    def do_DELETE(self):
        job_id, _ = self._parse_job_path(self.path)
        with _JOBS_LOCK:
            job = _JOBS.get(job_id or "")
        if job is None:
            self._send(404, b"job not found")
            return
        job["cancel_event"].set()
        self._json(202, {"state": "cancelling"})

    def do_POST(self):
        if self.path != "/v1/transcribe":
            self._send(404, b"not found")
            return
        if self.headers.get("X-Dialog-Txt-Protocol") != PROTOCOL_VERSION:
            self._send(400, b"unsupported protocol version")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.server.max_request_bytes:  # type: ignore[attr-defined]
                self._send(413, b"request is empty or too large")
                return
            raw = json.loads(self.headers.get("X-Dialog-Txt-Options", "{}"))
            if set(raw) != {field.name for field in fields(TranscriptionOptions)}:
                raise ValueError("invalid option fields")
            options = TranscriptionOptions(**raw)
            body = self.rfile.read(length)
        except Exception as exc:
            self._send(400, str(exc).encode())
            return
        job_id = uuid.uuid4().hex
        job = dict(
            lock=threading.Lock(), cancel_event=threading.Event(), state="queued",
            message="Queued", progress=2.0, error="", files={}, result=None,
        )
        with _JOBS_LOCK:
            _JOBS[job_id] = job
        threading.Thread(
            target=_run_job,
            args=(job_id, body, options, self.headers.get("X-Dialog-Txt-Ui-Language", "ru")),
            daemon=True,
        ).start()
        self._json(202, {"job_id": job_id})

    def log_message(self, format, *args):
        print(f"{self.client_address[0]} - {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dialog to TXT network Whisper server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-request-mb", type=int, default=4096)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.max_request_bytes = args.max_request_mb * 1024 * 1024
    print(f"Dialog TXT Whisper server: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
