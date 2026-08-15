from __future__ import annotations

import argparse
import hmac
import io
import json
import os
import tempfile
import threading
import zipfile
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import DESKTOP_FILE_NAME, MIC_FILE_NAME, MIX_FILE_NAME
from .models import TranscriptionOptions
from .network_whisper import PROTOCOL_VERSION, RESULT_FILES
from .transcription import WhisperTranscriber


MAX_REQUEST_BYTES = 4 * 1024 * 1024 * 1024
_TRANSCRIBER = WhisperTranscriber()
_GPU_JOB_LOCK = threading.Lock()


class WhisperRequestHandler(BaseHTTPRequestHandler):
    server_version = "DialogTxtWhisper/1"

    def _send(self, status: int, body: bytes, content_type: str = "text/plain; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = self.server.auth_token  # type: ignore[attr-defined]
        if not expected:
            return True
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, f"Bearer {expected}")

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b'{"status":"ok"}', "application/json")
            return
        self._send(404, b"not found")

    def do_POST(self):
        if self.path != "/v1/transcribe":
            self._send(404, b"not found")
            return
        if not self._authorized():
            self._send(401, b"invalid token")
            return
        if self.headers.get("X-Dialog-Txt-Protocol") != PROTOCOL_VERSION:
            self._send(400, b"unsupported protocol version")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > self.server.max_request_bytes:  # type: ignore[attr-defined]
            self._send(413, b"request is empty or too large")
            return
        try:
            raw_options = json.loads(self.headers.get("X-Dialog-Txt-Options", "{}"))
            allowed = {item.name for item in fields(TranscriptionOptions)}
            if set(raw_options) != allowed:
                raise ValueError("invalid option fields")
            options = TranscriptionOptions(**raw_options)
            ui_language = self.headers.get("X-Dialog-Txt-Ui-Language", "ru")
            request_body = self.rfile.read(length)
            result = self._process(request_body, options, ui_language)
        except Exception as exc:
            self.log_error("transcription failed: %s", exc)
            self._send(500, str(exc).encode("utf-8", errors="replace"))
            return
        self._send(200, result, "application/zip")

    @staticmethod
    def _process(body: bytes, options: TranscriptionOptions, ui_language: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="dialog-txt-whisper-") as temp_name:
            session_dir = Path(temp_name)
            with zipfile.ZipFile(io.BytesIO(body), "r") as archive:
                names = set(archive.namelist())
                if not {MIC_FILE_NAME, DESKTOP_FILE_NAME}.issubset(names):
                    raise ValueError("archive must contain mic.ogg and desktop.ogg")
                for name in (MIC_FILE_NAME, DESKTOP_FILE_NAME, MIX_FILE_NAME):
                    if name in names:
                        (session_dir / name).write_bytes(archive.read(name))

            cancel_event = threading.Event()
            with _GPU_JOB_LOCK:
                _TRANSCRIBER.transcribe_session(
                    session_dir=session_dir,
                    progress_cb=lambda message, pct: print(f"{pct:5.1f}% {message}", flush=True),
                    cancel_event=cancel_event,
                    options=options,
                    ui_language=ui_language,
                )
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name in RESULT_FILES:
                    path = session_dir / name
                    if path.is_file():
                        archive.write(path, name)
            return output.getvalue()

    def log_message(self, format, *args):
        print(f"{self.client_address[0]} - {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dialog to TXT network Whisper server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=os.environ.get("DIALOG_TXT_SERVER_TOKEN", ""))
    parser.add_argument("--max-request-mb", type=int, default=4096)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), WhisperRequestHandler)
    server.auth_token = args.token
    server.max_request_bytes = args.max_request_mb * 1024 * 1024
    print(f"Dialog TXT Whisper server: http://{args.host}:{args.port}", flush=True)
    if not args.token:
        print("WARNING: server is running without an access token", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
