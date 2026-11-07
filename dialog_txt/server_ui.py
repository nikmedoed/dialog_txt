from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from tkinter import ttk

from .config import APP_ROOT


_JOB_LOG = re.compile(
    r" dialog_txt\.network_server (?:Job (?P<job>[0-9a-f]+) "
    r"(?P<event>started:|completed|(?P<percent>\d+\.\d+)%)"
    r"|Transcription job (?P<failed>[0-9a-f]+) failed)(?: (?P<message>.*))?$"
)


def _launcher(action: str, *args: str) -> str:
    powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    # A detached Windows child can inherit output handles. A file lets us read
    # the result without waiting for the server to close a pipe.
    with tempfile.TemporaryFile(mode="w+t", errors="replace") as output:
        result = subprocess.run(
            [str(powershell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(APP_ROOT / "server.ps1"), action, *args],
            cwd=APP_ROOT, creationflags=subprocess.CREATE_NO_WINDOW, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, timeout=60, check=False,
        )
        output.seek(0)
        message = output.read().strip()
    if result.returncode:
        raise RuntimeError(message or f"{action}: PowerShell exited with code {result.returncode}")
    return message


class ServerControls:
    def __init__(self, app, row):
        self.app = app
        self.running = False
        self.pending = False
        self.owner_token = uuid.uuid4().hex
        self.start_requested = False
        self.close_callback = None
        self.closing = False
        self.log_path = APP_ROOT / ".server" / "server.log"
        self.log_position = 0
        self.log_identity = None
        self.log_timer = None
        self.progress_buckets: dict[str, int] = {}
        self.button = ttk.Button(row, command=self.toggle)
        self.button.grid(row=0, column=6, sticky="w", padx=(0, 4))
        self.render()
        if (APP_ROOT / ".server" / "process.json").is_file():
            self.app.after_idle(self.check_status)

    def check_status(self) -> None:
        if not self.pending and not self.closing:
            self.run("status")

    def toggle(self) -> None:
        if not self.pending:
            self.run("stop" if self.running else "start")

    def defer_close(self, callback) -> bool:
        if not self.start_requested:
            self.closing = True
            return False
        self.close_callback = callback
        if not self.pending:
            self.run("close")
        return True

    def render(self) -> None:
        key = "server_stop" if self.running else "server_button"
        self.button.configure(text=self.app._tr(key), state="disabled" if self.pending else "normal")

    def _read_job_logs(self) -> None:
        try:
            with self.log_path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = stat.st_dev, stat.st_ino
                if identity != self.log_identity or stat.st_size < self.log_position:
                    self.log_identity = identity
                    self.log_position = 0
                stream.seek(self.log_position)
                data = stream.read(65536)
        except OSError:
            return
        complete, newline, _ = data.rpartition(b"\n")
        if not newline:
            return
        self.log_position += len(complete) + 1
        for line in complete.decode("utf-8", errors="replace").splitlines():
            match = _JOB_LOG.search(line)
            if match is None:
                continue
            job = match["job"] or match["failed"]
            if match["percent"] is not None:
                percent = int(float(match["percent"]))
                bucket = percent // 10
                if bucket <= self.progress_buckets.get(job, -1):
                    continue
                self.progress_buckets[job] = bucket
                message = self.app._tr("log_transcription_progress", percent=percent, message=match["message"] or "")
            elif match["failed"]:
                self.progress_buckets.pop(job, None)
                message = self.app._tr("server_job_failed")
            elif match["event"] == "completed":
                self.progress_buckets.pop(job, None)
                message = self.app._tr("server_job_done")
            else:
                message = self.app._tr("server_job_started")
            self.app._log_event(f"[Whisper {job[:8]}] {message}", error=bool(match["failed"]))

    def _poll_job_logs(self) -> None:
        self.log_timer = None
        self._read_job_logs()
        if self.running:
            self.log_timer = self.app.after(1000, self._poll_job_logs)

    def run(self, action: str) -> None:
        self.pending = True
        if action == "start":
            self.start_requested = True
        if not self.running:
            self.progress_buckets.clear()
            try:
                with self.log_path.open("rb") as stream:
                    stat = os.fstat(stream.fileno())
                self.log_identity = stat.st_dev, stat.st_ino
                self.log_position = stat.st_size
            except OSError:
                self.log_identity = None
                self.log_position = 0
        status_key = {
            "start": "server_starting", "stop": "server_stopping",
            "close": "server_stopping", "status": "server_checking",
        }[action]
        self.render()
        self.app._set_status(self.app._tr(status_key), log=False)
        result = queue.Queue(maxsize=1)

        def worker():
            state = None
            error = None
            try:
                if action == "close":
                    _launcher("stop", "-OwnerToken", self.owner_token)
                elif action == "start":
                    _launcher("start", "-OwnerToken", self.owner_token)
                elif action == "stop":
                    _launcher(action)
                if action != "close":
                    state = json.loads(_launcher("status", "-ForApp"))
                    if action == "start" and not state["ready"]:
                        raise RuntimeError("The server did not respond to the health check.")
                    if action == "stop" and state["running"]:
                        raise RuntimeError("The server is still running.")
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                error = str(exc)
            result.put((state, error))

        def receive():
            try:
                state, error = result.get_nowait()
            except queue.Empty:
                self.app.after(75, receive)
                return
            self.pending = False
            if action == "close":
                callback = self.close_callback
                self.close_callback = None
                if error is None:
                    self.start_requested = False
                    callback()
                    return
            if state is not None:
                self.running = state["running"]
            if error is not None:
                key = {
                    "start": "server_start_failed", "stop": "server_stop_failed",
                    "close": "server_stop_failed", "status": "server_unknown",
                }[action]
                self.app._set_status(self.app._tr(key), error=True)
                self.app._log_event(error, error=True)
            elif self.running:
                key = "server_ready" if state["ready"] else "server_unresponsive"
                self.app._set_status(self.app._tr(key, port=state["port"]), error=not state["ready"])
            else:
                self.app._set_status(self.app._tr("server_stopped"))
            if self.close_callback is not None:
                self.run("close")
                return
            if self.running and self.log_timer is None:
                self._poll_job_logs()
            elif not self.running and self.log_timer is not None:
                self.app.after_cancel(self.log_timer)
                self._poll_job_logs()
            self.render()

        threading.Thread(target=worker, daemon=True).start()
        self.app.after(75, receive)
