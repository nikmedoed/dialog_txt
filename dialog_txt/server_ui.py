from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .config import APP_ROOT


def _launcher(action: str, *args: str) -> str:
    powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    # A detached Windows child can inherit the launcher's output handles.
    # A file lets us read the result without waiting for the server to close a pipe.
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


def _status() -> dict:
    status = json.loads(_launcher("status", "-ForApp"))
    host = status["host"]
    hosts = []
    if status["ready"] and host == "0.0.0.0":
        try:
            hosts = sorted({ip for ip in socket.gethostbyname_ex(socket.gethostname())[2]
                            if not ip.startswith(("127.", "169.254."))})
        except OSError:
            pass
        hosts.append(socket.gethostname())
    elif status["ready"] and host not in {"127.0.0.1", "localhost", "::1"}:
        hosts = [f"[{host}]" if ":" in host else host]
    status["urls"] = [f"http://{host}:{status['port']}" for host in hosts]
    return status


def _recent_logs() -> str:
    chunks = []
    for name in ("server.log", "stderr.log"):
        try:
            with (APP_ROOT / ".server" / name).open("rb") as stream:
                stream.seek(max(0, os.fstat(stream.fileno()).st_size - 16_384))
                lines = stream.read(16_384).decode("utf-8", errors="replace").splitlines()[-60:]
        except FileNotFoundError:
            continue
        if lines:
            chunks.append(f"{name}\n" + "\n".join(lines))
    return "\n\n".join(chunks)


def show_server_controls(app) -> None:
    panel = getattr(app, "_server_controls", None)
    if panel is not None and panel.winfo_exists():
        panel.lift()
        return
    app._server_controls = ServerControls(app)


class ServerControls(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.tr = app._tr
        self.title(self.tr("server_controls"))
        self.transient(app)
        self.minsize(460, 280)
        self._pending = False
        self._state = None
        self.protocol("WM_DELETE_WINDOW", self._close)

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        self.status = ttk.Label(frame, anchor=tk.W)
        self.status.pack(fill=tk.X)
        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X, pady=6)
        self.start = ttk.Button(buttons, text=self.tr("server_start"), command=lambda: self._run("start"))
        self.stop = ttk.Button(buttons, text=self.tr("server_stop"), command=self._stop)
        self.refresh = ttk.Button(buttons, text=self.tr("server_refresh"), command=lambda: self._run("status"))
        self.logs = ttk.Button(buttons, text=self.tr("server_logs"), command=lambda: self._run("logs"))
        for widget in (self.start, self.stop, self.refresh, self.logs):
            widget.pack(side=tk.LEFT, padx=(0, 4))

        address = ttk.Frame(frame)
        address.pack(fill=tk.X)
        ttk.Label(address, text=self.tr("server_address")).pack(side=tk.LEFT, padx=(0, 6))
        self.copy = ttk.Button(address, text=self.tr("server_copy"), command=self._copy)
        self.copy.pack(side=tk.RIGHT, padx=(6, 0))
        self.urls = ttk.Combobox(address, state="readonly")
        self.urls.pack(side=tk.LEFT, fill=tk.X, expand=True)

        settings = ttk.Frame(frame)
        settings.pack(fill=tk.X, pady=6)
        ttk.Label(settings, text=self.tr("server_port")).pack(side=tk.LEFT, padx=(0, 6))
        self.port = ttk.Entry(settings, width=6)
        self.port.pack(side=tk.LEFT)
        self.save = ttk.Button(settings, text=self.tr("server_save_port"), command=self._save)
        self.save.pack(side=tk.LEFT, padx=4)
        self.autostart = tk.BooleanVar(master=self)
        self.startup = ttk.Checkbutton(settings, text=self.tr("server_autostart"),
                                       variable=self.autostart, command=self._toggle_autostart)
        self.startup.pack(side=tk.LEFT, padx=4)

        self.details = ScrolledText(frame, width=58, height=7, wrap=tk.WORD, state=tk.DISABLED)
        self.details.pack(fill=tk.BOTH, expand=True)
        self._run("status")

    def _close(self) -> None:
        if not self._pending:
            self.destroy()

    def _copy(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.urls.get())

    def _stop(self) -> None:
        if messagebox.askokcancel(self.title(), self.tr("server_stop_confirm"), parent=self):
            self._run("stop")

    def _toggle_autostart(self) -> None:
        self._run("autostart-on" if self.autostart.get() else "autostart-off")

    def _save(self) -> None:
        try:
            port = int(self.port.get())
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            messagebox.showerror(self.title(), self.tr("server_invalid_port"), parent=self)
            return
        self._run("configure", "-Port", str(port))

    def _run(self, action: str, *args: str) -> None:
        self._pending = True
        self.status.configure(text=self.tr("server_working"))
        self._sync_controls()
        result = queue.Queue(maxsize=1)

        def worker():
            error = ""
            details = None
            state = None
            try:
                if action == "logs":
                    details = _recent_logs()
                elif action != "status":
                    _launcher(action, *args)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                error = str(exc)
            try:
                state = _status()
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                error = error or str(exc)
            result.put((state, details, error))

        def receive():
            try:
                state, details, error = result.get_nowait()
            except queue.Empty:
                self.after(75, receive)
                return
            self._pending = False
            self._state = state
            self._show_state()
            if error or details is not None or action != "status":
                text = error or (details or self.tr("server_no_logs") if action == "logs" else "")
                self.details.configure(state=tk.NORMAL)
                self.details.delete("1.0", tk.END)
                self.details.insert("1.0", text)
                self.details.configure(state=tk.DISABLED)
            self._sync_controls()

        threading.Thread(target=worker, daemon=True).start()
        self.after(75, receive)

    def _show_state(self) -> None:
        state = self._state
        if state is None:
            self.status.configure(text=self.tr("server_unknown"))
            self.urls.set("")
            return
        key = "server_ready" if state["ready"] else "server_unresponsive" if state["running"] else "server_stopped"
        text = self.tr(key, port=state["port"])
        if state["running"] and state["port"] != state["configured_port"]:
            text += " · " + self.tr("server_port_pending", port=state["configured_port"])
        self.status.configure(text=text)
        self.port.configure(state=tk.NORMAL)
        self.port.delete(0, tk.END)
        self.port.insert(0, state["configured_port"])
        self.autostart.set(state["autostart"])
        urls = state["urls"]
        selected = self.urls.get()
        self.urls.configure(values=urls)
        self.urls.set(selected if selected in urls else urls[0] if urls else "")

    def _sync_controls(self) -> None:
        state = self._state or {}
        enabled = not self._pending
        for widget, available in (
            (self.start, not state.get("running")), (self.stop, state.get("running")),
            (self.refresh, True), (self.logs, True), (self.port, bool(state)),
            (self.save, bool(state)), (self.startup, bool(state)), (self.copy, bool(state.get("urls"))),
        ):
            widget.configure(state=tk.NORMAL if enabled and available else tk.DISABLED)
