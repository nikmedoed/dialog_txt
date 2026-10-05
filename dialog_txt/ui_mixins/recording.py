from __future__ import annotations

import threading
import time
import re
import webbrowser
from datetime import datetime

import tkinter as tk
from tkinter import messagebox

from ..models import RecordingError
from ..recording import DualTrackRecorder
from ..storage import discard_failed_session, create_session_dir, update_session_metadata, write_initial_metadata
from ..ui_layout import (
    set_record_button_style,
    set_recording_ui_state,
    set_transcription_ui_state,
    update_status_line,
)
from ..utils import format_seconds


class RecordingMixin:
    def _start_recording(self) -> None:
        if self.recorder is not None:
            return
        if self.transcription_thread and self.transcription_thread.is_alive():
            messagebox.showwarning(
                self._tr("title_busy"),
                self._tr("msg_wait_transcription_complete"),
            )
            return

        selected_mic = self._resolve_selected_microphone()
        if selected_mic is None:
            messagebox.showerror(self._tr("title_error"), self._tr("msg_select_microphone"))
            return
        try:
            mic = self._recording_microphone(selected_mic)
        except Exception as exc:
            messagebox.showerror(self._tr("title_recording_error"), str(exc))
            self._log_event(self._tr("log_recording_error", error=exc), error=True)
            return

        try:
            speaker, desktop_loopback = self._resolve_desktop_loopback()
        except Exception as exc:  # pragma: no cover - hardware-specific
            messagebox.showerror(
                self._tr("title_error"),
                self._tr("msg_desktop_device_failed", error=exc),
            )
            self._log_event(self._tr("log_desktop_capture_error", error=exc), error=True)
            return
        self._stop_idle_level_monitor()

        created_at, session_dir = create_session_dir()
        self_label, other_label = self._current_speaker_labels()
        write_initial_metadata(
            session_dir=session_dir,
            created_at=created_at,
            mic_name=selected_mic.name,
            desktop_source=self._sound_device_name(speaker),
        )
        self._save_app_settings()

        recorder = DualTrackRecorder(
            mic=mic,
            desktop=desktop_loopback,
            session_dir=session_dir,
            level_callback=lambda source, level: self.event_queue.put(("level", source, level)),
            error_callback=lambda error: self.event_queue.put(("recording_error", error, recorder)),
        )
        self.recorder = recorder
        self.active_session_dir = session_dir
        self.recording_started_at = time.time()
        self._set_levels_to_zero()
        try:
            self.recorder.start()
        except RecordingError as exc:
            try:
                discard_failed_session(session_dir)
            except OSError as cleanup_error:
                self._log_event(str(cleanup_error), error=True)
            self.recorder = None
            self.recording_started_at = None
            self.active_session_dir = None
            self._set_levels_to_zero()
            self._set_status(self._tr("status_recording_error"), error=True)
            self._refresh_recordings()
            self._log_event(self._tr("log_recording_error", error=exc), error=True)
            messagebox.showerror(self._tr("title_recording_error"), str(exc))
            self._start_idle_level_monitor(restart=True)
            return
        self._set_recording_ui_state(is_recording=True)
        self._set_status(
            self._tr("status_recording_active", session=session_dir.name, duration="00:00:00")
        )
        self._log_event(self._tr("log_recording_started", session_dir=session_dir))
        self._log_event(self._tr("log_mic_source", mic=selected_mic.name))
        self._log_event(self._tr("log_desktop_source", desktop=self._sound_device_name(speaker)))
        self._log_event(
            self._tr("log_speaker_labels", self_label=self_label, other_label=other_label)
        )
        self._tick_recording_timer()

    def _stop_recording(
        self, auto_transcribe: bool | None = None, restart_level_monitor: bool = True,
        stopped_at: float | None = None, device_change: bool = False,
    ) -> None:
        if self.recorder is None:
            return

        if auto_transcribe is None:
            auto_transcribe = bool(self.auto_transcribe_var.get())

        recorder = self.recorder
        self.recorder = None
        try:
            recorder.stop()
        except Exception as exc:
            self._set_recording_ui_state(is_recording=False)
            self.recording_started_at = None
            self.active_session_dir = None
            self._set_levels_to_zero()
            self._set_status(self._tr("status_recording_error"), error=True)
            self._refresh_recordings()
            self._log_event(self._tr("log_recording_error", error=exc), error=True)
            messagebox.showerror(self._tr("title_recording_error"), str(exc))
            if restart_level_monitor:
                self._start_idle_level_monitor(restart=True)
            return

        try:
            recorder.raise_if_failed()
        except RecordingError as exc:
            self._log_event(self._tr("log_recording_error", error=exc), error=True)
            if not device_change:
                self._set_recording_ui_state(is_recording=False)
                self.recording_started_at = None
                self.active_session_dir = None
                self._set_levels_to_zero()
                self._set_status(self._tr("status_recording_error"), error=True)
                self._refresh_recordings()
                messagebox.showerror(self._tr("title_recording_error"), str(exc))
                if restart_level_monitor:
                    self._start_idle_level_monitor(restart=True)
                return

        duration = 0
        if self.recording_started_at is not None:
            duration = int((stopped_at if stopped_at is not None else time.time()) - self.recording_started_at)

        self.recording_started_at = None
        self._set_recording_ui_state(is_recording=False)
        self._set_levels_to_zero()

        session_dir = self.active_session_dir
        if session_dir:
            update_session_metadata(session_dir, duration, ended_at=stopped_at)
            self._apply_pending_short_name(session_dir)

        duration_text = format_seconds(duration)
        if auto_transcribe:
            self._set_status(self._tr("status_recording_stopped_transcribe", duration=duration_text))
            self._log_event(self._tr("log_recording_stopped_transcribe", duration=duration_text))
        else:
            self._set_status(self._tr("status_recording_stopped_no_auto", duration=duration_text))
            self._log_event(self._tr("log_recording_stopped_no_auto", duration=duration_text))
        self._refresh_recordings()
        if restart_level_monitor:
            self._start_idle_level_monitor(restart=True)

        self.active_session_dir = None
        if session_dir and auto_transcribe:
            self._enqueue_transcriptions([session_dir])
        elif stopped_at is None:
            self._start_next_transcription_from_queue()

    def _tick_recording_timer(self) -> None:
        if self.recorder is None or self.recording_started_at is None:
            return
        elapsed = time.time() - self.recording_started_at
        session_name = self.active_session_dir.name if self.active_session_dir else self._tr("session_fallback")
        self._set_status(
            self._tr("status_recording_active", session=session_name, duration=format_seconds(elapsed)),
            log=False,
        )
        self.after(250, self._tick_recording_timer)

    def _tick_level_meter(self) -> None:
        self._decay_levels()
        self._render_levels()
        self.after(250, self._tick_level_meter)

    def _toggle_recording(self) -> None:
        if self.recorder is None:
            self._start_recording()
        else:
            self._stop_recording()

    def _set_record_button_style(self, is_recording: bool) -> None:
        set_record_button_style(self, is_recording=is_recording)

    def _set_recording_ui_state(self, is_recording: bool) -> None:
        set_recording_ui_state(self, is_recording=is_recording)

    def _set_transcription_ui_state(self, is_running: bool) -> None:
        set_transcription_ui_state(self, is_running=is_running)

    def _set_status(self, text: str, *, error: bool = False, log: bool = True) -> None:
        self.status_text = text
        self.status_is_error = error
        self._update_status_line()
        if log:
            self._log_event(text, error=error)

    def _update_status_line(self) -> None:
        update_status_line(self)

    def _decay_levels(self) -> None:
        now = time.monotonic()
        for source in ("microphone", "desktop"):
            age = now - self.level_last_seen[source]
            if age > 0.8:
                self.level_values[source] *= 0.45
            else:
                self.level_values[source] *= 0.92

    def _set_levels_to_zero(self) -> None:
        self.level_values["microphone"] = 0.0
        self.level_values["desktop"] = 0.0
        self.level_last_seen["microphone"] = 0.0
        self.level_last_seen["desktop"] = 0.0
        self._render_levels()

    def _render_levels(self) -> None:
        mic_pct = int(max(0, min(100, self.level_values["microphone"] * 100)))
        desktop_pct = int(max(0, min(100, self.level_values["desktop"] * 100)))
        self.mic_level.configure(value=mic_pct)
        self.desktop_level.configure(value=desktop_pct)

    def _log_event(self, text: str, *, error: bool = False) -> None:
        # Some event handlers also report the same message as their status.
        if getattr(self, "_last_log_event", None) == (text, error):
            return
        self._last_log_event = (text, error)
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {text}\n"
        self.log_text.configure(state=tk.NORMAL)
        start = self.log_text.index("end-1c")
        self.log_text.tag_configure("error", foreground="#b42318")
        self.log_text.insert(tk.END, line, ("error",) if error else ())
        self.log_text.tag_configure("web_link", foreground="#1565c0", underline=True)
        self.log_text.tag_bind("web_link", "<Enter>", lambda _event: self.log_text.configure(cursor="hand2"))
        self.log_text.tag_bind("web_link", "<Leave>", lambda _event: self.log_text.configure(cursor="xterm"))
        self.log_text.tag_bind("web_link", "<Button-1>", self._open_log_link)
        for match in re.finditer(r"https?://[^\s<>]+", line):
            url = match.group().rstrip(".,;!?)\"'")
            self.log_text.tag_add("web_link", f"{start}+{match.start()}c",
                                  f"{start}+{match.start() + len(url)}c")
        self.log_text.see(tk.END)
        # Keep log widget responsive on long sessions.
        total_lines = int(self.log_text.index("end-1c").split(".")[0])
        if total_lines > 800:
            self.log_text.delete("1.0", "200.0")
        self.log_text.configure(state=tk.DISABLED)

    def _open_log_link(self, event) -> str:
        index = self.log_text.index(f"@{event.x},{event.y}")
        bounds = self.log_text.tag_prevrange("web_link", f"{index}+1c")
        if bounds and self.log_text.compare(bounds[0], "<=", index) and self.log_text.compare(index, "<", bounds[1]):
            webbrowser.open(self.log_text.get(*bounds))
        return "break"

    def _copy_log_selection(self, _event=None) -> str:
        try:
            text = self.log_text.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return "break"
        self.clipboard_clear()
        self.clipboard_append(text)
        return "break"

    def _copy_log_all(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.log_text.get("1.0", "end-1c"))

    def _select_log_all(self, _event=None) -> str:
        self.log_text.tag_add(tk.SEL, "1.0", "end-1c")
        self.log_text.focus_set()
        return "break"

    def _show_log_menu(self, event) -> str:
        self.log_text.focus_set()
        try:
            self.log_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.log_context_menu.grab_release()
        return "break"

    def _on_close(self) -> None:
        if self.trim_window is not None and not self.trim_window.close(restart_monitor=False):
            return
        if self.system_events is not None:
            self.system_events.close()
        self._save_app_settings()
        self._stop_idle_level_monitor()
        if self.recorder is not None:
            self._stop_recording(auto_transcribe=False, restart_level_monitor=False)
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.cancel_transcription_event.set()
            self._log_event(self._tr("log_window_close_cancel"))
        self._release_win32_icon_handles()
        self.destroy()
