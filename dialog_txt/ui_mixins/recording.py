from __future__ import annotations

import threading
import time
from datetime import datetime

import tkinter as tk
from tkinter import messagebox

from ..models import RecordingError
from ..recording import DualTrackRecorder
from ..storage import create_session_dir, update_session_metadata, write_initial_metadata
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

        mic = self._resolve_selected_microphone()
        if mic is None:
            messagebox.showerror(self._tr("title_error"), self._tr("msg_select_microphone"))
            return

        try:
            speaker, desktop_loopback = self._resolve_desktop_loopback()
        except Exception as exc:  # pragma: no cover - hardware-specific
            messagebox.showerror(
                self._tr("title_error"),
                self._tr("msg_desktop_device_failed", error=exc),
            )
            self._log_event(self._tr("log_desktop_capture_error", error=exc))
            return
        self._stop_idle_level_monitor()

        created_at, session_dir = create_session_dir()
        self_label, other_label = self._current_speaker_labels()
        write_initial_metadata(
            session_dir=session_dir,
            created_at=created_at,
            mic_name=mic.name,
            desktop_source=self._sound_device_name(speaker),
        )
        self._save_app_settings()

        self.recorder = DualTrackRecorder(
            mic=mic,
            desktop=desktop_loopback,
            session_dir=session_dir,
            level_callback=lambda source, level: self.event_queue.put(("level", source, level)),
        )
        self.active_session_dir = session_dir
        self.recording_started_at = time.time()
        self._set_levels_to_zero()
        self.recorder.start()
        self._set_recording_ui_state(is_recording=True)
        self._set_status(
            self._tr("status_recording_active", session=session_dir.name, duration="00:00:00")
        )
        self._log_event(self._tr("log_recording_started", session_dir=session_dir))
        self._log_event(self._tr("log_mic_source", mic=mic.name))
        self._log_event(self._tr("log_desktop_source", desktop=self._sound_device_name(speaker)))
        self._log_event(
            self._tr("log_speaker_labels", self_label=self_label, other_label=other_label)
        )
        self._tick_recording_timer()

    def _stop_recording(
        self, auto_transcribe: bool | None = None, restart_level_monitor: bool = True
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
            self._set_status(self._tr("status_recording_error"))
            self._refresh_recordings()
            self._log_event(self._tr("log_recording_error", error=exc))
            messagebox.showerror(self._tr("title_recording_error"), str(exc))
            if restart_level_monitor:
                self._start_idle_level_monitor(restart=True)
            return

        try:
            recorder.raise_if_failed()
        except RecordingError as exc:
            self._set_recording_ui_state(is_recording=False)
            self.recording_started_at = None
            self.active_session_dir = None
            self._set_levels_to_zero()
            self._set_status(self._tr("status_recording_error"))
            self._refresh_recordings()
            self._log_event(self._tr("log_recording_error", error=exc))
            messagebox.showerror(self._tr("title_recording_error"), str(exc))
            if restart_level_monitor:
                self._start_idle_level_monitor(restart=True)
            return

        duration = 0
        if self.recording_started_at is not None:
            duration = int(time.time() - self.recording_started_at)

        self.recording_started_at = None
        self._set_recording_ui_state(is_recording=False)
        self._set_levels_to_zero()

        session_dir = self.active_session_dir
        if session_dir:
            update_session_metadata(session_dir, duration)
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
        else:
            self._start_next_transcription_from_queue()

    def _tick_recording_timer(self) -> None:
        if self.recorder is None or self.recording_started_at is None:
            return
        elapsed = time.time() - self.recording_started_at
        session_name = self.active_session_dir.name if self.active_session_dir else self._tr("session_fallback")
        self._set_status(
            self._tr("status_recording_active", session=session_name, duration=format_seconds(elapsed))
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

    def _set_status(self, text: str) -> None:
        self.status_text = text
        self._update_status_line()

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

    def _log_event(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {text}\n"
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, line)
        self.log_text.see(tk.END)
        # Keep log widget responsive on long sessions.
        total_lines = int(self.log_text.index("end-1c").split(".")[0])
        if total_lines > 800:
            self.log_text.delete("1.0", "200.0")
        self.log_text.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        self._save_app_settings()
        self._stop_idle_level_monitor()
        if self.recorder is not None:
            self._stop_recording(auto_transcribe=False, restart_level_monitor=False)
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.cancel_transcription_event.set()
            self._log_event(self._tr("log_window_close_cancel"))
        self._release_win32_icon_handles()
        self.destroy()
