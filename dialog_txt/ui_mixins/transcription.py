from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from tkinter import messagebox

from ..models import TranscriptionCancelled, TranscriptionOptions
from ..storage import resolve_track_paths


class TranscriptionMixin:
    def _transcribe_selected(self) -> None:
        session = self._selected_session()
        if not session:
            messagebox.showwarning(
                self._tr("title_recording_selection"),
                self._tr("msg_select_recording_from_list"),
            )
            return
        self._start_transcription(session)

    def _start_transcription(self, session_dir: Path) -> None:
        if self.recorder is not None:
            messagebox.showwarning(self._tr("title_busy"), self._tr("msg_stop_recording_first"))
            return
        if self.transcription_thread and self.transcription_thread.is_alive():
            messagebox.showwarning(self._tr("title_busy"), self._tr("msg_transcription_running"))
            return

        mic_path, desktop_path = resolve_track_paths(session_dir)
        if mic_path is None or desktop_path is None:
            messagebox.showerror(
                self._tr("title_error"),
                self._tr("msg_session_missing_tracks"),
            )
            return

        self.cancel_transcription_event.clear()
        self.progress.configure(value=0)
        self.progress_label.configure(text="0%")
        self.last_progress_log_bucket = -1
        options = self._current_transcription_options()
        self._set_transcription_ui_state(is_running=True)
        self._set_status(self._tr("status_transcription_active", session=session_dir.name))
        self._log_event(
            self._tr(
                "log_transcription_started",
                session_dir=session_dir,
                model=options.model_name,
                lang=options.language,
                beam=options.beam_size,
                vad=self._tr("flag_on") if options.vad_filter else self._tr("flag_off"),
                compute=options.compute_type,
            )
        )

        self.transcription_thread = threading.Thread(
            target=self._transcribe_worker,
            args=(session_dir, options, self.ui_language),
            daemon=True,
        )
        self.transcription_thread.start()

    def _transcribe_worker(
        self,
        session_dir: Path,
        options: TranscriptionOptions,
        ui_language: str,
    ) -> None:
        try:
            out_path = self.transcriber.transcribe_session(
                session_dir=session_dir,
                progress_cb=lambda text, pct: self.event_queue.put(("progress", text, pct)),
                cancel_event=self.cancel_transcription_event,
                options=options,
                ui_language=ui_language,
            )
            self.event_queue.put(("done", session_dir, out_path))
        except TranscriptionCancelled:
            self.event_queue.put(("cancelled", session_dir))
        except Exception as exc:
            self.event_queue.put(("error", str(exc)))

    def _cancel_transcription(self) -> None:
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.cancel_transcription_event.set()
            self._set_status(self._tr("status_transcription_cancelling"))
            self._log_event(self._tr("log_transcription_cancel_requested"))

    def _poll_events(self) -> None:
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)
        self.after(150, self._poll_events)

    def _handle_event(self, event) -> None:
        kind = event[0]
        if kind == "level":
            _, source, level = event
            if source in self.level_values:
                self.level_values[source] = max(0.0, min(1.0, float(level)))
                self.level_last_seen[source] = time.monotonic()
                self._render_levels()
            return

        if kind == "level_monitor_error":
            _, source, message = event
            if source in self.level_values:
                self.level_values[source] = 0.0
                self.level_last_seen[source] = 0.0
                self._render_levels()
            self._log_event(self._tr("log_level_monitor_error", source=source, message=message))
            return

        if kind == "progress":
            _, message, pct = event
            pct_int = int(max(0, min(100, pct)))
            self.progress.configure(value=pct_int)
            self.progress_label.configure(text=f"{pct_int}%")
            self._set_status(message)
            progress_bucket = pct_int // 10
            if progress_bucket > self.last_progress_log_bucket:
                self.last_progress_log_bucket = progress_bucket
                self._log_event(
                    self._tr("log_transcription_progress", percent=pct_int, message=message)
                )
            return

        self._set_transcription_ui_state(is_running=False)
        self.progress.configure(value=0)
        self.progress_label.configure(text="0%")
        self._refresh_recordings()

        if kind == "done":
            _, session_dir, out_path = event
            self._set_status(self._tr("status_done", session=session_dir.name))
            self._log_event(self._tr("log_transcription_done", path=out_path))
            self._open_path_in_file_manager(session_dir)
            return
        if kind == "cancelled":
            _, session_dir = event
            self._set_status(self._tr("status_transcription_cancelled", session=session_dir.name))
            self._log_event(self._tr("log_transcription_cancelled", session_dir=session_dir))
            messagebox.showwarning(
                self._tr("title_cancelled"),
                self._tr("msg_transcription_cancelled"),
            )
            return
        if kind == "error":
            _, message = event
            self._set_status(self._tr("status_transcription_error"))
            self._log_event(self._tr("log_transcription_error", error=message))
            messagebox.showerror(self._tr("title_transcription_error"), message)
            return
