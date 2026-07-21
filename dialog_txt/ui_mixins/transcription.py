from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Literal

from tkinter import messagebox

from ..models import TranscriptionCancelled, TranscriptionOptions
from ..storage import discover_sessions, resolve_track_paths, transcript_path
from ..transcription_backends import (
    is_transcription_library_installed,
    normalize_transcription_library,
)


class TranscriptionMixin:
    def _transcribe_selected(self) -> None:
        sessions = self._selected_sessions()
        if not sessions:
            messagebox.showwarning(
                self._tr("title_recording_selection"),
                self._tr("msg_select_recording_from_list"),
            )
            return
        self._enqueue_transcriptions(sessions)

    def _queue_untranscribed(self) -> None:
        sessions = [
            session_dir
            for session_dir in discover_sessions()
            if self._session_audio_ready(session_dir) and not transcript_path(session_dir).exists()
        ]
        self._enqueue_transcriptions(sessions, announce_empty=True)

    @staticmethod
    def _queue_key(session_dir: Path) -> str:
        return str(session_dir)

    def _is_transcription_running(self) -> bool:
        return bool(self.transcription_thread and self.transcription_thread.is_alive())

    def _enqueue_transcriptions(
        self,
        sessions: list[Path],
        *,
        announce_empty: bool = False,
    ) -> None:
        unique_sessions: list[Path] = []
        seen_keys: set[str] = set()
        for session_dir in sessions:
            key = self._queue_key(session_dir)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_sessions.append(session_dir)

        added = 0
        skipped_missing = 0
        skipped_existing = 0
        for session_dir in unique_sessions:
            key = self._queue_key(session_dir)
            if self.current_transcription_session and key == self._queue_key(
                self.current_transcription_session
            ):
                skipped_existing += 1
                continue
            if key in self.transcription_queue_keys:
                skipped_existing += 1
                continue
            if not self._session_audio_ready(session_dir):
                skipped_missing += 1
                continue
            self.transcription_queue.append(session_dir)
            self.transcription_queue_keys.add(key)
            added += 1

        if added:
            self._set_status(
                self._tr(
                    "status_queue_added",
                    added=added,
                    pending=len(self.transcription_queue),
                )
            )
            self._log_event(
                self._tr(
                    "log_queue_added",
                    added=added,
                    pending=len(self.transcription_queue),
                )
            )
        elif announce_empty:
            self._set_status(self._tr("status_queue_empty"))

        if skipped_existing:
            self._log_event(self._tr("log_queue_skip_already", count=skipped_existing))
        if skipped_missing:
            self._log_event(self._tr("log_queue_skip_missing", count=skipped_missing))

        self._refresh_recordings()
        self._start_next_transcription_from_queue()

    def _clear_transcription_queue(self) -> int:
        cleared = len(self.transcription_queue)
        self.transcription_queue.clear()
        self.transcription_queue_keys.clear()
        if cleared:
            self._refresh_recordings()
        return cleared

    def _start_next_transcription_from_queue(self) -> None:
        if self.recorder is not None or self._is_transcription_running():
            return

        while self.transcription_queue:
            session_dir = self.transcription_queue[0]
            start_result = self._start_transcription(session_dir)
            if start_result == "started":
                self.transcription_queue.pop(0)
                self.transcription_queue_keys.discard(self._queue_key(session_dir))
                self._refresh_recordings()
                return
            if start_result == "skip":
                self.transcription_queue.pop(0)
                self.transcription_queue_keys.discard(self._queue_key(session_dir))
                self._refresh_recordings()
                continue
            return

    def _queue_position_for_session(self, session_dir: Path) -> int | None:
        key = self._queue_key(session_dir)
        for index, queued_session in enumerate(self.transcription_queue, start=1):
            if self._queue_key(queued_session) == key:
                return index
        return None

    def _queue_badge_for_session(self, session_dir: Path) -> str:
        key = self._queue_key(session_dir)
        if self.current_transcription_session and self._queue_key(self.current_transcription_session) == key:
            return "▶"
        position = self._queue_position_for_session(session_dir)
        return str(position) if position is not None else ""

    def _start_transcription(self, session_dir: Path) -> Literal["started", "skip", "pause"]:
        if self.recorder is not None:
            return "pause"
        if self._is_transcription_running():
            return "pause"

        mic_path, desktop_path = resolve_track_paths(session_dir)
        if mic_path is None or desktop_path is None:
            self._log_event(self._tr("log_queue_skip_session_missing_tracks", session=session_dir.name))
            return "skip"
        library = normalize_transcription_library(self.transcription_library_var.get())
        # Importing ctranslate2/faster-whisper can take tens of seconds. Only use
        # the interactive installer path when the package is actually absent;
        # full imports and model initialization happen in _transcribe_worker.
        if not is_transcription_library_installed(library):
            if not self._ensure_transcription_library_ready(interactive=True):
                return "pause"

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
                library=options.transcription_library,
                device=options.device,
                compute=options.compute_type,
            )
        )
        self.current_transcription_session = session_dir

        self.transcription_thread = threading.Thread(
            target=self._transcribe_worker,
            args=(session_dir, options, self.ui_language),
            daemon=True,
        )
        self.transcription_thread.start()
        return "started"

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
        if self._is_transcription_running():
            self.cancel_transcription_event.set()
            cleared = self._clear_transcription_queue()
            self._set_status(self._tr("status_transcription_cancelling"))
            self._log_event(self._tr("log_transcription_cancel_requested"))
            if cleared:
                self._log_event(self._tr("log_queue_cleared", count=cleared))
            return

        cleared = self._clear_transcription_queue()
        if cleared:
            self._set_status(self._tr("status_queue_cleared", count=cleared))
            self._log_event(self._tr("log_queue_cleared", count=cleared))

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

        if kind == "recording_error":
            # The capture thread has already stopped both tracks. Finalize now so
            # the user learns about the failure within one event-poll interval.
            if self.recorder is not None:
                self._stop_recording(auto_transcribe=False)
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

        self.current_transcription_session = None
        self._set_transcription_ui_state(is_running=False)
        self.progress.configure(value=0)
        self.progress_label.configure(text="0%")
        self._refresh_recordings()

        if kind == "done":
            _, session_dir, out_path = event
            pending = len(self.transcription_queue)
            if pending:
                self._set_status(
                    self._tr("status_done_queue_next", session=session_dir.name, pending=pending)
                )
            else:
                self._set_status(self._tr("status_done", session=session_dir.name))
            self._log_event(self._tr("log_transcription_done", path=out_path))
            if not pending:
                self._open_path_in_file_manager(session_dir)
            self._start_next_transcription_from_queue()
            return
        if kind == "cancelled":
            _, session_dir = event
            self._set_status(self._tr("status_transcription_cancelled", session=session_dir.name))
            self._log_event(self._tr("log_transcription_cancelled", session_dir=session_dir))
            messagebox.showwarning(
                self._tr("title_cancelled"),
                self._tr("msg_transcription_cancelled"),
            )
            self._start_next_transcription_from_queue()
            return
        if kind == "error":
            _, message = event
            pending = len(self.transcription_queue)
            if pending:
                self._set_status(self._tr("status_transcription_error_queue_next", pending=pending))
            else:
                self._set_status(self._tr("status_transcription_error"))
            self._log_event(self._tr("log_transcription_error", error=message))
            if not pending:
                messagebox.showerror(self._tr("title_transcription_error"), message)
            self._start_next_transcription_from_queue()
            return
