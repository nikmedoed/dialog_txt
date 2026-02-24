from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import tkinter as tk
from tkinter import messagebox

from ..storage import (
    discover_sessions,
    read_session_metadata,
    resolve_track_paths,
    session_title,
    transcript_path,
)
from ..utils import format_seconds


class RecordingsMixin:
    def _refresh_recordings(self) -> None:
        for row in self.recordings_tree.get_children():
            self.recordings_tree.delete(row)

        sessions = discover_sessions()
        for session_dir in sessions:
            audio_status, txt_status = self._session_status(session_dir)
            display_name = session_title(session_dir)
            folder_name = session_dir.name
            duration_text = self._session_duration_text(session_dir)
            self.recordings_tree.insert(
                "",
                tk.END,
                iid=str(session_dir),
                values=(display_name, folder_name, duration_text, audio_status, txt_status),
            )
        if sessions and not self.recordings_tree.selection():
            self.recordings_tree.selection_set(str(sessions[0]))
        self._log_event(self._tr("log_recordings_refreshed", count=len(sessions)))
        self._on_recording_selected()

    def _on_recording_selected(self, _event=None) -> None:
        session = self._selected_session()
        if not session:
            if not (self.transcription_thread and self.transcription_thread.is_alive()):
                self.transcribe_selected_button.configure(state=tk.DISABLED)
            self.open_folder_button.configure(state=tk.DISABLED)
            self.delete_selected_button.configure(state=tk.DISABLED)
            return

        can_transcribe = self._session_audio_ready(session)
        has_transcript = transcript_path(session).exists()
        if can_transcribe and has_transcript:
            self.transcribe_selected_button.configure(text=self._tr("btn_retranscribe"))
        elif can_transcribe:
            self.transcribe_selected_button.configure(text=self._tr("btn_transcribe_selected"))
        else:
            self.transcribe_selected_button.configure(
                text=self._tr("btn_cannot_transcribe_missing_tracks")
            )

        self.open_folder_button.configure(state=tk.NORMAL)
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.delete_selected_button.configure(state=tk.DISABLED)
        else:
            self.delete_selected_button.configure(state=tk.NORMAL)

        if not (self.transcription_thread and self.transcription_thread.is_alive()):
            self.transcribe_selected_button.configure(state=tk.NORMAL if can_transcribe else tk.DISABLED)

    def _on_recording_double_click(self, _event=None) -> None:
        self._open_selected_folder()

    def _on_recording_delete_key(self, _event=None) -> str:
        self._delete_selected_recordings()
        return "break"

    def _selected_session(self) -> Path | None:
        selected = self.recordings_tree.selection()
        if not selected:
            return None
        return Path(selected[0])

    def _selected_sessions(self) -> list[Path]:
        return [Path(value) for value in self.recordings_tree.selection()]

    def _open_selected_folder(self) -> None:
        session = self._selected_session()
        if not session:
            messagebox.showwarning(
                self._tr("title_open_folder"),
                self._tr("msg_select_recording_from_list"),
            )
            return
        self._open_path_in_file_manager(session)

    def _delete_selected_recordings(self) -> None:
        if self.transcription_thread and self.transcription_thread.is_alive():
            messagebox.showwarning(
                self._tr("title_busy"),
                self._tr("msg_wait_transcription_complete"),
            )
            return

        sessions = self._selected_sessions()
        if not sessions:
            messagebox.showwarning(
                self._tr("title_recording_selection"),
                self._tr("msg_select_recording_from_list"),
            )
            return

        if len(sessions) == 1:
            session = sessions[0]
            prompt = self._tr(
                "msg_delete_recording_confirm",
                session=session_title(session),
                folder=session.name,
            )
        else:
            prompt = self._tr("msg_delete_recordings_confirm", count=len(sessions))

        approved = messagebox.askyesno(
            self._tr("title_delete_recording"),
            prompt,
            icon=messagebox.WARNING,
        )
        if not approved:
            return

        deleted_count = 0
        errors: list[str] = []
        for session in sessions:
            try:
                shutil.rmtree(session)
                deleted_count += 1
                self._log_event(self._tr("log_recording_deleted", path=session))
            except Exception as exc:
                errors.append(self._tr("log_recording_delete_error", path=session, error=exc))

        if deleted_count:
            self._set_status(self._tr("status_recordings_deleted", count=deleted_count))
        for error_line in errors:
            self._log_event(error_line)

        self._refresh_recordings()

        if errors:
            messagebox.showerror(
                self._tr("title_delete_recording"),
                self._tr("msg_recording_delete_failed", errors="\n".join(errors)),
            )

    def _open_path_in_file_manager(self, path: Path) -> None:
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
            self._log_event(self._tr("log_folder_opened", path=path))
        except Exception as exc:
            self._log_event(self._tr("log_open_folder_error", path=path, error=exc))
            messagebox.showerror(
                self._tr("title_open_folder"),
                self._tr("msg_open_folder_failed", path=path, error=exc),
            )

    @staticmethod
    def _session_audio_ready(session_dir: Path) -> bool:
        mic_path, desktop_path = resolve_track_paths(session_dir)
        return mic_path is not None and desktop_path is not None

    @staticmethod
    def _session_duration_text(session_dir: Path) -> str:
        payload = read_session_metadata(session_dir)
        try:
            duration_seconds = int(payload.get("duration_seconds", ""))
        except (TypeError, ValueError):
            return "--:--:--"
        if duration_seconds < 0:
            return "--:--:--"
        return format_seconds(duration_seconds)

    def _session_status(self, session_dir: Path) -> tuple[str, str]:
        mic_path, desktop_path = resolve_track_paths(session_dir)
        has_transcript = transcript_path(session_dir).exists()

        if mic_path is not None and desktop_path is not None:
            audio_status = "OK"
        elif mic_path is None and desktop_path is None:
            audio_status = self._tr("audio_missing_both")
        elif mic_path is None:
            audio_status = self._tr("audio_missing_mic")
        else:
            audio_status = self._tr("audio_missing_desktop")

        txt_status = "✅" if has_transcript else "⬜"
        return audio_status, txt_status
