from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

from ..settings import save_app_settings
from ..storage import (
    discover_sessions,
    read_session_alias,
    read_session_metadata,
    resolve_track_paths,
    session_title,
    transcript_path,
    write_session_alias,
)
from ..utils import format_seconds


class RecordingsMixin:
    def _migrate_legacy_session_aliases(self) -> None:
        legacy_aliases = self.app_settings.get("session_aliases", {})
        if not isinstance(legacy_aliases, dict) or not legacy_aliases:
            return
        for session_dir in discover_sessions():
            alias = " ".join(str(legacy_aliases.get(session_dir.name, "")).split())
            if not alias:
                continue
            if read_session_alias(session_dir):
                continue
            write_session_alias(session_dir, alias)
        if "session_aliases" in self.app_settings:
            self.app_settings.pop("session_aliases", None)
            save_app_settings(self.app_settings)

    def _pending_short_name(self) -> str:
        return " ".join(self.pending_short_name_var.get().split())

    def _apply_pending_short_name(self, session_dir: Path) -> None:
        short_name = self._pending_short_name()
        if short_name:
            self._set_session_alias(session_dir, short_name)
        self.pending_short_name_var.set("")

    def _refresh_recordings(self) -> None:
        self._close_recording_alias_editor(commit=True)
        for row in self.recordings_tree.get_children():
            self.recordings_tree.delete(row)

        sessions = discover_sessions()
        for session_dir in sessions:
            audio_status, txt_status = self._session_status(session_dir)
            queue_status = self._queue_badge_for_session(session_dir)
            display_name = session_title(session_dir)
            short_name = self._session_alias(session_dir)
            duration_text = self._session_duration_text(session_dir)
            self.recordings_tree.insert(
                "",
                tk.END,
                iid=str(session_dir),
                values=(display_name, short_name, duration_text, audio_status, txt_status, queue_status),
            )
        if sessions and not self.recordings_tree.selection():
            self.recordings_tree.selection_set(str(sessions[0]))
        self._log_event(self._tr("log_recordings_refreshed", count=len(sessions)))
        self._on_recording_selected()

    def _on_recording_selected(self, _event=None) -> None:
        sessions = self._selected_sessions()
        if not sessions:
            self.transcribe_selected_button.configure(state=tk.DISABLED)
            self.open_folder_button.configure(state=tk.DISABLED)
            self.delete_selected_button.configure(state=tk.DISABLED)
            return

        can_queue_any = any(self._session_audio_ready(session) for session in sessions)
        self.transcribe_selected_button.configure(state=tk.NORMAL if can_queue_any else tk.DISABLED)
        self.open_folder_button.configure(state=tk.NORMAL if len(sessions) == 1 else tk.DISABLED)
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.delete_selected_button.configure(state=tk.DISABLED)
        elif self.transcription_queue:
            self.delete_selected_button.configure(state=tk.DISABLED)
        else:
            self.delete_selected_button.configure(state=tk.NORMAL)

    def _on_recording_double_click(self, event=None) -> str:
        if event is not None and self._begin_recording_alias_edit_from_event(event):
            return "break"
        self._open_selected_folder()
        return "break"

    def _on_recording_delete_key(self, _event=None) -> str:
        self._delete_selected_recordings()
        return "break"

    def _on_recording_rename_key(self, _event=None) -> str:
        session = self._selected_session()
        if session:
            self._begin_recording_alias_edit(session)
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
        self._close_recording_alias_editor(commit=True)
        if self.transcription_thread and self.transcription_thread.is_alive():
            messagebox.showwarning(
                self._tr("title_busy"),
                self._tr("msg_wait_transcription_complete"),
            )
            return
        if self.transcription_queue:
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

    def _session_alias(self, session_dir: Path) -> str:
        return read_session_alias(session_dir)

    def _recordings_tree_column_id(self, column_name: str) -> str | None:
        columns = tuple(self.recordings_tree["columns"])
        try:
            return f"#{columns.index(column_name) + 1}"
        except ValueError:
            return None

    def _begin_recording_alias_edit_from_event(self, event) -> bool:
        column_id = self.recordings_tree.identify_column(event.x)
        row_id = self.recordings_tree.identify_row(event.y)
        alias_column_id = self._recordings_tree_column_id("short_name")
        if not row_id or column_id != alias_column_id:
            return False
        self.recordings_tree.selection_set(row_id)
        self.recordings_tree.focus(row_id)
        return self._begin_recording_alias_edit(Path(row_id))

    def _begin_recording_alias_edit(self, session_dir: Path) -> bool:
        session_id = str(session_dir)
        if not self.recordings_tree.exists(session_id):
            return False

        self._close_recording_alias_editor(commit=True)
        self.recordings_tree.see(session_id)
        bbox = self.recordings_tree.bbox(session_id, "short_name")
        if not bbox:
            return False

        x, y, width, height = bbox
        if width <= 1 or height <= 1:
            return False

        original_value = self._session_alias(session_dir)
        editor = ttk.Entry(self.recordings_tree)
        editor.insert(0, original_value)
        editor.place(x=x, y=y, width=width, height=height)
        editor.focus_set()
        editor.select_range(0, tk.END)
        editor.bind("<Return>", self._submit_recording_alias_edit)
        editor.bind("<KP_Enter>", self._submit_recording_alias_edit)
        editor.bind("<Escape>", self._cancel_recording_alias_edit)
        editor.bind("<FocusOut>", self._commit_recording_alias_on_focus_out)

        self.recording_alias_editor = editor
        self.recording_alias_session = session_dir
        self.recording_alias_original_value = original_value
        return True

    def _submit_recording_alias_edit(self, _event=None) -> str:
        self._close_recording_alias_editor(commit=True)
        return "break"

    def _cancel_recording_alias_edit(self, _event=None) -> str:
        self._close_recording_alias_editor(commit=False)
        return "break"

    def _commit_recording_alias_on_focus_out(self, _event=None) -> None:
        self._close_recording_alias_editor(commit=True)

    def _close_recording_alias_editor(self, commit: bool) -> None:
        editor = self.recording_alias_editor
        session = self.recording_alias_session
        original_value = self.recording_alias_original_value
        if editor is None:
            return

        self.recording_alias_editor = None
        self.recording_alias_session = None
        self.recording_alias_original_value = ""

        new_value = original_value
        if commit and editor.winfo_exists():
            new_value = " ".join(editor.get().split())

        if editor.winfo_exists():
            editor.destroy()

        if commit and session is not None and new_value != original_value:
            self._set_session_alias(session, new_value)

    def _set_session_alias(self, session_dir: Path, alias: str) -> None:
        normalized_alias = " ".join(alias.split())
        write_session_alias(session_dir, normalized_alias)

        item_id = str(session_dir)
        if self.recordings_tree.exists(item_id):
            values = list(self.recordings_tree.item(item_id, "values"))
            if len(values) >= 2:
                values[1] = normalized_alias
                self.recordings_tree.item(item_id, values=values)


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
