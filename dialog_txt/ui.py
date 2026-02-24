from __future__ import annotations

import ctypes
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import messagebox

from .config import DEFAULT_OTHER_LABEL, DEFAULT_SELF_LABEL
from .localization import (
    UI_LANGUAGE_CODES,
    resolve_ui_language,
    system_microphone_label_prefixes,
    tr,
)
from .models import RecordingError, TranscriptionCancelled, TranscriptionOptions
from .numpy_compat import apply_numpy_fromstring_compat_patch
from .recording import DualTrackLevelMonitor, DualTrackRecorder
from .settings import load_app_settings, save_app_settings
from .settings import (
    ALLOWED_COMPUTE_TYPES,
    ALLOWED_MODELS,
    DEFAULT_BEAM_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
)
from .storage import (
    create_session_dir,
    discover_sessions,
    read_session_metadata,
    ensure_recordings_root,
    resolve_track_paths,
    session_title,
    transcript_path,
    update_session_metadata,
    write_initial_metadata,
)
from .transcription import WhisperTranscriber
from .ui_layout import (
    apply_localization,
    build_ui,
    set_record_button_style,
    set_recording_ui_state,
    set_transcription_ui_state,
    update_status_line,
)
from .utils import format_seconds

apply_numpy_fromstring_compat_patch()

import soundcard as sc

DESKTOP_SOURCE_NAME_HINTS = (
    "loopback",
    "monitor",
    "stereo mix",
    "what u hear",
    "blackhole",
    "soundflower",
    "cable output",
    "vb-cable",
    "loopback audio",
)


def _set_windows_app_user_model_id(app_id: str = "DialogTxt.App") -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        shell32 = ctypes.windll.shell32
        shell32.SetCurrentProcessExplicitAppUserModelID.argtypes = [ctypes.c_wchar_p]
        shell32.SetCurrentProcessExplicitAppUserModelID.restype = ctypes.c_long
        shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


class App(tk.Tk):
    SYSTEM_MICROPHONE_SETTING = "__system_default__"

    def __init__(self):
        _set_windows_app_user_model_id()
        super().__init__()
        self._icon_image = None
        self._win32_icon_handles: list[int] = []
        self.title("Dialog to TXT")
        self.geometry("620x760")
        self.minsize(560, 560)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._apply_window_icon()

        ensure_recordings_root()

        self.transcriber = WhisperTranscriber()
        self.recorder: DualTrackRecorder | None = None
        self.level_monitor: DualTrackLevelMonitor | None = None
        self.recording_started_at: float | None = None
        self.active_session_dir: Path | None = None

        self.event_queue: queue.Queue = queue.Queue()
        self.cancel_transcription_event = threading.Event()
        self.transcription_thread: threading.Thread | None = None

        self.app_settings = load_app_settings()
        self.ui_language = self._resolve_ui_language(self.app_settings.get("ui_language"))
        self.ui_language_code_var = tk.StringVar(
            value=next(
                (code for code, language in UI_LANGUAGE_CODES.items() if language == self.ui_language),
                "RU",
            )
        )
        self.level_values = {"microphone": 0.0, "desktop": 0.0}
        self.level_last_seen = {"microphone": 0.0, "desktop": 0.0}
        self.status_text = self._tr("status_idle")
        self.last_progress_log_bucket = -1
        self.settings_save_after_id: str | None = None

        self.microphones = []
        self.system_microphone_option = self._system_microphone_option_label()
        self.auto_transcribe_var = tk.BooleanVar(
            value=self.app_settings["auto_transcribe_after_record"]
        )
        self.self_label_var = tk.StringVar(value=self.app_settings["speaker_self"])
        self.other_label_var = tk.StringVar(value=self.app_settings["speaker_other"])
        self.model_var = tk.StringVar(value=self.app_settings["whisper_model"])
        self.language_var = tk.StringVar(value=self.app_settings["whisper_language"])
        self.beam_size_var = tk.IntVar(value=self.app_settings["whisper_beam_size"])
        self.vad_filter_var = tk.BooleanVar(value=self.app_settings["whisper_vad_filter"])
        self.compute_type_var = tk.StringVar(value=self.app_settings["whisper_compute_type"])
        self.include_timestamps_var = tk.BooleanVar(value=self.app_settings["include_timestamps"])
        self._build_ui()
        self.auto_transcribe_var.trace_add("write", self._schedule_settings_save)
        self.self_label_var.trace_add("write", self._schedule_settings_save)
        self.other_label_var.trace_add("write", self._schedule_settings_save)
        self.model_var.trace_add("write", self._schedule_settings_save)
        self.language_var.trace_add("write", self._schedule_settings_save)
        self.beam_size_var.trace_add("write", self._schedule_settings_save)
        self.vad_filter_var.trace_add("write", self._schedule_settings_save)
        self.compute_type_var.trace_add("write", self._schedule_settings_save)
        self.include_timestamps_var.trace_add("write", self._schedule_settings_save)
        self._refresh_microphones()
        self._refresh_recordings()
        self._set_levels_to_zero()
        self._start_idle_level_monitor()
        self.after(250, self._tick_level_meter)
        self.after(150, self._poll_events)

    def _apply_window_icon(self) -> None:
        icon_ico_path, icon_png_path = self._resolve_icon_paths()

        if sys.platform.startswith("win") and icon_ico_path is not None:
            try:
                self.iconbitmap(default=str(icon_ico_path))
            except tk.TclError:
                pass
            # Apply explicit small/big icons to avoid Windows sticking to 16px resource.
            self.after(0, lambda p=icon_ico_path: self._apply_win32_icon_handles(p))
            # Some Tk builds create/re-parent the native window after idle; re-apply once.
            self.after(250, lambda p=icon_ico_path: self._apply_win32_icon_handles(p))
            return

        # Keep PhotoImage reference to avoid garbage collection.
        self._icon_image = None
        photo_candidates = [candidate for candidate in (icon_png_path, icon_ico_path) if candidate]
        for candidate in photo_candidates:
            try:
                self._icon_image = tk.PhotoImage(file=str(candidate))
                self.iconphoto(True, self._icon_image)
                break
            except tk.TclError:
                self._icon_image = None

    @staticmethod
    def _first_existing(candidates: list[Path]) -> Path | None:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _apply_win32_icon_handles(self, icon_ico_path: Path) -> None:
        if not sys.platform.startswith("win"):
            return
        try:
            user32 = ctypes.windll.user32
            user32.LoadImageW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_wchar_p,
                ctypes.c_uint,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint,
            ]
            user32.LoadImageW.restype = ctypes.c_void_p
            user32.SendMessageW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            user32.SendMessageW.restype = ctypes.c_void_p
            user32.GetSystemMetrics.argtypes = [ctypes.c_int]
            user32.GetSystemMetrics.restype = ctypes.c_int
            user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            user32.GetAncestor.restype = ctypes.c_void_p

            set_class_icon = getattr(user32, "SetClassLongPtrW", None)
            if set_class_icon is None:
                set_class_icon = getattr(user32, "SetClassLongW", None)
            if set_class_icon is not None:
                set_class_icon.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
                set_class_icon.restype = ctypes.c_void_p

            hwnd = ctypes.c_void_p(self.winfo_id())
            if not hwnd.value:
                return
            GA_ROOT = 2
            root_hwnd_value = user32.GetAncestor(hwnd, GA_ROOT)
            root_hwnd = ctypes.c_void_p(root_hwnd_value) if root_hwnd_value else hwnd

            IMAGE_ICON = 1
            LR_LOADFROMFILE = 0x00000010
            WM_SETICON = 0x0080
            ICON_SMALL = 0
            ICON_BIG = 1
            GCLP_HICON = -14
            GCLP_HICONSM = -34
            SM_CXICON = 11
            SM_CYICON = 12
            SM_CXSMICON = 49
            SM_CYSMICON = 50

            big_w = max(32, int(user32.GetSystemMetrics(SM_CXICON) or 32))
            big_h = max(32, int(user32.GetSystemMetrics(SM_CYICON) or 32))
            small_w = max(16, int(user32.GetSystemMetrics(SM_CXSMICON) or 16))
            small_h = max(16, int(user32.GetSystemMetrics(SM_CYSMICON) or 16))

            hicon_big = user32.LoadImageW(
                None,
                str(icon_ico_path),
                IMAGE_ICON,
                big_w,
                big_h,
                LR_LOADFROMFILE,
            )
            hicon_small = user32.LoadImageW(
                None,
                str(icon_ico_path),
                IMAGE_ICON,
                small_w,
                small_h,
                LR_LOADFROMFILE,
            )

            if hicon_big:
                user32.SendMessageW(root_hwnd, WM_SETICON, ctypes.c_void_p(ICON_BIG), hicon_big)
                if hwnd.value != root_hwnd.value:
                    user32.SendMessageW(hwnd, WM_SETICON, ctypes.c_void_p(ICON_BIG), hicon_big)
                if set_class_icon is not None:
                    set_class_icon(root_hwnd, GCLP_HICON, hicon_big)
                self._win32_icon_handles.append(int(hicon_big))
            if hicon_small:
                user32.SendMessageW(root_hwnd, WM_SETICON, ctypes.c_void_p(ICON_SMALL), hicon_small)
                if hwnd.value != root_hwnd.value:
                    user32.SendMessageW(hwnd, WM_SETICON, ctypes.c_void_p(ICON_SMALL), hicon_small)
                if set_class_icon is not None:
                    set_class_icon(root_hwnd, GCLP_HICONSM, hicon_small)
                self._win32_icon_handles.append(int(hicon_small))
        except Exception:
            return

    def _release_win32_icon_handles(self) -> None:
        if not self._win32_icon_handles or not sys.platform.startswith("win"):
            return
        try:
            user32 = ctypes.windll.user32
            user32.DestroyIcon.argtypes = [ctypes.c_void_p]
            user32.DestroyIcon.restype = ctypes.c_bool
            for handle in self._win32_icon_handles:
                if handle:
                    user32.DestroyIcon(ctypes.c_void_p(handle))
        except Exception:
            pass
        self._win32_icon_handles.clear()

    def _resolve_icon_paths(self) -> tuple[Path | None, Path | None]:
        ico_candidates: list[Path] = []
        png_candidates: list[Path] = []
        if getattr(sys, "frozen", False):
            base_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
            ico_candidates.extend(
                [
                    base_dir / "icon.ico",
                    base_dir / "docs" / "icon.ico",
                ]
            )
            png_candidates.extend(
                [
                    base_dir / "icon.png",
                    base_dir / "docs" / "icon.png",
                ]
            )

        project_root = Path(__file__).resolve().parent.parent
        ico_candidates.extend(
            [
                project_root / "docs" / "icon.ico",
                project_root / "icon.ico",
            ]
        )
        png_candidates.extend(
            [
                project_root / "docs" / "icon.png",
                project_root / "icon.png",
            ]
        )
        return self._first_existing(ico_candidates), self._first_existing(png_candidates)

    def _build_ui(self) -> None:
        build_ui(self)

    def _resolve_ui_language(self, value: str | None) -> str:
        return resolve_ui_language(value)

    def _tr(self, key: str, **kwargs) -> str:
        return tr(self.ui_language, key, **kwargs)

    def _on_ui_language_selected(self, _event=None) -> None:
        selected_code = self.ui_language_code_var.get().strip().upper()
        target_language = UI_LANGUAGE_CODES.get(selected_code, resolve_ui_language(None))
        if target_language == self.ui_language:
            return
        self.ui_language = target_language
        self._apply_localization(refresh_data=True)
        self._save_app_settings()

    def _apply_localization(self, refresh_data: bool = False) -> None:
        apply_localization(self, refresh_data=refresh_data)

    def _refresh_microphones(self) -> None:
        try:
            mics = sc.all_microphones(include_loopback=False)
        except Exception as exc:  # pragma: no cover - hardware-specific
            messagebox.showerror(
                self._tr("title_error"),
                self._tr("msg_microphones_list_failed", error=exc),
            )
            self._log_event(self._tr("log_microphones_list_failed", error=exc))
            return

        previous_selection = self._selected_microphone_name()
        self.microphones = list(mics)
        self.system_microphone_option = self._system_microphone_option_label()
        values = [self.system_microphone_option] + [mic.name for mic in self.microphones]
        self.mic_combo["values"] = values
        if values:
            preferred = self._preferred_microphone_selection(previous_selection)
            selected_index = 0
            if preferred in values:
                selected_index = values.index(preferred)
            self.mic_combo.current(selected_index)
            self._set_status(self._tr("status_microphones_found", count=len(self.microphones)))
            self._log_event(self._tr("log_microphones_refreshed", count=len(self.microphones)))
            self._save_app_settings()
        else:
            self._set_status(self._tr("status_microphones_missing"))
            self._log_event(self._tr("log_microphones_missing"))
        self._start_idle_level_monitor(restart=True)

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

    def _on_mic_selected(self, _event=None) -> None:
        self._save_app_settings()
        self._start_idle_level_monitor(restart=True)

    def _selected_microphone_name(self) -> str:
        selected_name = self.mic_combo.get().strip() if hasattr(self, "mic_combo") else ""
        if selected_name:
            return selected_name

        idx = self.mic_combo.current() if hasattr(self, "mic_combo") else -1
        if idx == 0:
            return self.system_microphone_option
        mic_idx = idx - 1
        if mic_idx >= 0 and mic_idx < len(self.microphones):
            return self.microphones[mic_idx].name
        return ""

    def _resolve_selected_microphone(self):
        selected_name = self._selected_microphone_name()
        if self._is_system_microphone_selection(selected_name):
            return self._default_microphone()

        if selected_name:
            for mic in self.microphones:
                if mic.name == selected_name:
                    return mic

        idx = self.mic_combo.current()
        if idx == 0:
            return self._default_microphone()

        mic_idx = idx - 1
        if mic_idx >= 0 and mic_idx < len(self.microphones):
            return self.microphones[mic_idx]
        return None

    def _default_microphone(self):
        try:
            default_mic = sc.default_microphone()
        except Exception:
            return None
        return default_mic

    def _default_microphone_name(self) -> str:
        default_mic = self._default_microphone()
        if default_mic is None:
            return ""
        return str(getattr(default_mic, "name", "")).strip()

    def _system_microphone_option_label(self) -> str:
        default_name = self._default_microphone_name()
        label = self._tr("system_microphone")
        if not default_name:
            return f"{label} ({self._tr('system_not_defined')})"
        return f"{label} ({default_name})"

    def _is_system_microphone_selection(self, selected_name: str) -> bool:
        normalized_name = str(selected_name or "")
        if normalized_name == self.SYSTEM_MICROPHONE_SETTING:
            return True
        prefixes = system_microphone_label_prefixes()
        return any(normalized_name.startswith(prefix) for prefix in prefixes)

    def _preferred_microphone_selection(self, previous_selection: str) -> str:
        if self._is_system_microphone_selection(previous_selection):
            return self.system_microphone_option
        if previous_selection:
            return previous_selection

        preferred = self.app_settings.get("last_microphone", "")
        if self._is_system_microphone_selection(preferred):
            return self.system_microphone_option
        return preferred

    def _schedule_settings_save(self, *_args) -> None:
        if self.settings_save_after_id:
            self.after_cancel(self.settings_save_after_id)
        self.settings_save_after_id = self.after(250, self._save_app_settings)

    def _current_speaker_labels(self) -> tuple[str, str]:
        self_label = " ".join(self.self_label_var.get().split()) or DEFAULT_SELF_LABEL
        other_label = " ".join(self.other_label_var.get().split()) or DEFAULT_OTHER_LABEL
        if self.self_label_var.get() != self_label:
            self.self_label_var.set(self_label)
        if self.other_label_var.get() != other_label:
            self.other_label_var.set(other_label)
        return self_label, other_label

    def _current_transcription_options(self) -> TranscriptionOptions:
        model_name = self.model_var.get().strip()
        if model_name not in ALLOWED_MODELS:
            model_name = DEFAULT_MODEL
            self.model_var.set(model_name)

        language = " ".join(self.language_var.get().split()) or DEFAULT_LANGUAGE
        if self.language_var.get() != language:
            self.language_var.set(language)

        try:
            beam_size = int(self.beam_size_var.get())
        except (TypeError, ValueError, tk.TclError):
            beam_size = DEFAULT_BEAM_SIZE
        beam_size = max(1, min(10, beam_size))
        if self.beam_size_var.get() != beam_size:
            self.beam_size_var.set(beam_size)

        compute_type = self.compute_type_var.get().strip()
        if compute_type not in ALLOWED_COMPUTE_TYPES:
            compute_type = DEFAULT_COMPUTE_TYPE
            self.compute_type_var.set(compute_type)

        vad_filter = bool(self.vad_filter_var.get())
        include_timestamps = bool(self.include_timestamps_var.get())

        return TranscriptionOptions(
            model_name=model_name,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            compute_type=compute_type,
            include_timestamps=include_timestamps,
        )

    def _save_app_settings(self) -> None:
        if self.settings_save_after_id:
            self.settings_save_after_id = None
        selected_mic = self._selected_microphone_name()
        if self._is_system_microphone_selection(selected_mic):
            selected_mic = self.SYSTEM_MICROPHONE_SETTING

        self_label, other_label = self._current_speaker_labels()
        options = self._current_transcription_options()
        self.app_settings = {
            "last_microphone": selected_mic,
            "speaker_self": self_label,
            "speaker_other": other_label,
            "auto_transcribe_after_record": bool(self.auto_transcribe_var.get()),
            "whisper_model": options.model_name,
            "whisper_language": options.language,
            "whisper_beam_size": options.beam_size,
            "whisper_vad_filter": options.vad_filter,
            "whisper_compute_type": options.compute_type,
            "include_timestamps": options.include_timestamps,
            "ui_language": self.ui_language,
        }
        save_app_settings(self.app_settings)

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
    def _sound_device_name(device) -> str:
        if device is None:
            return ""
        name = str(getattr(device, "name", "")).strip()
        if name:
            return name
        return str(getattr(device, "id", "")).strip()

    @staticmethod
    def _sound_device_is_loopback(device) -> bool:
        try:
            return bool(getattr(device, "isloopback", False))
        except Exception:
            return False

    def _desktop_source_score(self, device, preferred_speaker_name: str) -> int:
        name = self._sound_device_name(device).lower()
        if not name:
            return 0

        score = 0
        if self._sound_device_is_loopback(device):
            score += 100

        preferred = preferred_speaker_name.lower()
        if preferred and preferred in name:
            score += 40
        if "monitor of" in name:
            score += 40
        if "monitor" in name or "loopback" in name:
            score += 30
        if any(hint in name for hint in DESKTOP_SOURCE_NAME_HINTS):
            score += 20
        return score

    def _pick_desktop_source_candidate(self, devices: list, preferred_speaker_name: str):
        best_device = None
        best_score = 0
        for device in devices:
            score = self._desktop_source_score(device, preferred_speaker_name)
            if score > best_score:
                best_device = device
                best_score = score
        return best_device

    def _resolve_desktop_loopback(self):
        speaker = None
        preferred_speaker_name = ""
        try:
            speaker = sc.default_speaker()
        except Exception:
            speaker = None

        if speaker is not None:
            preferred_speaker_name = self._sound_device_name(speaker)
            speaker_id = getattr(speaker, "id", None)
            if speaker_id:
                try:
                    loopback = sc.get_microphone(id=str(speaker_id), include_loopback=True)
                except Exception as exc:
                    loopback = None
                    loopback_open_error = exc
                else:
                    loopback_open_error = None

                if loopback is not None:
                    if self._desktop_source_score(loopback, preferred_speaker_name) > 0:
                        return speaker, loopback
                if loopback_open_error is not None:
                    self._log_event(self._tr("err_open_loopback_failed", error=loopback_open_error))
            else:
                self._log_event(self._tr("err_no_output_device_id"))
        else:
            self._log_event(self._tr("err_no_output_device"))

        include_loopback = sys.platform != "darwin"
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="macOS does not support loopback recording functionality",
                    category=Warning,
                )
                desktop_sources = list(sc.all_microphones(include_loopback=include_loopback))
        except Exception as exc:
            raise RuntimeError(self._tr("err_list_desktop_sources_failed", error=exc)) from exc

        candidate = self._pick_desktop_source_candidate(desktop_sources, preferred_speaker_name)
        if candidate is not None:
            source_info = speaker if speaker is not None else candidate
            return source_info, candidate

        if sys.platform == "darwin":
            raise RuntimeError(self._tr("err_desktop_source_not_found_macos"))
        raise RuntimeError(self._tr("err_loopback_not_found"))

    def _emit_level(self, source: str, level: float) -> None:
        self.event_queue.put(("level", source, level))

    def _emit_level_monitor_error(self, source: str, message: str) -> None:
        self.event_queue.put(("level_monitor_error", source, message))

    def _start_idle_level_monitor(self, restart: bool = False) -> None:
        if self.recorder is not None:
            return
        if self.level_monitor is not None and not restart:
            return
        if self.level_monitor is not None:
            self._stop_idle_level_monitor()

        mic = self._resolve_selected_microphone()
        if mic is None:
            self._set_levels_to_zero()
            return

        try:
            _, desktop_loopback = self._resolve_desktop_loopback()
        except Exception as exc:  # pragma: no cover - hardware-specific
            self._set_levels_to_zero()
            self._log_event(self._tr("log_level_monitor_unavailable", error=exc))
            return

        self.level_monitor = DualTrackLevelMonitor(
            mic=mic,
            desktop=desktop_loopback,
            level_callback=self._emit_level,
            error_callback=self._emit_level_monitor_error,
        )
        self.level_monitor.start()

    def _stop_idle_level_monitor(self) -> None:
        if self.level_monitor is None:
            return
        monitor = self.level_monitor
        self.level_monitor = None
        monitor.stop()

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
            speaker_self=self_label,
            speaker_other=other_label,
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
        self._set_status(self._tr("status_recording_active", session=session_dir.name, duration="00:00:00"))
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

        if self.active_session_dir:
            update_session_metadata(self.active_session_dir, duration)

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

        session_dir = self.active_session_dir
        self.active_session_dir = None
        if session_dir and auto_transcribe:
            self._start_transcription(session_dir)

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

    def _session_audio_ready(self, session_dir: Path) -> bool:
        mic_path, desktop_path = resolve_track_paths(session_dir)
        return mic_path is not None and desktop_path is not None

    def _session_duration_text(self, session_dir: Path) -> str:
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
