from __future__ import annotations

import queue
import threading
from pathlib import Path

import tkinter as tk

from .localization import UI_LANGUAGE_CODES
from .numpy_compat import apply_numpy_fromstring_compat_patch
from .recording import DualTrackLevelMonitor, DualTrackRecorder
from .settings import load_app_settings
from .storage import ensure_recordings_root
from .transcription import WhisperTranscriber
from .ui_mixins import (
    AudioMixin,
    LocalizationMixin,
    RecordingMixin,
    RecordingsMixin,
    SettingsMixin,
    TranscriptionMixin,
    WindowMixin,
    set_windows_app_user_model_id,
)

apply_numpy_fromstring_compat_patch()


class App(
    WindowMixin,
    LocalizationMixin,
    AudioMixin,
    SettingsMixin,
    RecordingsMixin,
    RecordingMixin,
    TranscriptionMixin,
    tk.Tk,
):
    SYSTEM_MICROPHONE_SETTING = "__system_default__"

    def __init__(self):
        set_windows_app_user_model_id()
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
        self.current_transcription_session: Path | None = None
        self.transcription_queue: list[Path] = []
        self.transcription_queue_keys: set[str] = set()

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
        self.recording_alias_editor: tk.Entry | None = None
        self.recording_alias_session: Path | None = None
        self.recording_alias_original_value = ""
        self.pending_short_name_var = tk.StringVar()

        self.microphones = []
        self.system_microphone_option = self._system_microphone_option_label()
        self.auto_transcribe_var = tk.BooleanVar(
            value=self.app_settings["auto_transcribe_after_record"]
        )
        self.self_label_var = tk.StringVar(value=self.app_settings["speaker_self"])
        self.other_label_var = tk.StringVar(value=self.app_settings["speaker_other"])
        self.transcription_library_var = tk.StringVar(
            value=self.app_settings["transcription_library"]
        )
        self.model_var = tk.StringVar(value=self.app_settings["whisper_model"])
        self.device_var = tk.StringVar(value=self.app_settings["whisper_device"])
        self.language_var = tk.StringVar(value=self.app_settings["whisper_language"])
        self.beam_size_var = tk.IntVar(value=self.app_settings["whisper_beam_size"])
        self.vad_filter_var = tk.BooleanVar(value=self.app_settings["whisper_vad_filter"])
        self.compute_type_var = tk.StringVar(value=self.app_settings["whisper_compute_type"])
        self.include_timestamps_var = tk.BooleanVar(value=self.app_settings["include_timestamps"])
        self.transcribe_mix_track_var = tk.BooleanVar(
            value=self.app_settings["transcribe_mix_track"]
        )
        self.transcription_mode_var = tk.StringVar(value=self.app_settings["transcription_mode"])
        self.network_whisper_url_var = tk.StringVar(value=self.app_settings["network_whisper_url"])
        self.network_whisper_token_var = tk.StringVar(value=self.app_settings["network_whisper_token"])
        self._build_ui()
        self._initialize_transcription_settings_ui()
        self.auto_transcribe_var.trace_add("write", self._schedule_settings_save)
        self.self_label_var.trace_add("write", self._schedule_settings_save)
        self.other_label_var.trace_add("write", self._schedule_settings_save)
        self.transcription_library_var.trace_add("write", self._schedule_settings_save)
        self.model_var.trace_add("write", self._schedule_settings_save)
        self.device_var.trace_add("write", self._schedule_settings_save)
        self.language_var.trace_add("write", self._schedule_settings_save)
        self.beam_size_var.trace_add("write", self._schedule_settings_save)
        self.vad_filter_var.trace_add("write", self._schedule_settings_save)
        self.compute_type_var.trace_add("write", self._schedule_settings_save)
        self.include_timestamps_var.trace_add("write", self._schedule_settings_save)
        self.transcribe_mix_track_var.trace_add("write", self._schedule_settings_save)
        self.transcription_mode_var.trace_add("write", self._schedule_settings_save)
        self.network_whisper_url_var.trace_add("write", self._schedule_settings_save)
        self.network_whisper_token_var.trace_add("write", self._schedule_settings_save)
        self._migrate_legacy_session_aliases()
        self._refresh_microphones()
        self._refresh_recordings()
        self._set_levels_to_zero()
        self._start_idle_level_monitor()
        self.after(250, self._tick_level_meter)
        self.after(150, self._poll_events)
