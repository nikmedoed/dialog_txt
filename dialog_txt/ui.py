from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .config import DEFAULT_OTHER_LABEL, DEFAULT_SELF_LABEL
from .models import RecordingError, TranscriptionCancelled, TranscriptionOptions
from .numpy_compat import apply_numpy_fromstring_compat_patch
from .recording import DualTrackLevelMonitor, DualTrackRecorder
from .settings import load_app_settings, save_app_settings
from .settings import (
    ALLOWED_COMPUTE_TYPES,
    ALLOWED_MODELS,
    ALLOWED_UI_LANGUAGES,
    DEFAULT_BEAM_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_UI_LANGUAGE,
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
from .utils import format_seconds

apply_numpy_fromstring_compat_patch()

import soundcard as sc


UI_LANGUAGE_CODES = {"RU": "ru", "EN": "en"}

UI_TEXTS = {
    "ru": {
        "group_recording": "Запись",
        "label_microphone": "Микрофон:",
        "label_ui_language": "UI:",
        "record_start": "● Начать запись",
        "record_stop": "■ Остановить запись",
        "status_prefix": "Статус",
        "status_idle": "ожидание",
        "label_level_mic": "Mic",
        "label_level_desktop": "Desktop",
        "group_transcription": "Транскрибация",
        "auto_transcribe_after_record": "Автотранскрибация после записи",
        "label_speaker_mic": "Метка mic:",
        "label_speaker_desktop": "Метка desktop:",
        "include_timestamps": "добавить таймметки",
        "label_model": "Модель:",
        "label_language": "Язык:",
        "label_beam": "Beam:",
        "label_compute": "Compute:",
        "vad_filter": "VAD фильтр",
        "label_progress": "Прогресс:",
        "cancel": "Отменить",
        "group_recordings": "Существующие записи",
        "btn_transcribe_selected": "Транскрибировать выбранную",
        "btn_retranscribe": "Перетранскрибировать",
        "btn_cannot_transcribe_missing_tracks": "Нельзя: нет обеих дорожек",
        "btn_refresh_list": "Обновить список",
        "btn_open_folder": "Открыть папку",
        "col_session": "Сессия",
        "col_duration": "Длительность",
        "col_audio": "Аудио",
        "col_txt": "TXT",
        "group_log": "Лог событий",
        "title_error": "Ошибка",
        "title_busy": "Занято",
        "title_open_folder": "Открыть папку",
        "title_recording_selection": "Выбор записи",
        "title_cancelled": "Отменено",
        "title_recording_error": "Ошибка записи",
        "title_transcription_error": "Ошибка транскрибации",
        "msg_microphones_list_failed": "Не удалось получить список микрофонов:\n{error}",
        "log_microphones_list_failed": "Ошибка списка микрофонов: {error}",
        "status_microphones_found": "Микрофонов найдено: {count}",
        "log_microphones_refreshed": "Список микрофонов обновлён: {count} устройств",
        "status_microphones_missing": "Микрофоны не найдены",
        "log_microphones_missing": "Микрофоны не найдены",
        "log_recordings_refreshed": "Список записей обновлён: {count} сессий",
        "msg_select_recording_from_list": "Выберите запись из списка.",
        "log_folder_opened": "Открыта папка: {path}",
        "log_open_folder_error": "Ошибка открытия папки {path}: {error}",
        "msg_open_folder_failed": "Не удалось открыть папку:\n{path}\n\n{error}",
        "err_no_output_device": "Не найдено устройство вывода для loopback-записи.",
        "err_no_output_device_id": "У устройства вывода отсутствует ID для loopback.",
        "err_open_loopback_failed": "Не удалось открыть loopback-источник: {error}",
        "err_loopback_not_found": "Loopback-источник для текущего устройства вывода не найден.",
        "log_level_monitor_unavailable": "Монитор уровней недоступен: {error}",
        "msg_wait_transcription_complete": "Сначала дождитесь завершения текущей транскрибации.",
        "msg_select_microphone": "Выберите микрофон.",
        "msg_desktop_device_failed": "Не удалось получить устройство рабочего стола:\n{error}",
        "log_desktop_capture_error": "Ошибка desktop-захвата: {error}",
        "status_recording_active": "Идёт запись: {session} · {duration}",
        "log_recording_started": "Старт записи: {session_dir}",
        "log_mic_source": "Mic: {mic}",
        "log_desktop_source": "Desktop (loopback): {desktop}",
        "log_speaker_labels": "Подписи: [{self_label}] / [{other_label}]",
        "status_recording_error": "Ошибка записи",
        "log_recording_error": "Ошибка записи: {error}",
        "status_recording_stopped_transcribe": "Запись остановлена ({duration}). Запускаю транскрибацию...",
        "log_recording_stopped_transcribe": "Запись остановлена ({duration}), запускаю транскрибацию",
        "status_recording_stopped_no_auto": "Запись остановлена ({duration}). Автотранскрибация отключена",
        "log_recording_stopped_no_auto": "Запись остановлена ({duration}), автотранскрибация отключена",
        "session_fallback": "сессия",
        "msg_stop_recording_first": "Сначала остановите текущую запись.",
        "msg_transcription_running": "Транскрибация уже выполняется.",
        "msg_session_missing_tracks": "В выбранной сессии отсутствуют дорожки mic.ogg и desktop.ogg.",
        "status_transcription_active": "Транскрибация: {session}",
        "log_transcription_started": (
            "Старт транскрибации: {session_dir} | model={model}, lang={lang}, "
            "beam={beam}, vad={vad}, compute={compute}"
        ),
        "status_transcription_cancelling": "Отмена транскрибации...",
        "log_transcription_cancel_requested": "Запрошена отмена транскрибации",
        "log_level_monitor_error": "Ошибка монитора уровней ({source}): {message}",
        "log_transcription_progress": "Транскрибация: {percent}% ({message})",
        "status_done": "Готово: {session}",
        "log_transcription_done": "Транскрибация завершена: {path}",
        "status_transcription_cancelled": "Транскрибация отменена: {session}",
        "log_transcription_cancelled": "Транскрибация отменена: {session_dir}",
        "msg_transcription_cancelled": "Транскрибация была отменена пользователем.",
        "status_transcription_error": "Ошибка транскрибации",
        "log_transcription_error": "Ошибка транскрибации: {error}",
        "audio_missing_both": "нет audio",
        "audio_missing_mic": "нет mic",
        "audio_missing_desktop": "нет desktop",
        "system_microphone": "Системный",
        "system_not_defined": "не определён",
        "log_window_close_cancel": "Окно закрывается: отправлен запрос на отмену транскрибации",
    },
    "en": {
        "group_recording": "Recording",
        "label_microphone": "Microphone:",
        "label_ui_language": "UI:",
        "record_start": "● Start recording",
        "record_stop": "■ Stop recording",
        "status_prefix": "Status",
        "status_idle": "idle",
        "label_level_mic": "Mic",
        "label_level_desktop": "Desktop",
        "group_transcription": "Transcription",
        "auto_transcribe_after_record": "Auto-transcribe after recording",
        "label_speaker_mic": "mic label:",
        "label_speaker_desktop": "desktop label:",
        "include_timestamps": "include timestamps",
        "label_model": "Model:",
        "label_language": "Language:",
        "label_beam": "Beam:",
        "label_compute": "Compute:",
        "vad_filter": "VAD filter",
        "label_progress": "Progress:",
        "cancel": "Cancel",
        "group_recordings": "Existing recordings",
        "btn_transcribe_selected": "Transcribe selected",
        "btn_retranscribe": "Re-transcribe",
        "btn_cannot_transcribe_missing_tracks": "Unavailable: missing both tracks",
        "btn_refresh_list": "Refresh list",
        "btn_open_folder": "Open folder",
        "col_session": "Session",
        "col_duration": "Duration",
        "col_audio": "Audio",
        "col_txt": "TXT",
        "group_log": "Event log",
        "title_error": "Error",
        "title_busy": "Busy",
        "title_open_folder": "Open folder",
        "title_recording_selection": "Select recording",
        "title_cancelled": "Cancelled",
        "title_recording_error": "Recording error",
        "title_transcription_error": "Transcription error",
        "msg_microphones_list_failed": "Failed to get microphone list:\n{error}",
        "log_microphones_list_failed": "Microphone list error: {error}",
        "status_microphones_found": "Microphones found: {count}",
        "log_microphones_refreshed": "Microphone list refreshed: {count} devices",
        "status_microphones_missing": "No microphones found",
        "log_microphones_missing": "No microphones found",
        "log_recordings_refreshed": "Recordings list refreshed: {count} sessions",
        "msg_select_recording_from_list": "Select a recording from the list.",
        "log_folder_opened": "Opened folder: {path}",
        "log_open_folder_error": "Failed to open folder {path}: {error}",
        "msg_open_folder_failed": "Failed to open folder:\n{path}\n\n{error}",
        "err_no_output_device": "No output device found for loopback recording.",
        "err_no_output_device_id": "Output device does not provide an ID for loopback.",
        "err_open_loopback_failed": "Failed to open loopback source: {error}",
        "err_loopback_not_found": "Loopback source for current output device was not found.",
        "log_level_monitor_unavailable": "Level monitor is unavailable: {error}",
        "msg_wait_transcription_complete": "Wait for the current transcription to complete first.",
        "msg_select_microphone": "Select a microphone.",
        "msg_desktop_device_failed": "Failed to access desktop device:\n{error}",
        "log_desktop_capture_error": "Desktop capture error: {error}",
        "status_recording_active": "Recording: {session} · {duration}",
        "log_recording_started": "Recording started: {session_dir}",
        "log_mic_source": "Mic: {mic}",
        "log_desktop_source": "Desktop (loopback): {desktop}",
        "log_speaker_labels": "Labels: [{self_label}] / [{other_label}]",
        "status_recording_error": "Recording error",
        "log_recording_error": "Recording error: {error}",
        "status_recording_stopped_transcribe": "Recording stopped ({duration}). Starting transcription...",
        "log_recording_stopped_transcribe": "Recording stopped ({duration}), starting transcription",
        "status_recording_stopped_no_auto": "Recording stopped ({duration}). Auto-transcribe is off",
        "log_recording_stopped_no_auto": "Recording stopped ({duration}), auto-transcribe is off",
        "session_fallback": "session",
        "msg_stop_recording_first": "Stop the current recording first.",
        "msg_transcription_running": "Transcription is already running.",
        "msg_session_missing_tracks": "Selected session is missing mic.ogg and desktop.ogg tracks.",
        "status_transcription_active": "Transcribing: {session}",
        "log_transcription_started": (
            "Transcription started: {session_dir} | model={model}, lang={lang}, "
            "beam={beam}, vad={vad}, compute={compute}"
        ),
        "status_transcription_cancelling": "Cancelling transcription...",
        "log_transcription_cancel_requested": "Transcription cancellation requested",
        "log_level_monitor_error": "Level monitor error ({source}): {message}",
        "log_transcription_progress": "Transcription: {percent}% ({message})",
        "status_done": "Done: {session}",
        "log_transcription_done": "Transcription finished: {path}",
        "status_transcription_cancelled": "Transcription cancelled: {session}",
        "log_transcription_cancelled": "Transcription cancelled: {session_dir}",
        "msg_transcription_cancelled": "Transcription was cancelled by user.",
        "status_transcription_error": "Transcription error",
        "log_transcription_error": "Transcription error: {error}",
        "audio_missing_both": "no audio",
        "audio_missing_mic": "no mic",
        "audio_missing_desktop": "no desktop",
        "system_microphone": "System",
        "system_not_defined": "not set",
        "log_window_close_cancel": "Window is closing: transcription cancellation requested",
    },
}

SYSTEM_MICROPHONE_LABEL_PREFIXES = (
    f"{UI_TEXTS['ru']['system_microphone']} (",
    f"{UI_TEXTS['en']['system_microphone']} (",
)


class App(tk.Tk):
    SYSTEM_MICROPHONE_SETTING = "__system_default__"

    def __init__(self):
        super().__init__()
        self.title("Dialog to TXT")
        self.geometry("620x760")
        self.minsize(560, 560)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

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

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("Treeview", rowheight=19)
        style.configure("Compact.Horizontal.TProgressbar", thickness=8)

        top = ttk.Frame(self, padding=6)
        top.pack(fill=tk.BOTH, expand=True)

        self.controls_box = ttk.LabelFrame(top, text=self._tr("group_recording"), padding=6)
        self.controls_box.pack(fill=tk.X)
        self.controls_box.columnconfigure(0, weight=1)

        mic_row = ttk.Frame(self.controls_box)
        mic_row.grid(row=0, column=0, sticky=tk.EW)
        mic_row.columnconfigure(3, weight=1)
        self.mic_label = ttk.Label(mic_row, text=self._tr("label_microphone"))
        self.mic_label.grid(row=0, column=0, sticky=tk.W)
        self.mic_combo = ttk.Combobox(mic_row, state="readonly", width=34)
        self.mic_combo.grid(row=0, column=1, sticky=tk.W, padx=(2, 0))
        self.mic_combo.bind("<<ComboboxSelected>>", self._on_mic_selected)

        self.refresh_mic_button = ttk.Button(
            mic_row, text="↻", width=3, command=self._refresh_microphones
        )
        self.refresh_mic_button.grid(row=0, column=2, padx=(4, 0), sticky=tk.W)

        self.ui_language_frame = ttk.Frame(mic_row)
        self.ui_language_frame.grid(row=0, column=3)
        self.ui_language_label = ttk.Label(self.ui_language_frame, text=self._tr("label_ui_language"))
        self.ui_language_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        self.ui_language_combo = ttk.Combobox(
            self.ui_language_frame,
            state="readonly",
            width=4,
            values=list(UI_LANGUAGE_CODES.keys()),
            textvariable=self.ui_language_code_var,
        )
        self.ui_language_combo.grid(row=0, column=1, sticky=tk.W)
        self.ui_language_combo.bind("<<ComboboxSelected>>", self._on_ui_language_selected)

        self.record_button = tk.Button(
            mic_row,
            text=self._tr("record_start"),
            command=self._toggle_recording,
            bg="#1f8b4c",
            fg="white",
            activebackground="#176a38",
            activeforeground="white",
            disabledforeground="#d8d8d8",
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=5,
        )
        self.record_button.grid(row=0, column=4, sticky=tk.E)

        self.status_label = ttk.Label(self.controls_box, text="")
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        self._update_status_line()

        levels_row = ttk.Frame(self.controls_box)
        levels_row.grid(row=2, column=0, sticky=tk.EW, pady=(2, 0))
        levels_row.columnconfigure(1, weight=1)
        levels_row.columnconfigure(4, weight=1)

        self.level_mic_label = ttk.Label(levels_row, text=self._tr("label_level_mic"))
        self.level_mic_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        self.mic_level = ttk.Progressbar(
            levels_row,
            mode="determinate",
            maximum=100,
            style="Compact.Horizontal.TProgressbar",
        )
        self.mic_level.grid(row=0, column=1, sticky=tk.EW, padx=(0, 8))

        self.level_desktop_label = ttk.Label(levels_row, text=self._tr("label_level_desktop"))
        self.level_desktop_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 4))
        self.desktop_level = ttk.Progressbar(
            levels_row,
            mode="determinate",
            maximum=100,
            style="Compact.Horizontal.TProgressbar",
        )
        self.desktop_level.grid(row=0, column=4, sticky=tk.EW)

        self.transcribe_box = ttk.LabelFrame(top, text=self._tr("group_transcription"), padding=6)
        self.transcribe_box.pack(fill=tk.X, pady=(6, 0))
        self.transcribe_box.columnconfigure(1, weight=1)

        transcribe_flags = ttk.Frame(self.transcribe_box)
        transcribe_flags.grid(row=0, column=0, columnspan=4, sticky=tk.W)
        self.auto_transcribe_check = ttk.Checkbutton(
            transcribe_flags,
            text=self._tr("auto_transcribe_after_record"),
            variable=self.auto_transcribe_var,
        )
        self.auto_transcribe_check.grid(row=0, column=0, sticky=tk.W)

        speakers_row = ttk.Frame(self.transcribe_box)
        speakers_row.grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=(4, 0))
        speakers_row.columnconfigure(1, weight=1)
        speakers_row.columnconfigure(3, weight=1)
        self.self_label_label = ttk.Label(speakers_row, text=self._tr("label_speaker_mic"))
        self.self_label_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        self.self_label_entry = ttk.Entry(speakers_row, textvariable=self.self_label_var)
        self.self_label_entry.grid(row=0, column=1, sticky=tk.EW, padx=(0, 8))
        self.other_label_label = ttk.Label(speakers_row, text=self._tr("label_speaker_desktop"))
        self.other_label_label.grid(row=0, column=2, sticky=tk.W, padx=(0, 4))
        self.other_label_entry = ttk.Entry(speakers_row, textvariable=self.other_label_var)
        self.other_label_entry.grid(row=0, column=3, sticky=tk.EW, padx=(0, 8))
        self.include_timestamps_check = ttk.Checkbutton(
            speakers_row,
            text=self._tr("include_timestamps"),
            variable=self.include_timestamps_var,
        )
        self.include_timestamps_check.grid(row=0, column=4, sticky=tk.W)

        whisper_row = ttk.Frame(self.transcribe_box)
        whisper_row.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))
        self.model_label = ttk.Label(whisper_row, text=self._tr("label_model"))
        self.model_label.grid(row=0, column=0, sticky=tk.W)
        self.model_combo = ttk.Combobox(
            whisper_row,
            state="readonly",
            width=10,
            values=list(ALLOWED_MODELS),
            textvariable=self.model_var,
        )
        self.model_combo.grid(row=0, column=1, sticky=tk.W, padx=(3, 8))
        self.language_label = ttk.Label(whisper_row, text=self._tr("label_language"))
        self.language_label.grid(row=0, column=2, sticky=tk.W)
        self.language_combo = ttk.Combobox(
            whisper_row,
            width=5,
            values=["ru", "en", "auto"],
            textvariable=self.language_var,
        )
        self.language_combo.grid(row=0, column=3, sticky=tk.W, padx=(3, 8))
        self.beam_label = ttk.Label(whisper_row, text=self._tr("label_beam"))
        self.beam_label.grid(row=0, column=4, sticky=tk.W)
        self.beam_spinbox = ttk.Spinbox(
            whisper_row,
            from_=1,
            to=10,
            width=4,
            textvariable=self.beam_size_var,
        )
        self.beam_spinbox.grid(row=0, column=5, sticky=tk.W, padx=(3, 8))
        self.compute_label = ttk.Label(whisper_row, text=self._tr("label_compute"))
        self.compute_label.grid(row=0, column=6, sticky=tk.W)
        self.compute_type_combo = ttk.Combobox(
            whisper_row,
            state="readonly",
            width=8,
            values=list(ALLOWED_COMPUTE_TYPES),
            textvariable=self.compute_type_var,
        )
        self.compute_type_combo.grid(row=0, column=7, sticky=tk.W, padx=(3, 8))
        self.vad_filter_check = ttk.Checkbutton(
            whisper_row,
            text=self._tr("vad_filter"),
            variable=self.vad_filter_var,
        )
        self.vad_filter_check.grid(row=0, column=8, sticky=tk.W)

        self.progress_label_title = ttk.Label(self.transcribe_box, text=self._tr("label_progress"))
        self.progress_label_title.grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
        self.progress = ttk.Progressbar(self.transcribe_box, mode="determinate", maximum=100)
        self.progress.grid(row=3, column=1, sticky=tk.EW, pady=(6, 0), padx=(6, 0))
        self.progress_label = ttk.Label(self.transcribe_box, text="0%")
        self.progress_label.grid(row=3, column=2, sticky=tk.W, padx=(6, 0), pady=(6, 0))

        self.cancel_transcribe_button = ttk.Button(
            self.transcribe_box,
            text=self._tr("cancel"),
            command=self._cancel_transcription,
            state=tk.DISABLED,
        )
        self.cancel_transcribe_button.grid(row=3, column=3, sticky=tk.E, padx=(6, 0), pady=(6, 0))

        self.recordings_box = ttk.LabelFrame(top, text=self._tr("group_recordings"), padding=6)
        self.recordings_box.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        recordings_actions = ttk.Frame(self.recordings_box)
        recordings_actions.pack(fill=tk.X, pady=(0, 4))
        self.transcribe_selected_button = ttk.Button(
            recordings_actions,
            text=self._tr("btn_transcribe_selected"),
            command=self._transcribe_selected,
        )
        self.transcribe_selected_button.grid(row=0, column=0, sticky=tk.W)
        self.refresh_recordings_button = ttk.Button(
            recordings_actions,
            text=self._tr("btn_refresh_list"),
            command=self._refresh_recordings,
        )
        self.refresh_recordings_button.grid(row=0, column=1, sticky=tk.W, padx=(4, 0))
        self.open_folder_button = ttk.Button(
            recordings_actions, text=self._tr("btn_open_folder"), command=self._open_selected_folder
        )
        self.open_folder_button.grid(row=0, column=2, sticky=tk.W, padx=(4, 0))

        columns = ("session", "duration", "audio", "txt")
        self.recordings_tree = ttk.Treeview(self.recordings_box, columns=columns, show="headings")
        self.recordings_tree.heading("session", text=self._tr("col_session"))
        self.recordings_tree.heading("duration", text=self._tr("col_duration"))
        self.recordings_tree.heading("audio", text=self._tr("col_audio"))
        self.recordings_tree.heading("txt", text=self._tr("col_txt"))
        self.recordings_tree.column("session", width=220, anchor=tk.W, stretch=True)
        self.recordings_tree.column("duration", width=100, anchor=tk.CENTER, stretch=False)
        self.recordings_tree.column("audio", width=72, anchor=tk.CENTER, stretch=False)
        self.recordings_tree.column("txt", width=48, anchor=tk.CENTER, stretch=False)
        self.recordings_tree.pack(fill=tk.BOTH, expand=True)
        self.recordings_tree.bind("<<TreeviewSelect>>", self._on_recording_selected)
        self.recordings_tree.bind("<Double-1>", self._on_recording_double_click)

        self.log_box = ttk.LabelFrame(top, text=self._tr("group_log"), padding=6)
        self.log_box.pack(fill=tk.BOTH, expand=False, pady=(6, 0))
        self.log_text = ScrolledText(self.log_box, height=4, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _resolve_ui_language(self, value: str | None) -> str:
        language = str(value or DEFAULT_UI_LANGUAGE).strip().lower()
        if language not in ALLOWED_UI_LANGUAGES:
            return DEFAULT_UI_LANGUAGE
        return language

    def _tr(self, key: str, **kwargs) -> str:
        fallback_pack = UI_TEXTS[DEFAULT_UI_LANGUAGE]
        language_pack = UI_TEXTS.get(self.ui_language, fallback_pack)
        template = language_pack.get(key, fallback_pack.get(key, key))
        return template.format(**kwargs) if kwargs else template

    def _on_ui_language_selected(self, _event=None) -> None:
        selected_code = self.ui_language_code_var.get().strip().upper()
        target_language = UI_LANGUAGE_CODES.get(selected_code, DEFAULT_UI_LANGUAGE)
        if target_language == self.ui_language:
            return
        self.ui_language = target_language
        self._apply_localization(refresh_data=True)
        self._save_app_settings()

    def _apply_localization(self, refresh_data: bool = False) -> None:
        self.controls_box.configure(text=self._tr("group_recording"))
        self.mic_label.configure(text=self._tr("label_microphone"))
        self.ui_language_label.configure(text=self._tr("label_ui_language"))
        self.level_mic_label.configure(text=self._tr("label_level_mic"))
        self.level_desktop_label.configure(text=self._tr("label_level_desktop"))
        self.transcribe_box.configure(text=self._tr("group_transcription"))
        self.auto_transcribe_check.configure(text=self._tr("auto_transcribe_after_record"))
        self.self_label_label.configure(text=self._tr("label_speaker_mic"))
        self.other_label_label.configure(text=self._tr("label_speaker_desktop"))
        self.include_timestamps_check.configure(text=self._tr("include_timestamps"))
        self.model_label.configure(text=self._tr("label_model"))
        self.language_label.configure(text=self._tr("label_language"))
        self.beam_label.configure(text=self._tr("label_beam"))
        self.compute_label.configure(text=self._tr("label_compute"))
        self.vad_filter_check.configure(text=self._tr("vad_filter"))
        self.progress_label_title.configure(text=self._tr("label_progress"))
        self.cancel_transcribe_button.configure(text=self._tr("cancel"))
        self.recordings_box.configure(text=self._tr("group_recordings"))
        self.transcribe_selected_button.configure(text=self._tr("btn_transcribe_selected"))
        self.refresh_recordings_button.configure(text=self._tr("btn_refresh_list"))
        self.open_folder_button.configure(text=self._tr("btn_open_folder"))
        self.recordings_tree.heading("session", text=self._tr("col_session"))
        self.recordings_tree.heading("duration", text=self._tr("col_duration"))
        self.recordings_tree.heading("audio", text=self._tr("col_audio"))
        self.recordings_tree.heading("txt", text=self._tr("col_txt"))
        self.log_box.configure(text=self._tr("group_log"))
        self._set_record_button_style(is_recording=self.recorder is not None)
        self._update_status_line()
        self._on_recording_selected()
        if refresh_data:
            self._refresh_microphones()
            self._refresh_recordings()

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
            duration_text = self._session_duration_text(session_dir)
            self.recordings_tree.insert(
                "",
                tk.END,
                iid=str(session_dir),
                values=(display_name, duration_text, audio_status, txt_status),
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
        if not (self.transcription_thread and self.transcription_thread.is_alive()):
            self.transcribe_selected_button.configure(state=tk.NORMAL if can_transcribe else tk.DISABLED)

    def _on_recording_double_click(self, _event=None) -> None:
        self._open_selected_folder()

    def _selected_session(self) -> Path | None:
        selected = self.recordings_tree.selection()
        if not selected:
            return None
        return Path(selected[0])

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
        return any(normalized_name.startswith(prefix) for prefix in SYSTEM_MICROPHONE_LABEL_PREFIXES)

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

    def _resolve_desktop_loopback(self):
        speaker = sc.default_speaker()
        if speaker is None:
            raise RuntimeError(self._tr("err_no_output_device"))

        speaker_id = getattr(speaker, "id", None)
        if not speaker_id:
            raise RuntimeError(self._tr("err_no_output_device_id"))

        try:
            loopback = sc.get_microphone(id=str(speaker_id), include_loopback=True)
        except Exception as exc:
            raise RuntimeError(self._tr("err_open_loopback_failed", error=exc)) from exc

        if loopback is None:
            raise RuntimeError(self._tr("err_loopback_not_found"))
        return speaker, loopback

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
            desktop_source=speaker.name,
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
        self._log_event(self._tr("log_desktop_source", desktop=speaker.name))
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
                vad="on" if options.vad_filter else "off",
                compute=options.compute_type,
            )
        )

        self.transcription_thread = threading.Thread(
            target=self._transcribe_worker,
            args=(session_dir, options),
            daemon=True,
        )
        self.transcription_thread.start()

    def _transcribe_worker(self, session_dir: Path, options: TranscriptionOptions) -> None:
        try:
            out_path = self.transcriber.transcribe_session(
                session_dir=session_dir,
                progress_cb=lambda text, pct: self.event_queue.put(("progress", text, pct)),
                cancel_event=self.cancel_transcription_event,
                options=options,
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
        if is_recording:
            self.record_button.configure(
                text=self._tr("record_stop"),
                bg="#b23b3b",
                activebackground="#8c2f2f",
            )
            return
        self.record_button.configure(
            text=self._tr("record_start"),
            bg="#1f8b4c",
            activebackground="#176a38",
        )

    def _set_recording_ui_state(self, is_recording: bool) -> None:
        self._set_record_button_style(is_recording)
        self.record_button.configure(state=tk.NORMAL)
        self.mic_combo.configure(state=tk.DISABLED if is_recording else "readonly")
        self.ui_language_combo.configure(state=tk.DISABLED if is_recording else "readonly")
        self.refresh_mic_button.configure(state=tk.DISABLED if is_recording else tk.NORMAL)

    def _set_transcription_ui_state(self, is_running: bool) -> None:
        if is_running:
            self.transcribe_selected_button.configure(state=tk.DISABLED)
        else:
            session = self._selected_session()
            can_transcribe = bool(session and self._session_audio_ready(session))
            self.transcribe_selected_button.configure(state=tk.NORMAL if can_transcribe else tk.DISABLED)
        self.self_label_entry.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.other_label_entry.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.auto_transcribe_check.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.model_combo.configure(state=tk.DISABLED if is_running else "readonly")
        self.language_combo.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.beam_spinbox.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.vad_filter_check.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.compute_type_combo.configure(state=tk.DISABLED if is_running else "readonly")
        self.include_timestamps_check.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        self.cancel_transcribe_button.configure(state=tk.NORMAL if is_running else tk.DISABLED)
        session = self._selected_session()
        self.refresh_recordings_button.configure(state=tk.NORMAL)
        self.open_folder_button.configure(state=tk.NORMAL if session else tk.DISABLED)

    def _set_status(self, text: str) -> None:
        self.status_text = text
        self._update_status_line()

    def _update_status_line(self) -> None:
        self.status_label.configure(text=f"{self._tr('status_prefix')}: {self.status_text}")

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
        self.destroy()
