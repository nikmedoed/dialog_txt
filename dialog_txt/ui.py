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
from .recording import DualTrackRecorder
from .settings import load_app_settings, save_app_settings
from .settings import (
    ALLOWED_COMPUTE_TYPES,
    ALLOWED_MODELS,
    DEFAULT_AUTO_TRANSCRIBE_AFTER_RECORD,
    DEFAULT_BEAM_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_VAD_FILTER,
)
from .storage import (
    create_session_dir,
    discover_sessions,
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


class App(tk.Tk):
    SYSTEM_MICROPHONE_SETTING = "__system_default__"
    SYSTEM_MICROPHONE_LABEL = "Системный"

    def __init__(self):
        super().__init__()
        self.title("Dialog TXT Recorder")
        self.geometry("1120x760")
        self.minsize(900, 600)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        ensure_recordings_root()

        self.transcriber = WhisperTranscriber()
        self.recorder: DualTrackRecorder | None = None
        self.recording_started_at: float | None = None
        self.active_session_dir: Path | None = None

        self.event_queue: queue.Queue = queue.Queue()
        self.cancel_transcription_event = threading.Event()
        self.transcription_thread: threading.Thread | None = None

        self.level_values = {"microphone": 0.0, "desktop": 0.0}
        self.level_last_seen = {"microphone": 0.0, "desktop": 0.0}
        self.last_progress_log_bucket = -1
        self.settings_save_after_id: str | None = None
        self.app_settings = load_app_settings()

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
        self.after(150, self._poll_events)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("Treeview", rowheight=24)

        top = ttk.Frame(self, padding=12)
        top.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(top, text="Запись", padding=10)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Микрофон:").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        self.mic_combo = ttk.Combobox(controls, state="readonly", width=62)
        self.mic_combo.grid(row=0, column=1, sticky=tk.EW)
        self.mic_combo.bind("<<ComboboxSelected>>", self._on_mic_selected)
        controls.columnconfigure(1, weight=1)

        self.refresh_mic_button = ttk.Button(
            controls, text="Обновить", command=self._refresh_microphones
        )
        self.refresh_mic_button.grid(row=0, column=2, padx=(8, 0))

        self.start_button = ttk.Button(controls, text="Начать запись", command=self._start_recording)
        self.start_button.grid(row=1, column=0, pady=(10, 0), sticky=tk.W)

        self.stop_button = ttk.Button(
            controls, text="Остановить запись", command=self._stop_recording, state=tk.DISABLED
        )
        self.stop_button.grid(row=1, column=1, pady=(10, 0), sticky=tk.W)

        self.recording_time_label = ttk.Label(controls, text="Длительность: 00:00:00")
        self.recording_time_label.grid(row=1, column=2, pady=(10, 0), sticky=tk.E)

        self.status_label = ttk.Label(controls, text="Статус: ожидание")
        self.status_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))

        ttk.Label(controls, text="Уровень mic:").grid(row=3, column=0, sticky=tk.W, pady=(8, 0))
        self.mic_level = ttk.Progressbar(controls, mode="determinate", maximum=100)
        self.mic_level.grid(row=3, column=1, sticky=tk.EW, pady=(8, 0))
        self.mic_level_value = ttk.Label(controls, text="0%")
        self.mic_level_value.grid(row=3, column=2, sticky=tk.E, pady=(8, 0))

        ttk.Label(controls, text="Уровень desktop:").grid(row=4, column=0, sticky=tk.W, pady=(4, 0))
        self.desktop_level = ttk.Progressbar(controls, mode="determinate", maximum=100)
        self.desktop_level.grid(row=4, column=1, sticky=tk.EW, pady=(4, 0))
        self.desktop_level_value = ttk.Label(controls, text="0%")
        self.desktop_level_value.grid(row=4, column=2, sticky=tk.E, pady=(4, 0))

        self.signal_label = ttk.Label(controls, text="Сигнал: запись не идёт")
        self.signal_label.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))

        ttk.Label(controls, text="Подпись (вы):").grid(row=6, column=0, sticky=tk.W, pady=(8, 0))
        self.self_label_entry = ttk.Entry(controls, textvariable=self.self_label_var)
        self.self_label_entry.grid(row=6, column=1, sticky=tk.EW, pady=(8, 0))

        ttk.Label(controls, text="Подпись (собеседник):").grid(row=7, column=0, sticky=tk.W, pady=(4, 0))
        self.other_label_entry = ttk.Entry(controls, textvariable=self.other_label_var)
        self.other_label_entry.grid(row=7, column=1, sticky=tk.EW, pady=(4, 0))

        self.auto_transcribe_check = ttk.Checkbutton(
            controls,
            text="Автотранскрибация после остановки записи",
            variable=self.auto_transcribe_var,
        )
        self.auto_transcribe_check.grid(row=8, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))

        transcribe_box = ttk.LabelFrame(top, text="Транскрибация", padding=10)
        transcribe_box.pack(fill=tk.X, pady=(12, 0))

        self.transcribe_selected_button = ttk.Button(
            transcribe_box, text="Транскрибировать выбранную запись", command=self._transcribe_selected
        )
        self.transcribe_selected_button.grid(row=0, column=0, sticky=tk.W)

        self.cancel_transcribe_button = ttk.Button(
            transcribe_box,
            text="Отменить транскрибацию",
            command=self._cancel_transcription,
            state=tk.DISABLED,
        )
        self.cancel_transcribe_button.grid(row=0, column=1, padx=(8, 0), sticky=tk.W)

        self.refresh_recordings_button = ttk.Button(
            transcribe_box, text="Обновить список", command=self._refresh_recordings
        )
        self.refresh_recordings_button.grid(row=0, column=2, padx=(8, 0), sticky=tk.W)

        self.open_folder_button = ttk.Button(
            transcribe_box, text="Открыть папку записи", command=self._open_selected_folder
        )
        self.open_folder_button.grid(row=0, column=3, padx=(8, 0), sticky=tk.W)

        self.progress = ttk.Progressbar(transcribe_box, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=(10, 0))
        transcribe_box.columnconfigure(0, weight=1)

        self.progress_label = ttk.Label(transcribe_box, text="Прогресс: 0%")
        self.progress_label.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(8, 0))

        whisper_box = ttk.LabelFrame(top, text="Настройки Whisper", padding=10)
        whisper_box.pack(fill=tk.X, pady=(12, 0))

        ttk.Label(whisper_box, text="Модель:").grid(row=0, column=0, sticky=tk.W)
        self.model_combo = ttk.Combobox(
            whisper_box,
            state="readonly",
            width=18,
            values=list(ALLOWED_MODELS),
            textvariable=self.model_var,
        )
        self.model_combo.grid(row=0, column=1, sticky=tk.W, padx=(6, 12))

        ttk.Label(whisper_box, text="Язык:").grid(row=0, column=2, sticky=tk.W)
        self.language_combo = ttk.Combobox(
            whisper_box,
            width=10,
            values=["ru", "en", "auto"],
            textvariable=self.language_var,
        )
        self.language_combo.grid(row=0, column=3, sticky=tk.W, padx=(6, 12))

        ttk.Label(whisper_box, text="Beam:").grid(row=0, column=4, sticky=tk.W)
        self.beam_spinbox = ttk.Spinbox(
            whisper_box,
            from_=1,
            to=10,
            width=5,
            textvariable=self.beam_size_var,
        )
        self.beam_spinbox.grid(row=0, column=5, sticky=tk.W, padx=(6, 12))

        ttk.Label(whisper_box, text="Compute:").grid(row=0, column=6, sticky=tk.W)
        self.compute_type_combo = ttk.Combobox(
            whisper_box,
            state="readonly",
            width=12,
            values=list(ALLOWED_COMPUTE_TYPES),
            textvariable=self.compute_type_var,
        )
        self.compute_type_combo.grid(row=0, column=7, sticky=tk.W, padx=(6, 12))

        self.vad_filter_check = ttk.Checkbutton(
            whisper_box,
            text="VAD фильтр",
            variable=self.vad_filter_var,
        )
        self.vad_filter_check.grid(row=0, column=8, sticky=tk.W)

        self.include_timestamps_check = ttk.Checkbutton(
            whisper_box,
            text="Добавлять таймметки в transcript.txt",
            variable=self.include_timestamps_var,
        )
        self.include_timestamps_check.grid(row=1, column=0, columnspan=9, sticky=tk.W, pady=(8, 0))

        recordings_box = ttk.LabelFrame(top, text="Существующие записи", padding=10)
        recordings_box.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        columns = ("session", "audio", "txt", "path")
        self.recordings_tree = ttk.Treeview(recordings_box, columns=columns, show="headings")
        self.recordings_tree.heading("session", text="Сессия")
        self.recordings_tree.heading("audio", text="Аудио")
        self.recordings_tree.heading("txt", text="TXT")
        self.recordings_tree.heading("path", text="Папка")
        self.recordings_tree.column("session", width=190, anchor=tk.W)
        self.recordings_tree.column("audio", width=120, anchor=tk.CENTER)
        self.recordings_tree.column("txt", width=80, anchor=tk.CENTER)
        self.recordings_tree.column("path", width=640, anchor=tk.W)
        self.recordings_tree.pack(fill=tk.BOTH, expand=True)
        self.recordings_tree.bind("<<TreeviewSelect>>", self._on_recording_selected)
        self.recordings_tree.bind("<Double-1>", self._on_recording_double_click)

        log_box = ttk.LabelFrame(top, text="Лог событий", padding=10)
        log_box.pack(fill=tk.BOTH, expand=False, pady=(12, 0))
        self.log_text = ScrolledText(log_box, height=9, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _refresh_microphones(self) -> None:
        try:
            mics = sc.all_microphones(include_loopback=False)
        except Exception as exc:  # pragma: no cover - hardware-specific
            messagebox.showerror("Ошибка", f"Не удалось получить список микрофонов:\n{exc}")
            self._log_event(f"Ошибка списка микрофонов: {exc}")
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
            self._set_status(f"Микрофонов найдено: {len(self.microphones)}")
            self._log_event(f"Список микрофонов обновлён: {len(self.microphones)} устройств")
            self._save_app_settings()
        else:
            self._set_status("Микрофоны не найдены")
            self._log_event("Микрофоны не найдены")

    def _refresh_recordings(self) -> None:
        for row in self.recordings_tree.get_children():
            self.recordings_tree.delete(row)

        sessions = discover_sessions()
        for session_dir in sessions:
            audio_status, txt_status = self._session_status(session_dir)
            display_name = session_title(session_dir)
            self.recordings_tree.insert(
                "",
                tk.END,
                iid=str(session_dir),
                values=(display_name, audio_status, txt_status, str(session_dir)),
            )
        if sessions and not self.recordings_tree.selection():
            self.recordings_tree.selection_set(str(sessions[0]))
        self._log_event(f"Список записей обновлён: {len(sessions)} сессий")
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
            self.transcribe_selected_button.configure(text="Перетранскрибировать выбранную запись")
        elif can_transcribe:
            self.transcribe_selected_button.configure(text="Транскрибировать выбранную запись")
        else:
            self.transcribe_selected_button.configure(text="Нельзя: нет обеих дорожек")
        if not (self.transcription_thread and self.transcription_thread.is_alive()):
            self.transcribe_selected_button.configure(state=tk.NORMAL if can_transcribe else tk.DISABLED)
            self.open_folder_button.configure(state=tk.NORMAL)

    def _on_recording_double_click(self, _event=None) -> None:
        self._open_selected_folder()

    def _selected_session(self) -> Path | None:
        selected = self.recordings_tree.selection()
        if not selected:
            return None
        return Path(selected[0])

    def _on_mic_selected(self, _event=None) -> None:
        self._save_app_settings()

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
        if not default_name:
            return f"{self.SYSTEM_MICROPHONE_LABEL} (не определён)"
        return f"{self.SYSTEM_MICROPHONE_LABEL} ({default_name})"

    def _is_system_microphone_selection(self, selected_name: str) -> bool:
        return selected_name == self.SYSTEM_MICROPHONE_SETTING or selected_name.startswith(
            f"{self.SYSTEM_MICROPHONE_LABEL} ("
        )

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
        }
        save_app_settings(self.app_settings)

    def _open_selected_folder(self) -> None:
        session = self._selected_session()
        if not session:
            messagebox.showwarning("Открыть папку", "Выберите запись из списка.")
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
            self._log_event(f"Открыта папка: {path}")
        except Exception as exc:
            self._log_event(f"Ошибка открытия папки {path}: {exc}")
            messagebox.showerror("Открыть папку", f"Не удалось открыть папку:\n{path}\n\n{exc}")

    def _resolve_desktop_loopback(self):
        speaker = sc.default_speaker()
        if speaker is None:
            raise RuntimeError("Не найдено устройство вывода для loopback-записи.")

        speaker_id = getattr(speaker, "id", None)
        if not speaker_id:
            raise RuntimeError("У устройства вывода отсутствует ID для loopback.")

        try:
            loopback = sc.get_microphone(id=str(speaker_id), include_loopback=True)
        except Exception as exc:
            raise RuntimeError(f"Не удалось открыть loopback-источник: {exc}") from exc

        if loopback is None:
            raise RuntimeError("Loopback-источник для текущего устройства вывода не найден.")
        return speaker, loopback

    def _start_recording(self) -> None:
        if self.recorder is not None:
            return
        if self.transcription_thread and self.transcription_thread.is_alive():
            messagebox.showwarning("Занято", "Сначала дождитесь завершения текущей транскрибации.")
            return

        mic = self._resolve_selected_microphone()
        if mic is None:
            messagebox.showerror("Ошибка", "Выберите микрофон.")
            return

        try:
            speaker, desktop_loopback = self._resolve_desktop_loopback()
        except Exception as exc:  # pragma: no cover - hardware-specific
            messagebox.showerror("Ошибка", f"Не удалось получить устройство рабочего стола:\n{exc}")
            self._log_event(f"Ошибка desktop-захвата: {exc}")
            return

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
        self._set_status(f"Идёт запись: {session_dir.name}")
        self._log_event(f"Старт записи: {session_dir}")
        self._log_event(f"Mic: {mic.name}")
        self._log_event(f"Desktop (loopback): {speaker.name}")
        self._log_event(f"Подписи: [{self_label}] / [{other_label}]")
        self._tick_recording_timer()

    def _stop_recording(self, auto_transcribe: bool | None = None) -> None:
        if self.recorder is None:
            return

        if auto_transcribe is None:
            auto_transcribe = bool(self.auto_transcribe_var.get())

        recorder = self.recorder
        self.recorder = None
        recorder.stop()

        try:
            recorder.raise_if_failed()
        except RecordingError as exc:
            self._set_recording_ui_state(is_recording=False)
            self.recording_started_at = None
            self.active_session_dir = None
            self._set_levels_to_zero()
            self._set_status("Ошибка записи")
            self._refresh_recordings()
            self._log_event(f"Ошибка записи: {exc}")
            messagebox.showerror("Ошибка записи", str(exc))
            return

        duration = 0
        if self.recording_started_at is not None:
            duration = int(time.time() - self.recording_started_at)

        self.recording_started_at = None
        self._set_recording_ui_state(is_recording=False)
        self._set_levels_to_zero()
        self.recording_time_label.configure(text=f"Длительность: {format_seconds(duration)}")

        if self.active_session_dir:
            update_session_metadata(self.active_session_dir, duration)

        if auto_transcribe:
            self._set_status("Запись остановлена. Запускаю транскрибацию...")
            self._log_event(f"Запись остановлена ({format_seconds(duration)}), запускаю транскрибацию")
        else:
            self._set_status("Запись остановлена. Автотранскрибация отключена")
            self._log_event(
                f"Запись остановлена ({format_seconds(duration)}), автотранскрибация отключена"
            )
        self._refresh_recordings()

        session_dir = self.active_session_dir
        self.active_session_dir = None
        if session_dir and auto_transcribe:
            self._start_transcription(session_dir)

    def _tick_recording_timer(self) -> None:
        if self.recorder is None or self.recording_started_at is None:
            return
        elapsed = time.time() - self.recording_started_at
        self._decay_levels()
        self._render_levels()
        self.recording_time_label.configure(text=f"Длительность: {format_seconds(elapsed)}")
        self.after(250, self._tick_recording_timer)

    def _transcribe_selected(self) -> None:
        session = self._selected_session()
        if not session:
            messagebox.showwarning("Выбор записи", "Выберите запись из списка.")
            return
        self._start_transcription(session)

    def _start_transcription(self, session_dir: Path) -> None:
        if self.recorder is not None:
            messagebox.showwarning("Занято", "Сначала остановите текущую запись.")
            return
        if self.transcription_thread and self.transcription_thread.is_alive():
            messagebox.showwarning("Занято", "Транскрибация уже выполняется.")
            return

        mic_path, desktop_path = resolve_track_paths(session_dir)
        if mic_path is None or desktop_path is None:
            messagebox.showerror(
                "Ошибка",
                "В выбранной сессии отсутствуют дорожки mic.ogg и desktop.ogg.",
            )
            return

        self.cancel_transcription_event.clear()
        self.progress.configure(value=0)
        self.progress_label.configure(text="Прогресс: 0%")
        self.last_progress_log_bucket = -1
        options = self._current_transcription_options()
        self._set_transcription_ui_state(is_running=True)
        self._set_status(f"Транскрибация: {session_dir.name}")
        self._log_event(
            "Старт транскрибации: "
            f"{session_dir} | model={options.model_name}, lang={options.language}, "
            f"beam={options.beam_size}, vad={'on' if options.vad_filter else 'off'}, "
            f"compute={options.compute_type}"
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
            self._set_status("Отмена транскрибации...")
            self._log_event("Запрошена отмена транскрибации")

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

        if kind == "progress":
            _, message, pct = event
            pct_int = int(max(0, min(100, pct)))
            self.progress.configure(value=pct_int)
            self.progress_label.configure(text=f"Прогресс: {pct_int}%")
            self._set_status(message)
            progress_bucket = pct_int // 10
            if progress_bucket > self.last_progress_log_bucket:
                self.last_progress_log_bucket = progress_bucket
                self._log_event(f"Транскрибация: {pct_int}% ({message})")
            return

        self._set_transcription_ui_state(is_running=False)
        self.progress.configure(value=0)
        self.progress_label.configure(text="Прогресс: 0%")
        self._refresh_recordings()

        if kind == "done":
            _, session_dir, out_path = event
            self._set_status(f"Готово: {session_dir.name}")
            self._log_event(f"Транскрибация завершена: {out_path}")
            self._open_path_in_file_manager(session_dir)
            return
        if kind == "cancelled":
            _, session_dir = event
            self._set_status(f"Транскрибация отменена: {session_dir.name}")
            self._log_event(f"Транскрибация отменена: {session_dir}")
            messagebox.showwarning("Отменено", "Транскрибация была отменена пользователем.")
            return
        if kind == "error":
            _, message = event
            self._set_status("Ошибка транскрибации")
            self._log_event(f"Ошибка транскрибации: {message}")
            messagebox.showerror("Ошибка транскрибации", message)
            return

    def _set_recording_ui_state(self, is_recording: bool) -> None:
        self.start_button.configure(state=tk.DISABLED if is_recording else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if is_recording else tk.DISABLED)
        self.mic_combo.configure(state=tk.DISABLED if is_recording else "readonly")
        self.refresh_mic_button.configure(state=tk.DISABLED if is_recording else tk.NORMAL)
        self.self_label_entry.configure(state=tk.DISABLED if is_recording else tk.NORMAL)
        self.other_label_entry.configure(state=tk.DISABLED if is_recording else tk.NORMAL)
        self.auto_transcribe_check.configure(state=tk.DISABLED if is_recording else tk.NORMAL)
        self.model_combo.configure(state=tk.DISABLED if is_recording else "readonly")
        self.language_combo.configure(state=tk.DISABLED if is_recording else tk.NORMAL)
        self.beam_spinbox.configure(state=tk.DISABLED if is_recording else tk.NORMAL)
        self.vad_filter_check.configure(state=tk.DISABLED if is_recording else tk.NORMAL)
        self.compute_type_combo.configure(state=tk.DISABLED if is_recording else "readonly")
        self.include_timestamps_check.configure(state=tk.DISABLED if is_recording else tk.NORMAL)

    def _set_transcription_ui_state(self, is_running: bool) -> None:
        if is_running:
            self.transcribe_selected_button.configure(state=tk.DISABLED)
            self.open_folder_button.configure(state=tk.DISABLED)
        else:
            session = self._selected_session()
            can_transcribe = bool(session and self._session_audio_ready(session))
            self.transcribe_selected_button.configure(state=tk.NORMAL if can_transcribe else tk.DISABLED)
            self.open_folder_button.configure(state=tk.NORMAL if session else tk.DISABLED)
        self.cancel_transcribe_button.configure(state=tk.NORMAL if is_running else tk.DISABLED)
        self.refresh_recordings_button.configure(state=tk.DISABLED if is_running else tk.NORMAL)

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=f"Статус: {text}")

    def _session_audio_ready(self, session_dir: Path) -> bool:
        mic_path, desktop_path = resolve_track_paths(session_dir)
        return mic_path is not None and desktop_path is not None

    def _session_status(self, session_dir: Path) -> tuple[str, str]:
        mic_path, desktop_path = resolve_track_paths(session_dir)
        has_transcript = transcript_path(session_dir).exists()

        if mic_path is not None and desktop_path is not None:
            audio_status = "OK"
        elif mic_path is None and desktop_path is None:
            audio_status = "нет audio"
        elif mic_path is None:
            audio_status = "нет mic"
        else:
            audio_status = "нет desktop"

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
        self.mic_level_value.configure(text=f"{mic_pct}%")
        self.desktop_level_value.configure(text=f"{desktop_pct}%")

        if self.recorder is None:
            self.signal_label.configure(text="Сигнал: запись не идёт")
            return

        now = time.monotonic()
        mic_alive = now - self.level_last_seen["microphone"] < 1.2 and mic_pct > 1
        desktop_alive = now - self.level_last_seen["desktop"] < 1.2 and desktop_pct > 1
        mic_state = "есть" if mic_alive else "нет"
        desktop_state = "есть" if desktop_alive else "нет"
        self.signal_label.configure(text=f"Сигнал: mic {mic_state}, desktop {desktop_state}")

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
        if self.recorder is not None:
            self._stop_recording(auto_transcribe=False)
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.cancel_transcription_event.set()
            self._log_event("Окно закрывается: отправлен запрос на отмену транскрибации")
        self.destroy()
