from __future__ import annotations

import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox

from ..config import DEFAULT_OTHER_LABEL, DEFAULT_SELF_LABEL
from ..models import TranscriptionOptions
from ..settings import (
    ALLOWED_COMPUTE_TYPES,
    DEFAULT_BEAM_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_LANGUAGE,
    save_app_settings,
)
from ..transcription_backends import (
    ALLOWED_DEVICES,
    DEFAULT_DEVICE,
    DEFAULT_TRANSCRIPTION_LIBRARY,
    TRANSCRIPTION_LIBRARY_FASTER,
    check_transcription_library_available,
    default_model_for_library,
    normalize_transcription_library,
    refresh_transcription_backend_caches,
    transcription_library_pip_package,
    transcription_models_for_library,
)


class SettingsMixin:
    def _initialize_transcription_settings_ui(self) -> None:
        library = normalize_transcription_library(self.transcription_library_var.get())
        if self.transcription_library_var.get() != library:
            self.transcription_library_var.set(library)
        self._last_valid_transcription_library = library
        self._refresh_model_choices_for_library(library)
        self._sync_transcription_settings_ui()

    def _schedule_settings_save(self, *_args) -> None:
        if self.settings_save_after_id:
            self.after_cancel(self.settings_save_after_id)
        self.settings_save_after_id = self.after(250, self._save_app_settings)

    def _refresh_model_choices_for_library(self, library: str) -> tuple[str, ...]:
        models = transcription_models_for_library(library)
        self.model_combo.configure(values=list(models))
        model_name = self.model_var.get().strip()
        if model_name not in models:
            model_name = default_model_for_library(library, models)
            self.model_var.set(model_name)
        return models

    def _sync_transcription_settings_ui(self) -> None:
        library = normalize_transcription_library(self.transcription_library_var.get())
        if self.transcription_thread and self.transcription_thread.is_alive():
            return
        compute_state = "readonly" if library == TRANSCRIPTION_LIBRARY_FASTER else tk.DISABLED
        self.compute_type_combo.configure(state=compute_state)

    def _ensure_transcription_library_ready(self, interactive: bool) -> bool:
        library = normalize_transcription_library(self.transcription_library_var.get())
        refresh_transcription_backend_caches()
        is_available, error = check_transcription_library_available(library)
        if is_available:
            return True
        self._log_event(
            self._tr("log_transcription_library_missing", library=library, error=error or "unknown")
        )
        if not interactive:
            return False

        package_name = transcription_library_pip_package(library)
        install_confirmed = messagebox.askyesno(
            self._tr("title_missing_library"),
            self._tr(
                "msg_missing_library_install",
                library=library,
                package=package_name,
                error=error or "unknown",
            ),
        )
        if not install_confirmed:
            return False
        return self._install_transcription_library(library)

    def _install_transcription_library(self, library: str) -> bool:
        package_name = transcription_library_pip_package(library)
        self._log_event(
            self._tr(
                "log_transcription_library_install_started",
                library=library,
                package=package_name,
            )
        )
        self._set_status(self._tr("status_installing_library", library=library))
        self.update_idletasks()

        commands: list[list[str]] = []
        uv_executable = shutil.which("uv")
        if uv_executable:
            commands.append(
                [
                    uv_executable,
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    package_name,
                ]
            )
        commands.append([sys.executable, "-m", "pip", "install", package_name])

        install_error_output = ""
        is_installed = False
        for command in commands:
            result = subprocess.run(command, capture_output=True, text=True)
            output_parts = [part.strip() for part in (result.stdout, result.stderr) if part and part.strip()]
            output = "\n".join(output_parts) or f"install exit code: {result.returncode}"
            if result.returncode == 0:
                is_installed = True
                break
            rendered_command = subprocess.list2cmdline(command)
            install_error_output = (
                f"{install_error_output}\n\n"
                if install_error_output
                else ""
            ) + f"$ {rendered_command}\n{output}"

        if not is_installed:
            self._log_event(
                self._tr(
                    "log_transcription_library_install_failed",
                    library=library,
                    error=install_error_output,
                )
            )
            messagebox.showerror(
                self._tr("title_error"),
                self._tr(
                    "msg_library_install_failed",
                    library=library,
                    package=package_name,
                    error=install_error_output,
                ),
            )
            self._set_status(self._tr("status_transcription_error"))
            return False

        refresh_transcription_backend_caches()
        is_available, error = check_transcription_library_available(library)
        if not is_available:
            self._log_event(
                self._tr(
                    "log_transcription_library_install_failed",
                    library=library,
                    error=error or "validation failed",
                )
            )
            messagebox.showerror(
                self._tr("title_error"),
                self._tr(
                    "msg_library_install_failed",
                    library=library,
                    package=package_name,
                    error=error or "validation failed",
                ),
            )
            self._set_status(self._tr("status_transcription_error"))
            return False

        self._log_event(
            self._tr(
                "log_transcription_library_install_ok",
                library=library,
                package=package_name,
            )
        )
        self._set_status(self._tr("status_idle"))
        return True

    def _on_transcription_library_selected(self, _event=None) -> None:
        requested = normalize_transcription_library(self.transcription_library_var.get())
        previous = getattr(self, "_last_valid_transcription_library", DEFAULT_TRANSCRIPTION_LIBRARY)
        if requested == previous:
            self._refresh_model_choices_for_library(requested)
            self._sync_transcription_settings_ui()
            return

        if not self._ensure_transcription_library_ready(interactive=True):
            self.transcription_library_var.set(previous)
            self._refresh_model_choices_for_library(previous)
            self._sync_transcription_settings_ui()
            return

        self._last_valid_transcription_library = requested
        self._refresh_model_choices_for_library(requested)
        self._sync_transcription_settings_ui()

    def _current_speaker_labels(self) -> tuple[str, str]:
        self_label = " ".join(self.self_label_var.get().split()) or DEFAULT_SELF_LABEL
        other_label = " ".join(self.other_label_var.get().split()) or DEFAULT_OTHER_LABEL
        if self.self_label_var.get() != self_label:
            self.self_label_var.set(self_label)
        if self.other_label_var.get() != other_label:
            self.other_label_var.set(other_label)
        return self_label, other_label

    def _current_transcription_options(self) -> TranscriptionOptions:
        transcription_library = normalize_transcription_library(self.transcription_library_var.get())
        if self.transcription_library_var.get() != transcription_library:
            self.transcription_library_var.set(transcription_library)
        models = self._refresh_model_choices_for_library(transcription_library)

        model_name = self.model_var.get().strip()
        if model_name not in models:
            model_name = default_model_for_library(transcription_library, models)
            self.model_var.set(model_name)

        device = str(self.device_var.get()).strip().lower()
        if device not in ALLOWED_DEVICES:
            device = DEFAULT_DEVICE
            self.device_var.set(device)

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

        self_label, other_label = self._current_speaker_labels()
        return TranscriptionOptions(
            transcription_library=transcription_library,
            model_name=model_name,
            device=device,
            language=language,
            speaker_self=self_label,
            speaker_other=other_label,
            beam_size=beam_size,
            vad_filter=bool(self.vad_filter_var.get()),
            compute_type=compute_type,
            include_timestamps=bool(self.include_timestamps_var.get()),
            transcribe_mix_track=bool(self.transcribe_mix_track_var.get()),
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
            "transcription_library": options.transcription_library,
            "whisper_model": options.model_name,
            "whisper_device": options.device,
            "whisper_language": options.language,
            "whisper_beam_size": options.beam_size,
            "whisper_vad_filter": options.vad_filter,
            "whisper_compute_type": options.compute_type,
            "include_timestamps": options.include_timestamps,
            "transcribe_mix_track": options.transcribe_mix_track,
            "ui_language": self.ui_language,
        }
        save_app_settings(self.app_settings)
