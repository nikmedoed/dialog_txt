from __future__ import annotations

import tkinter as tk

from ..config import DEFAULT_OTHER_LABEL, DEFAULT_SELF_LABEL
from ..models import TranscriptionOptions
from ..settings import (
    ALLOWED_COMPUTE_TYPES,
    ALLOWED_MODELS,
    DEFAULT_BEAM_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    save_app_settings,
)


class SettingsMixin:
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

        return TranscriptionOptions(
            model_name=model_name,
            language=language,
            beam_size=beam_size,
            vad_filter=bool(self.vad_filter_var.get()),
            compute_type=compute_type,
            include_timestamps=bool(self.include_timestamps_var.get()),
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
