from __future__ import annotations

import sys
import warnings

import soundcard as sc
from tkinter import messagebox

from ..localization import system_microphone_label_prefixes
from ..recording import DualTrackLevelMonitor, SoundDeviceMicrophone

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


class AudioMixin:
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
        if 0 <= mic_idx < len(self.microphones):
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
        if 0 <= mic_idx < len(self.microphones):
            return self.microphones[mic_idx]
        return None

    @staticmethod
    def _default_microphone():
        try:
            return sc.default_microphone()
        except Exception:
            return None

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

                if loopback is not None and self._desktop_source_score(loopback, preferred_speaker_name) > 0:
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

        selected_mic = self._resolve_selected_microphone()
        if selected_mic is None:
            self._set_levels_to_zero()
            return

        try:
            mic = self._recording_microphone(selected_mic)
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

    @staticmethod
    def _recording_microphone(mic):
        if mic is None or not sys.platform.startswith("win"):
            return mic
        return SoundDeviceMicrophone(str(getattr(mic, "name", "")))

    def _stop_idle_level_monitor(self) -> None:
        if self.level_monitor is None:
            return
        monitor = self.level_monitor
        self.level_monitor = None
        monitor.stop()
