from __future__ import annotations

import ctypes
import math
import queue
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Callable

import soundfile as sf
import sounddevice as sd

from .config import (
    AUDIO_WRITE_FLUSH_INTERVAL_SEC,
    BLOCK_FRAMES,
    DESKTOP_FILE_NAME,
    MIC_FILE_NAME,
    MIX_FILE_NAME,
    SAMPLE_RATE,
)
from .models import RecordingError
from .utils import to_mono


def _run_in_audio_thread(callback, *args) -> None:
    """Run audio work with COM initialized on the current Windows thread."""
    com_initialized = False
    if sys.platform.startswith("win"):
        ole32 = ctypes.windll.ole32
        ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        ole32.CoInitializeEx.restype = ctypes.c_long
        # COINIT_MULTITHREADED. S_OK and S_FALSE both require CoUninitialize.
        result = int(ole32.CoInitializeEx(None, 0))
        com_initialized = result in (0, 1)
    try:
        callback(*args)
    finally:
        if com_initialized:
            ctypes.windll.ole32.CoUninitialize()


class DualTrackRecorder:
    def __init__(
        self,
        mic,
        desktop,
        session_dir: Path,
        level_callback: Callable[[str, float], None] | None = None,
        error_callback: Callable[[str], None] | None = None,
    ):
        self.mic = mic
        self.desktop = desktop
        self.session_dir = session_dir
        self.level_callback = level_callback
        self.error_callback = error_callback
        self.stop_event = threading.Event()
        self.error_queue: queue.Queue[Exception] = queue.Queue()
        self.capture_threads: list[threading.Thread] = []
        self.mix_thread: threading.Thread | None = None
        self.mix_queues: dict[str, queue.Queue] = {}
        self.startup_events: dict[str, threading.Event] = {}

    @property
    def mic_path(self) -> Path:
        return self.session_dir / MIC_FILE_NAME

    @property
    def desktop_path(self) -> Path:
        return self.session_dir / DESKTOP_FILE_NAME

    @property
    def mix_path(self) -> Path:
        return self.session_dir / MIX_FILE_NAME

    def start(self) -> None:
        self.stop_event.clear()
        self.error_queue = queue.Queue()
        self.mix_queues = {
            "microphone": queue.Queue(),
            "desktop": queue.Queue(),
        }
        self.startup_events = {
            "microphone": threading.Event(),
            "desktop": threading.Event(),
        }
        self.capture_threads = [
            threading.Thread(
                target=_run_in_audio_thread,
                args=(self._capture_loop, self.mic, self.mic_path, "microphone"),
                daemon=True,
            ),
            threading.Thread(
                target=_run_in_audio_thread,
                args=(self._capture_loop, self.desktop, self.desktop_path, "desktop"),
                daemon=True,
            ),
        ]
        self.mix_thread = threading.Thread(target=self._mix_loop, daemon=True)

        for thread in self.capture_threads:
            thread.start()
        self.mix_thread.start()

        # Opening a WASAPI device happens inside the capture threads. Wait for both
        # streams to open so an unavailable device is reported before the UI claims
        # that recording has started.
        deadline = time.monotonic() + 3.0
        for startup_event in self.startup_events.values():
            startup_event.wait(max(0.0, deadline - time.monotonic()))
        try:
            self.raise_if_failed()
        except RecordingError:
            self.stop()
            raise
        if not all(event.is_set() for event in self.startup_events.values()):
            self.stop()
            raise RecordingError("Timed out while opening audio capture devices")

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.capture_threads:
            thread.join()
        if self.mix_thread is not None:
            self.mix_thread.join()

    def raise_if_failed(self) -> None:
        errors: list[str] = []
        while not self.error_queue.empty():
            errors.append(str(self.error_queue.get_nowait()))
        if errors:
            raise RecordingError("\n".join(errors))

    def _capture_loop(self, source, out_path: Path, source_name: str) -> None:
        mix_queue = self.mix_queues[source_name]
        try:
            with source.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCK_FRAMES) as rec:
                with sf.SoundFile(
                    str(out_path),
                    mode="w",
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    format="OGG",
                    subtype="VORBIS",
                ) as sound_file:
                    self.startup_events[source_name].set()
                    last_level_emit_at = 0.0
                    last_flush_at = time.monotonic()
                    while not self.stop_event.is_set():
                        chunk = rec.record(numframes=BLOCK_FRAMES)
                        mono = to_mono(chunk)
                        sound_file.write(mono)
                        mix_queue.put(mono.copy())
                        now = time.monotonic()
                        if now - last_flush_at >= AUDIO_WRITE_FLUSH_INTERVAL_SEC:
                            sound_file.flush()
                            last_flush_at = now
                        if self.level_callback:
                            if now - last_level_emit_at >= 0.12:
                                level = self._estimate_level(mono)
                                self.level_callback(source_name, level)
                                last_level_emit_at = now
        except Exception as exc:  # pragma: no cover - device-specific failures
            self.stop_event.set()
            self.startup_events[source_name].set()
            detail = str(exc).strip() or type(exc).__name__
            error = RuntimeError(f"Capture error ({source_name}): {detail}")
            self.error_queue.put(error)
            if self.error_callback:
                self.error_callback(str(error))
        finally:
            mix_queue.put(None)

    def _mix_loop(self) -> None:
        mic_queue = self.mix_queues["microphone"]
        desktop_queue = self.mix_queues["desktop"]
        try:
            with sf.SoundFile(
                str(self.mix_path),
                mode="w",
                samplerate=SAMPLE_RATE,
                channels=1,
                format="OGG",
                subtype="VORBIS",
            ) as out_file:
                last_flush_at = time.monotonic()
                while True:
                    mic_chunk = mic_queue.get()
                    desktop_chunk = desktop_queue.get()
                    if mic_chunk is None or desktop_chunk is None:
                        break

                    frames = min(len(mic_chunk), len(desktop_chunk))
                    if frames <= 0:
                        continue

                    mix = (mic_chunk[:frames] + desktop_chunk[:frames]) * 0.5
                    # Hard-clip overs to keep export stable and avoid NaN/inf propagation.
                    mix = mix.clip(-1.0, 1.0)
                    out_file.write(mix)

                    now = time.monotonic()
                    if now - last_flush_at >= AUDIO_WRITE_FLUSH_INTERVAL_SEC:
                        out_file.flush()
                        last_flush_at = now
        except Exception as exc:  # pragma: no cover - codec/filesystem failures
            self.stop_event.set()
            self.error_queue.put(RuntimeError(f"Mix render error: {exc}"))

    @staticmethod
    def _estimate_level(samples) -> float:
        if len(samples) == 0:
            return 0.0
        # Soundcard usually returns float samples in [-1, 1], so RMS gives a stable activity meter.
        squared = samples * samples
        rms = math.sqrt(float(squared.mean()))
        boosted = min(1.0, rms * 8.0)
        return boosted


class SoundDeviceMicrophone:
    """SoundCard-compatible microphone backed by PortAudio.

    SoundCard's Windows backend assumes WAVE_FORMAT_EXTENSIBLE. Some perfectly
    valid mono USB headsets expose plain WAVE_FORMAT_PCM and trip that assertion.
    PortAudio handles both formats, while SoundCard remains useful for WASAPI
    loopback capture.
    """

    def __init__(self, name: str):
        self.name = name
        self.id = self._find_device(name)

    @staticmethod
    def _normalized_name(value: str) -> str:
        return " ".join(str(value).lower().replace("microphone", "").split())

    @classmethod
    def _find_device(cls, name: str) -> int:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        target = cls._normalized_name(name)
        candidates: list[tuple[int, int]] = []
        for index, device in enumerate(devices):
            if int(device["max_input_channels"]) < 1:
                continue
            candidate = cls._normalized_name(device["name"])
            score = 0
            if candidate == target:
                score += 100
            elif candidate in target or target in candidate:
                score += 50
            host_name = str(hostapis[int(device["hostapi"])]["name"])
            if "WASAPI" in host_name:
                score += 20
            if score:
                candidates.append((score, index))
        if not candidates:
            raise RuntimeError(f'PortAudio could not match microphone "{name}"')
        return max(candidates)[1]

    def recorder(self, samplerate: int, blocksize: int):
        return _SoundDeviceRecorder(self.id, samplerate, blocksize)


class _SoundDeviceRecorder:
    def __init__(self, device: int, samplerate: int, blocksize: int):
        self.stream = sd.InputStream(
            device=device,
            samplerate=samplerate,
            blocksize=blocksize,
            channels=1,
            dtype="float32",
        )

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stream.close()

    def record(self, numframes: int):
        data, overflowed = self.stream.read(numframes)
        if overflowed:
            warnings.warn("audio input overflow", RuntimeWarning, stacklevel=2)
        return data


class DualTrackLevelMonitor:
    def __init__(
        self,
        mic,
        desktop,
        level_callback: Callable[[str, float], None] | None = None,
        error_callback: Callable[[str, str], None] | None = None,
    ):
        self.mic = mic
        self.desktop = desktop
        self.level_callback = level_callback
        self.error_callback = error_callback
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        self.stop_event.clear()
        self.threads = [
            threading.Thread(
                target=_run_in_audio_thread,
                args=(self._monitor_loop, self.mic, "microphone"),
                daemon=True,
            ),
            threading.Thread(
                target=_run_in_audio_thread,
                args=(self._monitor_loop, self.desktop, "desktop"),
                daemon=True,
            ),
        ]
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join()

    def _monitor_loop(self, source, source_name: str) -> None:
        try:
            # Metering is best-effort: ignore occasional MediaFoundation discontinuity warnings.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="data discontinuity in recording",
                    category=RuntimeWarning,
                    module=r"soundcard\.mediafoundation",
                )
                with source.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCK_FRAMES) as rec:
                    last_level_emit_at = 0.0
                    while not self.stop_event.is_set():
                        chunk = rec.record(numframes=BLOCK_FRAMES)
                        mono = to_mono(chunk)
                        if self.level_callback:
                            now = time.monotonic()
                            if now - last_level_emit_at >= 0.12:
                                level = DualTrackRecorder._estimate_level(mono)
                                self.level_callback(source_name, level)
                                last_level_emit_at = now
        except Exception as exc:  # pragma: no cover - device-specific failures
            self.stop_event.set()
            if self.error_callback:
                self.error_callback(source_name, str(exc))
