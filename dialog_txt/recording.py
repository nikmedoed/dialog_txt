from __future__ import annotations

import math
import queue
import threading
import time
import warnings
from pathlib import Path
from typing import Callable

import soundfile as sf

from .config import BLOCK_FRAMES, DESKTOP_FILE_NAME, MIC_FILE_NAME, MIX_FILE_NAME, SAMPLE_RATE
from .models import RecordingError
from .utils import to_mono


class DualTrackRecorder:
    def __init__(
        self,
        mic,
        desktop,
        session_dir: Path,
        level_callback: Callable[[str, float], None] | None = None,
    ):
        self.mic = mic
        self.desktop = desktop
        self.session_dir = session_dir
        self.level_callback = level_callback
        self.stop_event = threading.Event()
        self.error_queue: queue.Queue[Exception] = queue.Queue()
        self.threads: list[threading.Thread] = []

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
        self.threads = [
            threading.Thread(
                target=self._capture_loop,
                args=(self.mic, self.mic_path, "microphone"),
                daemon=True,
            ),
            threading.Thread(
                target=self._capture_loop,
                args=(self.desktop, self.desktop_path, "desktop"),
                daemon=True,
            ),
        ]
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join()
        if not self.error_queue.empty():
            return
        try:
            self._write_mix()
        except Exception as exc:
            self.error_queue.put(RuntimeError(f"Mix render error: {exc}"))

    def raise_if_failed(self) -> None:
        errors: list[str] = []
        while not self.error_queue.empty():
            errors.append(str(self.error_queue.get_nowait()))
        if errors:
            raise RecordingError("\n".join(errors))

    def _capture_loop(self, source, out_path: Path, source_name: str) -> None:
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
                    last_level_emit_at = 0.0
                    while not self.stop_event.is_set():
                        chunk = rec.record(numframes=BLOCK_FRAMES)
                        mono = to_mono(chunk)
                        sound_file.write(mono)
                        if self.level_callback:
                            now = time.monotonic()
                            if now - last_level_emit_at >= 0.12:
                                level = self._estimate_level(mono)
                                self.level_callback(source_name, level)
                                last_level_emit_at = now
        except Exception as exc:  # pragma: no cover - device-specific failures
            self.stop_event.set()
            self.error_queue.put(RuntimeError(f"Capture error ({source_name}): {exc}"))

    def _write_mix(self) -> None:
        if not self.mic_path.exists() or not self.desktop_path.exists():
            return

        with sf.SoundFile(str(self.mic_path), mode="r") as mic_file:
            with sf.SoundFile(str(self.desktop_path), mode="r") as desktop_file:
                if mic_file.samplerate != SAMPLE_RATE or desktop_file.samplerate != SAMPLE_RATE:
                    raise RuntimeError(
                        "Unexpected sample rate "
                        f"(mic={mic_file.samplerate}, desktop={desktop_file.samplerate})"
                    )

                with sf.SoundFile(
                    str(self.mix_path),
                    mode="w",
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    format="OGG",
                    subtype="VORBIS",
                ) as out_file:
                    while True:
                        mic_chunk = mic_file.read(BLOCK_FRAMES, dtype="float32", always_2d=False)
                        desktop_chunk = desktop_file.read(
                            BLOCK_FRAMES, dtype="float32", always_2d=False
                        )
                        mic_mono = to_mono(mic_chunk)
                        desktop_mono = to_mono(desktop_chunk)
                        frames = min(len(mic_mono), len(desktop_mono))
                        if frames <= 0:
                            break

                        mix = (mic_mono[:frames] + desktop_mono[:frames]) * 0.5
                        # Hard-clip overs to keep export stable and avoid NaN/inf propagation.
                        mix = mix.clip(-1.0, 1.0)
                        out_file.write(mix)

    @staticmethod
    def _estimate_level(samples) -> float:
        if len(samples) == 0:
            return 0.0
        # Soundcard usually returns float samples in [-1, 1], so RMS gives a stable activity meter.
        squared = samples * samples
        rms = math.sqrt(float(squared.mean()))
        boosted = min(1.0, rms * 8.0)
        return boosted


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
                target=self._monitor_loop,
                args=(self.mic, "microphone"),
                daemon=True,
            ),
            threading.Thread(
                target=self._monitor_loop,
                args=(self.desktop, "desktop"),
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
