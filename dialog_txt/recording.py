from __future__ import annotations

import math
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import soundfile as sf

from .config import BLOCK_FRAMES, DESKTOP_FILE_NAME, MIC_FILE_NAME, SAMPLE_RATE
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

    @staticmethod
    def _estimate_level(samples) -> float:
        if len(samples) == 0:
            return 0.0
        # Soundcard usually returns float samples in [-1, 1], so RMS gives a stable activity meter.
        squared = samples * samples
        rms = math.sqrt(float(squared.mean()))
        boosted = min(1.0, rms * 8.0)
        return boosted
