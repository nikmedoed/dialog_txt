from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import ctranslate2
import soundfile as sf
from faster_whisper import WhisperModel

from .config import MIN_SPEAKER_GAP
from .models import TranscriptSegment, TranscriptionCancelled, TranscriptionOptions
from .storage import resolve_track_paths, session_speaker_labels, transcript_path
from .utils import format_seconds, normalize_text


class WhisperTranscriber:
    COMPUTE_TYPE_FALLBACK_ORDER = (
        "int8_float16",
        "float16",
        "int8",
        "float32",
        "int8_float32",
        "bfloat16",
        "int8_bfloat16",
        "int16",
    )

    def __init__(self):
        self._models: dict[tuple[str, str, str], WhisperModel] = {}
        self._model_lock = threading.Lock()
        self._cuda_available = self._detect_cuda_availability()

    @staticmethod
    def _detect_cuda_availability() -> bool:
        try:
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    @staticmethod
    def _supported_compute_types(device: str) -> set[str]:
        try:
            return set(ctranslate2.get_supported_compute_types(device))
        except Exception:
            return set()

    def _model_load_plan(self, requested_compute_type: str) -> list[tuple[str, str]]:
        devices = ["cuda", "cpu"] if self._cuda_available else ["cpu"]
        requested = requested_compute_type.strip()
        plan: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for device in devices:
            supported = self._supported_compute_types(device)
            if not supported:
                continue

            compute_candidates: list[str] = []
            if requested in supported:
                compute_candidates.append(requested)
            for candidate in self.COMPUTE_TYPE_FALLBACK_ORDER:
                if candidate in supported and candidate not in compute_candidates:
                    compute_candidates.append(candidate)
            if not compute_candidates:
                compute_candidates.extend(sorted(supported))

            for compute_type in compute_candidates:
                key = (device, compute_type)
                if key in seen:
                    continue
                seen.add(key)
                plan.append(key)
        return plan

    def _load_model(
        self, options: TranscriptionOptions, progress_cb: Callable[[str, float], None]
    ) -> WhisperModel:
        load_plan = self._model_load_plan(options.compute_type)
        if not load_plan:
            raise RuntimeError("Не удалось определить поддерживаемые вычислительные устройства.")

        errors: list[str] = []
        for device, compute_type in load_plan:
            key = (options.model_name, device, compute_type)
            with self._model_lock:
                model = self._models.get(key)
                if model is not None:
                    return model

                progress_cb(
                    f"Загрузка модели {options.model_name} ({compute_type}) на {device.upper()}...",
                    1.0,
                )
                try:
                    model = WhisperModel(
                        options.model_name,
                        device=device,
                        compute_type=compute_type,
                    )
                except Exception as exc:
                    errors.append(f"{device}/{compute_type}: {exc}")
                    continue
                self._models[key] = model
                return model

        joined = "; ".join(errors) if errors else "unknown initialization error"
        raise RuntimeError(f"Не удалось загрузить модель {options.model_name}: {joined}")

    def transcribe_session(
        self,
        session_dir: Path,
        progress_cb: Callable[[str, float], None],
        cancel_event: threading.Event,
        options: TranscriptionOptions,
    ) -> Path:
        mic_path, desktop_path = resolve_track_paths(session_dir)
        if mic_path is None or desktop_path is None:
            raise FileNotFoundError("В выбранной папке нет обеих дорожек (mic.ogg и desktop.ogg).")
        self_label, other_label = session_speaker_labels(session_dir)

        model = self._load_model(options, progress_cb)

        mic_duration = sf.info(str(mic_path)).duration or 0.0
        desktop_duration = sf.info(str(desktop_path)).duration or 0.0

        progress_cb(f"Транскрибация дорожки [{other_label}]...", 5.0)
        desktop_segments = self._transcribe_track(
            model=model,
            audio_path=desktop_path,
            speaker=other_label,
            duration_hint=desktop_duration,
            options=options,
            progress_base=5.0,
            progress_span=45.0,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

        if cancel_event.is_set():
            raise TranscriptionCancelled()

        progress_cb(f"Транскрибация дорожки [{self_label}]...", 50.0)
        mic_segments = self._transcribe_track(
            model=model,
            audio_path=mic_path,
            speaker=self_label,
            duration_hint=mic_duration,
            options=options,
            progress_base=50.0,
            progress_span=45.0,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

        if cancel_event.is_set():
            raise TranscriptionCancelled()

        merged = self._merge_segments(desktop_segments + mic_segments)
        output_text = self._render_text(merged, options.include_timestamps)
        out_path = transcript_path(session_dir)
        out_path.write_text(output_text, encoding="utf-8")
        progress_cb("Готово", 100.0)
        return out_path

    def _transcribe_track(
        self,
        model: WhisperModel,
        audio_path: Path,
        speaker: str,
        duration_hint: float,
        options: TranscriptionOptions,
        progress_base: float,
        progress_span: float,
        progress_cb: Callable[[str, float], None],
        cancel_event: threading.Event,
    ) -> list[TranscriptSegment]:
        language = options.language.strip()
        language_arg = None if language.lower() in ("", "auto", "авто") else language
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language_arg,
            beam_size=options.beam_size,
            vad_filter=options.vad_filter,
            condition_on_previous_text=False,
            without_timestamps=False,
        )
        duration = info.duration or duration_hint or 1.0
        result: list[TranscriptSegment] = []

        for seg in segments_iter:
            if cancel_event.is_set():
                raise TranscriptionCancelled()
            text = normalize_text(seg.text)
            if text:
                result.append(
                    TranscriptSegment(
                        start=float(seg.start),
                        end=float(seg.end),
                        speaker=speaker,
                        text=text,
                    )
                )
            ratio = min(max(float(seg.end) / duration, 0.0), 1.0)
            progress = progress_base + ratio * progress_span
            progress_cb(f"Транскрибация: {speaker}", progress)

        return result

    def _merge_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        ordered = sorted(segments, key=lambda s: (s.start, s.end))
        merged: list[TranscriptSegment] = []

        for seg in ordered:
            if not merged:
                merged.append(seg)
                continue
            prev = merged[-1]
            same_speaker = prev.speaker == seg.speaker
            close_gap = seg.start - prev.end <= MIN_SPEAKER_GAP
            if same_speaker and close_gap:
                prev.end = max(prev.end, seg.end)
                prev.text = self._append_text(prev.text, seg.text)
            else:
                merged.append(seg)
        return merged

    @staticmethod
    def _append_text(left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        if left.endswith((".", "!", "?", ":")):
            return f"{left} {right}"
        return f"{left} {right}"

    def _render_text(self, segments: list[TranscriptSegment], include_timestamps: bool) -> str:
        lines: list[str] = []
        for seg in segments:
            if include_timestamps:
                timestamp = format_seconds(seg.start)
                lines.append(f"[{timestamp}] [{seg.speaker}] {seg.text}")
            else:
                lines.append(f"[{seg.speaker}] {seg.text}")
        return "\n".join(lines) + "\n"
