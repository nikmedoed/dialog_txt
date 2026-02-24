from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import ctranslate2
import soundfile as sf
from faster_whisper import WhisperModel

from .config import (
    MAX_MERGED_SEGMENT_DURATION,
    MAX_MERGED_SEGMENT_WORDS,
    MAX_OVERLAP_FOR_MERGE,
    MIN_AUDIO_RMS,
    MIN_SPEAKER_GAP,
    NO_SPEECH_PROB_THRESHOLD,
    SHORT_SEGMENT_BRIDGE_GAP,
    SHORT_SEGMENT_WORDS,
    WORD_PAUSE_SPLIT_GAP,
)
from .localization import tr
from .models import TranscriptSegment, TranscriptionCancelled, TranscriptionOptions
from .storage import resolve_track_paths, session_speaker_labels, transcript_path
from .utils import format_seconds, normalize_text, to_mono


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
        self,
        options: TranscriptionOptions,
        progress_cb: Callable[[str, float], None],
        ui_language: str,
    ) -> WhisperModel:
        load_plan = self._model_load_plan(options.compute_type)
        if not load_plan:
            raise RuntimeError(tr(ui_language, "transcriber_err_compute_types"))

        errors: list[str] = []
        for device, compute_type in load_plan:
            key = (options.model_name, device, compute_type)
            with self._model_lock:
                model = self._models.get(key)
                if model is not None:
                    return model

                progress_cb(
                    tr(
                        ui_language,
                        "transcriber_status_model_loading",
                        model=options.model_name,
                        compute=compute_type,
                        device=device.upper(),
                    ),
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
        raise RuntimeError(
            tr(
                ui_language,
                "transcriber_err_model_load",
                model=options.model_name,
                error=joined,
            )
        )

    def transcribe_session(
        self,
        session_dir: Path,
        progress_cb: Callable[[str, float], None],
        cancel_event: threading.Event,
        options: TranscriptionOptions,
        ui_language: str = "ru",
    ) -> Path:
        mic_path, desktop_path = resolve_track_paths(session_dir)
        if mic_path is None or desktop_path is None:
            raise FileNotFoundError(tr(ui_language, "transcriber_err_missing_tracks"))
        self_label, other_label = session_speaker_labels(session_dir)

        model = self._load_model(options, progress_cb, ui_language=ui_language)

        mic_duration = sf.info(str(mic_path)).duration or 0.0
        desktop_duration = sf.info(str(desktop_path)).duration or 0.0

        progress_cb(tr(ui_language, "transcriber_status_track", speaker=other_label), 5.0)
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
            ui_language=ui_language,
        )

        if cancel_event.is_set():
            raise TranscriptionCancelled()

        progress_cb(tr(ui_language, "transcriber_status_track", speaker=self_label), 50.0)
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
            ui_language=ui_language,
        )

        if cancel_event.is_set():
            raise TranscriptionCancelled()

        merged = self._merge_segments(desktop_segments + mic_segments)
        merged = self._collapse_consecutive_speaker_runs(merged)
        output_text = self._render_text(merged, options.include_timestamps)
        out_path = transcript_path(session_dir)
        out_path.write_text(output_text, encoding="utf-8")
        progress_cb(tr(ui_language, "transcriber_status_done"), 100.0)
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
        ui_language: str,
    ) -> list[TranscriptSegment]:
        language = options.language.strip()
        language_arg = None if language.lower() in ("", "auto", "авто") else language
        transcribe_kwargs = {
            "language": language_arg,
            "beam_size": options.beam_size,
            # Use soft post-filtering instead of hard VAD cuts to avoid clipping first words.
            "vad_filter": False,
            "condition_on_previous_text": False,
            "without_timestamps": False,
            "word_timestamps": True,
            "no_speech_threshold": NO_SPEECH_PROB_THRESHOLD,
        }

        segments_iter, info = model.transcribe(str(audio_path), **transcribe_kwargs)
        duration = info.duration or duration_hint or 1.0
        candidates: list[tuple[TranscriptSegment, float | None, float | None]] = []

        for seg in segments_iter:
            if cancel_event.is_set():
                raise TranscriptionCancelled()

            no_speech_prob = self._safe_float(getattr(seg, "no_speech_prob", None))
            avg_logprob = self._safe_float(getattr(seg, "avg_logprob", None))
            chunks = self._split_segment_by_pauses(seg=seg, speaker=speaker)
            for chunk in chunks:
                candidates.append((chunk, no_speech_prob, avg_logprob))

            ratio = min(max(float(seg.end) / duration, 0.0), 1.0)
            progress = progress_base + ratio * progress_span
            progress_cb(
                tr(ui_language, "transcriber_status_speaker_progress", speaker=speaker),
                progress,
            )

        if not candidates:
            return []

        measured: list[tuple[TranscriptSegment, float | None, float | None, float]] = []
        with sf.SoundFile(str(audio_path), mode="r") as audio_file:
            for chunk, no_speech_prob, avg_logprob in candidates:
                rms = self._segment_rms(audio_file=audio_file, start=chunk.start, end=chunk.end)
                measured.append((chunk, no_speech_prob, avg_logprob, rms))

        rms_floor = self._dynamic_rms_floor([rms for _, _, _, rms in measured])
        strict_filter = bool(options.vad_filter)

        result: list[TranscriptSegment] = []
        for chunk, no_speech_prob, avg_logprob, rms in measured:
            if self._should_drop_chunk(
                    chunk=chunk,
                    no_speech_prob=no_speech_prob,
                    avg_logprob=avg_logprob,
                    rms=rms,
                    rms_floor=rms_floor,
                    strict_filter=strict_filter,
            ):
                continue
            result.append(chunk)

        if result:
            return result
        return [chunk for chunk, _, _ in candidates]

    def _merge_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        ordered = sorted(segments, key=lambda s: (s.start, s.end))
        merged: list[TranscriptSegment] = []

        for seg in ordered:
            if not merged:
                merged.append(seg)
                continue
            prev = merged[-1]
            if self._can_merge_segments(prev, seg):
                prev.end = max(prev.end, seg.end)
                prev.text = self._append_text(prev.text, seg.text)
            else:
                merged.append(seg)
        return merged

    @staticmethod
    def _safe_float(value) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _split_segment_by_pauses(self, seg, speaker: str) -> list[TranscriptSegment]:
        words = getattr(seg, "words", None) or []
        if not words:
            text = normalize_text(getattr(seg, "text", ""))
            if not text:
                return []
            return [
                TranscriptSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    speaker=speaker,
                    text=text,
                )
            ]

        seg_start = float(seg.start)
        seg_end = float(seg.end)
        result: list[TranscriptSegment] = []
        current_tokens: list[str] = []
        current_start = seg_start
        current_end = seg_start

        def flush_chunk() -> None:
            nonlocal current_tokens, current_start, current_end
            if not current_tokens:
                return

            text = normalize_text("".join(current_tokens))
            if not text:
                current_tokens = []
                return

            chunk_start = max(seg_start, current_start)
            resolved_end = current_end
            if resolved_end <= chunk_start:
                resolved_end = seg_end
            chunk_end = min(seg_end, resolved_end)
            if chunk_end < chunk_start:
                chunk_end = chunk_start

            result.append(
                TranscriptSegment(
                    start=chunk_start,
                    end=chunk_end,
                    speaker=speaker,
                    text=text,
                )
            )
            current_tokens = []

        for word in words:
            token = str(getattr(word, "word", "") or "")
            if not token:
                continue

            word_start = self._safe_float(getattr(word, "start", None))
            word_end = self._safe_float(getattr(word, "end", None))

            if current_tokens and word_start is not None:
                gap = word_start - current_end
                if gap >= WORD_PAUSE_SPLIT_GAP:
                    flush_chunk()
                    current_start = word_start
                    current_end = word_start

            if not current_tokens:
                if word_start is not None:
                    current_start = word_start
                else:
                    current_start = max(seg_start, current_start)

            current_tokens.append(token)
            if word_end is not None:
                current_end = word_end
            elif word_start is not None:
                current_end = word_start

        flush_chunk()
        if result:
            return result

        text = normalize_text(getattr(seg, "text", ""))
        if not text:
            return []
        return [
            TranscriptSegment(
                start=seg_start,
                end=seg_end,
                speaker=speaker,
                text=text,
            )
        ]

    def _dynamic_rms_floor(self, rms_values: list[float]) -> float:
        usable = sorted(value for value in rms_values if value > 0.0)
        if not usable:
            return MIN_AUDIO_RMS
        median = usable[len(usable) // 2]
        adaptive = median * 0.18
        return max(MIN_AUDIO_RMS, adaptive)

    def _should_drop_chunk(
        self,
        chunk: TranscriptSegment,
        no_speech_prob: float | None,
        avg_logprob: float | None,
        rms: float,
        rms_floor: float,
        strict_filter: bool,
    ) -> bool:
        if rms >= rms_floor:
            return False

        duration = max(0.0, chunk.end - chunk.start)
        word_count = self._word_count(chunk.text)
        no_speech = no_speech_prob if no_speech_prob is not None else 0.0

        # Strict mode is used when "VAD filter" is enabled in UI:
        # remove more low-energy micro-phrases, but keep regular speech intact.
        if strict_filter:
            if no_speech >= 0.45 and duration <= 8.0:
                return True
            if duration <= 1.6 and word_count <= 5:
                return True
            if avg_logprob is not None and avg_logprob <= -1.0 and duration <= 5.0:
                return True
            return False

        if no_speech >= NO_SPEECH_PROB_THRESHOLD and duration <= 3.0:
            return True
        if duration <= 1.0 and word_count <= 3 and no_speech >= 0.35:
            return True
        if avg_logprob is not None and avg_logprob <= -1.2 and duration <= 2.0:
            return True

        return False

    @staticmethod
    def _segment_rms(audio_file: sf.SoundFile, start: float, end: float) -> float:
        sample_rate = int(audio_file.samplerate)
        start_frame = max(0, int(start * sample_rate))
        end_frame = min(audio_file.frames, int(end * sample_rate))
        frame_count = end_frame - start_frame
        if frame_count <= 0:
            return 0.0

        audio_file.seek(start_frame)
        samples = audio_file.read(frame_count, dtype="float32", always_2d=False)
        if len(samples) == 0:
            return 0.0

        mono = to_mono(samples)
        if len(mono) == 0:
            return 0.0

        squared = mono * mono
        return float(squared.mean() ** 0.5)

    def _can_merge_segments(self, left: TranscriptSegment, right: TranscriptSegment) -> bool:
        if left.speaker != right.speaker:
            return False

        gap = right.start - left.end
        if gap > MIN_SPEAKER_GAP:
            # Preserve conversational flow: tiny same-speaker tails are usually one turn.
            if gap > SHORT_SEGMENT_BRIDGE_GAP or self._word_count(right.text) > SHORT_SEGMENT_WORDS:
                return False
        if gap < -MAX_OVERLAP_FOR_MERGE:
            return False

        merged_duration = max(left.end, right.end) - min(left.start, right.start)
        if merged_duration > MAX_MERGED_SEGMENT_DURATION:
            return False

        merged_word_count = self._word_count(left.text) + self._word_count(right.text)
        if merged_word_count > MAX_MERGED_SEGMENT_WORDS:
            return False

        return True

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    def _collapse_consecutive_speaker_runs(
        self, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        collapsed: list[TranscriptSegment] = []
        for seg in segments:
            text = normalize_text(seg.text)
            if not text:
                continue

            if not collapsed:
                collapsed.append(
                    TranscriptSegment(
                        start=seg.start,
                        end=seg.end,
                        speaker=seg.speaker,
                        text=text,
                    )
                )
                continue

            prev = collapsed[-1]
            if prev.speaker == seg.speaker:
                prev.end = max(prev.end, seg.end)
                prev.text = self._append_text(prev.text, text)
                continue

            collapsed.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    speaker=seg.speaker,
                    text=text,
                )
            )
        return collapsed

    @staticmethod
    def _append_text(left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
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
