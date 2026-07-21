from __future__ import annotations

import contextlib
import importlib
import math
import os
import re
import sys
import threading
import tempfile
from pathlib import Path
from typing import Callable

import soundfile as sf

from .config import (
    BASELINE_CONTEXT_LOOKBACK_SEC,
    DIALOG_CONTEXT_PROMPT_WORDS,
    LANGUAGE_DETECT_MAX_SECONDS,
    MIX_FILE_NAME,
    MIX_REFERENCE_PAD_SEC,
    MAX_MERGED_SEGMENT_DURATION,
    MAX_MERGED_SEGMENT_WORDS,
    MAX_OVERLAP_FOR_MERGE,
    MIN_AUDIO_RMS,
    MIN_SPEAKER_GAP,
    NO_SPEECH_PROB_THRESHOLD,
    SPEECH_CHUNK_BRIDGE_SEC,
    SPEECH_CHUNK_MAX_SEC,
    SPEECH_GATE_FLOOR_RATIO,
    SPEECH_GATE_FRAME_MS,
    SPEECH_GATE_ONSET_RATIO,
    SPEECH_GATE_MIN_SILENCE_SEC,
    SPEECH_GATE_MIN_SPEECH_SEC,
    SPEECH_GATE_PAD_SEC,
    SPEECH_GATE_RELEASE_RATIO,
    SPEECH_GATE_THRESHOLD_MULTIPLIER,
    SHORT_SEGMENT_BRIDGE_GAP,
    SHORT_SEGMENT_WORDS,
    TRANSCRIPT_CHUNK_HARD_MAX_SEC,
    TRANSCRIPT_CHUNK_HARD_MAX_WORDS,
    TRANSCRIPT_CHUNK_SOFT_MAX_SEC,
    TRANSCRIPT_CHUNK_SOFT_MAX_WORDS,
    TRANSCRIPT_SOFT_SPLIT_GAP,
    WORD_PAUSE_SPLIT_GAP,
)
from .localization import tr
from .models import TranscriptSegment, TranscriptionCancelled, TranscriptionOptions
from .storage import debug_transcript_path, resolve_track_paths, transcript_path
from .transcription_backends import (
    TRANSCRIPTION_LIBRARY_FASTER,
    TRANSCRIPTION_LIBRARY_WHISPER,
    normalize_transcription_library,
)
from .utils import format_seconds, normalize_text, to_mono


class WhisperTranscriber:
    AUTO_COMPUTE_ORDER = {
        "cpu": ("int8", "int8_float32", "int16", "float32"),
        "cuda": ("float16", "int8_float16", "bfloat16", "int8", "float32"),
    }
    SILENCE_HALLUCINATION_FRAGMENTS = (
        "продолжение следует",
        "субтитры делал",
        "субтитры сделал",
        "субтитры создавал",
        "добавил субтитры",
        "субтитры подогнал",
        "редактор субтитров",
        "спасибо за просмотр",
        "спасибо за субтитры",
        "подписывайтесь на канал",
        "thank you for watching",
        "thanks for watching",
        "subtitles by",
    )
    HALLUCINATION_PATTERNS = (
        re.compile(r"(?iu)\bпродолжение\s+следует\b(?:\s*[.!?…]+)?"),
        re.compile(
            r"(?iu)\b(?:субтитры\s+(?:сделал|делал|создавал|подогнал)|"
            r"добавил\s+субтитры)\b(?:\s+[\"'«»a-zа-я0-9._-]+){0,6}"
        ),
        re.compile(
            r"(?iu)\bспасибо\s+за\s+(?:просмотр|субтитры)\b"
            r"(?:\s+[\"'«»a-zа-я0-9._-]+){0,6}(?:\s*[.!?…]+)?"
        ),
        re.compile(
            r"(?iu)\bредактор\s+субтитров\b(?:\s+[\"'«»a-zа-я0-9._-]+){0,8}"
            r"(?:\s+\bкорректор\b(?:\s+[\"'«»a-zа-я0-9._-]+){0,6})?"
        ),
        re.compile(r"(?iu)\bthanks?\s+for\s+watching\b(?:\s+[a-z0-9._-]+){0,6}(?:\s*[.!?…]+)?"),
        re.compile(r"(?iu)\bsubtitles\s+by\b(?:\s+[a-z0-9._-]+){0,6}(?:\s*[.!?…]+)?"),
    )

    def __init__(self):
        self._models: dict[tuple[str, str, str, str], object] = {}
        self._model_lock = threading.Lock()

    @staticmethod
    @contextlib.contextmanager
    def _redirect_missing_standard_streams():
        replacements: dict[str, object | None] = {}
        redirected = []
        try:
            for stream_name in ("stdout", "stderr"):
                current = getattr(sys, stream_name, None)
                if current is not None:
                    continue
                redirected_stream = open(os.devnull, "w", encoding="utf-8")
                replacements[stream_name] = current
                redirected.append(redirected_stream)
                setattr(sys, stream_name, redirected_stream)
            yield
        finally:
            for stream_name, original in replacements.items():
                setattr(sys, stream_name, original)
            for redirected_stream in redirected:
                redirected_stream.close()

    @staticmethod
    def _detect_faster_cuda_availability(ctranslate2_module) -> bool:
        try:
            return ctranslate2_module.get_cuda_device_count() > 0
        except Exception:
            return False

    @staticmethod
    def _detect_whisper_cuda_availability() -> tuple[bool, str]:
        try:
            torch = importlib.import_module("torch")
        except Exception:
            return False, "import-failed"

        try:
            if bool(torch.cuda.is_available()):
                return True, ""
        except Exception:
            return False, "runtime-unavailable"

        cuda_version = str(getattr(getattr(torch, "version", None), "cuda", "") or "").strip()
        if not cuda_version:
            return False, "cpu-only"
        return False, "runtime-unavailable"

    @staticmethod
    def _supported_compute_types(ctranslate2_module, device: str) -> set[str]:
        try:
            return set(ctranslate2_module.get_supported_compute_types(device))
        except Exception:
            return set()

    @staticmethod
    def _resolve_device_order(
        requested_device: str,
        cuda_available: bool,
        ui_language: str,
    ) -> list[str]:
        normalized = str(requested_device or "auto").strip().lower()
        if normalized == "gpu":
            if not cuda_available:
                raise RuntimeError(tr(ui_language, "transcriber_err_gpu_unavailable"))
            return ["cuda"]
        if normalized == "cpu":
            return ["cpu"]
        return ["cuda", "cpu"] if cuda_available else ["cpu"]

    @staticmethod
    def _resolve_single_device(
        requested_device: str,
        cuda_available: bool,
        ui_language: str,
        cuda_diagnostic: str = "",
    ) -> str:
        normalized = str(requested_device or "auto").strip().lower()
        if normalized == "gpu":
            if not cuda_available:
                if cuda_diagnostic == "cpu-only":
                    raise RuntimeError(tr(ui_language, "transcriber_err_gpu_torch_cpu_only"))
                raise RuntimeError(tr(ui_language, "transcriber_err_gpu_unavailable"))
            return "cuda"
        if normalized == "cpu":
            return "cpu"
        return "cuda" if cuda_available else "cpu"

    def _model_load_plan(
        self,
        requested_compute_type: str,
        requested_device: str,
        ctranslate2_module,
        ui_language: str,
    ) -> list[tuple[str, str]]:
        cuda_available = self._detect_faster_cuda_availability(ctranslate2_module)
        devices = self._resolve_device_order(requested_device, cuda_available, ui_language)
        requested = requested_compute_type.strip().lower()
        plan: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for device in devices:
            supported = self._supported_compute_types(ctranslate2_module, device)
            if not supported:
                continue

            if requested == "auto":
                compute_candidates = [
                    candidate
                    for candidate in self.AUTO_COMPUTE_ORDER.get(device, ())
                    if candidate in supported
                ]
            elif requested in supported:
                compute_candidates = [requested]
            else:
                # An explicit precision is a promise, not a hint. Do not silently
                # turn float16 into int8 (or vice versa) on another device.
                compute_candidates = []

            for compute_type in compute_candidates:
                key = (device, compute_type)
                if key in seen:
                    continue
                seen.add(key)
                plan.append(key)
        return plan

    def _load_faster_model(
        self,
        options: TranscriptionOptions,
        progress_cb: Callable[[str, float], None],
        ui_language: str,
    ) -> tuple[object, str, str]:
        try:
            ctranslate2_module = importlib.import_module("ctranslate2")
            faster_whisper = importlib.import_module("faster_whisper")
            whisper_model_cls = faster_whisper.WhisperModel
        except Exception as exc:
            raise RuntimeError(
                tr(
                    ui_language,
                    "transcriber_err_model_load",
                    model=options.model_name,
                    error=str(exc),
                )
            ) from exc

        load_plan = self._model_load_plan(
            options.compute_type,
            options.device,
            ctranslate2_module,
            ui_language,
        )
        if not load_plan:
            raise RuntimeError(tr(ui_language, "transcriber_err_compute_types"))

        errors: list[str] = []
        for device, compute_type in load_plan:
            key = (TRANSCRIPTION_LIBRARY_FASTER, options.model_name, device, compute_type)
            with self._model_lock:
                model = self._models.get(key)
                if model is not None:
                    actual_compute = str(
                        getattr(getattr(model, "model", None), "compute_type", compute_type)
                    )
                    return model, device, actual_compute

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
                    model = whisper_model_cls(
                        options.model_name,
                        device=device,
                        compute_type=compute_type,
                    )
                except Exception as exc:
                    errors.append(f"{device}/{compute_type}: {exc}")
                    continue
                self._models[key] = model
                actual_compute = str(
                    getattr(getattr(model, "model", None), "compute_type", compute_type)
                )
                return model, device, actual_compute

        joined = "; ".join(errors) if errors else "unknown initialization error"
        raise RuntimeError(
            tr(
                ui_language,
                "transcriber_err_model_load",
                model=options.model_name,
                error=joined,
            )
        )

    def _load_openai_whisper_model(
        self,
        options: TranscriptionOptions,
        progress_cb: Callable[[str, float], None],
        ui_language: str,
    ) -> tuple[object, str, str]:
        try:
            whisper_module = importlib.import_module("whisper")
        except Exception as exc:
            raise RuntimeError(
                tr(
                    ui_language,
                    "transcriber_err_model_load",
                    model=options.model_name,
                    error=str(exc),
                )
            ) from exc

        cuda_available, cuda_diagnostic = self._detect_whisper_cuda_availability()
        device = self._resolve_single_device(
            options.device,
            cuda_available,
            ui_language,
            cuda_diagnostic,
        )
        key = (TRANSCRIPTION_LIBRARY_WHISPER, options.model_name, device, "")
        with self._model_lock:
            model = self._models.get(key)
            if model is not None:
                return model, device, "torch"

            progress_cb(
                tr(
                    ui_language,
                    "transcriber_status_model_loading",
                    model=options.model_name,
                    compute="torch",
                    device=device.upper(),
                ),
                1.0,
            )
            try:
                # openai-whisper uses tqdm while downloading weights; under pythonw
                # sys.stderr/sys.stdout may be None and tqdm crashes on first write.
                with self._redirect_missing_standard_streams():
                    model = whisper_module.load_model(options.model_name, device=device)
            except Exception as exc:
                raise RuntimeError(
                    tr(
                        ui_language,
                        "transcriber_err_model_load",
                        model=options.model_name,
                        error=str(exc),
                    )
                ) from exc
            self._models[key] = model
            return model, device, "torch"

    def _load_model(
        self,
        options: TranscriptionOptions,
        progress_cb: Callable[[str, float], None],
        ui_language: str,
    ) -> tuple[object, str, str, str]:
        library = normalize_transcription_library(options.transcription_library)
        if library == TRANSCRIPTION_LIBRARY_FASTER:
            model, runtime_device, runtime_compute = self._load_faster_model(
                options,
                progress_cb,
                ui_language,
            )
            return model, library, runtime_device, runtime_compute
        if library == TRANSCRIPTION_LIBRARY_WHISPER:
            model, runtime_device, runtime_compute = self._load_openai_whisper_model(
                options,
                progress_cb,
                ui_language,
            )
            return model, library, runtime_device, runtime_compute
        raise RuntimeError(
            tr(ui_language, "transcriber_err_unsupported_library", library=library)
        )

    @staticmethod
    def _requested_language(language_value: str) -> str | None:
        normalized = str(language_value or "").strip()
        if normalized.lower() in ("", "auto", "авто"):
            return None
        return normalized

    def _detect_language_once(
        self,
        model,
        transcription_library: str,
        runtime_device: str,
        probe_audio_path: Path,
    ) -> str | None:
        if transcription_library == TRANSCRIPTION_LIBRARY_FASTER:
            kwargs = {
                "beam_size": 1,
                "vad_filter": False,
                "condition_on_previous_text": False,
                "without_timestamps": True,
                "word_timestamps": False,
                "no_speech_threshold": NO_SPEECH_PROB_THRESHOLD,
            }
            _segments, info = model.transcribe(str(probe_audio_path), **kwargs)
            detected = getattr(info, "language", None)
            detected_lang = str(detected or "").strip()
            return detected_lang or None

        if transcription_library == TRANSCRIPTION_LIBRARY_WHISPER:
            kwargs = {
                "beam_size": 1,
                "condition_on_previous_text": False,
                "word_timestamps": False,
                "fp16": runtime_device == "cuda",
                "verbose": None,
                "no_speech_threshold": NO_SPEECH_PROB_THRESHOLD,
            }
            audio_input = self._prepare_openai_whisper_audio_input(str(probe_audio_path))
            with self._redirect_missing_standard_streams():
                result = model.transcribe(audio_input, **kwargs)
            detected = result.get("language") if isinstance(result, dict) else None
            detected_lang = str(detected or "").strip()
            return detected_lang or None
        return None

    @staticmethod
    def _prepare_openai_whisper_audio_input(audio_input):
        if isinstance(audio_input, Path):
            audio_input = str(audio_input)
        if not isinstance(audio_input, str):
            return audio_input

        audio_data, _sample_rate = sf.read(audio_input, dtype="float32", always_2d=False)
        return to_mono(audio_data)

    def _resolve_session_language(
        self,
        model,
        transcription_library: str,
        runtime_device: str,
        options: TranscriptionOptions,
        session_dir: Path,
        fallback_ui_language: str,
    ) -> str | None:
        requested = self._requested_language(options.language)
        if requested:
            return requested

        fallback_lang = str(fallback_ui_language or "ru").strip() or "ru"
        mix_path = session_dir / MIX_FILE_NAME
        probe_source = mix_path if mix_path.exists() else None
        if probe_source is None:
            return fallback_lang

        with sf.SoundFile(str(probe_source), mode="r") as audio_file:
            speech_ranges = self._detect_speech_ranges(audio_file=audio_file, strict_filter=False)
            speech_ranges = self._coalesce_ranges_for_transcription(speech_ranges)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as probe_tmp:
                probe_path = Path(probe_tmp.name)
            try:
                with sf.SoundFile(
                    str(probe_path),
                    mode="w",
                    samplerate=audio_file.samplerate,
                    channels=1,
                    format="WAV",
                    subtype="PCM_16",
                ) as out_file:
                    written_seconds = 0.0
                    for start, end in speech_ranges:
                        if written_seconds >= LANGUAGE_DETECT_MAX_SECONDS:
                            break
                        chunk = self._read_audio_slice(audio_file=audio_file, start=start, end=end)
                        if len(chunk) == 0:
                            continue
                        chunk_sec = len(chunk) / max(1, int(audio_file.samplerate))
                        remaining = LANGUAGE_DETECT_MAX_SECONDS - written_seconds
                        if chunk_sec > remaining:
                            max_frames = int(remaining * int(audio_file.samplerate))
                            if max_frames <= 0:
                                break
                            chunk = chunk[:max_frames]
                            chunk_sec = len(chunk) / max(1, int(audio_file.samplerate))
                        out_file.write(chunk)
                        written_seconds += chunk_sec

                if sf.info(str(probe_path)).frames <= 0:
                    return fallback_lang

                detected = self._detect_language_once(
                    model=model,
                    transcription_library=transcription_library,
                    runtime_device=runtime_device,
                    probe_audio_path=probe_path,
                )
                return detected or fallback_lang
            finally:
                probe_path.unlink(missing_ok=True)

    @staticmethod
    def _speech_ranges_to_frame_ranges(
        speech_ranges: list[tuple[float, float]],
        sample_rate: int,
        total_frames: int,
    ) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        if sample_rate <= 0 or total_frames <= 0:
            return result
        for start_sec, end_sec in speech_ranges:
            start_frame = max(0, min(total_frames, int(start_sec * sample_rate)))
            end_frame = max(start_frame, min(total_frames, int(end_sec * sample_rate)))
            if end_frame <= start_frame:
                continue
            result.append((start_frame, end_frame))
        return result

    def _render_silence_masked_track(
        self,
        source_path: Path,
        speech_ranges: list[tuple[float, float]],
    ) -> Path | None:
        with sf.SoundFile(str(source_path), mode="r") as source:
            sample_rate = int(source.samplerate)
            speech_frames = self._speech_ranges_to_frame_ranges(
                speech_ranges=speech_ranges,
                sample_rate=sample_rate,
                total_frames=source.frames,
            )
            if not speech_frames:
                return None

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                masked_path = Path(tmp_audio.name)

            with sf.SoundFile(
                str(masked_path),
                mode="w",
                samplerate=sample_rate,
                channels=1,
                format="WAV",
                subtype="PCM_16",
            ) as out:
                block_size = max(1, sample_rate // 2)
                frame_pos = 0
                speech_idx = 0
                while True:
                    block = source.read(block_size, dtype="float32", always_2d=False)
                    if len(block) == 0:
                        break
                    mono = to_mono(block)
                    masked = mono * 0.0
                    block_start = frame_pos
                    block_end = block_start + len(mono)

                    while speech_idx < len(speech_frames) and speech_frames[speech_idx][1] <= block_start:
                        speech_idx += 1

                    idx = speech_idx
                    while idx < len(speech_frames):
                        speech_start, speech_end = speech_frames[idx]
                        if speech_start >= block_end:
                            break
                        copy_start = max(speech_start, block_start) - block_start
                        copy_end = min(speech_end, block_end) - block_start
                        if copy_end > copy_start:
                            masked[copy_start:copy_end] = mono[copy_start:copy_end]
                        if speech_end <= block_end:
                            idx += 1
                        else:
                            break
                    speech_idx = idx
                    out.write(masked)
                    frame_pos = block_end
        return masked_path

    def _collect_candidates_from_raw_segments(
        self,
        raw_segments: list,
        speaker: str,
        time_offset: float = 0.0,
    ) -> list[tuple[TranscriptSegment, float | None, float | None]]:
        candidates: list[tuple[TranscriptSegment, float | None, float | None]] = []
        for raw_seg in raw_segments:
            seg = self._segment_with_time_offset(raw_seg, time_offset) if time_offset else raw_seg
            no_speech_prob = self._safe_float(self._segment_field(seg, "no_speech_prob"))
            avg_logprob = self._safe_float(self._segment_field(seg, "avg_logprob"))
            chunks = self._split_segment_by_pauses(seg=seg, speaker=speaker)
            for chunk in chunks:
                candidates.append((chunk, no_speech_prob, avg_logprob))
        return candidates

    def _filter_track_candidates(
        self,
        audio_path: Path,
        candidates: list[tuple[TranscriptSegment, float | None, float | None]],
        strict_filter: bool,
    ) -> list[TranscriptSegment]:
        if not candidates:
            return []

        measured: list[tuple[TranscriptSegment, float | None, float | None, float]] = []
        with sf.SoundFile(str(audio_path), mode="r") as audio_file:
            for chunk, no_speech_prob, avg_logprob in candidates:
                rms = self._segment_rms(audio_file=audio_file, start=chunk.start, end=chunk.end)
                measured.append((chunk, no_speech_prob, avg_logprob, rms))

        rms_floor = self._dynamic_rms_floor([rms for _, _, _, rms in measured])
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
        return result

    @staticmethod
    def _prompt_from_segments(
        segments: list[TranscriptSegment],
        max_words: int,
    ) -> str:
        if not segments:
            return ""
        text = " ".join(f"[{seg.speaker}] {seg.text}" for seg in segments if seg.text)
        text = normalize_text(text)
        if not text:
            return ""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[-max_words:])

    @staticmethod
    def _segments_overlapping_range(
        segments: list[TranscriptSegment],
        speaker: str,
        start: float,
        end: float,
    ) -> list[TranscriptSegment]:
        result: list[TranscriptSegment] = []
        for seg in segments:
            if seg.speaker != speaker:
                continue
            if seg.end < start or seg.start > end:
                continue
            result.append(seg)
        return result

    def _build_dialogue_prompt(
        self,
        accepted_dialogue: list[TranscriptSegment],
        baseline_dialogue: list[TranscriptSegment],
        utterance_start: float,
    ) -> str:
        recent_confirmed = accepted_dialogue[-8:]
        if recent_confirmed:
            return self._prompt_from_segments(recent_confirmed, DIALOG_CONTEXT_PROMPT_WORDS)

        baseline_tail = [
            seg
            for seg in baseline_dialogue
            if seg.end <= utterance_start and seg.end >= (utterance_start - BASELINE_CONTEXT_LOOKBACK_SEC)
        ]
        return self._prompt_from_segments(baseline_tail[-8:], DIALOG_CONTEXT_PROMPT_WORDS)

    def _transcribe_track_baseline(
        self,
        model,
        transcription_library: str,
        runtime_device: str,
        audio_path: Path,
        speaker: str,
        speech_ranges: list[tuple[float, float]],
        beam_size: int,
        language: str | None,
        strict_filter: bool,
    ) -> list[TranscriptSegment]:
        masked_path = self._render_silence_masked_track(audio_path, speech_ranges)
        if masked_path is None:
            return []
        try:
            raw_segments = self._transcribe_audio(
                model=model,
                transcription_library=transcription_library,
                runtime_device=runtime_device,
                audio_input=str(masked_path),
                beam_size=beam_size,
                language=language,
                condition_on_previous_text=True,
                word_timestamps=True,
                without_timestamps=False,
            )
        finally:
            masked_path.unlink(missing_ok=True)

        candidates = self._collect_candidates_from_raw_segments(
            raw_segments=raw_segments,
            speaker=speaker,
            time_offset=0.0,
        )
        filtered = self._filter_track_candidates(
            audio_path=audio_path,
            candidates=candidates,
            strict_filter=strict_filter,
        )
        if filtered:
            return filtered
        return [chunk for chunk, _, _ in candidates]

    def _transcribe_utterance_range(
        self,
        model,
        transcription_library: str,
        runtime_device: str,
        audio_path: Path,
        speaker: str,
        utterance_start: float,
        utterance_end: float,
        beam_size: int,
        language: str | None,
        strict_filter: bool,
        prompt: str,
    ) -> list[TranscriptSegment]:
        with sf.SoundFile(str(audio_path), mode="r") as audio_file:
            sample_rate = int(audio_file.samplerate)
            audio_slice = self._read_audio_slice(
                audio_file=audio_file,
                start=utterance_start,
                end=utterance_end,
            )
        if len(audio_slice) == 0:
            return []

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            tmp_audio_path = Path(tmp_audio.name)
        try:
            sf.write(
                str(tmp_audio_path),
                audio_slice,
                sample_rate,
                format="WAV",
                subtype="PCM_16",
            )
            raw_segments = self._transcribe_audio(
                model=model,
                transcription_library=transcription_library,
                runtime_device=runtime_device,
                audio_input=str(tmp_audio_path),
                beam_size=beam_size,
                language=language,
                initial_prompt=prompt or None,
                condition_on_previous_text=False,
                word_timestamps=True,
                without_timestamps=False,
            )
        finally:
            tmp_audio_path.unlink(missing_ok=True)

        candidates = self._collect_candidates_from_raw_segments(
            raw_segments=raw_segments,
            speaker=speaker,
            time_offset=utterance_start,
        )
        filtered = self._filter_track_candidates(
            audio_path=audio_path,
            candidates=candidates,
            strict_filter=strict_filter,
        )
        if filtered:
            return filtered
        return [chunk for chunk, _, _ in candidates]

    def _transcribe_mix_reference(
        self,
        session_dir: Path,
        model,
        transcription_library: str,
        runtime_device: str,
        language: str | None,
        beam_size: int,
        cancel_event: threading.Event,
        progress_cb: Callable[[str, float], None],
        ui_language: str,
    ) -> list[TranscriptSegment]:
        if cancel_event.is_set():
            raise TranscriptionCancelled()

        mix_path = session_dir / MIX_FILE_NAME
        if not mix_path.exists():
            return []

        progress_cb(tr(ui_language, "transcriber_status_mix"), 97.0)
        with sf.SoundFile(str(mix_path), mode="r") as audio_file:
            speech_ranges = self._detect_speech_ranges(
                audio_file=audio_file,
                strict_filter=False,
            )
        speech_ranges = self._coalesce_ranges_for_transcription(speech_ranges)
        if not speech_ranges:
            return []

        return self._transcribe_track_baseline(
            model=model,
            transcription_library=transcription_library,
            runtime_device=runtime_device,
            audio_path=mix_path,
            speaker="",
            speech_ranges=speech_ranges,
            beam_size=beam_size,
            language=language,
            strict_filter=False,
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
        self_label = options.speaker_self
        other_label = options.speaker_other
        strict_filter = bool(options.vad_filter)

        model, library, runtime_device, _runtime_compute = self._load_model(
            options,
            progress_cb,
            ui_language=ui_language,
        )
        session_language = self._resolve_session_language(
            model=model,
            transcription_library=library,
            runtime_device=runtime_device,
            options=options,
            session_dir=session_dir,
            fallback_ui_language=ui_language,
        )

        tracks = [
            {
                "key": "desktop",
                "audio_path": desktop_path,
                "speaker": other_label,
            },
            {
                "key": "mic",
                "audio_path": mic_path,
                "speaker": self_label,
            },
        ]
        for track in tracks:
            with sf.SoundFile(str(track["audio_path"]), mode="r") as audio_file:
                speech_ranges = self._detect_speech_ranges(
                    audio_file=audio_file,
                    strict_filter=strict_filter,
                )
            track["speech_ranges"] = self._coalesce_ranges_for_transcription(speech_ranges)

        baseline_segments: list[TranscriptSegment] = []
        for index, track in enumerate(tracks):
            if cancel_event.is_set():
                raise TranscriptionCancelled()
            baseline_progress = 5.0 + index * 20.0
            progress_cb(
                tr(ui_language, "transcriber_status_track", speaker=track["speaker"]),
                baseline_progress,
            )
            speech_ranges = track["speech_ranges"] if isinstance(track["speech_ranges"], list) else []
            if speech_ranges:
                track_segments = self._transcribe_track_baseline(
                    model=model,
                    transcription_library=library,
                    runtime_device=runtime_device,
                    audio_path=track["audio_path"],
                    speaker=track["speaker"],
                    speech_ranges=speech_ranges,
                    beam_size=options.beam_size,
                    language=session_language,
                    strict_filter=strict_filter,
                )
                track["baseline_segments"] = track_segments
                baseline_segments.extend(track_segments)
            else:
                track["baseline_segments"] = []
            progress_cb(
                tr(ui_language, "transcriber_status_speaker_progress", speaker=track["speaker"]),
                baseline_progress + 20.0,
            )

        if cancel_event.is_set():
            raise TranscriptionCancelled()

        baseline_dialogue = self._merge_segments(baseline_segments)
        baseline_dialogue = self._collapse_consecutive_speaker_runs(baseline_dialogue)

        utterances: list[dict] = []
        for track in tracks:
            speech_ranges = track["speech_ranges"] if isinstance(track["speech_ranges"], list) else []
            for start, end in speech_ranges:
                utterances.append(
                    {
                        "audio_path": track["audio_path"],
                        "speaker": track["speaker"],
                        "start": start,
                        "end": end,
                    }
                )
        utterances.sort(key=lambda item: (item["start"], item["end"]))

        accepted_segments: list[TranscriptSegment] = []
        total_utterances = max(1, len(utterances))
        for index, utterance in enumerate(utterances):
            if cancel_event.is_set():
                raise TranscriptionCancelled()

            prompt = self._build_dialogue_prompt(
                accepted_dialogue=accepted_segments,
                baseline_dialogue=baseline_dialogue,
                utterance_start=float(utterance["start"]),
            )
            utterance_segments = self._transcribe_utterance_range(
                model=model,
                transcription_library=library,
                runtime_device=runtime_device,
                audio_path=utterance["audio_path"],
                speaker=str(utterance["speaker"]),
                utterance_start=float(utterance["start"]),
                utterance_end=float(utterance["end"]),
                beam_size=options.beam_size,
                language=session_language,
                strict_filter=strict_filter,
                prompt=prompt,
            )
            if not utterance_segments:
                utterance_segments = self._segments_overlapping_range(
                    segments=baseline_dialogue,
                    speaker=str(utterance["speaker"]),
                    start=float(utterance["start"]),
                    end=float(utterance["end"]),
                )
            if utterance_segments:
                accepted_segments.extend(utterance_segments)

            ratio = (index + 1) / total_utterances
            progress = 45.0 + ratio * 50.0
            progress_cb(
                tr(ui_language, "transcriber_status_speaker_progress", speaker=utterance["speaker"]),
                progress,
            )

        if cancel_event.is_set():
            raise TranscriptionCancelled()

        resolved_segments = accepted_segments if accepted_segments else baseline_dialogue
        mix_segments: list[TranscriptSegment] = []
        if options.transcribe_mix_track:
            mix_segments = self._transcribe_mix_reference(
                session_dir=session_dir,
                model=model,
                transcription_library=library,
                runtime_device=runtime_device,
                language=session_language,
                beam_size=options.beam_size,
                cancel_event=cancel_event,
                progress_cb=progress_cb,
                ui_language=ui_language,
            )

        polished_segments = self._polish_segments(
            resolved_segments,
            reference_segments=mix_segments,
        )
        merged = self._finalize_segments(polished_segments)
        for track in tracks:
            speaker = str(track["speaker"])
            track["resolved_segments"] = [seg for seg in merged if seg.speaker == speaker]

        out_path = transcript_path(session_dir)
        self._write_transcript_output(
            out_path,
            segments=merged,
            include_timestamps=options.include_timestamps,
        )
        self._write_debug_transcript_outputs(
            session_dir=session_dir,
            tracks=tracks,
            mix_segments=mix_segments,
            include_timestamps=options.include_timestamps,
            transcribe_mix_track=options.transcribe_mix_track,
        )
        progress_cb(tr(ui_language, "transcriber_status_done"), 100.0)
        return out_path

    def _write_debug_transcript_outputs(
        self,
        session_dir: Path,
        tracks: list[dict],
        mix_segments: list[TranscriptSegment],
        include_timestamps: bool,
        transcribe_mix_track: bool,
    ) -> None:
        for track in tracks:
            transcript_kind = str(track.get("key", "")).strip()
            if transcript_kind not in ("mic", "desktop"):
                continue
            resolved_segments = track.get("resolved_segments")
            segments = resolved_segments if isinstance(resolved_segments, list) else []
            self._write_transcript_output(
                debug_transcript_path(session_dir, transcript_kind),
                segments=segments,
                include_timestamps=include_timestamps,
                include_speakers=False,
            )

        if not transcribe_mix_track:
            debug_transcript_path(session_dir, "mix").unlink(missing_ok=True)
            return

        self._write_transcript_output(
            debug_transcript_path(session_dir, "mix"),
            segments=mix_segments,
            include_timestamps=include_timestamps,
            include_speakers=False,
        )

    def _transcribe_audio(
        self,
        model,
        transcription_library: str,
        runtime_device: str,
        audio_input,
        beam_size: int,
        language: str | None,
        initial_prompt: str | None = None,
        condition_on_previous_text: bool = False,
        word_timestamps: bool = True,
        without_timestamps: bool = False,
    ) -> list:
        language_arg = language.strip() if isinstance(language, str) else None
        if language_arg == "":
            language_arg = None

        if transcription_library == TRANSCRIPTION_LIBRARY_FASTER:
            transcribe_kwargs = {
                "beam_size": beam_size,
                # Keep hard VAD off: we pre-cut speech regions ourselves.
                "vad_filter": False,
                "condition_on_previous_text": condition_on_previous_text,
                "without_timestamps": without_timestamps,
                "word_timestamps": word_timestamps,
                "no_speech_threshold": NO_SPEECH_PROB_THRESHOLD,
            }
            if language_arg:
                transcribe_kwargs["language"] = language_arg
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt
            segments_iter, _info = model.transcribe(audio_input, **transcribe_kwargs)
            return list(segments_iter)

        if transcription_library == TRANSCRIPTION_LIBRARY_WHISPER:
            transcribe_kwargs = {
                "beam_size": beam_size,
                "condition_on_previous_text": condition_on_previous_text,
                "word_timestamps": word_timestamps,
                "fp16": runtime_device == "cuda",
                "verbose": None,
                "no_speech_threshold": NO_SPEECH_PROB_THRESHOLD,
            }
            if language_arg:
                transcribe_kwargs["language"] = language_arg
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt
            if without_timestamps:
                transcribe_kwargs["without_timestamps"] = True
            prepared_audio_input = self._prepare_openai_whisper_audio_input(audio_input)
            with self._redirect_missing_standard_streams():
                result = model.transcribe(prepared_audio_input, **transcribe_kwargs)
            segments = result.get("segments") if isinstance(result, dict) else []
            return segments if isinstance(segments, list) else []

        raise RuntimeError(f"Unsupported transcription library: {transcription_library}")

    def _detect_speech_ranges(
        self,
        audio_file: sf.SoundFile,
        strict_filter: bool,
    ) -> list[tuple[float, float]]:
        sample_rate = int(audio_file.samplerate)
        if sample_rate <= 0 or audio_file.frames <= 0:
            return []

        frame_samples = max(1, int(sample_rate * (SPEECH_GATE_FRAME_MS / 1000.0)))
        min_speech_frames = max(
            1,
            int(math.ceil(SPEECH_GATE_MIN_SPEECH_SEC * sample_rate / frame_samples)),
        )
        min_silence_frames = max(
            1,
            int(math.ceil(SPEECH_GATE_MIN_SILENCE_SEC * sample_rate / frame_samples)),
        )
        if strict_filter:
            min_speech_frames += 1

        audio_file.seek(0)
        rms_values: list[float] = []
        while True:
            block = audio_file.read(frame_samples, dtype="float32", always_2d=False)
            if len(block) == 0:
                break
            mono = to_mono(block)
            if len(mono) == 0:
                rms_values.append(0.0)
                continue
            squared = mono * mono
            rms_values.append(float(squared.mean() ** 0.5))

        if not rms_values:
            return []

        positive = sorted(value for value in rms_values if value > 0.0)
        if not positive:
            return []

        noise_floor = self._percentile(positive, 0.30)
        speech_peak = self._percentile(positive, 0.95)
        threshold = max(
            MIN_AUDIO_RMS * SPEECH_GATE_FLOOR_RATIO,
            noise_floor * SPEECH_GATE_THRESHOLD_MULTIPLIER,
        )
        if speech_peak > 0.0:
            threshold = min(threshold, speech_peak * 0.88)
        if strict_filter:
            threshold *= 1.08

        raw_ranges: list[tuple[int, int]] = []
        total_frames = len(rms_values)
        onset_threshold = threshold * SPEECH_GATE_ONSET_RATIO
        if strict_filter:
            onset_threshold *= 1.04
        release_threshold = max(
            MIN_AUDIO_RMS * 0.90,
            min(onset_threshold, threshold * SPEECH_GATE_RELEASE_RATIO),
        )

        frame_idx = 0
        while frame_idx < total_frames:
            if rms_values[frame_idx] < onset_threshold:
                frame_idx += 1
                continue

            start_idx = frame_idx
            last_active_idx = frame_idx + 1
            silence_run = 0
            frame_idx += 1
            while frame_idx < total_frames:
                if rms_values[frame_idx] >= release_threshold:
                    last_active_idx = frame_idx + 1
                    silence_run = 0
                    frame_idx += 1
                    continue

                silence_run += 1
                if silence_run > min_silence_frames:
                    break
                frame_idx += 1

            end_idx = last_active_idx
            if end_idx - start_idx >= min_speech_frames:
                raw_ranges.append((start_idx, end_idx))

        total_duration = float(audio_file.frames) / float(sample_rate)
        if not raw_ranges:
            average_rms = sum(rms_values) / len(rms_values)
            if average_rms >= MIN_AUDIO_RMS * 1.2:
                return [(0.0, total_duration)]
            return []

        merged_ranges: list[tuple[int, int]] = [raw_ranges[0]]
        for start_idx, end_idx in raw_ranges[1:]:
            prev_start, prev_end = merged_ranges[-1]
            if start_idx - prev_end <= min_silence_frames:
                merged_ranges[-1] = (prev_start, end_idx)
            else:
                merged_ranges.append((start_idx, end_idx))

        pad_sec = SPEECH_GATE_PAD_SEC
        if strict_filter:
            pad_sec *= 0.9

        resolved: list[tuple[float, float]] = []
        for start_idx, end_idx in merged_ranges:
            start_sec = max(0.0, (start_idx * frame_samples) / sample_rate - pad_sec)
            end_sec = min(total_duration, (end_idx * frame_samples) / sample_rate + pad_sec)
            if end_sec <= start_sec:
                continue
            resolved.append((start_sec, end_sec))

        if not resolved:
            return []

        collapsed: list[tuple[float, float]] = [resolved[0]]
        for start_sec, end_sec in resolved[1:]:
            prev_start, prev_end = collapsed[-1]
            if start_sec <= prev_end:
                collapsed[-1] = (prev_start, max(prev_end, end_sec))
            else:
                collapsed.append((start_sec, end_sec))
        return collapsed

    def _segment_with_time_offset(self, segment, offset: float) -> dict:
        start = (self._safe_float(self._segment_field(segment, "start")) or 0.0) + offset
        end = (self._safe_float(self._segment_field(segment, "end")) or 0.0) + offset
        text = str(self._segment_field(segment, "text", "") or "")
        words = self._segment_field(segment, "words") or []
        shifted_words: list[dict] = []
        for word in words:
            token = str(self._word_field(word, "word", "") or "")
            word_start = self._safe_float(self._word_field(word, "start"))
            word_end = self._safe_float(self._word_field(word, "end"))
            shifted_words.append(
                {
                    "word": token,
                    "start": (word_start + offset) if word_start is not None else None,
                    "end": (word_end + offset) if word_end is not None else None,
                }
            )
        return {
            "start": start,
            "end": max(start, end),
            "text": text,
            "words": shifted_words,
            "no_speech_prob": self._segment_field(segment, "no_speech_prob"),
            "avg_logprob": self._segment_field(segment, "avg_logprob"),
        }

    @staticmethod
    def _read_audio_slice(audio_file: sf.SoundFile, start: float, end: float):
        sample_rate = int(audio_file.samplerate)
        if sample_rate <= 0:
            return []

        start_frame = max(0, int(start * sample_rate))
        end_frame = min(audio_file.frames, int(end * sample_rate))
        frame_count = end_frame - start_frame
        if frame_count <= 0:
            return []

        audio_file.seek(start_frame)
        samples = audio_file.read(frame_count, dtype="float32", always_2d=False)
        if len(samples) == 0:
            return []
        return to_mono(samples)

    @staticmethod
    def _percentile(sorted_values: list[float], quantile: float) -> float:
        if not sorted_values:
            return 0.0
        clamped = min(max(quantile, 0.0), 1.0)
        index = int(round((len(sorted_values) - 1) * clamped))
        return sorted_values[index]

    @staticmethod
    def _coalesce_ranges_for_transcription(
        ranges: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if not ranges:
            return []

        merged: list[tuple[float, float]] = [ranges[0]]
        for start, end in ranges[1:]:
            prev_start, prev_end = merged[-1]
            candidate_end = max(prev_end, end)
            candidate_duration = candidate_end - prev_start
            if (
                start - prev_end <= SPEECH_CHUNK_BRIDGE_SEC
                and candidate_duration <= SPEECH_CHUNK_MAX_SEC
            ):
                merged[-1] = (prev_start, candidate_end)
                continue
            merged.append((start, end))
        return merged

    def _merge_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        ordered = sorted(segments, key=lambda s: (s.start, s.end))
        merged: list[TranscriptSegment] = []

        for seg in ordered:
            if not merged:
                merged.append(seg)
                continue
            prev = merged[-1]
            merged_overlap = self._merge_same_speaker_segments(prev, seg)
            if merged_overlap is not None:
                merged[-1] = merged_overlap
                continue
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

    @staticmethod
    def _segment_field(segment, field: str, default=None):
        if isinstance(segment, dict):
            return segment.get(field, default)
        return getattr(segment, field, default)

    @staticmethod
    def _word_field(word, field: str, default=None):
        if isinstance(word, dict):
            return word.get(field, default)
        return getattr(word, field, default)

    def _split_segment_by_pauses(self, seg, speaker: str) -> list[TranscriptSegment]:
        words = self._segment_field(seg, "words") or []
        if not words:
            text = normalize_text(str(self._segment_field(seg, "text", "") or ""))
            if not text:
                return []
            seg_start = self._safe_float(self._segment_field(seg, "start")) or 0.0
            seg_end = self._safe_float(self._segment_field(seg, "end")) or seg_start
            return [
                TranscriptSegment(
                    start=seg_start,
                    end=seg_end,
                    speaker=speaker,
                    text=text,
                )
            ]

        seg_start = self._safe_float(self._segment_field(seg, "start")) or 0.0
        seg_end = self._safe_float(self._segment_field(seg, "end")) or seg_start
        result: list[TranscriptSegment] = []
        current_tokens: list[str] = []
        current_token_count = 0
        current_start = seg_start
        current_end = seg_start

        def flush_chunk() -> None:
            nonlocal current_tokens, current_token_count, current_start, current_end
            if not current_tokens:
                return

            text = normalize_text("".join(current_tokens))
            if not text:
                current_tokens = []
                current_token_count = 0
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
            current_token_count = 0

        for word in words:
            token = str(self._word_field(word, "word", "") or "")
            if not token:
                continue

            word_start = self._safe_float(self._word_field(word, "start"))
            word_end = self._safe_float(self._word_field(word, "end"))

            if current_tokens and word_start is not None:
                gap = word_start - current_end
                current_duration = max(0.0, current_end - current_start)
                if gap >= WORD_PAUSE_SPLIT_GAP or (
                    gap >= TRANSCRIPT_SOFT_SPLIT_GAP
                    and (
                        current_duration >= TRANSCRIPT_CHUNK_SOFT_MAX_SEC
                        or current_token_count >= TRANSCRIPT_CHUNK_SOFT_MAX_WORDS
                    )
                ):
                    flush_chunk()
                    current_start = word_start
                    current_end = word_start

            if not current_tokens:
                if word_start is not None:
                    current_start = word_start
                else:
                    current_start = max(seg_start, current_start)

            current_tokens.append(token)
            current_token_count += 1
            if word_end is not None:
                current_end = word_end
            elif word_start is not None:
                current_end = word_start

            current_duration = max(0.0, current_end - current_start)
            token_tail = token.rstrip()
            soft_boundary = token_tail.endswith((".", "!", "?", ";", ":", ",", "…"))
            if (
                current_duration >= TRANSCRIPT_CHUNK_HARD_MAX_SEC
                or current_token_count >= TRANSCRIPT_CHUNK_HARD_MAX_WORDS
            ):
                flush_chunk()
                current_start = current_end
                continue
            if soft_boundary and (
                current_duration >= TRANSCRIPT_CHUNK_SOFT_MAX_SEC
                or current_token_count >= TRANSCRIPT_CHUNK_SOFT_MAX_WORDS
            ):
                flush_chunk()
                current_start = current_end

        flush_chunk()
        if result:
            return result

        text = normalize_text(str(self._segment_field(seg, "text", "") or ""))
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
        if self._looks_like_silence_hallucination(chunk.text) and rms < (rms_floor * 1.4):
            return True

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

    @classmethod
    def _looks_like_silence_hallucination(cls, text: str) -> bool:
        normalized = normalize_text(text).strip(" .,:;!?").lower()
        if not normalized:
            return False
        return any(fragment in normalized for fragment in cls.SILENCE_HALLUCINATION_FRAGMENTS)

    @classmethod
    def _strip_hallucination_phrases(cls, text: str) -> str:
        cleaned = normalize_text(text)
        if not cleaned:
            return ""

        for pattern in cls.HALLUCINATION_PATTERNS:
            cleaned = pattern.sub(" ", cleaned)

        cleaned = re.sub(r"\s+([,.;:!?…])", r"\1", cleaned)
        cleaned = re.sub(r"([(\[{])\s+", r"\1", cleaned)
        cleaned = re.sub(r"\s+([)\]}])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = cleaned.strip(" ,.;:!?…")
        return normalize_text(cleaned)

    @staticmethod
    def _tokenize_text(text: str) -> list[str]:
        normalized = normalize_text(text).lower()
        if not normalized:
            return []
        return [token for token in re.findall(r"[0-9a-zа-я]+", normalized, flags=re.IGNORECASE) if len(token) > 1]

    @classmethod
    def _word_overlap_ratio(cls, text: str, reference_text: str) -> float:
        text_tokens = cls._tokenize_text(text)
        reference_tokens = set(cls._tokenize_text(reference_text))
        if not text_tokens or not reference_tokens:
            return 0.0
        overlap = sum(1 for token in text_tokens if token in reference_tokens)
        return overlap / len(text_tokens)

    @staticmethod
    def _reference_text_for_range(
        segments: list[TranscriptSegment],
        start: float,
        end: float,
    ) -> str:
        if not segments:
            return ""
        start_bound = start - MIX_REFERENCE_PAD_SEC
        end_bound = end + MIX_REFERENCE_PAD_SEC
        relevant = [
            seg.text
            for seg in segments
            if seg.text and not (seg.end < start_bound or seg.start > end_bound)
        ]
        return normalize_text(" ".join(relevant))

    def _polish_segments(
        self,
        segments: list[TranscriptSegment],
        reference_segments: list[TranscriptSegment] | None = None,
    ) -> list[TranscriptSegment]:
        polished: list[TranscriptSegment] = []
        reference_segments = reference_segments or []

        for seg in sorted(segments, key=lambda item: (item.start, item.end)):
            original_text = normalize_text(seg.text)
            if not original_text:
                continue

            reference_text = self._reference_text_for_range(reference_segments, seg.start, seg.end)
            cleaned_text = self._strip_hallucination_phrases(original_text)
            if not cleaned_text:
                continue

            if cleaned_text != original_text and self._word_count(cleaned_text) <= 2:
                overlap = self._word_overlap_ratio(cleaned_text, reference_text)
                if reference_text and overlap < 0.34:
                    continue

            polished.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    speaker=seg.speaker,
                    text=cleaned_text,
                )
            )
        return polished

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

    @staticmethod
    def _overlap_words(text: str) -> tuple[list[str], list[str]]:
        raw_words = normalize_text(text).split()
        normalized_words = [
            token.strip(".,!?;:…\"'`()[]{}<>").lower()
            for token in raw_words
        ]
        return raw_words, normalized_words

    @staticmethod
    def _contains_word_sequence(haystack: list[str], needle: list[str]) -> bool:
        if not needle or len(needle) > len(haystack):
            return False
        last_start = len(haystack) - len(needle) + 1
        for start_idx in range(last_start):
            if haystack[start_idx : start_idx + len(needle)] == needle:
                return True
        return False

    def _merge_textual_overlap(self, left: str, right: str) -> str | None:
        left_raw, left_norm = self._overlap_words(left)
        right_raw, right_norm = self._overlap_words(right)
        if not left_raw:
            return normalize_text(right)
        if not right_raw:
            return normalize_text(left)

        if left_norm == right_norm:
            return normalize_text(left if len(left_raw) >= len(right_raw) else right)

        min_sequence_words = 6
        if len(right_norm) >= min_sequence_words and self._contains_word_sequence(left_norm, right_norm):
            return normalize_text(left)
        if len(left_norm) >= min_sequence_words and self._contains_word_sequence(right_norm, left_norm):
            return normalize_text(right)

        max_overlap = min(len(left_norm), len(right_norm), 48)
        for overlap_size in range(max_overlap, min_sequence_words - 1, -1):
            if left_norm[-overlap_size:] == right_norm[:overlap_size]:
                merged_words = left_raw + right_raw[overlap_size:]
                return normalize_text(" ".join(merged_words))

        smaller_norm, larger_text, larger_norm = (
            (left_norm, right, right_norm)
            if len(left_norm) <= len(right_norm)
            else (right_norm, left, left_norm)
        )
        smaller_set = {token for token in smaller_norm if token}
        if len(smaller_set) >= min_sequence_words:
            overlap_count = sum(1 for token in smaller_set if token in set(larger_norm))
            overlap_ratio = overlap_count / len(smaller_set)
            if overlap_ratio >= 0.78 and len(larger_norm) >= (len(smaller_norm) + min_sequence_words):
                return normalize_text(larger_text)
        return None

    def _merge_same_speaker_segments(
        self,
        left: TranscriptSegment,
        right: TranscriptSegment,
    ) -> TranscriptSegment | None:
        if left.speaker != right.speaker:
            return None

        gap = right.start - left.end
        if gap > MIN_SPEAKER_GAP:
            return None

        merged_text = self._merge_textual_overlap(left.text, right.text)
        if merged_text is None:
            return None

        return TranscriptSegment(
            start=min(left.start, right.start),
            end=max(left.end, right.end),
            speaker=left.speaker,
            text=merged_text,
        )

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
            merged_overlap = self._merge_same_speaker_segments(prev, seg)
            if merged_overlap is not None:
                collapsed[-1] = merged_overlap
                continue
            if prev.speaker == seg.speaker and self._can_merge_segments(prev, seg):
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

    def _finalize_segments(
        self,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        merged = self._merge_segments(segments)
        return self._collapse_consecutive_speaker_runs(merged)

    @staticmethod
    def _append_text(left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        return f"{left} {right}"

    def _render_text(
        self,
        segments: list[TranscriptSegment],
        include_timestamps: bool,
        include_speakers: bool = True,
    ) -> str:
        lines: list[str] = []
        for seg in segments:
            if include_timestamps:
                timestamp = format_seconds(seg.start)
                if include_speakers:
                    lines.append(f"[{timestamp}] [{seg.speaker}] {seg.text}")
                else:
                    lines.append(f"[{timestamp}] {seg.text}")
            else:
                if include_speakers:
                    lines.append(f"[{seg.speaker}] {seg.text}")
                else:
                    lines.append(seg.text)
        return "\n".join(lines) + "\n"

    def _write_transcript_output(
        self,
        out_path: Path,
        segments: list[TranscriptSegment],
        include_timestamps: bool,
        include_speakers: bool = True,
    ) -> None:
        output_text = self._render_text(
            segments,
            include_timestamps,
            include_speakers=include_speakers,
        )
        out_path.write_text(output_text, encoding="utf-8")
