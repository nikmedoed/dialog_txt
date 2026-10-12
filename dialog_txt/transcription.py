from __future__ import annotations

import contextlib
import importlib
import math
import logging
import json
from dataclasses import asdict, replace
from difflib import SequenceMatcher
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

        audio_data, sample_rate = sf.read(audio_input, dtype="float32", always_2d=False)
        if sample_rate != 16000:
            # Whisper treats arrays as 16 kHz. Passing the recorder's 48 kHz
            # samples unchanged distorts duration, pitch and all timestamps.
            decoder = importlib.import_module("faster_whisper.audio")
            return decoder.decode_audio(audio_input, sampling_rate=16000)
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
        loop_indexes = self._repetitive_segment_indexes(raw_segments)
        for index, raw_seg in enumerate(raw_segments):
            if index in loop_indexes:
                logging.getLogger(__name__).warning(
                    "Rejected decoding loop on %s at %s", speaker,
                    self._segment_field(raw_seg, "start", 0.0),
                )
                continue
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
        text = " ".join(seg.text for seg in segments if seg.text)
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
        speaker: str,
    ) -> str:
        # Cross-speaker prompts make Whisper parrot the other audio track when
        # the current chunk is quiet or unclear. Context is useful, but only
        # from the same physical source/speaker.
        recent_confirmed = [
            seg for seg in accepted_dialogue
            if seg.speaker == speaker
            and utterance_start - BASELINE_CONTEXT_LOOKBACK_SEC <= seg.end <= utterance_start
        ][-8:]
        if recent_confirmed:
            return self._prompt_from_segments(recent_confirmed, DIALOG_CONTEXT_PROMPT_WORDS)

        baseline_tail = [
            seg
            for seg in baseline_dialogue
            if seg.speaker == speaker
            and seg.end <= utterance_start
            and seg.end >= (utterance_start - BASELINE_CONTEXT_LOOKBACK_SEC)
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
        cancel_event: threading.Event | None = None,
        max_chunk_sec: float = 30.0,
    ) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        for range_start, range_end in speech_ranges:
            start = range_start
            while start < range_end:
                if cancel_event is not None and cancel_event.is_set():
                    raise TranscriptionCancelled()
                end = min(range_end, start + max_chunk_sec)
                segments.extend(self._transcribe_utterance_range(
                    model=model,
                    transcription_library=transcription_library,
                    runtime_device=runtime_device,
                    audio_path=audio_path,
                    speaker=speaker,
                    utterance_start=start,
                    utterance_end=end,
                    beam_size=beam_size,
                    language=language,
                    strict_filter=strict_filter,
                    prompt="",
                ))
                if end >= range_end:
                    break
                start = end - min(2.0, max_chunk_sec / 4)
        return segments

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
        return filtered

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

        # A mix track is a full third transcription pass, not the last 3%.
        progress_cb(tr(ui_language, "transcriber_status_mix"), 67.0)
        with sf.SoundFile(str(mix_path), mode="r") as audio_file:
            speech_ranges = self._detect_speech_ranges(
                audio_file=audio_file,
                strict_filter=False,
            )
            duration = audio_file.frames / audio_file.samplerate
        speech_ranges = self._coalesce_ranges_for_transcription(speech_ranges)
        if not speech_ranges:
            return []

        return self._transcribe_track_baseline(
            model=model,
            transcription_library=transcription_library,
            runtime_device=runtime_device,
            audio_path=mix_path,
            speaker="",
            # Preserve both sides and pauses as acoustic context in the mix.
            # Physical speech support still gates the resulting document.
            speech_ranges=[(max(0.0, speech_ranges[0][0] - 1.0),
                            min(duration, speech_ranges[-1][1] + 1.0))],
            beam_size=beam_size,
            language=language,
            strict_filter=False,
            cancel_event=cancel_event,
            max_chunk_sec=60.0,
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
            track["activity_ranges"] = speech_ranges
            track["speech_ranges"] = self._coalesce_ranges_for_transcription(speech_ranges)

        baseline_segments: list[TranscriptSegment] = []
        for index, track in enumerate(tracks):
            if cancel_event.is_set():
                raise TranscriptionCancelled()
            baseline_progress = 5.0 + index * 10.0
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
                    cancel_event=cancel_event,
                )
                track["baseline_segments"] = track_segments
                baseline_segments.extend(track_segments)
            else:
                track["baseline_segments"] = []
            progress_cb(
                tr(ui_language, "transcriber_status_speaker_progress", speaker=track["speaker"]),
                baseline_progress + 10.0,
            )
            # Make each independent track useful immediately. It will be
            # refined/overwritten as utterance-level recognition progresses.
            self._write_transcript_output(
                debug_transcript_path(session_dir, str(track["key"])),
                segments=list(track["baseline_segments"]),
                include_timestamps=options.include_timestamps,
                include_speakers=False,
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
                speaker=str(utterance["speaker"]),
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
            # Empty recognition is evidence of silence/uncertainty. Restoring
            # the baseline here used to resurrect rejected hallucinations.
            if utterance_segments:
                accepted_segments.extend(self._polish_segments(utterance_segments))

            # Persist incremental per-track results. Besides being safer on a
            # long run, the network server can return these files while the job
            # is still running.
            speaker_segments = [
                seg for seg in accepted_segments if seg.speaker == str(utterance["speaker"])
            ]
            track_key = next(
                str(track["key"])
                for track in tracks
                if str(track["speaker"]) == str(utterance["speaker"])
            )
            self._write_transcript_output(
                debug_transcript_path(session_dir, track_key),
                segments=self._finalize_segments(speaker_segments),
                include_timestamps=options.include_timestamps,
                include_speakers=False,
            )
            self._write_transcript_output(
                transcript_path(session_dir),
                segments=self._finalize_segments(accepted_segments),
                include_timestamps=options.include_timestamps,
            )

            ratio = (index + 1) / total_utterances
            utterance_end_progress = 66.0 if options.transcribe_mix_track else 98.0
            progress = 25.0 + ratio * (utterance_end_progress - 25.0)
            progress_cb(
                tr(ui_language, "transcriber_status_speaker_progress", speaker=utterance["speaker"]),
                progress,
            )

        if cancel_event.is_set():
            raise TranscriptionCancelled()

        resolved_segments = accepted_segments
        for track in tracks:
            track["recognized_segments"] = [
                seg for seg in accepted_segments if seg.speaker == track["speaker"]
            ]
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
            if (session_dir / MIX_FILE_NAME).exists():
                # The mixed signal normally gives Whisper the cleanest lexical
                # result. Use the separate tracks for diarization instead of
                # throwing the expensive mix pass away as a debug-only output.
                resolved_segments = self._assign_mix_speakers(mix_segments, tracks)

        polished_segments = self._polish_segments(
            resolved_segments,
            reference_segments=mix_segments,
        )
        polished_segments = self._remove_cross_track_echoes(
            polished_segments,
            preferred_speaker=other_label,
        )
        merged = self._finalize_segments(polished_segments)
        for track in tracks:
            speaker = str(track["speaker"])
            track["resolved_segments"] = self._finalize_segments(self._polish_segments(
                [seg for seg in accepted_segments if seg.speaker == speaker]
            ))

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
        (session_dir / "transcription_debug.json").write_text(json.dumps({
            "pipeline_version": 3,
            "options": asdict(options),
            "runtime": {"library": library, "device": runtime_device,
                        "compute_type": _runtime_compute, "language": session_language},
            "tracks": [{"key": track["key"], "speaker": track["speaker"],
                        "activity_ranges": track["activity_ranges"],
                        "segments": [asdict(seg) for seg in track["recognized_segments"]]}
                       for track in tracks],
            "mix_segments": [asdict(seg) for seg in mix_segments],
            "final_segments": [asdict(seg) for seg in merged],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        progress_cb(tr(ui_language, "transcriber_status_done"), 100.0)
        return out_path

    def _assign_mix_speakers(
        self,
        mix_segments: list[TranscriptSegment],
        tracks: list[dict],
    ) -> list[TranscriptSegment]:
        with contextlib.ExitStack() as stack:
            measured_tracks = []
            for track in tracks:
                measured = dict(track)
                if track.get("audio_path"):
                    measured["audio_file"] = stack.enter_context(
                        sf.SoundFile(str(track["audio_path"]), mode="r")
                    )
                measured_tracks.append(measured)
            return self._assign_mix_speakers_with_evidence(mix_segments, measured_tracks)

    def _assign_mix_speakers_with_evidence(
        self,
        mix_segments: list[TranscriptSegment],
        tracks: list[dict],
    ) -> list[TranscriptSegment]:
        if not tracks:
            return []
        assigned: list[TranscriptSegment] = []
        # Keep lexical content from the mix, but attribute individual timed
        # words. A model sentence can contain both sides of a conversation.
        units: list[TranscriptSegment] = []
        for segment in mix_segments:
            if segment.words:
                # Whisper can attach valid function words to a zero-length
                # timestamp. Give those words a small attribution interval.
                units.extend(TranscriptSegment(max(0.0, start - .02) if end <= start else start,
                                               end + .02 if end <= start else end, "", text)
                             for start, end, text in segment.words)
            else:
                units.append(segment)
        for segment in units:
            energy = [self._segment_rms(track["audio_file"], segment.start, segment.end)
                      if "audio_file" in track else 0.0 for track in tracks]
            total_energy = sum(energy)
            best_speaker = str(tracks[0]["speaker"])
            best_score = -1.0
            supported = False
            for track_index, track in enumerate(tracks):
                ranges = track.get("activity_ranges", track.get("speech_ranges"))
                if not isinstance(ranges, list):
                    ranges = []
                overlap = sum(
                    max(0.0, min(segment.end, end + .75) - max(segment.start, start - .75))
                    for start, end in ranges
                )
                if overlap <= 0:
                    continue
                supported = True
                recognized = track.get("recognized_segments", [])
                local_text = normalize_text(" ".join(
                    text for seg in recognized for start, end, text in seg.words
                    if end >= segment.start - .75 and start <= segment.end + .75
                )) or self._reference_text_for_range(recognized, segment.start, segment.end)
                lexical = self._word_overlap_ratio(segment.text, local_text)
                score = overlap / max(.01, segment.end - segment.start) + 2.0 * lexical
                if total_energy:
                    score += 1.5 * energy[track_index] / total_energy
                if score > best_score:
                    best_score = score
                    best_speaker = str(track["speaker"])
            # Mixed audio supplies words, physical tracks supply speech support.
            # Never attribute a phrase recognized during silence to a default speaker.
            if not supported:
                logging.getLogger(__name__).info(
                    "Rejected unsupported mix segment %.2f-%.2f", segment.start, segment.end
                )
                continue
            if (assigned and assigned[-1].speaker == best_speaker
                    and -.04 <= segment.start - assigned[-1].end <= WORD_PAUSE_SPLIT_GAP
                    and segment.end - assigned[-1].start <= TRANSCRIPT_CHUNK_HARD_MAX_SEC
                    and self._word_count(assigned[-1].text) < TRANSCRIPT_CHUNK_HARD_MAX_WORDS):
                assigned[-1].text = normalize_text(assigned[-1].text + " " + segment.text)
                assigned[-1].end = segment.end
                continue
            assigned.append(
                TranscriptSegment(
                    start=segment.start,
                    end=segment.end,
                    speaker=best_speaker,
                    text=normalize_text(segment.text),
                )
            )
        return assigned

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
        _retry_loop: bool = True,
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
            if not _retry_loop:
                transcribe_kwargs["temperature"] = 0.0
            segments_iter, _info = model.transcribe(audio_input, **transcribe_kwargs)
            segments = list(segments_iter)
            return self._retry_repetitive_audio(
                segments, model, transcription_library, runtime_device, audio_input,
                beam_size, language, word_timestamps, without_timestamps, _retry_loop,
            )

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
            if not _retry_loop:
                transcribe_kwargs["temperature"] = 0.0
            if without_timestamps:
                transcribe_kwargs["without_timestamps"] = True
            prepared_audio_input = self._prepare_openai_whisper_audio_input(audio_input)
            with self._redirect_missing_standard_streams():
                result = model.transcribe(prepared_audio_input, **transcribe_kwargs)
            segments = result.get("segments") if isinstance(result, dict) else []
            return self._retry_repetitive_audio(
                segments if isinstance(segments, list) else [], model,
                transcription_library, runtime_device, audio_input, beam_size,
                language, word_timestamps, without_timestamps, _retry_loop,
            )

        raise RuntimeError(f"Unsupported transcription library: {transcription_library}")

    def _retry_repetitive_audio(
        self, segments, model, library, device, audio_input, beam_size,
        language, word_timestamps, without_timestamps, retry,
    ) -> list:
        if retry and self._repetitive_segment_indexes(segments):
            logging.getLogger(__name__).warning("Retrying repetitive decoding without text context")
            return self._transcribe_audio(
                model, library, device, audio_input, max(8, beam_size), language,
                initial_prompt=None, condition_on_previous_text=False,
                word_timestamps=word_timestamps, without_timestamps=without_timestamps,
                _retry_loop=False,
            )
        return segments

    def _repetitive_segment_indexes(self, segments: list) -> set[int]:
        rejected = {index for index, seg in enumerate(segments)
                    if self._has_repetition_loop(str(self._segment_field(seg, "text", "")))}
        run: list[int] = []
        previous_tokens: list[str] = []
        previous_end = 0.0

        def finish_run() -> None:
            if len(run) >= 4 and len(run) * len(previous_tokens) >= 12:
                rejected.update(run)

        for index, seg in enumerate(segments):
            tokens = self._overlap_words(str(self._segment_field(seg, "text", "")))[1]
            start = self._safe_float(self._segment_field(seg, "start")) or 0.0
            if tokens != previous_tokens or start - previous_end > 1.5:
                finish_run()
                run = []
            run.append(index)
            previous_tokens = tokens
            previous_end = self._safe_float(self._segment_field(seg, "end")) or start
        finish_run()
        return rejected

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
        # Incremental rendering must not rewrite recognition evidence used by
        # later prompts and word-level speaker attribution.
        segments = [replace(segment, words=list(segment.words)) for segment in segments]
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
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
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
        current_words: list[tuple[float, float, str]] = []
        current_token_count = 0
        current_start = seg_start
        current_end = seg_start

        def flush_chunk() -> None:
            nonlocal current_tokens, current_words, current_token_count, current_start, current_end
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
                    words=list(current_words),
                )
            )
            current_tokens = []
            current_words = []
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
            if word_start is not None and word_end is not None:
                current_words.append((word_start, word_end, token))
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
        # Absolute silence and failed decoding remain invalid even for long
        # segments, and even when the optional stricter VAD mode is disabled.
        if rms < MIN_AUDIO_RMS:
            return True
        if (no_speech_prob is not None and no_speech_prob >= NO_SPEECH_PROB_THRESHOLD
                and avg_logprob is not None and avg_logprob <= -1.0):
            return True
        if self._has_repetition_loop(chunk.text):
            return True
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
    def _has_repetition_loop(cls, text: str) -> bool:
        """Detect sustained exact loops, preserving normal double repetitions."""
        tokens = re.findall(r"\w+", text.casefold())
        for width in range(1, min(24, len(tokens) // 4) + 1):
            for start in range(len(tokens) - width * 4 + 1):
                phrase = tokens[start:start + width]
                count = 1
                while tokens[start + count * width:start + (count + 1) * width] == phrase:
                    count += 1
                if count >= 4 and count * width >= 12:
                    return True
        return False

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

        for seg in sorted(segments, key=lambda item: (item.start, item.end)):
            original_text = normalize_text(seg.text)
            if not original_text:
                continue

            # Phrase lists are acoustic warning signs, not editing rules:
            # someone may actually say "thank you for watching".
            cleaned_text = original_text
            polished.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    speaker=seg.speaker,
                    text=cleaned_text,
                    words=list(seg.words),
                )
            )
        return polished

    def _remove_cross_track_echoes(
        self,
        segments: list[TranscriptSegment],
        preferred_speaker: str,
    ) -> list[TranscriptSegment]:
        """Remove the same utterance recognized on both physical tracks.

        Desktop is preferred for a simultaneous near-verbatim duplicate: its
        digital capture is authoritative, while the microphone copy is usually
        speaker/headphone bleed. Non-overlapping repetitions are preserved.
        """
        kept: list[TranscriptSegment] = []
        for candidate in sorted(segments, key=lambda item: (item.start, item.end)):
            duplicate_index: int | None = None
            for index in range(len(kept) - 1, -1, -1):
                existing = kept[index]
                if existing.end < candidate.start:
                    continue
                if existing.speaker == candidate.speaker:
                    continue
                overlap_duration = min(existing.end, candidate.end) - max(
                    existing.start, candidate.start
                )
                shorter_duration = min(
                    max(0.01, existing.end - existing.start),
                    max(0.01, candidate.end - candidate.start),
                )
                if overlap_duration / shorter_duration < 0.50:
                    continue
                left_tokens = self._overlap_words(existing.text)[1]
                right_tokens = self._overlap_words(candidate.text)[1]
                text_overlap = SequenceMatcher(None, left_tokens, right_tokens, autojunk=False).ratio()
                def factual_tokens(tokens):
                    return [token for token in tokens if any(char.isdigit() for char in token)
                            or token in {"not", "no", "never", "cannot", "не", "нет", "нельзя"}
                            or token.endswith("n't")]
                if left_tokens == right_tokens or (
                    min(len(left_tokens), len(right_tokens)) >= 6 and text_overlap >= 0.92
                    and factual_tokens(left_tokens) == factual_tokens(right_tokens)
                ):
                    duplicate_index = index
                    break

            if duplicate_index is None:
                kept.append(candidate)
                continue
            if candidate.speaker == preferred_speaker:
                kept[duplicate_index] = candidate
        return sorted(kept, key=lambda item: (item.start, item.end))

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
        for overlap_size in range(max_overlap, 1, -1):
            if left_norm[-overlap_size:] == right_norm[:overlap_size]:
                # Use the continuing window's punctuation on shared words,
                # rather than retaining a premature sentence end at the cut.
                merged_words = left_raw[:-overlap_size] + right_raw
                return normalize_text(" ".join(merged_words))

        # A shared vocabulary is not duplication: reordered words or changed
        # negation/numbers may represent a different claim.
        return None

    def _merge_same_speaker_segments(
        self,
        left: TranscriptSegment,
        right: TranscriptSegment,
    ) -> TranscriptSegment | None:
        if left.speaker != right.speaker:
            return None

        gap = right.start - left.end
        if gap >= 0:
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
