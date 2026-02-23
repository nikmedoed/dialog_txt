from __future__ import annotations

import json
from pathlib import Path

from .config import APP_SETTINGS_FILE, DEFAULT_OTHER_LABEL, DEFAULT_SELF_LABEL


DEFAULT_MODEL = "large-v3"
DEFAULT_LANGUAGE = "ru"
DEFAULT_BEAM_SIZE = 5
DEFAULT_VAD_FILTER = True
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_AUTO_TRANSCRIBE_AFTER_RECORD = True
DEFAULT_INCLUDE_TIMESTAMPS = False

ALLOWED_MODELS = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "distil-large-v3",
)
ALLOWED_COMPUTE_TYPES = ("float16", "int8_float16", "int8")


def _default_settings() -> dict:
    return {
        "last_microphone": "",
        "speaker_self": DEFAULT_SELF_LABEL,
        "speaker_other": DEFAULT_OTHER_LABEL,
        "auto_transcribe_after_record": DEFAULT_AUTO_TRANSCRIBE_AFTER_RECORD,
        "whisper_model": DEFAULT_MODEL,
        "whisper_language": DEFAULT_LANGUAGE,
        "whisper_beam_size": DEFAULT_BEAM_SIZE,
        "whisper_vad_filter": DEFAULT_VAD_FILTER,
        "whisper_compute_type": DEFAULT_COMPUTE_TYPE,
        "include_timestamps": DEFAULT_INCLUDE_TIMESTAMPS,
    }


def _normalize_label(value: str, fallback: str) -> str:
    cleaned = " ".join((value or "").split())
    return cleaned or fallback


def _normalize_bool(value, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    return fallback


def _normalize_int(value, fallback: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, parsed))


def _sanitize_settings(raw: dict | None) -> dict:
    payload = raw or {}
    model = str(payload.get("whisper_model", DEFAULT_MODEL)).strip() or DEFAULT_MODEL
    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL

    language = " ".join(str(payload.get("whisper_language", DEFAULT_LANGUAGE)).split())
    language = language or DEFAULT_LANGUAGE

    compute_type = str(payload.get("whisper_compute_type", DEFAULT_COMPUTE_TYPE)).strip()
    if compute_type not in ALLOWED_COMPUTE_TYPES:
        compute_type = DEFAULT_COMPUTE_TYPE

    return {
        "last_microphone": " ".join(str(payload.get("last_microphone", "")).split()),
        "speaker_self": _normalize_label(payload.get("speaker_self", ""), DEFAULT_SELF_LABEL),
        "speaker_other": _normalize_label(payload.get("speaker_other", ""), DEFAULT_OTHER_LABEL),
        "auto_transcribe_after_record": _normalize_bool(
            payload.get("auto_transcribe_after_record", DEFAULT_AUTO_TRANSCRIBE_AFTER_RECORD),
            DEFAULT_AUTO_TRANSCRIBE_AFTER_RECORD,
        ),
        "whisper_model": model,
        "whisper_language": language,
        "whisper_beam_size": _normalize_int(
            payload.get("whisper_beam_size", DEFAULT_BEAM_SIZE),
            DEFAULT_BEAM_SIZE,
            1,
            10,
        ),
        "whisper_vad_filter": _normalize_bool(
            payload.get("whisper_vad_filter", DEFAULT_VAD_FILTER),
            DEFAULT_VAD_FILTER,
        ),
        "whisper_compute_type": compute_type,
        "include_timestamps": _normalize_bool(
            payload.get("include_timestamps", DEFAULT_INCLUDE_TIMESTAMPS),
            DEFAULT_INCLUDE_TIMESTAMPS,
        ),
    }


def load_app_settings(path: Path = APP_SETTINGS_FILE) -> dict:
    if not path.exists():
        return _default_settings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _default_settings()
    return _sanitize_settings(raw)


def save_app_settings(settings: dict, path: Path = APP_SETTINGS_FILE) -> None:
    payload = _sanitize_settings(settings)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
