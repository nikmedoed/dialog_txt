from __future__ import annotations

import json
from pathlib import Path

from .config import APP_SETTINGS_FILE, DEFAULT_OTHER_LABEL, DEFAULT_SELF_LABEL
from .transcription_backends import (
    ALLOWED_DEVICES,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_BY_LIBRARY,
    DEFAULT_TRANSCRIPTION_LIBRARY,
    default_model_for_library,
    normalize_transcription_library,
    transcription_models_for_library,
)


DEFAULT_MODEL = DEFAULT_MODEL_BY_LIBRARY[DEFAULT_TRANSCRIPTION_LIBRARY]
DEFAULT_LANGUAGE = "ru"
DEFAULT_BEAM_SIZE = 5
DEFAULT_VAD_FILTER = True
DEFAULT_COMPUTE_TYPE = "auto"
DEFAULT_AUTO_TRANSCRIBE_AFTER_RECORD = True
DEFAULT_INCLUDE_TIMESTAMPS = False
DEFAULT_TRANSCRIBE_MIX_TRACK = False
DEFAULT_UI_LANGUAGE = "ru"
DEFAULT_TRANSCRIPTION_MODE = "local"
DEFAULT_NETWORK_WHISPER_URL = "http://127.0.0.1:8765"
DEFAULT_NETWORK_WHISPER_TOKEN = ""

ALLOWED_COMPUTE_TYPES = (
    "auto",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32",
)
ALLOWED_UI_LANGUAGES = ("ru", "en")


def _default_settings() -> dict:
    return {
        "last_microphone": "",
        "speaker_self": DEFAULT_SELF_LABEL,
        "speaker_other": DEFAULT_OTHER_LABEL,
        "auto_transcribe_after_record": DEFAULT_AUTO_TRANSCRIBE_AFTER_RECORD,
        "transcription_library": DEFAULT_TRANSCRIPTION_LIBRARY,
        "whisper_model": DEFAULT_MODEL,
        "whisper_device": DEFAULT_DEVICE,
        "whisper_language": DEFAULT_LANGUAGE,
        "whisper_beam_size": DEFAULT_BEAM_SIZE,
        "whisper_vad_filter": DEFAULT_VAD_FILTER,
        "whisper_compute_type": DEFAULT_COMPUTE_TYPE,
        "include_timestamps": DEFAULT_INCLUDE_TIMESTAMPS,
        "transcribe_mix_track": DEFAULT_TRANSCRIBE_MIX_TRACK,
        "ui_language": DEFAULT_UI_LANGUAGE,
        "transcription_mode": DEFAULT_TRANSCRIPTION_MODE,
        "network_whisper_url": DEFAULT_NETWORK_WHISPER_URL,
        "network_whisper_token": DEFAULT_NETWORK_WHISPER_TOKEN,
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
    transcription_library = normalize_transcription_library(
        payload.get("transcription_library", DEFAULT_TRANSCRIPTION_LIBRARY)
    )
    allowed_models = transcription_models_for_library(transcription_library)
    default_model = default_model_for_library(transcription_library, allowed_models)
    model = str(payload.get("whisper_model", default_model)).strip() or default_model
    if model not in allowed_models:
        model = default_model

    language = " ".join(str(payload.get("whisper_language", DEFAULT_LANGUAGE)).split())
    language = language or DEFAULT_LANGUAGE

    device = str(payload.get("whisper_device", DEFAULT_DEVICE)).strip().lower()
    if device not in ALLOWED_DEVICES:
        device = DEFAULT_DEVICE

    compute_type = str(payload.get("whisper_compute_type", DEFAULT_COMPUTE_TYPE)).strip()
    if compute_type not in ALLOWED_COMPUTE_TYPES:
        compute_type = DEFAULT_COMPUTE_TYPE

    ui_language = str(payload.get("ui_language", DEFAULT_UI_LANGUAGE)).strip().lower()
    if ui_language not in ALLOWED_UI_LANGUAGES:
        ui_language = DEFAULT_UI_LANGUAGE

    return {
        "last_microphone": " ".join(str(payload.get("last_microphone", "")).split()),
        "speaker_self": _normalize_label(payload.get("speaker_self", ""), DEFAULT_SELF_LABEL),
        "speaker_other": _normalize_label(payload.get("speaker_other", ""), DEFAULT_OTHER_LABEL),
        "auto_transcribe_after_record": _normalize_bool(
            payload.get("auto_transcribe_after_record", DEFAULT_AUTO_TRANSCRIBE_AFTER_RECORD),
            DEFAULT_AUTO_TRANSCRIBE_AFTER_RECORD,
        ),
        "transcription_library": transcription_library,
        "whisper_model": model,
        "whisper_device": device,
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
        "transcribe_mix_track": _normalize_bool(
            payload.get("transcribe_mix_track", DEFAULT_TRANSCRIBE_MIX_TRACK),
            DEFAULT_TRANSCRIBE_MIX_TRACK,
        ),
        "ui_language": ui_language,
        "transcription_mode": (
            "network" if str(payload.get("transcription_mode", "local")).strip().lower() == "network"
            else "local"
        ),
        "network_whisper_url": str(
            payload.get("network_whisper_url", DEFAULT_NETWORK_WHISPER_URL)
        ).strip().rstrip("/") or DEFAULT_NETWORK_WHISPER_URL,
        "network_whisper_token": str(payload.get("network_whisper_token", "")).strip(),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
