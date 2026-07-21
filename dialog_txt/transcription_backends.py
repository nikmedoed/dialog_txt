from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache


TRANSCRIPTION_LIBRARY_FASTER = "faster-whisper"
TRANSCRIPTION_LIBRARY_WHISPER = "whisper"

ALLOWED_TRANSCRIPTION_LIBRARIES = (
    TRANSCRIPTION_LIBRARY_FASTER,
    TRANSCRIPTION_LIBRARY_WHISPER,
)
DEFAULT_TRANSCRIPTION_LIBRARY = TRANSCRIPTION_LIBRARY_FASTER

FASTER_WHISPER_MODELS = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "distil-small.en",
    "medium",
    "medium.en",
    "distil-medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-large-v2",
    "distil-large-v3",
    "distil-large-v3.5",
    "large-v3-turbo",
    "turbo",
)
OPENAI_WHISPER_MODELS = (
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "large-v3-turbo",
    "turbo",
)

DEFAULT_MODEL_BY_LIBRARY: dict[str, str] = {
    TRANSCRIPTION_LIBRARY_FASTER: "turbo",
    TRANSCRIPTION_LIBRARY_WHISPER: "turbo",
}

ALLOWED_DEVICES = ("auto", "cpu", "gpu")
DEFAULT_DEVICE = "auto"

_LIBRARY_PIP_PACKAGES: dict[str, str] = {
    TRANSCRIPTION_LIBRARY_FASTER: "faster-whisper",
    TRANSCRIPTION_LIBRARY_WHISPER: "openai-whisper",
}


def normalize_transcription_library(value: str | None) -> str:
    resolved = str(value or DEFAULT_TRANSCRIPTION_LIBRARY).strip().lower()
    if resolved not in ALLOWED_TRANSCRIPTION_LIBRARIES:
        return DEFAULT_TRANSCRIPTION_LIBRARY
    return resolved


def transcription_library_pip_package(library: str) -> str:
    normalized = normalize_transcription_library(library)
    return _LIBRARY_PIP_PACKAGES[normalized]


def transcription_models_for_library(library: str) -> tuple[str, ...]:
    normalized = normalize_transcription_library(library)
    if normalized == TRANSCRIPTION_LIBRARY_WHISPER:
        discovered = _discover_openai_whisper_models()
        if discovered:
            return discovered
        return OPENAI_WHISPER_MODELS
    return FASTER_WHISPER_MODELS


def default_model_for_library(
    library: str,
    models: tuple[str, ...] | None = None,
) -> str:
    normalized = normalize_transcription_library(library)
    available_models = models if models is not None else transcription_models_for_library(normalized)
    preferred = DEFAULT_MODEL_BY_LIBRARY.get(
        normalized,
        DEFAULT_MODEL_BY_LIBRARY[DEFAULT_TRANSCRIPTION_LIBRARY],
    )
    if preferred in available_models:
        return preferred
    if available_models:
        return available_models[0]
    return preferred


@lru_cache(maxsize=len(ALLOWED_TRANSCRIPTION_LIBRARIES))
def check_transcription_library_available(library: str) -> tuple[bool, str]:
    normalized = normalize_transcription_library(library)
    try:
        if normalized == TRANSCRIPTION_LIBRARY_FASTER:
            ctranslate2 = importlib.import_module("ctranslate2")
            faster_whisper = importlib.import_module("faster_whisper")
            if not hasattr(ctranslate2, "get_cuda_device_count"):
                return False, "module ctranslate2 is incomplete"
            if not hasattr(faster_whisper, "WhisperModel"):
                return False, "module faster_whisper is incomplete"
        elif normalized == TRANSCRIPTION_LIBRARY_WHISPER:
            whisper = importlib.import_module("whisper")
            if not hasattr(whisper, "load_model"):
                return False, "module whisper is incomplete"
        else:
            return False, f"unsupported transcription library: {normalized}"
    except Exception as exc:
        return False, str(exc)
    return True, ""


def is_transcription_library_installed(library: str) -> bool:
    """Cheap package-presence check that does not import heavy ML runtimes."""
    normalized = normalize_transcription_library(library)
    module_names = (
        ("ctranslate2", "faster_whisper")
        if normalized == TRANSCRIPTION_LIBRARY_FASTER
        else ("whisper",)
    )
    return all(importlib.util.find_spec(module_name) is not None for module_name in module_names)


def refresh_transcription_backend_caches() -> None:
    check_transcription_library_available.cache_clear()
    _discover_openai_whisper_models.cache_clear()


@lru_cache(maxsize=1)
def _discover_openai_whisper_models() -> tuple[str, ...]:
    is_available, _ = check_transcription_library_available(TRANSCRIPTION_LIBRARY_WHISPER)
    if not is_available:
        return tuple()

    try:
        whisper = importlib.import_module("whisper")
        raw_models = whisper.available_models()
    except Exception:
        return tuple()

    seen: set[str] = set()
    models: list[str] = []
    for item in raw_models:
        model = str(item).strip()
        if not model or model in seen:
            continue
        seen.add(model)
        models.append(model)
    return tuple(models)
