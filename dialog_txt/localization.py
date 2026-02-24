from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .settings import ALLOWED_UI_LANGUAGES, DEFAULT_UI_LANGUAGE


UI_LANGUAGE_CODES = {"RU": "ru", "EN": "en"}
_LOCALES_DIR = Path(__file__).resolve().parent / "locales"


@lru_cache(maxsize=len(ALLOWED_UI_LANGUAGES))
def _load_language_pack(language: str) -> dict[str, str]:
    target = _LOCALES_DIR / f"{language}.json"
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid locale payload: {target}")
    return {str(key): str(value) for key, value in payload.items()}


@lru_cache(maxsize=1)
def _fallback_pack() -> dict[str, str]:
    return _load_language_pack(DEFAULT_UI_LANGUAGE)


def resolve_ui_language(value: str | None) -> str:
    language = str(value or DEFAULT_UI_LANGUAGE).strip().lower()
    if language not in ALLOWED_UI_LANGUAGES:
        return DEFAULT_UI_LANGUAGE
    return language


def tr(language: str, key: str, **kwargs) -> str:
    resolved = resolve_ui_language(language)
    fallback_pack = _fallback_pack()
    language_pack = _load_language_pack(resolved)
    template = language_pack.get(key, fallback_pack.get(key, key))
    return template.format(**kwargs) if kwargs else template


def system_microphone_label_prefixes() -> tuple[str, ...]:
    return tuple(f"{tr(language, 'system_microphone')} (" for language in ALLOWED_UI_LANGUAGES)
