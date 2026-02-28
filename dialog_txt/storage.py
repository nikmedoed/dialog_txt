from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .config import (
    DESKTOP_FILE_NAME,
    DESKTOP_TRANSCRIPT_FILE_NAME,
    METADATA_FILE_NAME,
    MIC_FILE_NAME,
    MIC_TRANSCRIPT_FILE_NAME,
    MIX_TRANSCRIPT_FILE_NAME,
    RECORDINGS_ROOT,
    TRANSCRIPT_FILE_NAME,
)


def ensure_recordings_root() -> None:
    RECORDINGS_ROOT.mkdir(parents=True, exist_ok=True)
    migrate_legacy_layout()
    migrate_flat_old_names()


def create_session_dir(now: datetime | None = None) -> tuple[datetime, Path]:
    created_at = now or datetime.now()
    base_name = created_at.strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = RECORDINGS_ROOT / base_name
    suffix = 1
    while session_dir.exists():
        session_dir = RECORDINGS_ROOT / f"{base_name}_{suffix}"
        suffix += 1
    session_dir.mkdir(parents=True, exist_ok=True)
    return created_at, session_dir


def write_initial_metadata(
    session_dir: Path,
    created_at: datetime,
    mic_name: str,
    desktop_source: str,
) -> None:
    payload = {
        "created_at": created_at.isoformat(timespec="seconds"),
        "mic_name": mic_name,
        "desktop_source": desktop_source,
    }
    (session_dir / METADATA_FILE_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def update_session_metadata(session_dir: Path, duration_seconds: int) -> None:
    meta_path = session_dir / METADATA_FILE_NAME
    payload = {}
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload["ended_at"] = datetime.now().isoformat(timespec="seconds")
    payload["duration_seconds"] = duration_seconds
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_track_paths(session_dir: Path) -> tuple[Path | None, Path | None]:
    mic_candidate = session_dir / MIC_FILE_NAME
    desktop_candidate = session_dir / DESKTOP_FILE_NAME
    mic_path = mic_candidate if mic_candidate.exists() else None
    desktop_path = desktop_candidate if desktop_candidate.exists() else None
    return mic_path, desktop_path


def is_session_dir(session_dir: Path) -> bool:
    if not session_dir.is_dir():
        return False
    for file_name in (MIC_FILE_NAME, DESKTOP_FILE_NAME, TRANSCRIPT_FILE_NAME, METADATA_FILE_NAME):
        if (session_dir / file_name).exists():
            return True
    return False


def _is_legacy_date_dir(path: Path) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.name))


def migrate_legacy_layout() -> None:
    if not RECORDINGS_ROOT.exists():
        return

    for date_dir in RECORDINGS_ROOT.iterdir():
        if not date_dir.is_dir() or not _is_legacy_date_dir(date_dir):
            continue

        moved_any = False
        for session_dir in date_dir.iterdir():
            if not is_session_dir(session_dir):
                continue

            target = RECORDINGS_ROOT / session_dir.name
            if target.exists():
                suffix = 1
                while (RECORDINGS_ROOT / f"{session_dir.name}_{suffix}").exists():
                    suffix += 1
                target = RECORDINGS_ROOT / f"{session_dir.name}_{suffix}"
            session_dir.rename(target)
            moved_any = True

        if moved_any and not any(date_dir.iterdir()):
            date_dir.rmdir()


def migrate_flat_old_names() -> None:
    if not RECORDINGS_ROOT.exists():
        return

    for session_dir in RECORDINGS_ROOT.iterdir():
        if not session_dir.is_dir():
            continue

        match = re.fullmatch(r"(?P<date>\d{8})_(?P<time>\d{6})(?:_(?P<suffix>\d+))?", session_dir.name)
        if not match:
            continue

        try:
            dt = datetime.strptime(
                f"{match.group('date')}_{match.group('time')}",
                "%Y%m%d_%H%M%S",
            )
        except ValueError:
            continue

        base_name = dt.strftime("%Y-%m-%d_%H-%M-%S")
        suffix = match.group("suffix")
        target_name = f"{base_name}_{suffix}" if suffix else base_name
        target = RECORDINGS_ROOT / target_name

        if target == session_dir:
            continue

        if target.exists():
            idx = 1
            while (RECORDINGS_ROOT / f"{base_name}_{idx}").exists():
                idx += 1
            target = RECORDINGS_ROOT / f"{base_name}_{idx}"
        session_dir.rename(target)


def transcript_path(session_dir: Path) -> Path:
    return session_dir / TRANSCRIPT_FILE_NAME


def debug_transcript_path(session_dir: Path, transcript_kind: str) -> Path:
    mapping = {
        "mic": MIC_TRANSCRIPT_FILE_NAME,
        "desktop": DESKTOP_TRANSCRIPT_FILE_NAME,
        "mix": MIX_TRANSCRIPT_FILE_NAME,
    }
    try:
        file_name = mapping[transcript_kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported transcript kind: {transcript_kind}") from exc
    return session_dir / file_name


def read_session_metadata(session_dir: Path) -> dict:
    meta_path = session_dir / METADATA_FILE_NAME
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def discover_sessions() -> list[Path]:
    if not RECORDINGS_ROOT.exists():
        return []

    result = [
        child
        for child in RECORDINGS_ROOT.iterdir()
        if not child.name.startswith("_") and is_session_dir(child)
    ]
    result.sort(reverse=True)
    return result


def session_title(session_dir: Path) -> str:
    stamp = session_dir.name
    candidates = [stamp]
    if re.fullmatch(r".+_\d+", stamp):
        candidates.append(stamp.rsplit("_", 1)[0])

    for candidate in candidates:
        for fmt in ("%Y-%m-%d_%H-%M-%S", "%Y%m%d_%H%M%S"):
            try:
                dt = datetime.strptime(candidate, fmt)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
    return stamp
