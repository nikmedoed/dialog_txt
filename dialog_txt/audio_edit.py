"""Streaming audio editing; originals and transcripts are retained for undo."""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import (MIC_FILE_NAME, DESKTOP_FILE_NAME, MIX_FILE_NAME, METADATA_FILE_NAME,
                     TRANSCRIPT_FILE_NAME, MIC_TRANSCRIPT_FILE_NAME,
                     DESKTOP_TRANSCRIPT_FILE_NAME, MIX_TRANSCRIPT_FILE_NAME)
from .storage import read_session_metadata

AUDIO_FILES = (MIC_FILE_NAME, DESKTOP_FILE_NAME, MIX_FILE_NAME)
MANAGED_FILES = AUDIO_FILES + (METADATA_FILE_NAME, TRANSCRIPT_FILE_NAME,
                              MIC_TRANSCRIPT_FILE_NAME, DESKTOP_TRANSCRIPT_FILE_NAME,
                              MIX_TRANSCRIPT_FILE_NAME)


def parse_duration(value: str) -> float:
    parts = value.strip().replace(",", ".").split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError("Use seconds, MM:SS or HH:MM:SS")
    numbers = [float(part) for part in parts]
    if any(not math.isfinite(n) or n < 0 for n in numbers):
        raise ValueError("Invalid duration")
    if len(numbers) > 1 and any(n >= 60 for n in numbers[1:]):
        raise ValueError("Minutes and seconds must be below 60")
    return sum(n * 60 ** i for i, n in enumerate(reversed(numbers)))


def time_text(seconds: float) -> str:
    milliseconds = round(max(0, seconds) * 1000)
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def session_start(session: Path) -> datetime | None:
    value = read_session_metadata(session).get("created_at")
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        try:
            return datetime.strptime(session.name[:19], "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            return None


def clock_offset(value: str, start: datetime, duration: float) -> float:
    """Accept a full date, or resolve a clock time within a recording (incl. midnight)."""
    value = value.strip()
    if " " in value or "T" in value:
        end = datetime.fromisoformat(value)
        if end.tzinfo is None:
            end = end.replace(tzinfo=start.tzinfo)
        return (end - start).total_seconds()
    seconds = parse_duration(value)
    if len(value.split(":")) not in (2, 3) or seconds >= 86400:
        raise ValueError("Use HH:MM[:SS] or YYYY-MM-DD HH:MM:SS")
    # Two fields represent hours and minutes, unlike duration input.
    if len(value.split(":")) == 2:
        seconds *= 60
    if seconds >= 86400:
        raise ValueError("Invalid clock time")
    end = start.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=seconds)
    if end < start:
        end += timedelta(days=1)
    offset = (end - start).total_seconds()
    if offset + 86400 <= duration:
        raise ValueError("Ambiguous clock time; enter the date too")
    return offset


def audio_duration(session: Path) -> float:
    durations = [sf.info(str(session / name)).duration for name in AUDIO_FILES
                 if (session / name).is_file()]
    if not durations:
        raise ValueError("No audio tracks")
    return max(durations)


def mixed_peaks(session: Path, start: float, end: float, bins: int = 1200, cancel=None) -> np.ndarray:
    """Bounded-memory envelope, also supports older sessions without mix.ogg."""
    mix = session / MIX_FILE_NAME
    paths = [mix] if mix.exists() else [session / n for n in AUDIO_FILES[:2]
                                      if (session / n).exists()]
    peaks = np.zeros(bins, dtype=np.float32)
    span = max(end - start, 0.001)
    for path in paths:
        with sf.SoundFile(str(path)) as source:
            first = min(source.frames, round(start * source.samplerate))
            last = min(source.frames, math.ceil(end * source.samplerate))
            source.seek(first)
            position = first
            while position < last:
                if cancel is not None and cancel.is_set():
                    return peaks
                data = source.read(min(65536, last - position), dtype="float32", always_2d=True)
                if not len(data):
                    break
                indices = ((np.arange(position, position + len(data)) / source.samplerate - start)
                           * bins / span).astype(int).clip(0, bins - 1)
                np.maximum.at(peaks, indices, np.max(np.abs(data), axis=1))
                position += len(data)
    return peaks


def _snapshot(session: Path) -> Path:
    root = session / "_trim_backups"
    root.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=root))
    for name in MANAGED_FILES:
        if (session / name).is_file():
            shutil.copy2(session / name, backup / name)
    (backup / "_snapshot.json").write_text(json.dumps({"files": [name for name in MANAGED_FILES
                                                               if (backup / name).is_file()]}), encoding="utf-8")
    return backup


def _install(session: Path, prepared: Path, backup: Path) -> None:
    """Roll back the entire set on an ordinary filesystem/codec failure."""
    try:
        for name in MANAGED_FILES:
            target = session / name
            if (prepared / name).exists():
                (prepared / name).replace(target)
            elif target.exists():
                target.unlink()
    except Exception:
        for name in MANAGED_FILES:
            if (backup / name).exists():
                shutil.copy2(backup / name, session / name)
            elif (session / name).exists():
                (session / name).unlink()
        raise


def trim_session(session: Path, end: float) -> Path:
    duration = audio_duration(session)
    if not math.isfinite(end) or not 0 < end < duration:
        raise ValueError("The end must be inside the recording")
    with tempfile.TemporaryDirectory(prefix="_trim_", dir=session) as staging:
        prepared = Path(staging)
        for name in AUDIO_FILES:
            path = session / name
            if not path.exists():
                continue
            with sf.SoundFile(str(path)) as source:
                frames = min(source.frames, round(end * source.samplerate))
                if source.frames and frames < 1:
                    raise ValueError("The retained audio must be at least one sample long")
                if frames == source.frames:
                    # Preserve a shorter track without unnecessarily re-encoding it.
                    shutil.copy2(path, prepared / name)
                    continue
                with sf.SoundFile(str(prepared / name), "w", samplerate=source.samplerate,
                                  channels=source.channels, format=source.format,
                                  subtype=source.subtype) as output:
                    remaining = frames
                    while remaining:
                        block = source.read(min(65536, remaining), dtype="float32", always_2d=True)
                        if not len(block):
                            raise OSError(f"Unexpected end of {name}")
                        output.write(block)
                        remaining -= len(block)
            sf.info(str(prepared / name))  # Verify output before touching originals.
        backup = _snapshot(session)
        metadata = read_session_metadata(session)
        metadata["duration_seconds"] = audio_duration(prepared)
        start = session_start(session)
        if start:
            metadata["ended_at"] = (start + timedelta(seconds=metadata["duration_seconds"])).isoformat(timespec="milliseconds")
        metadata["trim_backup"] = backup.relative_to(session).as_posix()
        metadata["trimmed_at"] = datetime.now().isoformat(timespec="seconds")
        (prepared / METADATA_FILE_NAME).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        # Old transcripts describe the removed tail. They live in the backup, and
        # disappear from the active session so it can be transcribed again.
        _install(session, prepared, backup)
        return backup


def trim_backup(session: Path) -> Path | None:
    value = read_session_metadata(session).get("trim_backup")
    if not isinstance(value, str):
        return None
    path = (session / value).resolve()
    root = (session / "_trim_backups").resolve()
    if path.parent != root or not (path / "_snapshot.json").is_file():
        return None
    return path


def restore_session(session: Path) -> None:
    original = trim_backup(session)
    if original is None:
        raise ValueError("No trim backup")
    with tempfile.TemporaryDirectory(prefix="_restore_", dir=session) as staging:
        prepared = Path(staging)
        for name in MANAGED_FILES:
            if (original / name).exists():
                shutil.copy2(original / name, prepared / name)
        current = _snapshot(session)
        _install(session, prepared, current)
