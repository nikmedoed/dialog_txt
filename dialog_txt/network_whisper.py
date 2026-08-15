from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .config import (
    DESKTOP_FILE_NAME,
    DESKTOP_TRANSCRIPT_FILE_NAME,
    MIC_FILE_NAME,
    MIC_TRANSCRIPT_FILE_NAME,
    MIX_FILE_NAME,
    MIX_TRANSCRIPT_FILE_NAME,
    TRANSCRIPT_FILE_NAME,
)
from .models import TranscriptionCancelled, TranscriptionOptions
from .localization import tr
from .storage import resolve_track_paths, transcript_path


PROTOCOL_VERSION = "1"
RESULT_FILES = (
    TRANSCRIPT_FILE_NAME,
    MIC_TRANSCRIPT_FILE_NAME,
    DESKTOP_TRANSCRIPT_FILE_NAME,
    MIX_TRANSCRIPT_FILE_NAME,
)


def _request_archive(session_dir: Path) -> bytes:
    mic_path, desktop_path = resolve_track_paths(session_dir)
    if mic_path is None or desktop_path is None:
        raise RuntimeError("Missing mic.ogg or desktop.ogg")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(mic_path, MIC_FILE_NAME)
        archive.write(desktop_path, DESKTOP_FILE_NAME)
        mix_path = session_dir / MIX_FILE_NAME
        if mix_path.is_file():
            archive.write(mix_path, MIX_FILE_NAME)
    return buffer.getvalue()


def transcribe_over_network(
    *,
    session_dir: Path,
    options: TranscriptionOptions,
    ui_language: str,
    server_url: str,
    progress_cb: Callable[[str, float], None],
    cancel_event: threading.Event,
    timeout: float = 4 * 60 * 60,
) -> Path:
    if cancel_event.is_set():
        raise TranscriptionCancelled()
    base_url = str(server_url or "").strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise RuntimeError("Network Whisper URL must start with http:// or https://")

    progress_cb(tr(ui_language, "network_status_uploading"), 2.0)
    payload = _request_archive(session_dir)
    headers = {
        "Content-Type": "application/zip",
        "X-Dialog-Txt-Protocol": PROTOCOL_VERSION,
        "X-Dialog-Txt-Options": json.dumps(asdict(options), ensure_ascii=True),
        "X-Dialog-Txt-Ui-Language": ui_language,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/transcribe", data=payload, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"Whisper server returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Whisper server is unavailable: {exc}") from exc

    if cancel_event.is_set():
        raise TranscriptionCancelled()
    progress_cb(tr(ui_language, "network_status_downloading"), 98.0)
    try:
        with zipfile.ZipFile(io.BytesIO(result), "r") as archive:
            names = set(archive.namelist())
            if TRANSCRIPT_FILE_NAME not in names:
                raise RuntimeError("Server response does not contain transcript.txt")
            for name in RESULT_FILES:
                if name in names:
                    (session_dir / name).write_bytes(archive.read(name))
    except zipfile.BadZipFile as exc:
        raise RuntimeError("Whisper server returned an invalid response") from exc
    progress_cb(tr(ui_language, "transcriber_status_done"), 100.0)
    return transcript_path(session_dir)
