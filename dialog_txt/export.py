"""Save transcripts to the user's Downloads folder."""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

from .storage import read_session_alias, session_title, transcript_path


def downloads_directory() -> Path:
    if sys.platform == "win32":
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as key:
            value, _ = winreg.QueryValueEx(key, "{374DE290-123F-4565-9164-39C4925E467B}")
        return Path(os.path.expandvars(value))
    return Path.home() / "Downloads"


def export_transcript(session: Path) -> Path:
    source = transcript_path(session)
    stamp = session_title(session).replace(" ", "_").replace(":", "-")
    alias = read_session_alias(session) or "transcript"
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", f"{stamp}-{alias}").rstrip(" .")[:180]
    directory = downloads_directory()
    directory.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        target = directory / (f"{name}.txt" if suffix == 0 else f"{name} ({suffix}).txt")
        try:
            output = target.open("xb")
        except FileExistsError:
            suffix += 1
            continue
        try:
            with output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return target
