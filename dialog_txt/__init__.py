"""Dialog TXT package.

The GUI is imported lazily so command-line tools such as the Whisper server do
not require audio-capture dependencies just to start.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ui import App


def __getattr__(name: str):
    if name == "App":
        from .ui import App

        return App
    raise AttributeError(name)

__all__ = ["App"]
