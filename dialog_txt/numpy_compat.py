from __future__ import annotations

import threading

import numpy as np


_PATCH_LOCK = threading.Lock()
_PATCHED = False
_ORIGINAL_FROMSTRING = np.fromstring


def apply_numpy_fromstring_compat_patch() -> None:
    global _PATCHED
    with _PATCH_LOCK:
        if _PATCHED:
            return

        def _compat_fromstring(string, dtype=float, count=-1, sep="", *, like=None):
            if sep == "":
                try:
                    # numpy.fromstring(binary) produced an independent array.
                    # numpy.frombuffer returns a view, but soundcard releases the WASAPI buffer
                    # right after read, so we must copy immediately.
                    return np.frombuffer(string, dtype=dtype, count=count).copy()
                except Exception:
                    # Fall back to original parser for text and unsupported buffer types.
                    pass
            return _ORIGINAL_FROMSTRING(string, dtype=dtype, count=count, sep=sep, like=like)

        np.fromstring = _compat_fromstring
        _PATCHED = True
