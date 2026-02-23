from __future__ import annotations

def to_mono(samples):
    if samples.ndim == 1:
        return samples
    # Use first channel to avoid phase-cancellation artifacts on some stereo sources.
    return samples[:, 0]


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def format_seconds(value: float) -> str:
    value = max(0, int(value))
    hours = value // 3600
    minutes = (value % 3600) // 60
    seconds = value % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
