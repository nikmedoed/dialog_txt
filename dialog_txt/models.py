from dataclasses import dataclass


class RecordingError(RuntimeError):
    pass


class TranscriptionCancelled(Exception):
    pass


@dataclass
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True)
class TranscriptionOptions:
    model_name: str
    language: str
    beam_size: int
    vad_filter: bool
    compute_type: str
    include_timestamps: bool
