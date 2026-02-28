import os
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent.parent


def _resolve_app_data_root() -> Path:
    # Allows launcher scripts to redirect writable app data (settings/recordings)
    # to a user profile directory, similar to installed desktop apps.
    explicit_home = os.environ.get("DIALOG_TXT_HOME", "").strip()
    if not explicit_home:
        return APP_ROOT

    candidate = Path(explicit_home).expanduser()
    if candidate.is_absolute():
        return candidate
    return (APP_ROOT / candidate).resolve()


APP_DATA_ROOT = _resolve_app_data_root()
RECORDINGS_ROOT = APP_DATA_ROOT / "recordings"
APP_SETTINGS_FILE = APP_DATA_ROOT / "app_settings.json"

MIC_FILE_NAME = "mic.ogg"
DESKTOP_FILE_NAME = "desktop.ogg"
MIX_FILE_NAME = "mix.ogg"
TRANSCRIPT_FILE_NAME = "transcript.txt"
MIC_TRANSCRIPT_FILE_NAME = "mic_transcript.txt"
DESKTOP_TRANSCRIPT_FILE_NAME = "desktop_transcript.txt"
MIX_TRANSCRIPT_FILE_NAME = "mix_transcript.txt"
METADATA_FILE_NAME = "meta.json"

SAMPLE_RATE = 48_000
BLOCK_FRAMES = 4_800
AUDIO_WRITE_FLUSH_INTERVAL_SEC = 2.0
MIN_SPEAKER_GAP = 6.0
MAX_MERGED_SEGMENT_DURATION = 120.0
MAX_MERGED_SEGMENT_WORDS = 220
MAX_OVERLAP_FOR_MERGE = 2.5
WORD_PAUSE_SPLIT_GAP = 1.35
TRANSCRIPT_SOFT_SPLIT_GAP = 0.45
TRANSCRIPT_CHUNK_SOFT_MAX_SEC = 7.5
TRANSCRIPT_CHUNK_HARD_MAX_SEC = 12.0
TRANSCRIPT_CHUNK_SOFT_MAX_WORDS = 28
TRANSCRIPT_CHUNK_HARD_MAX_WORDS = 42
SHORT_SEGMENT_WORDS = 4
SHORT_SEGMENT_BRIDGE_GAP = 12.0
VAD_MIN_SILENCE_DURATION_MS = 450
VAD_SPEECH_PAD_MS = 120
MIN_AUDIO_RMS = 0.0025
NO_SPEECH_PROB_THRESHOLD = 0.60
SPEECH_GATE_FRAME_MS = 30
SPEECH_GATE_MIN_SPEECH_SEC = 0.28
SPEECH_GATE_MIN_SILENCE_SEC = 0.34
SPEECH_GATE_PAD_SEC = 0.18
SPEECH_GATE_THRESHOLD_MULTIPLIER = 2.4
SPEECH_GATE_FLOOR_RATIO = 0.70
SPEECH_GATE_ONSET_RATIO = 1.10
SPEECH_GATE_RELEASE_RATIO = 0.72
SPEECH_CHUNK_BRIDGE_SEC = 1.80
SPEECH_CHUNK_MAX_SEC = 40.0
DIALOG_CONTEXT_PROMPT_WORDS = 96
BASELINE_CONTEXT_LOOKBACK_SEC = 14.0
LANGUAGE_DETECT_MAX_SECONDS = 75.0
MIX_REFERENCE_PAD_SEC = 0.75

DEFAULT_SELF_LABEL = "Я"
DEFAULT_OTHER_LABEL = "Собеседник"
