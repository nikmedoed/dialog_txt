from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_ROOT = APP_ROOT / "recordings"
APP_SETTINGS_FILE = APP_ROOT / "app_settings.json"

MIC_FILE_NAME = "mic.ogg"
DESKTOP_FILE_NAME = "desktop.ogg"
TRANSCRIPT_FILE_NAME = "transcript.txt"
METADATA_FILE_NAME = "meta.json"

SAMPLE_RATE = 48_000
BLOCK_FRAMES = 4_800
MIN_SPEAKER_GAP = 1.0

DEFAULT_SELF_LABEL = "Я"
DEFAULT_OTHER_LABEL = "Собеседник"
