import tempfile
import unittest
from pathlib import Path

from dialog_txt.storage import (
    latest_transcription_batch_sessions,
    mark_transcription_batch,
    read_session_metadata,
    transcript_path,
    write_session_metadata,
)


class TranscriptionBatchTests(unittest.TestCase):
    def test_latest_batch_survives_reload_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = [Path(directory) / str(i) for i in range(4)]
            for session in sessions:
                session.mkdir()
                transcript_path(session).write_text("text", encoding="utf-8")
            write_session_metadata(sessions[1], {"short_name": "Meeting"})
            mark_transcription_batch(sessions[0], "2026-11-06_old")
            for session in sessions[1:3]:
                mark_transcription_batch(session, "2026-11-07_new")
            self.assertEqual(latest_transcription_batch_sessions(sessions), set(sessions[1:3]))
            self.assertEqual(read_session_metadata(sessions[1])["short_name"], "Meeting")
            # Missing transcripts and malformed metadata must not select a newer batch.
            mark_transcription_batch(sessions[3], "2026-11-08_missing")
            transcript_path(sessions[3]).unlink()
            self.assertEqual(latest_transcription_batch_sessions(sessions), set(sessions[1:3]))
            write_session_metadata(sessions[3], {"transcription_batch": []})
            transcript_path(sessions[3]).touch()
            self.assertEqual(latest_transcription_batch_sessions(sessions), set(sessions[1:3]))

    def test_legacy_sessions_are_not_highlighted(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            transcript_path(session).touch()
            self.assertEqual(latest_transcription_batch_sessions([session]), set())


if __name__ == "__main__":
    unittest.main()
