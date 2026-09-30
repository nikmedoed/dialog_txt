import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from dialog_txt.audio_edit import (MANAGED_FILES, audio_duration, clock_offset, mixed_peaks,
                                   parse_duration, restore_session, time_text, trim_session)
from dialog_txt.storage import read_session_metadata, write_session_metadata
from dialog_txt.system_events import stop_reason
from dialog_txt.trim_ui import MixedPlayer
import queue


class TimeSelectionTests(unittest.TestCase):
    def test_duration_formats(self):
        self.assertEqual(parse_duration("1:02:03.125"), 3723.125)
        self.assertEqual(parse_duration("02:03"), 123)
        self.assertEqual(parse_duration("123,5"), 123.5)
        self.assertEqual(time_text(3599.9996), "01:00:00.000")
        for value in ("nan", "inf", "-1", "1:60", "a", "1:2:3:4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_duration(value)

    def test_clock_midnight_and_date(self):
        start = datetime(2026, 9, 30, 23, 50)
        self.assertEqual(clock_offset("00:10", start, 3600), 1200)
        self.assertEqual(clock_offset("00:10:00.125", start, 3600), 1200.125)
        self.assertEqual(clock_offset("2026-10-01 00:10:00", start, 3600), 1200)
        with self.assertRaises(ValueError):
            clock_offset("00:10", start, 90000)
        with self.assertRaises(ValueError):
            clock_offset("24:00", start, 3600)

    def test_only_lock_and_suspend_stop_recording(self):
        self.assertEqual(stop_reason(0x02B1, 7), "lock")
        self.assertEqual(stop_reason(0x0218, 4), "sleep")
        for message, event in ((0x02B1, 8), (0x0218, 18), (0x0218, 10), (0, 7)):
            self.assertIsNone(stop_reason(message, event))


class AudioEditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = Path(self.temp.name)
        self.rate = 8000
        samples = (np.sin(np.arange(3 * self.rate) * 0.1) * 0.4).astype("float32")
        for name, length in (("mic.ogg", 24000), ("desktop.ogg", 23900), ("mix.ogg", 23800)):
            sf.write(str(self.session / name), samples[:length], self.rate, format="OGG", subtype="VORBIS")
        write_session_metadata(self.session, {"created_at": "2026-09-30T23:59:59", "duration_seconds": 3,
                                              "short_name": "Meeting", "mic_name": "Test"})
        (self.session / "transcript.txt").write_text("old transcript", encoding="utf-8")
        (self.session / "mic_transcript.txt").write_text("old mic", encoding="utf-8")
        self.originals = {name: (self.session / name).read_bytes() for name in MANAGED_FILES
                          if (self.session / name).exists()}

    def test_all_tracks_trim_at_same_sample_and_undo_exact_originals(self):
        backup = trim_session(self.session, 1.25)
        for name in ("mic.ogg", "desktop.ogg", "mix.ogg"):
            self.assertEqual(sf.info(str(self.session / name)).frames, 10000)
            self.assertEqual((backup / name).read_bytes(), self.originals[name])
        metadata = read_session_metadata(self.session)
        self.assertEqual(metadata["duration_seconds"], 1.25)
        self.assertEqual(metadata["ended_at"], "2026-10-01T00:00:00.250")
        self.assertEqual(metadata["short_name"], "Meeting")
        self.assertFalse((self.session / "transcript.txt").exists())
        restore_session(self.session)
        for name, content in self.originals.items():
            self.assertEqual((self.session / name).read_bytes(), content)

    def test_multiple_trims_can_be_undone_in_order(self):
        trim_session(self.session, 2)
        first_trim = (self.session / "mic.ogg").read_bytes()
        trim_session(self.session, 1)
        restore_session(self.session)
        self.assertEqual((self.session / "mic.ogg").read_bytes(), first_trim)
        restore_session(self.session)
        self.assertEqual((self.session / "mic.ogg").read_bytes(), self.originals["mic.ogg"])

    def test_rollback_if_replacement_fails(self):
        replace = Path.replace
        failed = False

        def fail_once(path, target):
            nonlocal failed
            if path.name == "desktop.ogg" and path.parent.name.startswith("_trim_") and not failed:
                failed = True
                raise OSError("simulated disk failure")
            return replace(path, target)

        with patch.object(Path, "replace", fail_once), self.assertRaises(OSError):
            trim_session(self.session, 1)
        for name, content in self.originals.items():
            self.assertEqual((self.session / name).read_bytes(), content)

    def test_invalid_cut_never_changes_originals(self):
        for end in (0, -1, float("nan"), 3, 10, 0.000001):
            with self.subTest(end=end), self.assertRaises(ValueError):
                trim_session(self.session, end)
        for name, content in self.originals.items():
            self.assertEqual((self.session / name).read_bytes(), content)

    def test_missing_mix_and_shorter_tracks(self):
        (self.session / "mix.ogg").unlink()
        peaks = mixed_peaks(self.session, 0, 3, bins=100)
        self.assertEqual(len(peaks), 100)
        self.assertGreater(float(peaks.max()), 0.2)
        desktop = (self.session / "desktop.ogg").read_bytes()
        trim_session(self.session, 2.99)
        self.assertEqual((self.session / "desktop.ogg").read_bytes(), desktop)
        self.assertAlmostEqual(audio_duration(self.session), 2.99)

    def test_backup_path_cannot_escape_session(self):
        write_session_metadata(self.session, {"trim_backup": "../somewhere"})
        with self.assertRaises(ValueError):
            restore_session(self.session)

    def test_legacy_session_without_metadata_can_be_restored(self):
        (self.session / "meta.json").unlink()
        trim_session(self.session, 1)
        restore_session(self.session)
        self.assertFalse((self.session / "meta.json").exists())
        self.assertEqual((self.session / "mic.ogg").read_bytes(), self.originals["mic.ogg"])

    def test_streaming_playback_seek_and_mix_without_mix_file(self):
        (self.session / "mix.ogg").unlink()
        blocks = []

        class Output:
            def __init__(self, **kwargs):
                self.options = kwargs
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def write(self, data):
                blocks.append(data.copy())

        events = queue.Queue()
        player = MixedPlayer(self.session, events)
        with patch("dialog_txt.trim_ui.sd.OutputStream", Output):
            player.play(.5, 1.25)
            player.thread.join(timeout=3)
            self.assertFalse(player.thread.is_alive())
        played = np.concatenate(blocks)[:, 0]
        self.assertEqual(len(played), 6000)
        self.assertTrue(all(len(block) <= 2048 for block in blocks))
        expected = np.zeros(6000, dtype=np.float32)
        for name in ("mic.ogg", "desktop.ogg"):
            with sf.SoundFile(str(self.session / name)) as source:
                source.seek(4000)
                expected += source.read(6000, dtype="float32") * .5
        np.testing.assert_allclose(played, expected, atol=1e-7)
        self.assertEqual(player.position, 1.25)
        self.assertEqual(events.get_nowait()[0], "play_done")
        player.stop()


if __name__ == "__main__":
    unittest.main()
