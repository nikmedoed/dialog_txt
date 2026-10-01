import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dialog_txt.ui_mixins.audio import AudioMixin
from dialog_txt.recording import SoundDeviceMicrophone


class DeviceSelectionTests(unittest.TestCase):
    def app(self, selected, speakers):
        app = AudioMixin()
        app.output_combo = Mock()
        app.output_combo.get.return_value = selected
        app.speakers = speakers
        return app

    def test_explicit_output_overrides_default(self):
        explicit = SimpleNamespace(name='Headset')
        app = self.app('Headset', [explicit])
        with patch('dialog_txt.ui_mixins.audio.sc.default_speaker') as default:
            self.assertIs(app._selected_speaker(), explicit)
            default.assert_not_called()

    def test_missing_output_falls_back_to_default(self):
        default = SimpleNamespace(name='Speakers')
        app = self.app('Disconnected headset', [default])
        with patch('dialog_txt.ui_mixins.audio.sc.default_speaker', return_value=default):
            self.assertIs(app._selected_speaker(), default)

    def test_missing_default_uses_available_output(self):
        available = SimpleNamespace(name='Speakers')
        app = self.app('System', [available])
        with patch('dialog_txt.ui_mixins.audio.sc.default_speaker', return_value=None):
            self.assertIs(app._selected_speaker(), available)

    def test_unchanged_devices_do_not_restart_streams(self):
        app = AudioMixin()
        app._audio_signature = ('unchanged',)
        app._audio_device_signature = Mock(return_value=app._audio_signature)
        app._refresh_microphones = Mock()
        app.after = Mock()
        app._poll_audio_devices()
        app._refresh_microphones.assert_not_called()
        app.after.assert_called_once_with(1500, app._poll_audio_devices)

    def test_changed_devices_refresh(self):
        app = AudioMixin()
        app._audio_signature = ('old',)
        app._audio_device_signature = Mock(return_value=('new',))
        app._refresh_microphones = Mock()
        app.after = Mock()
        app._poll_audio_devices()
        app._refresh_microphones.assert_called_once()

    def test_unrelated_wasapi_microphone_is_not_a_match(self):
        with patch('dialog_txt.recording.sd.query_devices', return_value=[
            {'name': 'Unrelated device', 'max_input_channels': 1, 'hostapi': 0}
        ]), patch('dialog_txt.recording.sd.query_hostapis', return_value=[{'name': 'Windows WASAPI'}]):
            with self.assertRaises(RuntimeError):
                SoundDeviceMicrophone._find_device('Disconnected headset')

class NativeRateTests(unittest.TestCase):
    def test_native_rate_fallback_preserves_block_length_and_boundaries(self):
        import numpy as np
        import sounddevice as sd
        from dialog_txt.recording import _SoundDeviceRecorder
        stream = Mock()
        offset = 0
        def read(frames):
            nonlocal offset
            data = np.arange(offset, offset + frames, dtype=np.float32).reshape(-1, 1)
            offset += frames
            return data, False
        stream.read.side_effect = read
        with patch('dialog_txt.recording.sd.check_input_settings',
                   side_effect=[sd.PortAudioError('Invalid sample rate'), None]), \
             patch('dialog_txt.recording.sd.query_devices',
                   return_value={'default_samplerate': 16000}), \
             patch('dialog_txt.recording.sd.InputStream', return_value=stream) as factory:
            with _SoundDeviceRecorder(31, 48000, 4800) as recorder:
                blocks = [recorder.record(4800) for _ in range(3)]
            self.assertEqual(factory.call_args.kwargs['samplerate'], 16000)
            self.assertEqual(factory.call_args.kwargs['blocksize'], 1600)
            actual = np.concatenate(blocks)[:, 0]
            self.assertEqual(len(actual), 14400)
            np.testing.assert_allclose(actual, np.arange(14400) / 3, atol=.001)
            stream.close.assert_called_once()

    def test_supported_rate_is_used_directly(self):
        import numpy as np
        from dialog_txt.recording import _SoundDeviceRecorder
        stream = Mock()
        data = np.zeros((4800, 1), dtype=np.float32)
        stream.read.return_value = data, False
        with patch('dialog_txt.recording.sd.check_input_settings'), \
             patch('dialog_txt.recording.sd.InputStream', return_value=stream):
            with _SoundDeviceRecorder(0, 48000, 4800) as recorder:
                np.testing.assert_array_equal(recorder.record(4800), data)
        stream.read.assert_called_once_with(4800)

    def test_failed_session_cleanup_removes_only_new_session(self):
        import tempfile
        from pathlib import Path
        from dialog_txt.storage import discard_failed_session
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = root / 'failed'
            failed.mkdir()
            for name in ('mic.ogg', 'desktop.ogg', 'mix.ogg', 'meta.json'):
                (failed / name).write_bytes(b'partial startup')
            existing = root / 'existing'
            existing.mkdir()
            (existing / 'mic.ogg').write_bytes(b'keep')
            with patch('dialog_txt.storage.recordings_root', return_value=root):
                discard_failed_session(failed)
            self.assertFalse(failed.exists())
            self.assertEqual((existing / 'mic.ogg').read_bytes(), b'keep')

    def test_failed_session_cleanup_rejects_other_directory(self):
        import tempfile
        from pathlib import Path
        from dialog_txt.storage import discard_failed_session
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('dialog_txt.storage.recordings_root', return_value=root / 'recordings'):
                with self.assertRaises(ValueError):
                    discard_failed_session(root)
            self.assertTrue(root.exists())

    def test_stereo_only_microphone_uses_native_channels(self):
        import numpy as np
        import sounddevice as sd
        from dialog_txt.recording import _SoundDeviceRecorder
        stream = Mock()
        stream.read.return_value = np.column_stack((np.ones(8), np.zeros(8))).astype('float32'), False
        def check(**kwargs):
            if kwargs['channels'] == 1:
                raise sd.PortAudioError('Invalid number of channels', -9998)
        with patch('dialog_txt.recording.sd.query_devices', return_value={
                'default_samplerate': 48000, 'max_input_channels': 2}), \
             patch('dialog_txt.recording.sd.check_input_settings', side_effect=check), \
             patch('dialog_txt.recording.sd.InputStream', return_value=stream) as factory:
            with _SoundDeviceRecorder(0, 48000, 8) as recorder:
                data = recorder.record(8)
            self.assertEqual(factory.call_args.kwargs['channels'], 2)
            self.assertEqual(data.shape, (8, 1))
            np.testing.assert_array_equal(data[:, 0], np.ones(8))

    def test_actual_stream_open_failure_tries_native_rate(self):
        import sounddevice as sd
        from dialog_txt.recording import _SoundDeviceRecorder
        stream = Mock()
        with patch('dialog_txt.recording.sd.query_devices', return_value={
                'default_samplerate': 16000, 'max_input_channels': 1}), \
             patch('dialog_txt.recording.sd.check_input_settings'), \
             patch('dialog_txt.recording.sd.InputStream',
                   side_effect=[sd.PortAudioError('Invalid sample rate', -9997), stream]) as factory:
            with _SoundDeviceRecorder(0, 48000, 4800) as recorder:
                self.assertEqual(recorder.input_rate, 16000)
            self.assertEqual(factory.call_count, 2)
