import unittest
from unittest.mock import Mock
import threading
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from dialog_txt.models import TranscriptionOptions, TranscriptionCancelled

from dialog_txt.models import TranscriptSegment
from dialog_txt.transcription import WhisperTranscriber


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.transcriber = WhisperTranscriber()

    def test_loops_rejected_before_pause_splitting(self):
        raw = {"start": 0, "end": 50, "text": "We need to discuss this. " * 20}
        self.assertEqual(self.transcriber._collect_candidates_from_raw_segments([raw], "mic"), [])
        self.assertFalse(self.transcriber._has_repetition_loop("No, no, no. Thank you, thank you."))

    def test_long_silence_and_loud_failed_decode(self):
        segment = TranscriptSegment(0, 60, "mic", "This was never spoken")
        for strict in (False, True):
            self.assertTrue(self.transcriber._should_drop_chunk(segment, .1, -.1, 0, .01, strict))
            self.assertTrue(self.transcriber._should_drop_chunk(segment, .9, -1.5, .1, .01, strict))
            self.assertFalse(self.transcriber._should_drop_chunk(segment, .1, -.1, .1, .01, strict))

    def test_mix_requires_physical_speech_support(self):
        tracks = [{"speaker": "desktop", "speech_ranges": [(0, 2)]},
                  {"speaker": "mic", "speech_ranges": [(3, 5)]}]
        mix = [TranscriptSegment(3, 4, "", "Actual speech"),
               TranscriptSegment(8, 9, "", "Invented speech")]
        result = self.transcriber._assign_mix_speakers(mix, tracks)
        self.assertEqual([(s.speaker, s.text) for s in result], [("mic", "Actual speech")])

    def test_legitimate_outro_is_preserved(self):
        segment = TranscriptSegment(0, 2, "mic", "Thank you for watching.")
        self.assertEqual(self.transcriber._polish_segments([segment])[0].text, segment.text)

    def test_stale_prompt_not_reused(self):
        segment = TranscriptSegment(0, 2, "mic", "Old unrelated topic")
        self.assertEqual(self.transcriber._build_dialogue_prompt([segment], [], 1000, "mic"), "")

    def test_decoding_retries_once_without_prompt(self):
        loop = {"start": 0, "end": 10, "text": "Repeat this phrase " * 8}
        model = Mock()
        model.transcribe.return_value = (iter([]), None)
        # Use fresh iterators, as actual faster-whisper returns a generator.
        model.transcribe.side_effect = lambda *args, **kwargs: (iter([loop]), None)
        self.transcriber._transcribe_audio(
            model, "faster-whisper", "cpu", "test.wav", 5, "en",
            initial_prompt="previous potentially contaminated words",
        )
        self.assertEqual(model.transcribe.call_count, 2)
        retry_kwargs = model.transcribe.call_args.kwargs
        self.assertNotIn("initial_prompt", retry_kwargs)
        self.assertFalse(retry_kwargs["condition_on_previous_text"])

    def test_mix_sentence_split_at_speaker_change(self):
        mix = [TranscriptSegment(0, 4, "", "Hello. Good morning.",
                                 [(0, 1, " Hello."), (2, 3, " Good"), (3, 4, " morning.")])]
        tracks = [{"speaker": "A", "activity_ranges": [(0, 1.2)]},
                  {"speaker": "B", "activity_ranges": [(1.8, 4.2)]}]
        result = self.transcriber._assign_mix_speakers(mix, tracks)
        self.assertEqual([(s.speaker, s.text.strip()) for s in result],
                         [("A", "Hello."), ("B", "Good morning.")])

    def test_word_timings_survive_pause_splitting(self):
        raw = {"start": 0, "end": 4, "text": "Hello there",
               "words": [{"start": 0, "end": 1, "word": " Hello"},
                         {"start": 2, "end": 3, "word": " there"}]}
        chunks = self.transcriber._split_segment_by_pauses(raw, "")
        self.assertEqual(sum(len(s.words) for s in chunks), 2)

    def test_actual_repetition_at_different_times_preserved(self):
        segments = [TranscriptSegment(0, 1, "A", "Thank you."),
                    TranscriptSegment(2, 3, "A", "Thank you.")]
        text = " ".join(s.text for s in self.transcriber._finalize_segments(segments))
        self.assertEqual(text.count("Thank you."), 2)

    def test_similar_vocabulary_does_not_erase_claim(self):
        self.assertIsNone(self.transcriber._merge_textual_overlap(
            "We will send the report to the client today",
            "We will not send the report to the client today because it is incomplete",
        ))

    def test_session_uses_mix_and_keeps_independent_tracks(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name, volume in (("mic.ogg", .1), ("desktop.ogg", .2), ("mix.ogg", .3)):
                sf.write(str(folder/name), np.ones(32000, dtype=np.float32) * volume,
                         16000, format="WAV", subtype="PCM_16")
            model = Mock()
            def recognize(*args, **kwargs):
                audio, _ = sf.read(args[0])
                volume = float(np.mean(audio))
                text = ("Independent microphone text" if volume < .15 else
                        "Independent desktop text" if volume < .25 else "Mix base text")
                return iter([{"start": .1, "end": 1.8, "text": text,
                              "avg_logprob": -.1, "no_speech_prob": .1}]), None
            model.transcribe.side_effect = recognize
            self.transcriber._load_model = Mock(return_value=(model, "faster-whisper", "cpu", "int8"))
            options = TranscriptionOptions("faster-whisper", "tiny", "cpu", "en", "A", "B", 5, True, "int8", False, True)
            self.transcriber.transcribe_session(folder, lambda *args: None, threading.Event(), options)
            self.assertIn("Mix base text", (folder/"transcript.txt").read_text(encoding="utf-8"))
            self.assertIn("Independent microphone text", (folder/"mic_transcript.txt").read_text(encoding="utf-8"))
            self.assertIn("Independent desktop text", (folder/"desktop_transcript.txt").read_text(encoding="utf-8"))
            self.assertTrue((folder/"transcription_debug.json").is_file())
            original = recognize
            def empty_mix(*args, **kwargs):
                audio, _ = sf.read(args[0])
                return (iter([]), None) if np.mean(audio) > .25 else original(*args, **kwargs)
            model.transcribe.side_effect = empty_mix
            self.transcriber.transcribe_session(folder, lambda *args: None, threading.Event(), options)
            self.assertEqual((folder/"transcript.txt").read_text(encoding="utf-8").strip(), "")

    def test_baseline_bounded_and_cancellable(self):
        self.transcriber._transcribe_utterance_range = Mock(return_value=[])
        self.transcriber._transcribe_track_baseline(None, "faster-whisper", "cpu", Path("audio"),
                                                  "A", [(0, 80)], 5, "en", True)
        calls = self.transcriber._transcribe_utterance_range.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(c.kwargs["utterance_end"] - c.kwargs["utterance_start"] <= 30 for c in calls))
        event = threading.Event()
        event.set()
        with self.assertRaises(TranscriptionCancelled):
            self.transcriber._transcribe_track_baseline(None, "faster-whisper", "cpu", Path("audio"),
                                                      "A", [(0, 80)], 5, "en", True, event)

    def test_mix_speaker_uses_local_words_with_alignment_tolerance(self):
        mix = [TranscriptSegment(1, 2, "", "You'll choose", [(1, 1.3, " You'll"), (1.3, 2, " choose")])]
        tracks = [{"speaker": "A", "activity_ranges": [(0, 1.4)],
                   "recognized_segments": [TranscriptSegment(0, 1.4, "A", "my lessons")]},
                  {"speaker": "B", "activity_ranges": [(1.4, 3)],
                   "recognized_segments": [TranscriptSegment(1.1, 3, "B", "You'll choose")]}]
        result = self.transcriber._assign_mix_speakers(mix, tracks)
        self.assertTrue(all(s.speaker == "B" for s in result))

    def test_temporally_overlapping_boundary_words_deduplicated(self):
        segments = [TranscriptSegment(0, 5, "A", "Tell me about the last topic"),
                    TranscriptSegment(4, 8, "A", "last topic we discussed yesterday")]
        text = " ".join(s.text for s in self.transcriber._finalize_segments(segments))
        self.assertEqual(text.count("last topic"), 1)

    def test_loop_across_decoder_segments(self):
        segments = [{"start": i, "end": i + 1, "text": "I don't know"} for i in range(6)]
        segments.append({"start": 6, "end": 8, "text": "The actual next sentence"})
        self.assertEqual(self.transcriber._repetitive_segment_indexes(segments), set(range(6)))
        self.assertEqual(len(self.transcriber._collect_candidates_from_raw_segments(segments, "A")), 1)
        spaced = [{"start": i * 10, "end": i * 10 + 1, "text": "I don't know"} for i in range(6)]
        self.assertEqual(self.transcriber._repetitive_segment_indexes(spaced), set())

    def test_zero_length_mix_word_preserved(self):
        segment = TranscriptSegment(0, 2, "", "We can", [(0, 0, " We"), (0, 1, " can")])
        result = self.transcriber._assign_mix_speakers([segment], [{"speaker": "A", "activity_ranges": [(0, 2)]}])
        self.assertEqual(" ".join(s.text for s in result).strip(), "We can")

    def test_overlap_keeps_continuation_punctuation(self):
        self.assertEqual(self.transcriber._merge_textual_overlap(
            "We should discuss how to mitigate.", "how to mitigate risks today."
        ), "We should discuss how to mitigate risks today.")

    def test_cross_track_changed_negation_is_not_echo(self):
        first = "We are going to send the entire complete final report to the client tomorrow morning"
        second = "We are not going to send the entire complete final report to the client tomorrow morning"
        result = self.transcriber._remove_cross_track_echoes(
            [TranscriptSegment(0, 10, "A", first), TranscriptSegment(0, 10, "B", second)], "B")
        self.assertEqual(len(result), 2)

    def test_whisper_array_resampled_to_16khz(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"audio.wav"
            samples = np.sin(np.arange(48000, dtype=np.float32) * (2*np.pi*440/48000)) * .2
            sf.write(str(path), samples, 48000)
            decoded = self.transcriber._prepare_openai_whisper_audio_input(path)
            self.assertEqual(len(decoded), 16000)
            self.assertEqual(decoded.ndim, 1)

    def test_incremental_rendering_does_not_mutate_recognition(self):
        segments = [TranscriptSegment(0, 1, "A", "First sentence."),
                    TranscriptSegment(2, 3, "A", "Second sentence.")]
        first = self.transcriber._finalize_segments(segments)
        second = self.transcriber._finalize_segments(segments)
        self.assertEqual([(s.start, s.end, s.text) for s in first],
                         [(s.start, s.end, s.text) for s in second])
        self.assertEqual([(s.end, s.text) for s in segments],
                         [(1, "First sentence."), (3, "Second sentence.")])


if __name__ == "__main__":
    unittest.main()
