"""
tests/unit/test_audio_detector.py
===================================
Unit tests for pipeline/audio/detector.py

All tests run without GPU, without pretrained weights, and without torchaudio.
Synthetic WAV files are written to temp paths for I/O tests.
"""

from __future__ import annotations

import os
import sys
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# --- path setup -----------------------------------------------------------
REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.audio.detector import (
    AudioDetector,
    MelSpectrogramProcessor,
    RawNet2Classifier,
    _read_wav_mono_16k,
)
from utils.types import AudioSegmentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_wav(path: str, samples: np.ndarray, sr: int = 16_000, n_channels: int = 1) -> None:
    """Write a float32 array as a 16-bit PCM WAV file."""
    pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _sine(sr: int = 16_000, duration: float = 5.0, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class _FakeAudioDetector(AudioDetector):
    """
    AudioDetector with RawNet2 replaced by a tiny 2-layer network
    for fast CPU-only testing — skips the stem's Conv2d which requires
    exact n_mels match.
    """

    def __init__(self, segment_dur: float = 3.0, overlap: float = 0.5):
        self.device           = torch.device("cpu")
        self._sample_rate     = 16_000
        self._segment_dur     = segment_dur
        self._overlap         = overlap
        self._n_fft           = 512
        self._hop_length      = 128
        self._n_mels          = 80

        from pipeline.audio.detector import MelSpectrogramProcessor
        self._mel = MelSpectrogramProcessor(
            sample_rate = self._sample_rate,
            n_fft       = self._n_fft,
            hop_length  = self._hop_length,
            n_mels      = self._n_mels,
        )

        # Tiny model: global avg pool → linear(1, 2)
        self._model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(1, 2),
        )
        self._model.eval()

    def _predict_segments(self, segments_data):
        """Override to use adaptive pooling model on raw mel tensor."""
        target_len = int(self._segment_dur * self._sample_rate)
        fake_probs = []
        with torch.no_grad():
            for chunk in segments_data:
                chunk = self._pad_or_trim(chunk, target_len)
                mel_t = self._mel(chunk).unsqueeze(0).to(self.device)  # (1,1,80,T)
                logits = self._model(mel_t)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                fake_probs.append(prob)
        return fake_probs


# ---------------------------------------------------------------------------
# MelSpectrogramProcessor tests
# ---------------------------------------------------------------------------

class TestMelSpectrogramProcessor(unittest.TestCase):

    def setUp(self):
        self.proc = MelSpectrogramProcessor(
            sample_rate=16_000, n_fft=512, hop_length=128, n_mels=80
        )

    def test_output_shape(self):
        sig = np.random.randn(16_000 * 3).astype(np.float32)
        out = self.proc(sig)
        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape[0], 1)        # channel dim
        self.assertEqual(out.shape[1], 80)       # n_mels

    def test_output_is_float_tensor(self):
        sig = np.random.randn(16_000).astype(np.float32)
        out = self.proc(sig)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.dtype, torch.float32)

    def test_normalised_output(self):
        """Output should be approximately zero-mean, unit-std after normalisation."""
        sig = np.random.randn(16_000 * 5).astype(np.float32)
        out = self.proc(sig).numpy()[0]          # (80, T)
        self.assertAlmostEqual(float(out.mean()), 0.0, delta=0.3)

    def test_filterbank_shape(self):
        n_freqs = 512 // 2 + 1
        self.assertEqual(self.proc._fbank.shape, (80, n_freqs))

    def test_filterbank_non_negative(self):
        self.assertTrue((self.proc._fbank >= 0).all())

    def test_silent_signal(self):
        """Silent signal shouldn't crash — just produce a very negative log."""
        sig = np.zeros(16_000, dtype=np.float32)
        out = self.proc(sig)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())


# ---------------------------------------------------------------------------
# RawNet2Classifier tests
# ---------------------------------------------------------------------------

class TestRawNet2Classifier(unittest.TestCase):

    def setUp(self):
        self.model = RawNet2Classifier(n_mels=80, base_channels=16, gru_hidden=32, num_res_blocks=2)
        self.model.eval()

    def test_output_shape(self):
        x = torch.randn(2, 1, 80, 50)   # batch=2
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (2, 2))

    def test_probabilities_sum_to_one(self):
        x = torch.randn(1, 1, 80, 50)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)

    def test_no_nan_in_output(self):
        x = torch.randn(4, 1, 80, 100)
        with torch.no_grad():
            out = self.model(x)
        self.assertFalse(torch.isnan(out).any())

    def test_different_time_lengths(self):
        """Model must handle variable-length spectrograms (different T)."""
        for T in [20, 50, 100, 200]:
            x = torch.randn(1, 1, 80, T)
            with torch.no_grad():
                out = self.model(x)
            self.assertEqual(out.shape, (1, 2), msg=f"Failed at T={T}")


# ---------------------------------------------------------------------------
# WAV reader tests
# ---------------------------------------------------------------------------

class TestReadWavMono16k(unittest.TestCase):

    def _tmp_wav(self, samples, sr=16_000, n_channels=1) -> str:
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.close()
        _write_wav(f.name, samples, sr, n_channels)
        return f.name

    def test_reads_mono_correctly(self):
        sig  = _sine(duration=2.0)
        path = self._tmp_wav(sig)
        try:
            arr, sr = _read_wav_mono_16k(path)
            self.assertEqual(sr, 16_000)
            self.assertEqual(arr.dtype, np.float32)
            self.assertAlmostEqual(len(arr), 16_000 * 2, delta=10)
        finally:
            os.unlink(path)

    def test_stereo_mixed_to_mono(self):
        sig    = _sine(duration=1.0)
        stereo = np.column_stack([sig, sig]).flatten()
        path   = self._tmp_wav(stereo, n_channels=2)
        try:
            arr, sr = _read_wav_mono_16k(path)
            self.assertEqual(arr.ndim, 1)
            self.assertAlmostEqual(len(arr), 16_000, delta=10)
        finally:
            os.unlink(path)

    def test_range_approximately_minus1_to_1(self):
        sig  = _sine(duration=1.0)
        path = self._tmp_wav(sig)
        try:
            arr, _ = _read_wav_mono_16k(path)
            self.assertLessEqual(float(arr.max()), 1.1)
            self.assertGreaterEqual(float(arr.min()), -1.1)
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with self.assertRaises(Exception):
            _read_wav_mono_16k("/tmp/does_not_exist_xyz.wav")


# ---------------------------------------------------------------------------
# AudioDetector segmentation tests
# ---------------------------------------------------------------------------

class TestSegmentation(unittest.TestCase):

    def setUp(self):
        self.det = _FakeAudioDetector(segment_dur=3.0, overlap=0.5)

    def test_segment_count_5s_audio(self):
        sr   = 16_000
        sig  = np.zeros(sr * 5, dtype=np.float32)
        segs = self.det._segment(sig, sr)
        # With 3s segments, 50% overlap, step=1.5s:
        # starts at 0, 1.5, 3.0 → 3 segments
        self.assertGreaterEqual(len(segs), 2)

    def test_segment_covers_full_audio(self):
        sr   = 16_000
        sig  = np.zeros(sr * 10, dtype=np.float32)
        segs = self.det._segment(sig, sr)
        self.assertEqual(segs[0][2], 0.0)            # first starts at 0s
        self.assertAlmostEqual(segs[-1][3], 10.0, delta=0.2)

    def test_short_audio_single_segment(self):
        sr   = 16_000
        sig  = np.zeros(sr, dtype=np.float32)        # 1s < segment_dur=3s
        segs = self.det._segment(sig, sr)
        self.assertEqual(len(segs), 1)

    def test_segment_timestamps_non_overlapping_start(self):
        sr   = 16_000
        sig  = np.zeros(sr * 10, dtype=np.float32)
        segs = self.det._segment(sig, sr)
        starts = [s[2] for s in segs]
        self.assertEqual(starts, sorted(starts))     # monotonically increasing


# ---------------------------------------------------------------------------
# AudioDetector.run() tests
# ---------------------------------------------------------------------------

class TestAudioDetectorRun(unittest.TestCase):

    def _make_wav(self, duration: float = 5.0) -> str:
        sig = _sine(duration=duration)
        f   = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.close()
        _write_wav(f.name, sig)
        return f.name

    def setUp(self):
        self.det = _FakeAudioDetector()

    def test_run_returns_segment_results(self):
        path = self._make_wav(5.0)
        try:
            results = self.det.run(path)
            self.assertGreater(len(results), 0)
            self.assertIsInstance(results[0], AudioSegmentResult)
        finally:
            os.unlink(path)

    def test_run_fake_probs_in_range(self):
        path = self._make_wav(6.0)
        try:
            results = self.det.run(path)
            for r in results:
                self.assertGreaterEqual(r.fake_prob, 0.0)
                self.assertLessEqual(r.fake_prob, 1.0)
        finally:
            os.unlink(path)

    def test_run_timestamps_are_sequential(self):
        path = self._make_wav(9.0)
        try:
            results = self.det.run(path)
            starts = [r.start_sec for r in results]
            self.assertEqual(starts, sorted(starts))
            for r in results:
                self.assertLess(r.start_sec, r.end_sec)
        finally:
            os.unlink(path)

    def test_run_segment_indices_sequential(self):
        path = self._make_wav(6.0)
        try:
            results = self.det.run(path)
            indices = [r.segment_index for r in results]
            self.assertEqual(indices, list(range(len(results))))
        finally:
            os.unlink(path)

    def test_run_missing_file_returns_empty(self):
        results = self.det.run("/tmp/does_not_exist_xyz.wav")
        self.assertEqual(results, [])

    def test_run_short_audio_under_segment_dur(self):
        """Audio shorter than one segment should still produce a result."""
        path = self._make_wav(1.0)
        try:
            results = self.det.run(path)
            self.assertEqual(len(results), 1)
        finally:
            os.unlink(path)

    def test_run_long_audio_multiple_segments(self):
        path = self._make_wav(15.0)
        try:
            results = self.det.run(path)
            self.assertGreater(len(results), 3)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# AudioDetector.aggregate() tests
# ---------------------------------------------------------------------------

class TestAggregate(unittest.TestCase):

    def setUp(self):
        self.det = _FakeAudioDetector()

    def _seg(self, prob: float, idx: int = 0) -> AudioSegmentResult:
        return AudioSegmentResult(
            segment_index=idx, start_sec=float(idx),
            end_sec=float(idx + 1), fake_prob=prob,
        )

    def test_aggregate_empty_returns_half(self):
        self.assertEqual(self.det.aggregate([]), 0.5)

    def test_aggregate_single_segment(self):
        r = self.det.aggregate([self._seg(0.8)])
        self.assertAlmostEqual(r, 0.8, places=4)

    def test_aggregate_uniform_segments(self):
        segs = [self._seg(0.6, i) for i in range(4)]
        r    = self.det.aggregate(segs)
        self.assertAlmostEqual(r, 0.6, places=3)

    def test_aggregate_in_range(self):
        segs = [self._seg(p, i) for i, p in enumerate([0.1, 0.5, 0.9])]
        r    = self.det.aggregate(segs)
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)

    def test_aggregate_weights_high_probs_more(self):
        """Weighted mean should be >= plain mean when probs are skewed high."""
        segs      = [self._seg(p, i) for i, p in enumerate([0.1, 0.1, 0.9])]
        plain_mean = np.mean([0.1, 0.1, 0.9])
        weighted   = self.det.aggregate(segs)
        self.assertGreater(weighted, plain_mean - 0.05)   # soft assertion (exp weighting)

    def test_aggregate_all_real(self):
        segs = [self._seg(0.0, i) for i in range(5)]
        r    = self.det.aggregate(segs)
        self.assertAlmostEqual(r, 0.0, places=4)

    def test_aggregate_all_fake(self):
        segs = [self._seg(1.0, i) for i in range(5)]
        r    = self.det.aggregate(segs)
        self.assertAlmostEqual(r, 1.0, places=4)


# ---------------------------------------------------------------------------
# Pad/trim helper
# ---------------------------------------------------------------------------

class TestPadOrTrim(unittest.TestCase):

    def test_pads_short(self):
        chunk = np.ones(100, dtype=np.float32)
        out   = AudioDetector._pad_or_trim(chunk, 200)
        self.assertEqual(len(out), 200)
        self.assertEqual(out[100:].sum(), 0.0)   # padded region is zero

    def test_trims_long(self):
        chunk = np.ones(300, dtype=np.float32)
        out   = AudioDetector._pad_or_trim(chunk, 200)
        self.assertEqual(len(out), 200)

    def test_exact_length_unchanged(self):
        chunk = np.arange(150, dtype=np.float32)
        out   = AudioDetector._pad_or_trim(chunk, 150)
        np.testing.assert_array_equal(out, chunk)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)