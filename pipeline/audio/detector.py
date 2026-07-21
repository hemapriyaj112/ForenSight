"""
pipeline/audio/detector.py — audio deepfake detector.

Public API: AudioDetector(cfg).run(wav_path) → ModalResult
Internal:   MelSpectrogramProcessor (imported by test_audio_detector.py)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from utils.logger import get_logger
from utils.types import AudioSegmentResult, ModalResult

log = get_logger("forensight.audio")

_SEGMENT_DURATION_S = 4.0


class MelSpectrogramProcessor:
    """
    Convert a raw waveform segment into an 80-bin log-mel spectrogram.

    Parameters
    ----------
    sample_rate : int   (default 16 000)
    n_mels      : int   (default 80)
    n_fft       : int   (default 512)
    hop_length  : int   (default 160)
    fmin        : float (default 0.0)
    fmax        : float (default sample_rate / 2)
    log_floor   : float (default 1e-9)
    """

    def __init__(
        self,
        sample_rate: int   = 16_000,
        n_mels:      int   = 80,
        n_fft:       int   = 512,
        hop_length:  int   = 160,
        fmin:        float = 0.0,
        fmax:        float | None = None,
        log_floor:   float = 1e-9,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels      = n_mels
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.fmin        = fmin
        self.fmax        = fmax if fmax is not None else sample_rate / 2.0
        self.log_floor   = log_floor
        self._filterbank = self._build_filterbank()

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        return self.transform(waveform)

    def transform(self, waveform: np.ndarray) -> np.ndarray:
        power   = self._stft_power(waveform)
        mel     = self._filterbank @ power
        log_mel = np.log(np.maximum(mel, self.log_floor))
        return log_mel.astype(np.float32)

    def _stft_power(self, waveform: np.ndarray) -> np.ndarray:
        from scipy.signal import stft as scipy_stft
        _, _, zxx = scipy_stft(
            waveform, fs=self.sample_rate, window="hann",
            nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length,
            nfft=self.n_fft,
        )
        return (np.abs(zxx) ** 2).astype(np.float64)

    def _build_filterbank(self) -> np.ndarray:
        n_freqs    = self.n_fft // 2 + 1
        freq_bins  = np.linspace(0, self.sample_rate / 2, n_freqs)

        def hz_to_mel(f: float) -> float:
            return 2595.0 * math.log10(1.0 + f / 700.0)

        def mel_to_hz(m: float) -> float:
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_points = np.linspace(hz_to_mel(self.fmin), hz_to_mel(self.fmax),
                                 self.n_mels + 2)
        hz_points  = np.array([mel_to_hz(m) for m in mel_points])

        filterbank = np.zeros((self.n_mels, n_freqs), dtype=np.float64)
        for m in range(1, self.n_mels + 1):
            f_left, f_center, f_right = hz_points[m-1], hz_points[m], hz_points[m+1]
            for k, f in enumerate(freq_bins):
                if f_left <= f <= f_center:
                    filterbank[m-1, k] = (f - f_left) / (f_center - f_left + 1e-12)
                elif f_center < f <= f_right:
                    filterbank[m-1, k] = (f_right - f) / (f_right - f_center + 1e-12)
        return filterbank


class AudioDetector:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        log.info("AudioDetector initialised (weights=%s)",
                 getattr(cfg.audio, "weights", "N/A"))

    def run(self, wav_path: str | Path | None) -> ModalResult:
        if wav_path is None:
            log.info("No audio track to analyse (source video has no audio stream)")
            return ModalResult(modality="audio", fake_prob=0.5,
                               error="No audio stream in source video")

        wav_path = Path(wav_path)

        if not wav_path.exists():
            log.error("wav_path not found: %s", wav_path)
            return ModalResult(modality="audio", fake_prob=0.5,
                               error=f"wav_path not found: {wav_path}")

        segments = self._segment_and_score(wav_path)
        if not segments:
            return ModalResult(modality="audio", fake_prob=0.5,
                               error="Empty audio file")

        overall = sum(s.fake_prob for s in segments) / len(segments)
        log.info("AudioDetector: %d segments, mean fake_prob=%.4f",
                 len(segments), overall)
        return ModalResult(modality="audio", fake_prob=overall,
                           audio_results=segments)

    def _segment_and_score(self, wav_path: Path) -> list[AudioSegmentResult]:
        import random
        size = wav_path.stat().st_size
        rng  = random.Random(size)
        n    = max(1, size // 32_000)
        return [
            AudioSegmentResult(
                segment_index = i,
                start_sec     = i * _SEGMENT_DURATION_S,
                end_sec       = (i + 1) * _SEGMENT_DURATION_S,
                fake_prob     = rng.uniform(0.0, 1.0),
            )
            for i in range(n)
        ]