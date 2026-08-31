"""
pipeline/audio/audio_forensics.py

Real audio forensics heuristics for deepfake detection.

AI-generated or manipulated speech typically exhibits:
  - Unnatural spectral stability (generators tend to produce consistent
    spectral envelopes frame-to-frame, while real speech varies naturally)
  - Phase incoherence between harmonics (neural vocoders often struggle
    with phase consistency)
  - Abnormal MFCC distributions (generated speech sits in a different
    region of the feature space than natural speech)
  - Reduced spectral flux (less natural variation between frames)
  - Unnatural zero-crossing rate patterns

All heuristics are computed with librosa (already in requirements) and
return a 0-1 suspicion score plus human-readable findings.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

_SEGMENT_DURATION_S = 2.0  # default; authoritative value comes from config.audio.segment_duration
_SAMPLE_RATE = 16_000
_N_FFT = 512
_HOP_LENGTH = 160


def _load_waveform(wav_path) -> np.ndarray:
    try:
        import soundfile as sf
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=-1)
        if sr != _SAMPLE_RATE:
            t = np.linspace(0, len(data) / sr, len(data), endpoint=False)
            new_t = np.linspace(0, t[-1], int(len(data) * _SAMPLE_RATE / sr), endpoint=False)
            data = np.interp(new_t, t, data)
        return data.astype(np.float32)
    except Exception:
        return None


def _stft(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.signal import stft as scipy_stft
    f, t, zxx = scipy_stft(
        waveform, fs=_SAMPLE_RATE, window="hann",
        nperseg=_N_FFT, noverlap=_N_FFT - _HOP_LENGTH,
        nfft=_N_FFT,
    )
    mag = np.abs(zxx)
    phase = np.angle(zxx)
    return f, t, mag, phase


def _spectral_centroid_stability(mag: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    """Measure how stable spectral centroids are across frames.

    Real speech has natural variation in spectral centroid from frame to
    frame. AI-generated or vocoded speech tends to have unnaturally stable
    centroids because the generator produces a consistent spectral envelope.

    Returns (stability_score, mean_centroid) where stability_score is 0-1
    (higher = more suspiciously stable).
    """
    epsilon = 1e-9
    power = np.maximum(mag, epsilon)
    centroids = (f[:, np.newaxis] * power).sum(axis=0) / (power.sum(axis=0) + epsilon)
    if centroids.size < 3:
        return 0.3, float(centroids.mean()) if centroids.size > 0 else 0.0

    diffs = np.abs(np.diff(centroids))
    mean_centroid = float(np.mean(centroids))
    if mean_centroid <= 0:
        return 0.3, 0.0

    relative_variation = float(np.std(diffs) / (mean_centroid + epsilon))
    score = float(np.clip(1.0 - relative_variation * 8.0, 0.0, 1.0))
    return score, mean_centroid


def _phase_coherence(phase: np.ndarray) -> float:
    """Measure phase coherence between adjacent frames.

    Real audio has smooth phase evolution. AI-generated audio (especially
    from neural vocoders) can show phase discontinuities or inconsistent
    phase relationships between harmonics.

    Returns 0-1 score (higher = more suspicious phase pattern).
    """
    if phase.shape[1] < 3:
        return 0.3

    phase_diffs = np.diff(phase, axis=1)
    wrapped = np.mod(phase_diffs + math.pi, 2 * math.pi) - math.pi

    magnitudes = np.abs(wrapped)
    median_mag = float(np.median(magnitudes))
    if median_mag < 1e-6:
        return 0.3

    frame_stds = np.std(magnitudes, axis=0)
    consistency = float(np.mean(frame_stds) / (median_mag + 1e-9))
    score = float(np.clip(consistency / 3.0, 0.0, 1.0))
    return score


def _spectral_flux(mag: np.ndarray) -> float:
    """Measure spectral flux (frame-to-frame spectral change).

    Real speech has natural variation in spectral content. AI-generated
    speech often has reduced spectral flux — the spectrum changes less
    from frame to frame.

    Returns 0-1 score (higher = more suspiciously low flux).
    """
    if mag.shape[1] < 3:
        return 0.3

    diff = np.diff(mag, axis=1)
    flux = np.sqrt(np.sum(diff ** 2, axis=0))
    if flux.max() > 0:
        flux = flux / flux.max()

    mean_flux = float(np.mean(flux))
    score = float(np.clip(1.0 - mean_flux * 4.0, 0.0, 1.0))
    return score


def _mfcc_distribution(mag: np.ndarray, f: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute MFCC-like features and measure how unusual their distribution is.

    Real speech has MFCCs with specific statistical properties. AI speech
    tends to cluster in different regions of MFCC space.

    Returns (score, mfcc_mean) where score is 0-1 (higher = more suspicious).
    """
    if mag.shape[1] < 5:
        return 0.3, np.array([])

    power = mag ** 2
    mel_filterbank = _build_mel_filterbank(f, n_mels=13)
    mel_spec = mel_filterbank @ power
    log_mel = np.log(np.maximum(mel_spec, 1e-9))

    mfcc = _dct(log_mel, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    mfcc_std = np.std(mfcc, axis=1)
    normalized_std = mfcc_std / (np.abs(mfcc_mean) + 1e-9)
    avg_normalized_std = float(np.mean(normalized_std))

    score = float(np.clip(1.0 - avg_normalized_std * 3.0, 0.0, 1.0))
    return score, mfcc_mean


def _build_mel_filterbank(freqs: np.ndarray, n_mels: int = 13) -> np.ndarray:
    n_freqs = len(freqs)
    mel_min = 2595.0 * math.log10(1.0 + freqs[1] / 700.0) if len(freqs) > 1 else 0.0
    mel_max = 2595.0 * math.log10(1.0 + freqs[-1] / 700.0)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([700.0 * (10.0 ** (m / 2595.0) - 1.0) for m in mel_points])

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(1, n_mels + 1):
        for k, f in enumerate(freqs):
            if hz_points[m - 1] <= f <= hz_points[m]:
                filterbank[m - 1, k] = (f - hz_points[m - 1]) / (hz_points[m] - hz_points[m - 1] + 1e-12)
            elif hz_points[m] < f <= hz_points[m + 1]:
                filterbank[m - 1, k] = (hz_points[m + 1] - f) / (hz_points[m + 1] - hz_points[m] + 1e-12)
    return filterbank


def _dct(log_mel: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    n_frames, n_mels = log_mel.shape
    mfcc = np.zeros((n_mfcc, n_frames), dtype=np.float64)
    for m in range(n_mfcc):
        for n in range(n_frames):
            s = sum(log_mel[n, k] * math.cos(math.pi * m * (2 * k + 1) / (2 * n_mels)) for k in range(n_mels))
            mfcc[m, n] = s * math.sqrt(2.0 / n_mels) if m > 0 else s * math.sqrt(1.0 / n_mels)
    return mfcc


def _zero_crossing_rate(waveform: np.ndarray, frame_length: int = 512, hop_length: int = 160) -> float:
    if len(waveform) < frame_length:
        return 0.3

    n_frames = 1 + (len(waveform) - frame_length) // hop_length
    if n_frames < 3:
        return 0.3

    zcrs = []
    for i in range(n_frames):
        start = i * hop_length
        frame = waveform[start:start + frame_length]
        zc = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0
        zcrs.append(zc)

    zcrs = np.array(zcrs)
    mean_zcr = float(np.mean(zcrs))
    std_zcr = float(np.std(zcrs))

    if mean_zcr <= 0:
        return 0.3

    cv = std_zcr / (mean_zcr + 1e-9)
    if cv < 0.15:
        return 0.7
    elif cv < 0.3:
        return 0.4
    return 0.2


def analyse_segment(waveform: np.ndarray) -> dict[str, Any]:
    """Run all audio forensics heuristics on a waveform segment.

    Returns dict with score (0-1), findings (list[str]), sub_scores.
    """
    if waveform is None or len(waveform) < _N_FFT:
        return {"score": 0.5, "findings": [], "sub_scores": {}}

    f, t, mag, phase = _stft(waveform)
    if mag.shape[1] < 3:
        return {"score": 0.5, "findings": [], "sub_scores": {}}

    stability_score, mean_centroid = _spectral_centroid_stability(mag, f)
    phase_score = _phase_coherence(phase)
    flux_score = _spectral_flux(mag)
    mfcc_score, _ = _mfcc_distribution(mag, f)
    zcr_score = _zero_crossing_rate(waveform)

    sub_scores = {
        "spectral_stability": stability_score,
        "phase_coherence": phase_score,
        "spectral_flux": flux_score,
        "mfcc_distribution": mfcc_score,
        "zcr_pattern": zcr_score,
    }

    weights = {
        "spectral_stability": 0.30,
        "phase_coherence": 0.25,
        "spectral_flux": 0.20,
        "mfcc_distribution": 0.15,
        "zcr_pattern": 0.10,
    }

    score = sum(sub_scores[k] * weights[k] for k in sub_scores)
    score = float(np.clip(score, 0.0, 1.0))

    findings = []
    if stability_score > 0.6:
        findings.append(
            "Spectral centroid shows unusually low variation across frames, "
            "consistent with AI-generated or vocoded speech."
        )
    if phase_score > 0.6:
        findings.append(
            "Phase relationships between harmonics are unusually consistent, "
            "a pattern often seen in neural vocoder output."
        )
    if flux_score > 0.6:
        findings.append(
            "Spectral content changes less between frames than natural speech "
            "typically does."
        )
    if mfcc_score > 0.6:
        findings.append(
            "MFCC distribution falls outside the typical range for natural speech."
        )
    if zcr_score > 0.6:
        findings.append(
            "Zero-crossing rate pattern is unnaturally consistent across frames."
        )

    if not findings and score < 0.4:
        findings.append(
            "Audio spectral characteristics look consistent with natural speech."
        )

    return {"score": score, "findings": findings, "sub_scores": sub_scores}


def segment_audio(wav_path, segment_duration: float = _SEGMENT_DURATION_S):
    """Split audio into overlapping segments for analysis.

    Returns list of (start_sec, end_sec, waveform_array).
    """
    waveform = _load_waveform(wav_path)
    if waveform is None or len(waveform) == 0:
        return []

    samples_per_segment = int(segment_duration * _SAMPLE_RATE)
    hop = samples_per_segment // 2
    if hop < 1:
        hop = samples_per_segment

    segments = []
    n = len(waveform)

    if n < samples_per_segment:
        segments.append((0.0, n / _SAMPLE_RATE, waveform))
        return segments

    idx = 0
    while idx < n:
        end = min(idx + samples_per_segment, n)
        seg = waveform[idx:end]
        if len(seg) > 0:
            segments.append((idx / _SAMPLE_RATE, end / _SAMPLE_RATE, seg))
        if end >= n:
            break
        idx += hop

    return segments
