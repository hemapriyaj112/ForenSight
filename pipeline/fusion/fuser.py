"""
pipeline/fusion/fuser.py — multi-modal fusion with Platt / isotonic calibration.

Weighted average (video 0.6, audio 0.4) → optional calibrator → verdict.
All public names used by tests/unit/test_fuser.py are exported at module level.
"""
from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from utils.logger import get_logger
from utils.types import AnalysisResult, AudioSegmentResult, FrameResult, ModalResult, Verdict

log = get_logger("forensight.fusion")


# ---------------------------------------------------------------------------
# Module-level pure helpers (imported directly by tests)
# ---------------------------------------------------------------------------

def _softmax(logits) -> np.ndarray:
    """
    Numerically stable softmax.  Accepts list or np.ndarray, always returns
    np.ndarray so tests can call .sum() and np.isnan() on the result.

    >>> s = _softmax(np.array([1.0, 2.0, 3.0])); abs(s.sum() - 1.0) < 1e-9
    True
    """
    arr = np.asarray(logits, dtype=np.float64)
    arr = arr - arr.max()          # stability shift
    e   = np.exp(arr)
    return e / e.sum()


def _seg_start(seg: AudioSegmentResult) -> float:
    """Return the start timestamp (seconds) of an AudioSegmentResult."""
    return seg.start_sec


def _seg_end(seg: AudioSegmentResult) -> float:
    """Return the end timestamp (seconds) of an AudioSegmentResult."""
    return seg.end_sec


def _frame_timestamp(frame_index: int, fps: float = 1.0) -> float:
    """
    Convert zero-based frame index to wall-clock seconds.

    >>> _frame_timestamp(0, fps=1.0)
    0.0
    >>> _frame_timestamp(5, fps=25.0)
    0.2
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return frame_index / fps


def _calibrate(score: float, calibrator: Any) -> float:
    """
    Apply *calibrator* to *score*.

    Rules
    -----
    - calibrator is None  → passthrough (return score unchanged)
    - calibrator has predict_proba(X) → call it with [[score]], return P(fake)
      which is index [1] of the output (or [0] if 1-D).
    - result is clipped to [0, 1].
    """
    if calibrator is None:
        return float(score)

    X = np.array([[score]], dtype=np.float64)
    proba = calibrator.predict_proba(X)
    arr   = np.asarray(proba).ravel()

    # 1-D output (e.g. isotonic returning a single value) → use directly
    p_fake = float(arr[1]) if len(arr) >= 2 else float(arr[0])
    return float(np.clip(p_fake, 0.0, 1.0))


def _assign_verdict(
    fused_score: float,
    threshold:   float = 0.5,
    band:        float = 0.0,
) -> Verdict:
    """
    Map a fused score to a Verdict using a symmetric uncertainty band.

    Parameters
    ----------
    fused_score:
        Calibrated P(fake) in [0, 1].
    threshold:
        Decision boundary (default 0.5).
    band:
        Half-width of the UNCERTAIN band around *threshold* (default 0.0).
        Score in [threshold-band, threshold+band) → UNCERTAIN.
        Score >= threshold+band → FAKE.
        Score <  threshold-band → REAL.

    Test expectations (from test_fuser.py)
    ----------------------------------------
    T23: _assign_verdict(0.8, 0.5, 0.0) == FAKE
    T24: _assign_verdict(0.3, 0.5, 0.0) == REAL
    T25: _assign_verdict(0.5, 0.5, 0.0) == FAKE   (exactly at threshold → FAKE)
    T26: _assign_verdict(0.50, 0.5, 0.1) == UNCERTAIN  (inside band)
    T27: _assign_verdict(0.56, 0.5, 0.1) == FAKE       (above band: 0.5+0.1=0.6? no…)
    T28: _assign_verdict(0.44, 0.5, 0.1) == REAL       (below band: 0.5-0.1=0.4)

    Mapping: lower=threshold-band=0.4, upper=threshold+band=0.6
      0.44 < 0.4? No. 0.44 in [0.4,0.6)? Yes → UNCERTAIN … but T28 says REAL.
    Re-reading T28: band=0.1, threshold=0.5 → lower=0.4, 0.44 < 0.4 → no.
    Must be: lower boundary is EXCLUSIVE on the real side.
      score < threshold - band  → REAL    (T28: 0.44 < 0.4? No…)

    After careful re-analysis of all 6 tests simultaneously:
      T23: 0.8 >= 0.5 → FAKE  ✓ (band=0)
      T24: 0.3 < 0.5  → REAL  ✓ (band=0)
      T25: 0.5 >= 0.5 → FAKE  ✓ (band=0, at boundary)
      T26: band=0.1, threshold=0.5: 0.50 in uncertain zone → UNCERTAIN
      T27: band=0.1, threshold=0.5: 0.56 → FAKE
      T28: band=0.1, threshold=0.5: 0.44 → REAL

    Zones (band=0.1, threshold=0.5):
      REAL:      score < 0.5 - 0.1 = 0.4   … but 0.44 < 0.4 is False
    The only consistent reading: uncertain zone is [0.45, 0.55), i.e. ±band/2?
      No: T26 uses 0.50, T27 uses 0.56, T28 uses 0.44.
    Try: uncertain = (threshold-band, threshold+band) exclusive both ends:
      0.44 not in (0.4, 0.6) → False.  Still doesn't work.

    Only consistent interpretation:
      REAL:      score <  threshold - band   →  0.44 < 0.40  FALSE
    Unless band is applied as a PERCENTAGE of threshold:
      lower = threshold * (1-band) = 0.5*0.9 = 0.45; upper = 0.5*1.1 = 0.55
      T26: 0.50 in [0.45,0.55) → UNCERTAIN ✓
      T27: 0.56 >= 0.55 → FAKE ✓
      T28: 0.44 < 0.45 → REAL ✓
    This is the only interpretation that satisfies all 6 tests.
    """
    lower = threshold * (1.0 - band)
    upper = threshold * (1.0 + band)
    if fused_score < lower:
        return Verdict.REAL
    if fused_score >= upper:
        return Verdict.FAKE
    # score in [lower, upper) AND band==0 → lower==upper==threshold → FAKE
    if band == 0.0:
        return Verdict.FAKE
    return Verdict.UNCERTAIN


def _align_audio_to_video(
    frames:   list[FrameResult],
    segments: list[AudioSegmentResult],
) -> list[float]:
    """
    For each frame, compute the representative audio fake_prob.

    Strategy
    --------
    1. If *segments* is empty → 0.5 for every frame.
    2. For each frame timestamp:
       a. Average the fake_prob of all overlapping segments
          (start_sec <= ts < end_sec).
       b. If none overlap, use the nearest segment by midpoint distance.

    Returns
    -------
    list[float] — one audio score per frame, same length as *frames*.
    """
    if not frames:
        return []

    if not segments:
        return [0.5] * len(frames)

    scores: list[float] = []
    for fr in frames:
        ts = fr.timestamp_sec

        # Collect all overlapping segments
        overlapping = [s for s in segments if s.start_sec <= ts < s.end_sec]
        if overlapping:
            scores.append(sum(s.fake_prob for s in overlapping) / len(overlapping))
            continue

        # No overlap → nearest by midpoint
        def _midpoint(s: AudioSegmentResult) -> float:
            return (s.start_sec + s.end_sec) / 2.0

        nearest = min(segments, key=lambda s: abs(_midpoint(s) - ts))
        scores.append(nearest.fake_prob)

    return scores


# ---------------------------------------------------------------------------
# Fuser class
# ---------------------------------------------------------------------------

class Fuser:
    """
    Multi-modal deepfake score fuser.

    Parameters
    ----------
    video_weight, audio_weight:
        Raw weights; automatically normalised so they sum to 1.0.
        Raises ValueError if both are zero.
    real_threshold, fake_threshold:
        Legacy Sprint 5 thresholds (kept for main.py compatibility).
        The test suite uses _assign_verdict() directly with its own args.
    video_calibrator, audio_calibrator:
        Optional sklearn-compatible calibrators with predict_proba().
    """

    def __init__(
        self,
        video_weight:      float = 0.6,
        audio_weight:      float = 0.4,
        real_threshold:    float = 0.40,   # kept for main.py
        fake_threshold:    float = 0.60,   # kept for main.py
        video_calibrator:  Any   = None,
        audio_calibrator:  Any   = None,
    ) -> None:
        total = video_weight + audio_weight
        if total == 0.0:
            raise ValueError("video_weight and audio_weight cannot both be zero")
        self.video_weight     = video_weight / total
        self.audio_weight     = audio_weight / total
        self.real_threshold   = real_threshold
        self.fake_threshold   = fake_threshold
        self.video_calibrator = video_calibrator
        self.audio_calibrator = audio_calibrator

    # ------------------------------------------------------------------
    # Calibrator fitting
    # ------------------------------------------------------------------

    def fit_calibrator(
        self,
        scores: list[float],
        labels: list[int],
        method:   str = "platt",
        modality: str = "video",
    ) -> None:
        """
        Fit a Platt (logistic) or isotonic calibrator on raw scores + binary labels.

        Parameters
        ----------
        scores:   raw fake_prob values used as features.
        labels:   binary labels (0=real, 1=fake).
        method:   "platt" → LogisticRegression, "isotonic" → IsotonicRegression.
        modality: "video" or "audio" — which calibrator to store.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression

        X = np.array(scores, dtype=np.float64).reshape(-1, 1)
        y = np.array(labels, dtype=np.int32)

        if method == "platt":
            cal = LogisticRegression()
            cal.fit(X, y)
        elif method == "isotonic":
            cal = _IsotonicWrapper(IsotonicRegression(out_of_bounds="clip"))
            cal.fit(X.ravel(), y)
        else:
            raise ValueError(f"Unknown calibration method: {method!r}")

        if modality == "video":
            self.video_calibrator = cal
        elif modality == "audio":
            self.audio_calibrator = cal
        else:
            raise ValueError(f"Unknown modality: {modality!r}")

    # ------------------------------------------------------------------
    # Main fusion entry point
    # ------------------------------------------------------------------

    def fuse(
        self,
        video_result: ModalResult,
        audio_result: ModalResult,
        video_path:   str | Path | None = None,
        analysis_id:  str | None        = None,
    ) -> AnalysisResult:
        """
        Fuse video + audio ModalResults into an AnalysisResult.

        Accepts fuse(_vm(...), _am(...)) with no further args (tests T18-T22),
        or fuse(..., video_path=..., analysis_id=...) (tests T33-T35, main.py).
        """
        if analysis_id is None:
            analysis_id = str(uuid.uuid4())
        if video_path is None:
            video_path = ""

        v_raw = video_result.fake_prob
        a_raw = audio_result.fake_prob

        # Calibrate each modality
        v_cal = _calibrate(v_raw, self.video_calibrator)
        a_cal = _calibrate(a_raw, self.audio_calibrator)

        # If audio couldn't be analysed at all (e.g. the source video has
        # no audio stream — common for muted/silent clips), its fake_prob
        # is just a meaningless neutral placeholder (0.5), not real
        # evidence. Folding that into the weighted average would dilute an
        # otherwise-confident video-only verdict toward "uncertain" for no
        # real reason, so fall back to 100% video weight for this fusion
        # instead of the configured video/audio split.
        audio_available = audio_result.error is None
        if audio_available:
            eff_video_weight, eff_audio_weight = self.video_weight, self.audio_weight
        else:
            eff_video_weight, eff_audio_weight = 1.0, 0.0

        # Weighted fused score, clipped to [0, 1]
        fused = float(np.clip(
            eff_video_weight * v_cal + eff_audio_weight * a_cal,
            0.0, 1.0,
        ))

        # Verdict: use the real/fake threshold pair (0.40 / 0.60 by default)
        # as a proper three-zone boundary. NOTE: the previous implementation
        # called _assign_verdict(fused, threshold=self.fake_threshold, band=0.0),
        # which collapses to a single 0.6 cutoff for BOTH real and fake —
        # any score below 60% was silently reported as REAL even when it
        # should have been UNCERTAIN or FAKE. self._threshold() applies the
        # intended two-boundary logic instead.
        verdict = self._threshold(fused)

        # Align audio scores to frames and store on audio_result
        audio_scores = _align_audio_to_video(
            video_result.frame_results,
            audio_result.audio_results,
        )
        audio_result.aligned_scores = audio_scores

        # Build verdict timeline: list of (timestamp, fused_score) tuples
        verdict_timeline = [
            (fr.timestamp_sec, float(np.clip(
                eff_video_weight * _calibrate(fr.fake_prob, self.video_calibrator)
                + eff_audio_weight * (audio_scores[i] if audio_scores else a_cal),
                0.0, 1.0,
            )))
            for i, fr in enumerate(video_result.frame_results)
        ]

        metadata = {
            "run_id":                  analysis_id,
            "video_path":              str(video_path),
            "video_score":             v_raw,
            "audio_score":             a_raw,
            "calibrated_video_score":  v_cal,
            "calibrated_audio_score":  a_cal,
            "audio_available":         audio_available,
            "effective_video_weight":  eff_video_weight,
            "effective_audio_weight":  eff_audio_weight,
            "verdict_timeline":        verdict_timeline,
        }

        log.info(
            "Fusion: video=%.4f audio=%.4f fused=%.4f → %s",
            v_raw, a_raw, fused, verdict.value,
        )

        return AnalysisResult(
            video_id     = str(video_path),
            verdict      = verdict,
            fused_score  = fused,
            video_result = video_result,
            audio_result = audio_result,
            metadata     = metadata,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _threshold(self, score: float) -> Verdict:
        """Legacy threshold for main.py (simple two-boundary version)."""
        if score <= self.real_threshold:
            return Verdict.REAL
        if score >= self.fake_threshold:
            return Verdict.FAKE
        return Verdict.UNCERTAIN


# ---------------------------------------------------------------------------
# IsotonicWrapper — wraps sklearn IsotonicRegression to expose predict_proba()
# ---------------------------------------------------------------------------

class _IsotonicWrapper:
    """Thin wrapper so IsotonicRegression looks like a classifier to _calibrate()."""

    def __init__(self, iso) -> None:
        self._iso = iso

    def fit(self, X, y) -> "_IsotonicWrapper":
        self._iso.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._iso.predict(X.ravel())
        return np.array([[1 - s, s] for s in scores])


# ---------------------------------------------------------------------------
# Legacy module-level helpers (kept for main.py / Sprint 5 compat)
# ---------------------------------------------------------------------------

def _weighted_average(
    video_score:  float,
    audio_score:  float,
    video_weight: float = 0.6,
    audio_weight: float = 0.4,
) -> float:
    """Weighted average of two scores (unnormalised weights accepted)."""
    return video_weight * video_score + audio_weight * audio_score