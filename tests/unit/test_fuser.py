"""
tests/unit/test_fuser.py  —  Sprint 4 (38 tests)
==================================================
Exact Sprint 1 field names (confirmed from discovery stdout):

  FrameResult:        frame_index, timestamp_sec, fake_prob, face_detected,
                      bbox, face_crop, gradcam_overlay, no_face_strategy_applied
  AudioSegmentResult: segment_index, start_sec, end_sec, fake_prob
  ModalResult:        modality, fake_prob, frame_results, audio_results, metadata
  AnalysisResult:     video_id, verdict, fused_score, video_result, audio_result, metadata

Extended fuser outputs (raw scores, calibrated scores, timeline) live in
AnalysisResult.metadata under keys: video_score, audio_score,
calibrated_video_score, calibrated_audio_score, verdict_timeline.
"""
from __future__ import annotations

import dataclasses
import sys, os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pipeline.fusion.fuser import (
    Fuser,
    _align_audio_to_video,
    _assign_verdict,
    _calibrate,
    _softmax,
    _frame_timestamp,
    _seg_start,
    _seg_end,
)
from utils.types import (
    AnalysisResult,
    AudioSegmentResult,
    FrameResult,
    ModalResult,
    Verdict,
)


# ---------------------------------------------------------------------------
# Fixtures  (exact Sprint 1 field names)
# ---------------------------------------------------------------------------

def _fr(idx: int, ts: float, prob: float, face: bool = True) -> FrameResult:
    return FrameResult(frame_index=idx, timestamp_sec=ts, fake_prob=prob, face_detected=face)


def _seg(idx: int, start: float, end: float, prob: float) -> AudioSegmentResult:
    return AudioSegmentResult(segment_index=idx, start_sec=start, end_sec=end, fake_prob=prob)


def _vm(prob: float, frames=None) -> ModalResult:
    return ModalResult(modality="video", fake_prob=prob,
                       frame_results=frames or [], audio_results=[])


def _am(prob: float, segs=None) -> ModalResult:
    return ModalResult(modality="audio", fake_prob=prob,
                       frame_results=[], audio_results=segs or [])


# ---------------------------------------------------------------------------
# Helpers to read extended metadata from AnalysisResult
# ---------------------------------------------------------------------------

def _meta(r: AnalysisResult, key: str, default=None):
    return r.metadata.get(key, default)


# ---------------------------------------------------------------------------
# Field discovery — confirms Sprint 1 schema is still intact
# ---------------------------------------------------------------------------

def test_field_discovery():
    fr_fields = {f.name for f in dataclasses.fields(FrameResult)}
    as_fields = {f.name for f in dataclasses.fields(AudioSegmentResult)}
    mr_fields = {f.name for f in dataclasses.fields(ModalResult)}
    ar_fields = {f.name for f in dataclasses.fields(AnalysisResult)}

    print(f"\nFrameResult        fields: {sorted(fr_fields)}")
    print(f"AudioSegmentResult fields: {sorted(as_fields)}")
    print(f"ModalResult        fields: {sorted(mr_fields)}")
    print(f"AnalysisResult     fields: {sorted(ar_fields)}")

    # FrameResult
    assert "timestamp_sec"  in fr_fields, f"FrameResult missing timestamp_sec. Got: {sorted(fr_fields)}"
    assert "fake_prob"       in fr_fields, f"FrameResult missing fake_prob. Got: {sorted(fr_fields)}"
    assert "face_detected"   in fr_fields, f"FrameResult missing face_detected. Got: {sorted(fr_fields)}"
    # AudioSegmentResult
    assert "start_sec"       in as_fields, f"AudioSegmentResult missing start_sec. Got: {sorted(as_fields)}"
    assert "end_sec"         in as_fields, f"AudioSegmentResult missing end_sec. Got: {sorted(as_fields)}"
    assert "fake_prob"       in as_fields, f"AudioSegmentResult missing fake_prob. Got: {sorted(as_fields)}"
    # ModalResult
    assert "fake_prob"       in mr_fields, f"ModalResult missing fake_prob. Got: {sorted(mr_fields)}"
    assert "frame_results"   in mr_fields, f"ModalResult missing frame_results. Got: {sorted(mr_fields)}"
    assert "audio_results"   in mr_fields, f"ModalResult missing audio_results. Got: {sorted(mr_fields)}"
    # AnalysisResult
    assert "video_id"        in ar_fields, f"AnalysisResult missing video_id. Got: {sorted(ar_fields)}"
    assert "verdict"         in ar_fields, f"AnalysisResult missing verdict. Got: {sorted(ar_fields)}"
    assert "fused_score"     in ar_fields, f"AnalysisResult missing fused_score. Got: {sorted(ar_fields)}"
    assert "video_result"    in ar_fields, f"AnalysisResult missing video_result. Got: {sorted(ar_fields)}"
    assert "audio_result"    in ar_fields, f"AnalysisResult missing audio_result. Got: {sorted(ar_fields)}"
    assert "metadata"        in ar_fields, f"AnalysisResult missing metadata. Got: {sorted(ar_fields)}"


# ---------------------------------------------------------------------------
# T01-T05  Construction & weight normalisation
# ---------------------------------------------------------------------------

class TestFuserConstruction:

    def test_T01_default_weights_sum_to_one(self):
        f = Fuser()
        assert abs(f.video_weight + f.audio_weight - 1.0) < 1e-9

    def test_T02_custom_weights_normalised(self):
        f = Fuser(video_weight=3.0, audio_weight=1.0)
        assert abs(f.video_weight - 0.75) < 1e-9
        assert abs(f.audio_weight - 0.25) < 1e-9

    def test_T03_equal_weights_normalised(self):
        f = Fuser(video_weight=1.0, audio_weight=1.0)
        assert abs(f.video_weight - 0.5) < 1e-9

    def test_T04_zero_weights_raise(self):
        with pytest.raises(ValueError):
            Fuser(video_weight=0.0, audio_weight=0.0)

    def test_T05_calibrators_stored(self):
        class Cal:
            def predict_proba(self, X): return np.array([[0.3, 0.7]])
            def fit(self, X, y): return self
        cal = Cal()
        f = Fuser(video_calibrator=cal, audio_calibrator=cal)
        assert f.video_calibrator is cal
        assert f.audio_calibrator is cal


# ---------------------------------------------------------------------------
# T06-T12  Timestamp alignment
# ---------------------------------------------------------------------------

class TestTimestampAlignment:

    def test_T06_empty_frames_returns_empty(self):
        assert len(_align_audio_to_video([], [_seg(0, 0, 3, 0.8)])) == 0

    def test_T07_empty_segments_returns_neutral(self):
        result = _align_audio_to_video([_fr(0, 0.0, 0.9), _fr(1, 1.0, 0.1)], [])
        assert list(result) == [0.5, 0.5]

    def test_T08_frame_inside_segment(self):
        result = _align_audio_to_video([_fr(0, 1.5, 0.9)], [_seg(0, 0.0, 3.0, 0.7)])
        assert abs(result[0] - 0.7) < 1e-9

    def test_T09_overlapping_segments_averaged(self):
        result = _align_audio_to_video(
            [_fr(0, 1.5, 0.9)],
            [_seg(0, 0.0, 3.0, 0.6), _seg(1, 1.5, 4.5, 0.8)],
        )
        assert abs(result[0] - 0.7) < 1e-9   # mean(0.6, 0.8)

    def test_T10_frame_before_all_segments_uses_nearest(self):
        result = _align_audio_to_video([_fr(0, 0.0, 0.5)], [_seg(0, 2.0, 5.0, 0.9)])
        assert abs(result[0] - 0.9) < 1e-9

    def test_T11_frame_after_all_segments_uses_nearest(self):
        result = _align_audio_to_video(
            [_fr(0, 20.0, 0.5)],
            [_seg(0, 0.0, 3.0, 0.2), _seg(1, 3.0, 6.0, 0.4)],
        )
        assert abs(result[0] - 0.4) < 1e-9   # seg1 midpoint=4.5 is nearest

    def test_T12_multiple_frames_aligned_independently(self):
        result = _align_audio_to_video(
            [_fr(0, 0.5, 0.9), _fr(1, 4.0, 0.1)],
            [_seg(0, 0.0, 3.0, 0.3), _seg(1, 3.0, 6.0, 0.8)],
        )
        assert abs(result[0] - 0.3) < 1e-9
        assert abs(result[1] - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# T13-T17  Calibration
# ---------------------------------------------------------------------------

class TestCalibration:

    def test_T13_no_calibrator_passthrough(self):
        assert abs(_calibrate(0.75, None) - 0.75) < 1e-9

    def test_T14_calibrator_called(self):
        class FakeCal:
            def predict_proba(self, X): return np.array([[0.2, 0.8]])
            def fit(self, X, y): return self
        assert abs(_calibrate(0.5, FakeCal()) - 0.8) < 1e-9

    def test_T15_isotonic_1d_output_handled(self):
        class OneDCal:
            def predict_proba(self, X): return np.array([0.6])
            def fit(self, X, y): return self
        assert abs(_calibrate(0.5, OneDCal()) - 0.6) < 1e-9

    def test_T16_fit_calibrator_platt_sklearn(self):
        pytest.importorskip("sklearn")
        f = Fuser()
        f.fit_calibrator([0.1, 0.2, 0.7, 0.85, 0.9], [0, 0, 1, 1, 1],
                         method="platt", modality="video")
        assert f.video_calibrator is not None
        assert 0.0 <= _calibrate(0.8, f.video_calibrator) <= 1.0

    def test_T17_fit_calibrator_isotonic_sklearn(self):
        pytest.importorskip("sklearn")
        f = Fuser()
        f.fit_calibrator([0.1, 0.3, 0.6, 0.9], [0, 0, 1, 1],
                         method="isotonic", modality="audio")
        assert f.audio_calibrator is not None
        assert 0.0 <= _calibrate(0.5, f.audio_calibrator) <= 1.0


# ---------------------------------------------------------------------------
# T18-T22  Weighted fusion arithmetic
# ---------------------------------------------------------------------------

class TestFusionArithmetic:

    def test_T18_fused_score_is_weighted_sum(self):
        result = Fuser(video_weight=0.6, audio_weight=0.4).fuse(_vm(0.8), _am(0.5))
        assert abs(result.fused_score - (0.6 * 0.8 + 0.4 * 0.5)) < 1e-6

    def test_T19_fused_score_clipped_upper(self):
        class HighCal:
            def predict_proba(self, X): return np.array([[0.0, 1.05]])
            def fit(self, X, y): return self
        f = Fuser(video_calibrator=HighCal(), audio_calibrator=HighCal())
        assert f.fuse(_vm(1.0), _am(1.0)).fused_score <= 1.0

    def test_T20_fused_score_clipped_lower(self):
        class NegCal:
            def predict_proba(self, X): return np.array([[1.05, -0.05]])
            def fit(self, X, y): return self
        f = Fuser(video_calibrator=NegCal(), audio_calibrator=NegCal())
        assert f.fuse(_vm(0.0), _am(0.0)).fused_score >= 0.0

    def test_T21_raw_scores_stored_in_metadata(self):
        result = Fuser().fuse(_vm(0.7), _am(0.3))
        assert abs(_meta(result, "video_score") - 0.7) < 1e-9
        assert abs(_meta(result, "audio_score") - 0.3) < 1e-9

    def test_T22_calibrated_scores_stored_in_metadata(self):
        result = Fuser().fuse(_vm(0.6), _am(0.4))
        assert _meta(result, "calibrated_video_score") is not None
        assert abs(_meta(result, "calibrated_video_score") - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# T23-T28  Verdict assignment
# ---------------------------------------------------------------------------

class TestVerdictAssignment:

    def test_T23_above_threshold_is_fake(self):
        assert _assign_verdict(0.8, 0.5, 0.0) == Verdict.FAKE

    def test_T24_below_threshold_is_real(self):
        assert _assign_verdict(0.3, 0.5, 0.0) == Verdict.REAL

    def test_T25_exactly_at_threshold_is_fake(self):
        assert _assign_verdict(0.5, 0.5, 0.0) == Verdict.FAKE

    def test_T26_uncertain_band_middle(self):
        assert _assign_verdict(0.50, 0.5, 0.1) == Verdict.UNCERTAIN

    def test_T27_uncertain_band_above_upper_is_fake(self):
        assert _assign_verdict(0.56, 0.5, 0.1) == Verdict.FAKE

    def test_T28_uncertain_band_below_lower_is_real(self):
        assert _assign_verdict(0.44, 0.5, 0.1) == Verdict.REAL


# ---------------------------------------------------------------------------
# T29-T32  Timeline builder
# ---------------------------------------------------------------------------

class TestTimelineBuilder:

    def test_T29_empty_frames_gives_empty_timeline(self):
        result = Fuser().fuse(_vm(0.6), _am(0.4))
        assert _meta(result, "verdict_timeline") == []

    def test_T30_timeline_length_equals_frame_count(self):
        frames = [_fr(i, float(i), 0.5) for i in range(5)]
        result = Fuser().fuse(_vm(0.6, frames=frames), _am(0.4))
        assert len(_meta(result, "verdict_timeline")) == 5

    def test_T31_timeline_timestamps_match_frames(self):
        frames = [_fr(0, 0.0, 0.9), _fr(1, 1.0, 0.2)]
        result = Fuser().fuse(_vm(0.55, frames=frames), _am(0.4))
        ts = [t for t, _ in _meta(result, "verdict_timeline")]
        assert ts == [0.0, 1.0]

    def test_T32_timeline_scores_in_unit_interval(self):
        frames = [_fr(i, float(i), float(i) / 10) for i in range(10)]
        segs   = [_seg(0, 0.0, 5.0, 0.3), _seg(1, 5.0, 10.0, 0.7)]
        result = Fuser().fuse(_vm(0.5, frames=frames), _am(0.5, segs=segs))
        for _, s in _meta(result, "verdict_timeline"):
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# T33-T35  End-to-end
# ---------------------------------------------------------------------------

class TestFuseEndToEnd:

    def test_T33_full_pipeline_fake_verdict(self):
        f      = Fuser(video_weight=0.6, audio_weight=0.4)
        frames = [_fr(i, float(i), 0.9) for i in range(3)]
        segs   = [_seg(0, 0.0, 3.0, 0.85)]
        result = f.fuse(_vm(0.9, frames=frames), _am(0.85, segs=segs),
                        video_path="test.mp4", analysis_id="abc-123")
        assert result.verdict      == Verdict.FAKE
        assert result.fused_score   > 0.5
        assert result.video_id     == "test.mp4"
        assert _meta(result, "run_id") == "abc-123"

    def test_T34_full_pipeline_real_verdict(self):
        f      = Fuser(video_weight=0.6, audio_weight=0.4)
        frames = [_fr(i, float(i), 0.1) for i in range(3)]
        segs   = [_seg(0, 0.0, 3.0, 0.15)]
        result = f.fuse(_vm(0.1, frames=frames), _am(0.15, segs=segs))
        assert result.verdict     == Verdict.REAL
        assert result.fused_score  < 0.5

    def test_T35_audio_aligned_scores_populated(self):
        frames = [_fr(0, 0.5, 0.8), _fr(1, 1.5, 0.7)]
        segs   = [_seg(0, 0.0, 3.0, 0.6)]
        result = Fuser().fuse(_vm(0.75, frames=frames), _am(0.6, segs=segs))
        assert result.audio_result.aligned_scores is not None
        assert len(result.audio_result.aligned_scores) == 2


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def test_softmax_sums_to_one():
    s = _softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(s.sum() - 1.0) < 1e-9

def test_softmax_numerical_stability():
    s = _softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert not any(np.isnan(s))
    assert abs(s.sum() - 1.0) < 1e-9