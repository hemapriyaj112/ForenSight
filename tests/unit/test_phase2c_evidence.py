"""Unit tests for Phase 2C evidence-analysis utilities.

These test PURE ANALYSIS helpers in pipeline/analysis/evidence.py. They never
instantiate a detector, model, fuser, or config — no weights are touched.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pipeline.analysis import evidence  # noqa: E402


def make_row(**over):
    """Minimal eval-result row (shape matches scripts/evaluate_testing_videos.py)."""
    row = {
        "filename": "x.mp4", "ground_truth": "FAKE", "duration_seconds": 10.0,
        "fps": 24.0, "has_audio": True, "verdict": "FAKE", "fusion_score": 0.7,
        "video_score": 0.6, "audio_score": 0.1, "texture_score": 0.5,
        "frequency_score": 0.5, "provenance_score": 0.55,
        "provenance_override": False, "provenance_reason": None,
        "classifier_status": "INACTIVE", "classifier_score": None,
        "classifier_reason": "disabled in eval", "error": None,
        "genconvit_status": "ACTIVE_MODEL", "genconvit_score": 0.48,
        "genconvit_predicted_label": "FAKE", "genconvit_frames_used": 10,
        "genconvit_frames_requested": 10, "genconvit_frames_sampled": 10,
        "genconvit_face_frames_found": 10, "genconvit_face_coverage": 1.0,
        "genconvit_average_face_confidence": 0.95,
        "genconvit_coverage_status": "GOOD_FACE_COVERAGE",
        "aasist_status": "ACTIVE_MODEL", "aasist_spoof_score": 0.99,
        "aasist_bonafide_score": 0.01, "aasist_predicted_label": "SPOOF",
        "aasist_window_count": 2, "aasist_window_scores": [0.99, 0.98],
        "aasist_note": None, "aasist_load_error": None,
        "audio_status": "ACTIVE", "audio_segments": [
            {"segment_index": 0, "start": 0.0, "end": 2.0, "fake_prob": 0.1},
            {"segment_index": 1, "start": 2.0, "end": 4.0, "fake_prob": 0.2},
        ],
        "provenance": {
            "generator_tag": None, "has_c2pa": False, "provenance_score": 0.55,
            "uncorroborated_video_evidence": False, "temporal_analysis": {"score": 0.5},
        },
    }
    row.update(over)
    return row


def ev_for(row):
    v = evidence.assemble_video_evidence(row)
    v["evidence_categories"] = evidence.classify_evidence(v)
    v["agreement"] = evidence.agreement_pattern(v)
    return v


# --------------------------------------------------------------------------
# AASIST window timing
# --------------------------------------------------------------------------

class TestAasistWindowTimestamps:
    def test_window_timestamps_geometry(self):
        assert evidence.AASIST_WINDOW_SECONDS == pytest.approx(4.0375)
        ts = evidence.aasist_window_timestamps(3)
        # Detector rounds window starts to 3 decimals (pipeline/aasist/detector.py).
        assert ts == [pytest.approx(0.0), pytest.approx(4.037), pytest.approx(8.075)]

    def test_zero_windows(self):
        assert evidence.aasist_window_timestamps(0) == []
        assert evidence.aasist_window_timestamps(None) == []


# --------------------------------------------------------------------------
# AASIST segment pattern classification (Task 6 helper)
# --------------------------------------------------------------------------

class TestAasistSegmentPattern:
    def test_persistent_high(self):
        p = evidence.aasist_segment_pattern([0.96, 0.98, 0.99, 0.97, 0.95])
        assert p["pattern"] == "persistent_high_spoof"
        assert p["pct_high_ge_0_8"] == 1.0

    def test_persistent_low(self):
        p = evidence.aasist_segment_pattern([0.01, 0.02, 0.05, 0.03])
        assert p["pattern"] == "persistent_low_spoof"
        assert p["pct_low_lt_0_2"] == 1.0

    def test_mixed_localized(self):
        p = evidence.aasist_segment_pattern([0.995, 0.01, 0.401])
        assert p["pattern"] == "mixed_or_localized"
        assert p["high_windows"] == [0]

    def test_no_windows(self):
        assert evidence.aasist_segment_pattern([])["pattern"] == "no_windows"
        assert evidence.aasist_segment_pattern(None)["pattern"] == "no_windows"


# --------------------------------------------------------------------------
# Evidence taxonomy (Task 2)
# --------------------------------------------------------------------------

class TestClassifyEvidence:
    def test_no_audio_category(self):
        v = ev_for(make_row(has_audio=False, aasist_note="No audio track",
                            aasist_status="INACTIVE", aasist_spoof_score=None,
                            aasist_predicted_label=None, audio_status="INACTIVE",
                            audio_score=None))
        cats = {c["category"] for c in v["evidence_categories"]}
        assert "NO_AUDIO" in cats

    def test_audio_spoof_vs_authentic(self):
        cats = {c["category"] for c in ev_for(make_row())["evidence_categories"]}
        assert "AUDIO_SPOOF_EVIDENCE" in cats
        assert "AUTHENTIC_AUDIO_EVIDENCE" not in cats

        bonafide = ev_for(make_row(aasist_spoof_score=0.02,
                                   aasist_bonafide_score=0.98,
                                   aasist_predicted_label="BONAFIDE"))
        cats = {c["category"] for c in bonafide["evidence_categories"]}
        assert "AUTHENTIC_AUDIO_EVIDENCE" in cats

    def test_no_face_and_low_coverage(self):
        v = ev_for(make_row(genconvit_face_frames_found=0, genconvit_face_coverage=0.0,
                            genconvit_coverage_status="NO_FACE_DETECTED"))
        cats = {c["category"] for c in v["evidence_categories"]}
        assert "NO_FACE" in cats

    def test_compression_risk_only_when_real_label(self):
        # AASIST SPOOF + low heuristic on a FAKE video -> disagreement, NOT
        # an auto compression-risk tag.
        fake = ev_for(make_row(ground_truth="FAKE"))
        fake_cats = {c["category"] for c in fake["evidence_categories"]}
        assert "COMPRESSION_OR_CODEC_RISK" not in fake_cats
        assert "MODALITY_DISAGREEMENT" in fake_cats

        # Same signal pattern but provided label REAL -> compression hypothesis.
        real = ev_for(make_row(ground_truth="REAL"))
        real_cats = {c["category"] for c in real["evidence_categories"]}
        comp = [c for c in real["evidence_categories"]
                if c["category"] == "COMPRESSION_OR_CODEC_RISK"]
        assert comp and comp[0]["status"] == "hypothesis"

    def test_modality_disagreement_neural_vs_neural(self):
        row = make_row(genconvit_predicted_label="REAL", genconvit_score=0.58)
        cats = {c["category"] for c in ev_for(row)["evidence_categories"]}
        assert "MODALITY_DISAGREEMENT" in cats

    def test_multimodal_agreement_caveat_on_real(self):
        real = ev_for(make_row(ground_truth="REAL", genconvit_predicted_label="FAKE",
                               texture_score=0.7))
        ag = [c for c in real["evidence_categories"]
              if c["category"] == "MULTIMODAL_AGREEMENT"]
        assert ag and "CONFLICTS" in ag[0]["reason"]

    def test_provenance_evidence(self):
        row = make_row(provenance={
            "generator_tag": "AI tool", "has_c2pa": True,
            "provenance_score": 0.97, "uncorroborated_video_evidence": True,
        }, provenance_override=True)
        cats = {c["category"] for c in ev_for(row)["evidence_categories"]}
        assert "PROVENANCE_EVIDENCE" in cats

    def test_agreement_pattern_inference(self):
        v = ev_for(make_row(verdict="FAKE"))
        assert v["agreement"]["pattern"] == "STRONG_MULTIMODAL_AGREEMENT"

        noaud = ev_for(make_row(has_audio=False, aasist_status="INACTIVE",
                                aasist_spoof_score=None, aasist_predicted_label=None,
                                audio_status="INACTIVE", audio_score=None,
                                genconvit_face_frames_found=0,
                                genconvit_coverage_status="NO_FACE_DETECTED",
                                verdict="UNCERTAIN"))
        assert noaud["agreement"]["pattern"] in ("SINGLE_WEAK_MODALITY", "AMBI")


# --------------------------------------------------------------------------
# Production invariance (Task 14)
# --------------------------------------------------------------------------

class TestProductionInvariance:
    BASELINE_ROWS = [
        ("ai_noaud2.mp4", "FAKE", "UNCERTAIN"),
        ("ai_vid_without_audio.mp4", "FAKE", "FAKE"),
        ("avengers ai vid 1.mp4", "FAKE", "REAL"),
        ("avengers ai vid 2.mp4", "FAKE", "UNCERTAIN"),
        ("crackers_withaud.mp4", "FAKE", "FAKE"),
        ("gemini ai video.mp4", "FAKE", "FAKE"),
        ("girl walkin video.mp4", "FAKE", "FAKE"),
        ("girl_speaking video gemini.mp4", "FAKE", "FAKE"),
        ("insta vid.mp4", "FAKE", "REAL"),
        ("real vid ai aud.mp4", "FAKE", "REAL"),
        ("south indian women speaking.mp4", "FAKE", "FAKE"),
        ("WhatsApp …6.15.45 PM.mp4", "REAL", "REAL"),
        ("WhatsApp …6.15.46 PM.mp4", "REAL", "REAL"),
        ("real_vid_to_test_withaudio.mp4", "REAL", "UNCERTAIN"),
        ("vid_with_noaudio.mp4", "REAL", "UNCERTAIN"),
    ]

    def _rows(self):
        return [{"filename": fn, "ground_truth": gt, "verdict": v,
                 "fusion_score": 0.5} for fn, gt, v in self.BASELINE_ROWS]

    def test_baseline_confusion_matches(self):
        r = evidence.verify_production_invariance(self._rows())
        assert r["computed"]["confusion"] == {"TP": 6, "TN": 4, "FP": 0, "FN": 5}
        assert r["matches_phase2b_baseline"] is True
        assert not r["metric_deltas"]

    def test_change_detected(self):
        rows = self._rows()
        rows[1]["verdict"] = "REAL"  # any change must flip the flag
        r = evidence.verify_production_invariance(rows)
        assert r["matches_phase2b_baseline"] is False