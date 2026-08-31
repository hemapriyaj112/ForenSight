"""Unit tests for Phase 3A reliability and domain-risk reporting layer.

These test PURE ANALYSIS helpers in pipeline/analysis/reliability.py.
They never instantiate a detector, model, fuser, or config — no weights
are touched, and no production verdict is ever modified.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pipeline.analysis import reliability  # noqa: E402
from pipeline.analysis.reliability import (  # noqa: E402
    CoverageStatus,
    EvidenceDirection,
    ReliabilityState,
    analyse_domain_risk,
    assign_reliability_state,
    build_reliability_report,
    classify_genconvit_coverage,
    compute_cross_modal_agreement,
    compute_evidence_direction,
    report_aasist_temporal,
)


# ---------------------------------------------------------------------------
# Shared fixture: minimal eval-result row (same shape as evaluate script)
# ---------------------------------------------------------------------------

def make_row(**over):
    row = {
        "filename": "clip.mp4", "ground_truth": "FAKE", "duration_seconds": 10.0,
        "fps": 24.0, "has_audio": True, "verdict": "FAKE", "fusion_score": 0.72,
        "video_score": 0.7, "audio_score": 0.1, "texture_score": 0.65,
        "frequency_score": 0.5, "provenance_score": 0.55,
        "provenance_override": False, "provenance_reason": None,
        "classifier_status": "INACTIVE", "classifier_score": None,
        "classifier_reason": None, "error": None, "bitrate_kbps": None,
        "resolution": None,
        "genconvit_status": "ACTIVE_MODEL", "genconvit_score": 0.48,
        "genconvit_predicted_label": "FAKE", "genconvit_frames_used": 10,
        "genconvit_frames_requested": 10, "genconvit_frames_sampled": 10,
        "genconvit_face_frames_found": 8, "genconvit_face_coverage": 0.8,
        "genconvit_average_face_confidence": 0.95,
        "genconvit_min_face_confidence": 0.9, "genconvit_max_face_confidence": 0.99,
        "genconvit_coverage_status": "GOOD_FACE_COVERAGE",
        "aasist_status": "ACTIVE_MODEL", "aasist_spoof_score": 0.99,
        "aasist_bonafide_score": 0.01, "aasist_predicted_label": "SPOOF",
        "aasist_window_count": 3, "aasist_window_scores": [0.99, 0.98, 0.97],
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


# fmt: off
# ---------------------------------------------------------------------------
# Task 13a — Missing modality
# ---------------------------------------------------------------------------

class TestMissingModality:
    def test_no_audio_flag(self):
        dr = analyse_domain_risk(
            has_audio=False, has_face=True, bitrate_kbps=2000,
            resolution="1280x720", filename="clip.mp4",
        )
        flag_names = [f["flag"] for f in dr["flags"]]
        assert "MISSING_AUDIO" in flag_names
        assert all(f["severity"] == "HIGH" for f in dr["flags"]
                   if f["flag"] == "MISSING_AUDIO")

    def test_no_face_flag(self):
        dr = analyse_domain_risk(
            has_audio=True, has_face=False, bitrate_kbps=2000,
            resolution="1280x720", filename="clip.mp4",
        )
        flag_names = [f["flag"] for f in dr["flags"]]
        assert "MISSING_FACE" in flag_names

    def test_no_audio_and_no_face_domain_risk(self):
        dr = analyse_domain_risk(has_audio=False, has_face=False,
                                 bitrate_kbps=2000, resolution="1280x720",
                                 filename="clip.mp4")
        assert dr["state"] == ReliabilityState.DOMAIN_RISK.value


# ---------------------------------------------------------------------------
# Task 13b — GenConViT coverage
# ---------------------------------------------------------------------------

class TestGenconvitCoverage:
    def test_good_coverage(self):
        c = classify_genconvit_coverage(
            face_frames_found=10, frames_requested=15, face_coverage=0.67,
            coverage_status="GOOD_FACE_COVERAGE",
        )
        assert c["status"] == CoverageStatus.GOOD.value

    def test_partial_coverage(self):
        c = classify_genconvit_coverage(
            face_frames_found=4, frames_requested=15, face_coverage=0.27,
            coverage_status="PARTIAL_FACE_COVERAGE",
        )
        assert c["status"] == CoverageStatus.PARTIAL.value

    def test_low_coverage(self):
        c = classify_genconvit_coverage(
            face_frames_found=1, frames_requested=15, face_coverage=0.07,
            coverage_status="LOW_FACE_COVERAGE",
        )
        assert c["status"] == CoverageStatus.LOW.value

    def test_no_face(self):
        c = classify_genconvit_coverage(
            face_frames_found=0, frames_requested=15, face_coverage=0.0,
            coverage_status="NO_FACE_DETECTED",
        )
        assert c["status"] == CoverageStatus.NO_FACE.value

    def test_numeric_derivation(self):
        # No coverage_status provided → derive from numeric values
        c = classify_genconvit_coverage(
            face_frames_found=15, frames_requested=15, face_coverage=1.0,
            coverage_status=None,
        )
        assert c["status"] == CoverageStatus.GOOD.value

        c2 = classify_genconvit_coverage(
            face_frames_found=0, frames_requested=15, face_coverage=0.0,
            coverage_status=None,
        )
        assert c2["status"] == CoverageStatus.NO_FACE.value


# ---------------------------------------------------------------------------
# Task 13c — AASIST patterns
# ---------------------------------------------------------------------------

class TestAasistTemporal:
    def test_persistent_high_spoof(self):
        t = report_aasist_temporal(
            window_scores=[0.99, 0.98, 0.97, 0.96], predicted_label="SPOOF",
        )
        assert t["segment_pattern"] == "persistent_high_spoof"
        assert t["window_count"] == 4
        assert t["high_spoof_window_count"] == 4
        assert t["high_spoof_fraction"] == 1.0

    def test_persistent_low_spoof(self):
        t = report_aasist_temporal(
            window_scores=[0.01, 0.02, 0.05, 0.03], predicted_label="BONAFIDE",
        )
        assert t["segment_pattern"] == "persistent_low_spoof"
        assert t["low_spoof_window_count"] == 4

    def test_mixed_or_localized(self):
        t = report_aasist_temporal(
            window_scores=[0.995, 0.01, 0.401], predicted_label=None,
        )
        assert t["segment_pattern"] == "mixed_or_localized"

    def test_no_windows(self):
        t = report_aasist_temporal(window_scores=[])
        assert t["segment_pattern"] == "no_windows"
        assert t["window_count"] == 0
        assert t["mean_spoof"] is None

    def test_timestamps(self):
        t = report_aasist_temporal(
            window_scores=[0.9, 0.8, 0.7], window_timestamps=[0.0, 4.037, 8.075],
        )
        assert t["timestamps"] == [0.0, 4.037, 8.075]


# ---------------------------------------------------------------------------
# Task 13d — Domain-risk indicators
# ---------------------------------------------------------------------------

class TestDomainRisk:
    def test_low_bitrate(self):
        dr = analyse_domain_risk(
            has_audio=True, has_face=True, bitrate_kbps=700,
            resolution="1280x720", filename="clip.mp4",
        )
        flags = {f["flag"]: f for f in dr["flags"]}
        assert "LOW_BITRATE" in flags
        assert flags["LOW_BITRATE"]["severity"] == "HIGH"
        assert flags["LOW_BITRATE"]["observed_value"] == 700

    def test_low_resolution(self):
        dr = analyse_domain_risk(
            has_audio=True, has_face=True, bitrate_kbps=2000,
            resolution="480x360", filename="clip.mp4",
        )
        flags = {f["flag"]: f for f in dr["flags"]}
        assert "LOW_RESOLUTION" in flags

    def test_whatsapp_relink(self):
        dr = analyse_domain_risk(
            has_audio=True, has_face=True, bitrate_kbps=2000,
            resolution="1280x720", filename="WhatsApp video.mp4",
        )
        flags = {f["flag"]: f for f in dr["flags"]}
        assert "WHATSAPP_OR_CHAT_RELINK" in flags
        assert "WHATSAPP_OR_CHAT_RELINK" in flags

    def test_neural_audio_heuristic_disagreement(self):
        dr = analyse_domain_risk(
            has_audio=True, has_face=True, bitrate_kbps=2000,
            resolution="1280x720", filename="clip.mp4",
            aasist_label="SPOOF", audio_heuristic_mean=0.1,
        )
        flags = {f["flag"]: f for f in dr["flags"]}
        assert "NEURAL_AUDIO_HEURISTIC_DISAGREEMENT" in flags

    def test_neural_modality_disagreement(self):
        dr = analyse_domain_risk(
            has_audio=True, has_face=True, bitrate_kbps=2000,
            resolution="1280x720", filename="clip.mp4",
            genconvit_label="REAL", aasist_label="SPOOF",
        )
        flags = {f["flag"]: f for f in dr["flags"]}
        assert "NEURAL_MODALITY_DISAGREEMENT" in flags


# ---------------------------------------------------------------------------
# Task 13e — Evidence direction
# ---------------------------------------------------------------------------

class TestEvidenceDirection:
    def test_fake_evidence(self):
        v = compute_evidence_direction("aasist", "SPOOF", 0.99, True)
        assert v.evidence_direction == EvidenceDirection.FAKE_EVIDENCE.value

    def test_real_evidence(self):
        v = compute_evidence_direction("genconvit", "REAL", 0.5, True)
        assert v.evidence_direction == EvidenceDirection.REAL_EVIDENCE.value

    def test_no_evidence(self):
        v = compute_evidence_direction("genconvit", None, None, False)
        assert v.evidence_direction == EvidenceDirection.NO_EVIDENCE.value
        assert v.reliability_state == ReliabilityState.UNAVAILABLE.value


# ---------------------------------------------------------------------------
# Task 13f — Reliability states
# ---------------------------------------------------------------------------

class TestReliabilityState:
    def test_high(self):
        s = assign_reliability_state(
            domain_risk_state=ReliabilityState.HIGH.value,
            genconvit_available=True,
            genconvit_coverage=CoverageStatus.GOOD.value,
            aasist_available=True,
            agreement_summary="MULTIMODAL_AGREEMENT",
        )
        assert s == ReliabilityState.HIGH.value

    def test_moderate(self):
        s = assign_reliability_state(
            domain_risk_state=ReliabilityState.HIGH.value,
            genconvit_available=True,
            genconvit_coverage=CoverageStatus.GOOD.value,
            aasist_available=True,
            texture_available=True,
            agreement_summary="PARTIAL_AGREEMENT",
        )
        assert s == ReliabilityState.MODERATE.value

    def test_low(self):
        s = assign_reliability_state(
            domain_risk_state=ReliabilityState.HIGH.value,
            genconvit_available=True,
            genconvit_coverage=CoverageStatus.LOW.value,
            agreement_summary="INSUFFICIENT_DATA",
        )
        assert s == ReliabilityState.LOW.value

    def test_domain_risk_caps(self):
        s = assign_reliability_state(
            domain_risk_state=ReliabilityState.DOMAIN_RISK.value,
            genconvit_available=True,
            genconvit_coverage=CoverageStatus.GOOD.value,
            aasist_available=True,
            agreement_summary="MULTIMODAL_AGREEMENT",
        )
        assert s == ReliabilityState.DOMAIN_RISK.value

    def test_unavailable(self):
        s = assign_reliability_state(
            domain_risk_state=ReliabilityState.HIGH.value,
        )
        assert s == ReliabilityState.UNAVAILABLE.value


# ---------------------------------------------------------------------------
# Task 13g — Evidence-vs-verdict separation
# ---------------------------------------------------------------------------

class TestEvidenceVerdictSeparation:
    def test_aasist_spoof_does_not_change_verdict(self):
        # Row has production verdict FAKE and AASIST SPOOF — both align here.
        row = make_row(verdict="FAKE", aasist_predicted_label="SPOOF")
        report = build_reliability_report(row)
        # Production verdict must be preserved exactly
        assert report.production_verdict == "FAKE"
        # AASIST evidence is FAKE_EVIDENCE, but that's separate from verdict
        assert report.modalities["aasist"]["evidence_direction"] != {}
        # No separate verdict key is introduced
        assert "final_verdict" not in report.modalities

    def test_aasist_spoof_on_real_verdict(self):
        # A REAL production verdict with AASIST=SPOOF — the verdict must
        # remain REAL; only the evidence is reported as spoof-oriented.
        row = make_row(
            verdict="REAL", fusion_score=0.3,
            ground_truth="REAL",
            aasist_predicted_label="SPOOF", aasist_spoof_score=0.99,
            aasist_window_scores=[0.99, 0.98, 0.97],
            genconvit_predicted_label="REAL", genconvit_score=0.4,
            genconvit_coverage_status="GOOD_FACE_COVERAGE",
        )
        report = build_reliability_report(row)
        assert report.production_verdict == "REAL"
        assert report.modalities["aasist"]["evidence_direction"]["evidence_direction"] \
            == EvidenceDirection.FAKE_EVIDENCE.value
        # Verdict is NOT changed by AASIST spoofing
        assert report.production_verdict == "REAL"

    def test_genconvit_fake_does_not_change_verdict(self):
        # A REAL production verdict with GenConViT=FAKE — verdict stays REAL.
        row = make_row(
            verdict="REAL", fusion_score=0.2,
            ground_truth="REAL",
            genconvit_predicted_label="FAKE", genconvit_score=0.8,
            genconvit_coverage_status="GOOD_FACE_COVERAGE",
            aasist_predicted_label="BONAFIDE", aasist_spoof_score=0.05,
        )
        report = build_reliability_report(row)
        assert report.production_verdict == "REAL"
        assert report.modalities["genconvit"]["evidence_direction"]["evidence_direction"] \
            == EvidenceDirection.FAKE_EVIDENCE.value
        assert report.production_verdict == "REAL"


# ---------------------------------------------------------------------------
# Task 8 — Per-video report structure / Task 9 — explanations
# ---------------------------------------------------------------------------

class TestReportStructure:
    def test_full_structure(self):
        report = build_reliability_report(make_row())
        d = report.to_dict()
        assert d["filename"] == "clip.mp4"
        assert d["production_verdict"] == "FAKE"
        assert "domain_risk" in d
        assert "modalities" in d
        assert "agreement" in d
        assert "evidence_summary" in d
        assert "explanations" in d
        assert "genconvit" in d["modalities"]
        assert "aasist" in d["modalities"]

    def test_explanations_do_not_claim_fakeness(self):
        row = make_row(
            verdict="REAL", aasist_predicted_label="SPOOF",
            aasist_spoof_score=0.99,
        )
        report = build_reliability_report(row)
        for exp in report.explanations:
            assert "video is fake" not in exp.lower()
            assert "video is real" not in exp.lower()

    def test_bonafide_not_real_proof(self):
        row = make_row(
            verdict="FAKE", ground_truth="REAL",
            aasist_predicted_label="BONAFIDE", aasist_spoof_score=0.01,
        )
        report = build_reliability_report(row)
        assert report.production_verdict == "FAKE"
        assert report.modalities["aasist"]["evidence_direction"]["evidence_direction"] \
            == EvidenceDirection.REAL_EVIDENCE.value


# ---------------------------------------------------------------------------
# Task 5 — Cross-modal agreement
# ---------------------------------------------------------------------------

class TestCrossModalAgreement:
    def test_agreement(self):
        a = compute_cross_modal_agreement(
            genconvit_label="FAKE", genconvit_available=True,
            aasist_label="SPOOF", aasist_available=True,
            audio_heuristic_mean=0.1, audio_heuristic_available=True,
        )
        assert a["comparisons"].get("genconvit_vs_aasist") == "AGREE"
        assert a["summary"] in (
            "MULTIMODAL_AGREEMENT", "PARTIAL_AGREEMENT",
            "MULTIMODAL_DISAGREEMENT", "MIXED_AGREEMENT",
        )

    def test_disagreement(self):
        a = compute_cross_modal_agreement(
            genconvit_label="REAL", genconvit_available=True,
            aasist_label="SPOOF", aasist_available=True,
            audio_heuristic_mean=0.1, audio_heuristic_available=True,
        )
        assert a["summary"] == "MULTIMODAL_DISAGREEMENT"
        assert a["neural_agreement"] == "NEURAL_DISAGREEMENT"
        assert a["neural_vs_heuristic_disagreement"] is True

    def test_only_available_modalities(self):
        # Only AASIST available → no genconvit comparison should be present
        a = compute_cross_modal_agreement(
            genconvit_available=False,
            aasist_label="SPOOF", aasist_available=True,
            audio_heuristic_mean=None, audio_heuristic_available=False,
        )
        assert "genconvit_vs_aasist" not in a["comparisons"]
        assert a["summary"] == "INSUFFICIENT_DATA"
