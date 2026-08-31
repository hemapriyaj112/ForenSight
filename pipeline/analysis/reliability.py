"""pipeline/analysis/reliability.py

Phase 3A — Evidence reliability and domain-risk reporting layer.

INSTRUMENTATION / REPORTING ONLY.

* Consumes already-produced detector outputs (never runs detectors).
* Produces reliability states, domain-risk flags, evidence-vote
  representations, cross-modal agreement metadata, and human-readable
  forensic explanations.
* Does NOT modify the production verdict, fusion formula, thresholds,
  or any detector configuration.
* GenConViT remains evaluation-only.  AASIST remains evaluation-only.

Three separate concepts are maintained throughout:
  1. DETECTOR OUTPUT — what did the detector say?
  2. EVIDENCE RELIABILITY — how trustworthy is that output in this context?
  3. FINAL FORENSIC VERDICT — what does the complete system conclude?

These must never be conflated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Reliability states  (NOT prediction classes — these describe evidence quality)
# ---------------------------------------------------------------------------

class ReliabilityState(str, Enum):
    """Evidence reliability states proposed in Phase 2C, Section 12."""
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    DOMAIN_RISK = "DOMAIN-RISK"
    UNAVAILABLE = "UNAVAILABLE"


# ---------------------------------------------------------------------------
# Evidence direction (NOT a verdict — describes what the evidence points toward)
# ---------------------------------------------------------------------------

class EvidenceDirection(str, Enum):
    """Direction a modality's evidence points in, independent of final verdict."""
    FAKE_EVIDENCE = "FAKE_EVIDENCE"
    REAL_EVIDENCE = "REAL_EVIDENCE"
    NO_EVIDENCE = "NO_EVIDENCE"


# ---------------------------------------------------------------------------
# Coverage status for GenConViT (from Phase 2A.1)
# ---------------------------------------------------------------------------

class CoverageStatus(str, Enum):
    """GenConViT face coverage classification from Phase 2A.1."""
    GOOD = "GOOD"
    PARTIAL = "PARTIAL"
    LOW = "LOW"
    NO_FACE = "NO_FACE"


# ---------------------------------------------------------------------------
# Domain-risk flag severity
# ---------------------------------------------------------------------------

class FlagSeverity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ---------------------------------------------------------------------------
# Constants — thresholds and boundaries
# ---------------------------------------------------------------------------
# These are derived from Phase 2C observations on 15 videos and Phase 2A.1
# coverage analysis.  They are DOCUMENTED as provisional/heuristic and
# explicitly NOT tuned on the corpus.

# Bitrate thresholds (kbps)
LOW_BITRATE_THRESHOLD = 1200    # Phase 2C: videos ≤ ~1200 kbps flagged

# Resolution thresholds
LOW_RESOLUTION_WIDTH = 720      # < 720p width flagged as low-res

# GenConViT coverage thresholds (fraction of requested frames with faces)
GOOD_FACE_COVERAGE_MIN = 0.5    # >= 50% frames with faces
PARTIAL_FACE_COVERAGE_MIN = 0.2 # >= 20% frames with faces
# < 20% but > 0% = LOW, 0% = NO_FACE

# AASIST pattern classification (from evidence.py aasist_segment_pattern)
HIGH_SPOOF_THRESHOLD = 0.8      # window score >= 0.8 is "high"
LOW_SPOOF_THRESHOLD = 0.2       # window score < 0.2 is "low"

# Cross-modal disagreement thresholds
NEURAL_HEURISTIC_DISAGREEMENT_HEUR_MAX = 0.4  # heuristic mean <= this while AASIST=SPOOF


# ---------------------------------------------------------------------------
# Domain-risk flags
# ---------------------------------------------------------------------------

@dataclass
class DomainRiskFlag:
    """A single domain-risk indicator."""
    flag: str
    severity: str
    observed_value: Any
    reason: str
    evidence_source: str

    def to_dict(self) -> dict:
        return {
            "flag": self.flag,
            "severity": self.severity,
            "observed_value": self.observed_value,
            "reason": self.reason,
            "evidence_source": self.evidence_source,
        }


# ---------------------------------------------------------------------------
# Evidence vote for a single modality
# ---------------------------------------------------------------------------

@dataclass
class EvidenceVote:
    """Reporting-only representation for one modality's evidence direction."""
    modality: str
    evidence_direction: str
    evidence_strength: str
    reliability_state: str
    score: Optional[float] = None
    detail: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "modality": self.modality,
            "evidence_direction": self.evidence_direction,
            "evidence_strength": self.evidence_strength,
            "reliability_state": self.reliability_state,
        }
        if self.score is not None:
            d["score"] = self.score
        if self.detail is not None:
            d["detail"] = self.detail
        return d


# ---------------------------------------------------------------------------
# Cross-modal agreement result
# ---------------------------------------------------------------------------

@dataclass
class AgreementResult:
    """Structured cross-modal agreement/disagreement metadata."""
    comparisons: dict = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict:
        return {"comparisons": self.comparisons, "summary": self.summary}


# ---------------------------------------------------------------------------
# Per-video reliability report
# ---------------------------------------------------------------------------

@dataclass
class ReliabilityReport:
    """Structured reliability report for a single video."""
    filename: str
    production_verdict: str
    production_fusion_score: Optional[float] = None
    ground_truth: Optional[str] = None
    domain_risk: dict = field(default_factory=dict)
    modalities: dict = field(default_factory=dict)
    agreement: dict = field(default_factory=dict)
    evidence_summary: list = field(default_factory=list)
    explanations: list = field(default_factory=list)
    overall_reliability: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "ground_truth": self.ground_truth,
            "production_verdict": self.production_verdict,
            "production_fusion_score": self.production_fusion_score,
            "domain_risk": self.domain_risk,
            "modalities": self.modalities,
            "agreement": self.agreement,
            "evidence_summary": self.evidence_summary,
            "explanations": self.explanations,
            "overall_reliability": self.overall_reliability,
        }


# ===========================================================================
# DOMAIN-RISK ANALYSIS  (Task 3)
# ===========================================================================

def analyse_domain_risk(
    has_audio: bool,
    has_face: bool,
    bitrate_kbps: Optional[float] = None,
    resolution: Optional[str] = None,
    filename: str = "",
    audio_heuristic_mean: Optional[float] = None,
    aasist_label: Optional[str] = None,
    aasist_spoof_score: Optional[float] = None,
    genconvit_label: Optional[str] = None,
    aasist_pattern: Optional[str] = None,
    face_coverage: Optional[float] = None,
    provenance_override: bool = False,
) -> dict:
    """Produce structured domain-risk flags for one video.

    Each flag represents a *domain-risk indicator*, NOT proof of
    manipulation.  The flags describe media conditions that historically
    correlate with detector unreliability.

    Classification of each rule:
      validated  — grounded in domain knowledge (e.g. missing modality)
      observed   — seen repeatedly in the 15-video corpus
      provisional — extrapolated from observations, may need revision

    Returns dict with keys: state, flags, flag_count.
    """
    flags: list[DomainRiskFlag] = []

    # --- LOW_BITRATE (observed heuristic) ---
    if bitrate_kbps is not None and bitrate_kbps <= LOW_BITRATE_THRESHOLD:
        flags.append(DomainRiskFlag(
            flag="LOW_BITRATE",
            severity="HIGH",
            observed_value=bitrate_kbps,
            reason=(f"Input bitrate ({bitrate_kbps:.0f} kbps) is within the "
                    f"previously observed low-bitrate risk regime "
                    f"(threshold: {LOW_BITRATE_THRESHOLD} kbps)."),
            evidence_source="ffprobe",
        ))

    # --- LOW_RESOLUTION (observed heuristic) ---
    if resolution:
        try:
            w = int(resolution.split("x")[0])
            if w < LOW_RESOLUTION_WIDTH:
                flags.append(DomainRiskFlag(
                    flag="LOW_RESOLUTION",
                    severity="MEDIUM",
                    observed_value=resolution,
                    reason=(f"Resolution ({resolution}) is below 720p; "
                            f"downscaled/screen-recorded media may lose "
                            f"detection-relevant texture information."),
                    evidence_source="ffprobe",
                ))
        except (ValueError, IndexError):
            pass

    # --- WHATSAPP_OR_CHAT_RELINK (observed heuristic) ---
    fn_lower = (filename or "").lower()
    if "whatsapp" in fn_lower or "chat" in fn_lower:
        flags.append(DomainRiskFlag(
            flag="WHATSAPP_OR_CHAT_RELINK",
            severity="HIGH",
            observed_value=filename,
            reason=("Filename suggests WhatsApp/chat relink. Such media "
                    "undergoes platform transcoding (AAC re-encode, "
                    "resolution/bitrate reduction) that may create "
                    "artefacts correlated with false detector responses."),
            evidence_source="filename_pattern",
        ))

    # --- MISSING_AUDIO (validated rule) ---
    if not has_audio:
        flags.append(DomainRiskFlag(
            flag="MISSING_AUDIO",
            severity="HIGH",
            observed_value=False,
            reason=("No audio stream in container; audio authenticity "
                    "could not be assessed by any detector."),
            evidence_source="container_metadata",
        ))

    # --- MISSING_FACE (validated rule) ---
    if not has_face:
        flags.append(DomainRiskFlag(
            flag="MISSING_FACE",
            severity="HIGH",
            observed_value=False,
            reason=("No usable face frames detected; face-dependent "
                    "neural detectors (GenConViT) cannot produce "
                    "a visual authenticity reading."),
            evidence_source="face_detector",
        ))

    # --- LOW_FACE_COVERAGE (observed heuristic) ---
    if has_face and face_coverage is not None:
        if face_coverage < PARTIAL_FACE_COVERAGE_MIN:
            flags.append(DomainRiskFlag(
                "LOW_FACE_COVERAGE", "MEDIUM",
                face_coverage,
                (f"Face coverage is {face_coverage:.1%} of sampled frames; "
                 f"GenConViT's visual reading is based on limited data."),
                "face_detector",
            ))

    # --- NEURAL_AUDIO_HEURISTIC_DISAGREEMENT (observed) ---
    if (aasist_label == "SPOOF"
            and audio_heuristic_mean is not None
            and audio_heuristic_mean <= NEURAL_HEURISTIC_DISAGREEMENT_HEUR_MAX):
        flags.append(DomainRiskFlag(
            "NEURAL_AUDIO_HEURISTIC_DISAGREEMENT", "MEDIUM",
            {"aasist_label": aasist_label, "heuristic_mean": audio_heuristic_mean},
            ("AASIST reads SPOOF but the production audio heuristic reads "
             "natural (mean {:.2f}). The two audio detectors disagree; "
             "this pattern is consistent with either genuine synthetic audio "
             "or codec/telephony artefacts — further investigation needed."),
            "cross_detector",
        ))

    # --- NEURAL_MODALITY_DISAGREEMENT (observed) ---
    if genconvit_label in ("FAKE", "REAL") and aasist_label in ("SPOOF", "BONAFIDE"):
        gc_video = "FAKE" if genconvit_label == "FAKE" else "REAL"
        aa_video = "FAKE" if aasist_label == "SPOOF" else "REAL"
        if gc_video != aa_video:
            flags.append(DomainRiskFlag(
                "NEURAL_MODALITY_DISAGREEMENT", "MEDIUM",
                {"genconvit": genconvit_label, "aasist": aasist_label},
                (f"GenConViT reads {genconvit_label} while AASIST reads "
                 f"{aasist_label}; the two neural detectors disagree on the "
                 f"direction of manipulation evidence."),
                "cross_detector",
            ))

    # --- PERSISTENT_AASIST_SPOOF_ON_RISKY_MEDIA (observed) ---
    if (aasist_label == "SPOOF"
            and aasist_pattern == "persistent_high_spoof"
            and not has_face):
        flags.append(DomainRiskFlag(
            "PERSISTENT_AASIST_SPOOF_ON_RISKY_MEDIA", "HIGH",
            {"aasist_pattern": aasist_pattern, "has_face": has_face},
            ("AASIST shows persistent high spoof across all windows on a "
             "video with no detectable faces; on faceless media, this "
             "pattern has historically produced false positives due to "
             "codec/telephony artefacts."),
            "cross_detector_observed",
        ))
    elif (aasist_label == "SPOOF"
            and aasist_pattern == "persistent_high_spoof"
            and bitrate_kbps is not None and bitrate_kbps <= LOW_BITRATE_THRESHOLD):
        flags.append(DomainRiskFlag(
            "PERSISTENT_AASIST_SPOOF_ON_RISKY_MEDIA", "MEDIUM",
            {"aasist_pattern": aasist_pattern, "bitrate_kbps": bitrate_kbps},
            ("AASIST shows persistent high spoof on low-bitrate media; "
             "compression artefacts may inflate spoof readings."),
            "cross_detector_observed",
        ))

    # --- Determine overall domain-risk state ---
    high_severity = sum(1 for f in flags if f.severity == "HIGH")
    any_flags = len(flags) > 0

    if high_severity >= 2:
        state = ReliabilityState.DOMAIN_RISK
    elif high_severity == 1:
        state = ReliabilityState.DOMAIN_RISK
    elif any_flags:
        state = ReliabilityState.MODERATE  # flags present but all MEDIUM/LOW
    else:
        state = ReliabilityState.HIGH

    return {
        "state": state.value,
        "flags": [f.to_dict() for f in flags],
        "flag_count": len(flags),
        "high_severity_count": high_severity,
    }


# ===========================================================================
# GENCONVIT COVERAGE RELIABILITY  (Task 7)
# ===========================================================================

def classify_genconvit_coverage(
    face_frames_found: Optional[int] = None,
    frames_requested: Optional[int] = None,
    face_coverage: Optional[float] = None,
    coverage_status: Optional[str] = None,
) -> dict:
    """Map GenConViT input coverage into reliability-relevant status.

    Uses the existing Phase 2A.1 classification without introducing
    new thresholds.  The coverage_status field from the detector is
    preferred when available; otherwise derives from numeric values.

    Returns dict with keys: status, face_coverage, frames_used,
    frames_requested, average_face_confidence, reliability_note.
    """
    # Prefer the detector's own status when available
    if coverage_status:
        status_map = {
            "GOOD_FACE_COVERAGE": CoverageStatus.GOOD.value,
            "PARTIAL_FACE_COVERAGE": CoverageStatus.PARTIAL.value,
            "LOW_FACE_COVERAGE": CoverageStatus.LOW.value,
            "NO_FACE_DETECTED": CoverageStatus.NO_FACE.value,
        }
        status = status_map.get(coverage_status, CoverageStatus.LOW.value)
    elif face_frames_found is not None and frames_requested:
        if face_frames_found == 0:
            status = CoverageStatus.NO_FACE.value
        elif face_coverage is not None:
            if face_coverage >= GOOD_FACE_COVERAGE_MIN:
                status = CoverageStatus.GOOD.value
            elif face_coverage >= PARTIAL_FACE_COVERAGE_MIN:
                status = CoverageStatus.PARTIAL.value
            else:
                status = CoverageStatus.LOW.value
        else:
            status = CoverageStatus.LOW.value
    else:
        status = CoverageStatus.NO_FACE.value

    # Reliability note
    notes = {
        CoverageStatus.GOOD.value: (
            "GenConViT had adequate visual coverage; its output can be "
            "interpreted with moderate confidence."
        ),
        CoverageStatus.PARTIAL.value: (
            "GenConViT had partial visual coverage; its output should "
            "be interpreted with some caution."
        ),
        CoverageStatus.LOW.value: (
            "GenConViT had limited visual coverage; its output should "
            "be interpreted cautiously."
        ),
        CoverageStatus.NO_FACE.value: (
            "GenConViT could not detect any usable face frames; its "
            "visual deepfake detector produced no evidence."
        ),
    }

    return {
        "status": status,
        "face_coverage": face_coverage,
        "frames_used": face_frames_found,
        "frames_requested": frames_requested,
        "reliability_note": notes.get(status, ""),
    }


# ===========================================================================
# EVIDENCE DIRECTION  (Task 4)
# ===========================================================================

def compute_evidence_direction(
    modality: str,
    label: Optional[str] = None,
    score: Optional[float] = None,
    available: bool = True,
) -> EvidenceVote:
    """Compute the evidence direction and strength for a single modality.

    This is a REPORTING representation only — it does NOT constitute a
    verdict or modify any decision.

    Args:
        modality: "genconvit", "aasist", "audio_heuristic", "texture", etc.
        label: detector's predicted label (FAKE/REAL/SPOOF/BONAFIDE/etc.)
        score: detector's confidence/suspicion score
        available: whether the detector produced any output

    Returns:
        EvidenceVote with direction, strength, reliability placeholder.
    """
    if not available or label is None:
        return EvidenceVote(
            modality=modality,
            evidence_direction=EvidenceDirection.NO_EVIDENCE.value,
            evidence_strength="N/A",
            reliability_state=ReliabilityState.UNAVAILABLE.value,
            score=score,
            detail=f"{modality} did not produce evidence (unavailable or no output).",
        )

    # Map label to evidence direction
    label_upper = label.upper()
    if label_upper in ("FAKE", "SPOOF"):
        direction = EvidenceDirection.FAKE_EVIDENCE.value
    elif label_upper in ("REAL", "BONAFIDE"):
        direction = EvidenceDirection.REAL_EVIDENCE.value
    else:
        direction = EvidenceDirection.NO_EVIDENCE.value

    # Strength from score
    if score is not None:
        if abs(score - 0.5) >= 0.3:
            strength = "HIGH"
        elif abs(score - 0.5) >= 0.15:
            strength = "MODERATE"
        else:
            strength = "LOW"
    else:
        strength = "UNKNOWN"

    return EvidenceVote(
        modality=modality,
        evidence_direction=direction,
        evidence_strength=strength,
        reliability_state="PENDING",  # filled in later by reliability assignment
        score=score,
        detail=None,
    )


# ===========================================================================
# AASIST TEMPORAL EVIDENCE REPORTING  (Task 6)
# ===========================================================================

def report_aasist_temporal(
    window_scores: list[float],
    spoof_score: Optional[float] = None,
    predicted_label: Optional[str] = None,
    window_timestamps: Optional[list[float]] = None,
) -> dict:
    """Expose existing AASIST segment-level temporal evidence.

    Does NOT change AASIST aggregation or add a new decision threshold.
    The purpose is to surface the existing temporal evidence structure.

    Returns dict with: mean_spoof, min_spoof, max_spoof, window_count,
    segment_pattern, high_spoof_window_count, low_spoof_window_count,
    high_spoof_fraction, timestamps, detail.
    """
    if not window_scores:
        return {
            "mean_spoof": None,
            "min_spoof": None,
            "max_spoof": None,
            "window_count": 0,
            "segment_pattern": "no_windows",
            "high_spoof_window_count": 0,
            "low_spoof_window_count": 0,
            "high_spoof_fraction": None,
            "timestamps": [],
            "detail": "No AASIST window scores available.",
        }

    xs = [float(s) for s in window_scores]
    n = len(xs)
    mean_s = sum(xs) / n
    high_count = sum(1 for s in xs if s >= HIGH_SPOOF_THRESHOLD)
    low_count = sum(1 for s in xs if s < LOW_SPOOF_THRESHOLD)
    high_frac = high_count / n

    # Pattern classification (mirrors evidence.aasist_segment_pattern)
    if high_frac >= 0.8 and mean_s >= 0.8:
        pattern = "persistent_high_spoof"
    elif (low_count / n) >= 0.8 and mean_s <= 0.2:
        pattern = "persistent_low_spoof"
    else:
        pattern = "mixed_or_localized"

    # Timestamps
    if window_timestamps is None:
        window_timestamps = [round(i * 4.0375, 3) for i in range(n)]

    # Build human-readable detail
    label_str = predicted_label or "unknown"
    detail = (
        f"AASIST produced {label_str}-oriented audio evidence across "
        f"{n} analyzed windows "
        f"(mean spoof score: {mean_s:.4f}, range: {min(xs):.4f}–{max(xs):.4f}). "
        f"Segment pattern: {pattern}. "
        f"{high_count}/{n} windows scored >= {HIGH_SPOOF_THRESHOLD} "
        f"(high spoof), {low_count}/{n} scored < {LOW_SPOOF_THRESHOLD} (low spoof)."
    )
    if pattern == "persistent_high_spoof":
        detail += (
            " This is audio-level evidence; persistent spoof detection "
            "does NOT by itself establish video-level manipulation."
        )

    return {
        "mean_spoof": round(mean_s, 6),
        "min_spoof": round(min(xs), 6),
        "max_spoof": round(max(xs), 6),
        "window_count": n,
        "segment_pattern": pattern,
        "high_spoof_window_count": high_count,
        "low_spoof_window_count": low_count,
        "high_spoof_fraction": round(high_frac, 4),
        "timestamps": window_timestamps,
        "detail": detail,
    }


# ===========================================================================
# CROSS-MODAL AGREEMENT  (Task 5)
# ===========================================================================

def compute_cross_modal_agreement(
    genconvit_label: Optional[str] = None,
    genconvit_score: Optional[float] = None,
    genconvit_available: bool = False,
    aasist_label: Optional[str] = None,
    aasist_spoof_score: Optional[float] = None,
    aasist_available: bool = False,
    audio_heuristic_mean: Optional[float] = None,
    audio_heuristic_available: bool = False,
    texture_score: Optional[float] = None,
    provenance_override: bool = False,
) -> dict:
    """Calculate cross-modal agreement/disagreement metadata.

    Only includes comparisons when both modalities are available.
    Does NOT reduce to a single score — prefer explicit structured info.

    Returns dict with comparisons, neural_agreement, summary.
    """
    comparisons = {}

    # Helper to map labels to FAKE/REAL for comparison
    def _to_fake_real(detector, label):
        if label is None:
            return None
        upper = label.upper()
        if upper in ("FAKE", "SPOOF"):
            return "FAKE"
        if upper in ("REAL", "BONAFIDE"):
            return "REAL"
        return None

    gc_fr = _to_fake_real("genconvit", genconvit_label)
    aa_fr = _to_fake_real("aasist", aasist_label)

    # GenConViT vs AASIST
    if genconvit_available and aasist_available and gc_fr and aa_fr:
        comparisons["genconvit_vs_aasist"] = "AGREE" if gc_fr == aa_fr else "DISAGREE"

    # AASIST vs audio heuristic
    if (aasist_available and audio_heuristic_available
            and aa_fr is not None and audio_heuristic_mean is not None):
        heur_fr = "FAKE" if audio_heuristic_mean >= 0.6 else (
            "REAL" if audio_heuristic_mean <= 0.4 else "UNCERTAIN"
        )
        if heur_fr == "UNCERTAIN":
            comparisons["aasist_vs_audio_heuristic"] = "NEITHER"
        elif aa_fr == heur_fr:
            comparisons["aasist_vs_audio_heuristic"] = "AGREE"
        else:
            comparisons["aasist_vs_audio_heuristic"] = "DISAGREE"

    # GenConViT vs texture
    if genconvit_available and gc_fr is not None and texture_score is not None:
        tex_fr = "FAKE" if texture_score >= 0.6 else (
            "REAL" if texture_score <= 0.4 else "UNCERTAIN"
        )
        if tex_fr != "UNCERTAIN":
            comparisons["genconvit_vs_texture"] = "AGREE" if gc_fr == tex_fr else "DISAGREE"

    # Provenance vs others
    if provenance_override:
        if gc_fr == "FAKE":
            comparisons["provenance_vs_genconvit"] = "AGREE"
        elif gc_fr == "REAL":
            comparisons["provenance_vs_genconvit"] = "DISAGREE"

    # Neural agreement (both neural detectors available and same direction)
    if genconvit_available and aasist_available and gc_fr and aa_fr:
        neural_agreement = "NEURAL_AGREEMENT" if gc_fr == aa_fr else "NEURAL_DISAGREEMENT"
    else:
        neural_agreement = "INSUFFICIENT_MODALITIES"

    # Neural vs heuristic disagreement
    neural_heuristic = False
    if (aasist_available and audio_heuristic_available
            and aa_fr == "FAKE"
            and audio_heuristic_mean is not None
            and audio_heuristic_mean <= 0.4):
        neural_heuristic = True

    # Summary
    agree_count = sum(1 for v in comparisons.values() if v == "AGREE")
    disagree_count = sum(1 for v in comparisons.values() if v == "DISAGREE")

    if disagree_count > 0 and agree_count > 0:
        summary = "MIXED_AGREEMENT"
    elif disagree_count > 0:
        summary = "MULTIMODAL_DISAGREEMENT"
    elif agree_count >= 2:
        summary = "MULTIMODAL_AGREEMENT"
    elif agree_count == 1:
        summary = "PARTIAL_AGREEMENT"
    else:
        summary = "INSUFFICIENT_DATA"

    return {
        "comparisons": comparisons,
        "neural_agreement": neural_agreement,
        "neural_vs_heuristic_disagreement": neural_heuristic,
        "agree_count": agree_count,
        "disagree_count": disagree_count,
        "summary": summary,
    }


# ===========================================================================
# RELIABILITY STATE ASSIGNMENT  (Task 2)
# ===========================================================================

def assign_reliability_state(
    domain_risk_state: str,
    genconvit_available: bool = False,
    genconvit_coverage: Optional[str] = None,
    aasist_available: bool = False,
    aasist_pattern: Optional[str] = None,
    texture_available: bool = False,
    provenance_available: bool = False,
    agreement_summary: Optional[str] = None,
) -> str:
    """Assign an overall evidence reliability state.

    Decision rules (ordered by priority):

    1. If no detector produced usable evidence → UNAVAILABLE
    2. If domain-risk state is DOMAIN-RISK → DOMAIN-RISK (caps both FAKE
       and REAL confidence)
    3. If two or more independent modalities agree and no domain-risk → HIGH
    4. If exactly one modality or mixed agreement → MODERATE
    5. If only weak/unreliable evidence → LOW

    These rules are derived from Phase 2C findings but avoid hard-coding
    arbitrary sample-specific rules.  The state is a REPORTING label, not
    a prediction class.

    Returns a string value of ReliabilityState.
    """
    # Count available evidence sources
    available_count = sum([
        genconvit_available and genconvit_coverage != CoverageStatus.NO_FACE.value,
        aasist_available and aasist_pattern != "no_windows",
        texture_available,
        provenance_available,
    ])

    # Rule 1: nothing available
    if available_count == 0:
        return ReliabilityState.UNAVAILABLE.value

    # Rule 2: domain risk caps reliability
    if domain_risk_state == ReliabilityState.DOMAIN_RISK.value:
        return ReliabilityState.DOMAIN_RISK.value

    # Rule 3: strong multi-source corroboration
    if agreement_summary == "MULTIMODAL_AGREEMENT" and available_count >= 2:
        return ReliabilityState.HIGH.value

    # Rule 4: partial or mixed
    if available_count >= 2 and agreement_summary in ("PARTIAL_AGREEMENT", "MIXED_AGREEMENT"):
        return ReliabilityState.MODERATE.value

    if available_count >= 2 and agreement_summary == "MULTIMODAL_DISAGREEMENT":
        return ReliabilityState.MODERATE.value

    # Single source
    if available_count == 1:
        # Good coverage single source → MODERATE
        if (genconvit_available and genconvit_coverage == CoverageStatus.GOOD.value):
            return ReliabilityState.MODERATE.value
        return ReliabilityState.LOW.value

    # Default
    return ReliabilityState.LOW.value


# ===========================================================================
# PER-VIDEO RELIABILITY REPORT  (Task 8)
# ===========================================================================

def build_reliability_report(row: dict) -> ReliabilityReport:
    """Build a complete per-video reliability report from an eval result row.

    Consumes the same row shape produced by evaluate_testing_videos.py.
    Produces metadata/reporting only — never modifies the verdict.
    """
    # Extract fields
    filename = row.get("filename", "")
    ground_truth = row.get("ground_truth")
    verdict = row.get("verdict", "UNKNOWN")
    fusion_score = row.get("fusion_score")
    has_audio = bool(row.get("has_audio", True))
    bitrate_kbps = row.get("bitrate_kbps")
    resolution = row.get("resolution")

    # GenConViT fields
    gc_status = row.get("genconvit_status")
    gc_label = row.get("genconvit_predicted_label")
    gc_score = row.get("genconvit_score")
    gc_available = gc_status == "ACTIVE_MODEL" and gc_label is not None
    gc_coverage_status = row.get("genconvit_coverage_status") or row.get("genconvit_face_quality")
    gc_face_coverage = row.get("genconvit_face_coverage")
    gc_frames_found = row.get("genconvit_face_frames_found")
    gc_frames_requested = row.get("genconvit_frames_requested")
    gc_avg_conf = row.get("genconvit_average_face_confidence")
    gc_min_conf = row.get("genconvit_min_face_confidence")
    gc_max_conf = row.get("genconvit_max_face_confidence")

    # AASIST fields
    aa_status = row.get("aasist_status")
    aa_label = row.get("aasist_predicted_label")
    aa_spoof = row.get("aasist_spoof_score")
    aa_windows = row.get("aasist_window_scores") or []
    aa_window_count = row.get("aasist_window_count", 0)
    aa_available = aa_status == "ACTIVE_MODEL" and aa_label is not None

    # Audio heuristic
    audio_heur_mean = None
    heur_segs = row.get("audio_segments") or []
    if heur_segs:
        scores = [float(s.get("fake_prob", 0)) for s in heur_segs]
        audio_heur_mean = sum(scores) / len(scores) if scores else None

    texture_score = row.get("texture_score")
    provenance_override = bool(row.get("provenance_override", False))

    # Has face?
    has_face = gc_available and gc_frames_found is not None and gc_frames_found > 0

    # --- Domain-risk analysis ---
    domain_risk = analyse_domain_risk(
        has_audio=has_audio,
        has_face=has_face,
        bitrate_kbps=bitrate_kbps,
        resolution=resolution,
        filename=filename,
        audio_heuristic_mean=audio_heur_mean,
        aasist_label=aa_label,
        aasist_spoof_score=aa_spoof,
        genconvit_label=gc_label,
        aasist_pattern=None,  # will compute below
        face_coverage=gc_face_coverage,
        provenance_override=provenance_override,
    )

    # --- GenConViT coverage ---
    gc_coverage_info = classify_genconvit_coverage(
        face_frames_found=gc_frames_found,
        frames_requested=gc_frames_requested,
        face_coverage=gc_face_coverage,
        coverage_status=gc_coverage_status,
    )

    # --- AASIST temporal ---
    aa_temporal = report_aasist_temporal(
        window_scores=aa_windows,
        spoof_score=aa_spoof,
        predicted_label=aa_label,
    )

    # --- Evidence directions ---
    gc_direction = compute_evidence_direction(
        "genconvit", gc_label, gc_score, gc_available,
    )
    aa_direction = compute_evidence_direction(
        "aasist", aa_label, aa_spoof, aa_available,
    )

    # --- Cross-modal agreement ---
    agreement = compute_cross_modal_agreement(
        genconvit_label=gc_label,
        genconvit_score=gc_score,
        genconvit_available=gc_available,
        aasist_label=aa_label,
        aasist_spoof_score=aa_spoof,
        aasist_available=aa_available,
        audio_heuristic_mean=audio_heur_mean,
        audio_heuristic_available=(audio_heur_mean is not None),
        texture_score=texture_score,
        provenance_override=provenance_override,
    )

    # --- Reliability state assignment ---
    overall_reliability = assign_reliability_state(
        domain_risk_state=domain_risk["state"],
        genconvit_available=gc_available,
        genconvit_coverage=gc_coverage_info["status"],
        aasist_available=aa_available,
        aasist_pattern=aa_temporal["segment_pattern"],
        texture_available=(texture_score is not None),
        provenance_available=provenance_override,
        agreement_summary=agreement["summary"],
    )

    # Assign per-modality reliability
    gc_reliability = _assign_modality_reliability(
        "genconvit", gc_available, gc_coverage_info["status"],
        domain_risk["state"], agreement,
    )
    aa_reliability = _assign_modality_reliability(
        "aasist", aa_available, None,
        domain_risk["state"], agreement,
    )
    gc_direction.reliability_state = gc_reliability
    aa_direction.reliability_state = aa_reliability

    # --- Modality details ---
    modalities = {
        "genconvit": {
            "available": gc_available,
            "evidence_direction": gc_direction.to_dict(),
            "score": gc_score,
            "coverage": gc_coverage_info,
            "reliability": gc_reliability,
        },
        "aasist": {
            "available": aa_available,
            "evidence_direction": aa_direction.to_dict(),
            "score": aa_spoof,
            "segment_pattern": aa_temporal,
            "reliability": aa_reliability,
        },
    }

    # --- Human-readable explanations ---
    explanations = _generate_explanations(
        gc_available=gc_available,
        gc_label=gc_label,
        gc_coverage=gc_coverage_info,
        aa_available=aa_available,
        aa_label=aa_label,
        aa_temporal=aa_temporal,
        has_audio=has_audio,
        has_face=has_face,
        domain_risk_flags=domain_risk["flags"],
        agreement=agreement,
        overall_reliability=overall_reliability,
    )

    # --- Evidence summary (compact) ---
    evidence_summary = []
    if gc_available:
        evidence_summary.append(
            f"GenConViT: {gc_label} (score={gc_score}, "
            f"coverage={gc_coverage_info['status']})"
        )
    if aa_available:
        evidence_summary.append(
            f"AASIST: {aa_label} (spoof={aa_spoof}, "
            f"pattern={aa_temporal['segment_pattern']}, "
            f"n={aa_temporal['window_count']})"
        )
    if not has_audio:
        evidence_summary.append("Audio: unavailable (no audio track)")
    if not has_face:
        evidence_summary.append("Face: unavailable (no detectable faces)")
    if domain_risk["flags"]:
        flag_names = [f["flag"] for f in domain_risk["flags"]]
        evidence_summary.append(f"Domain-risk flags: {', '.join(flag_names)}")

    return ReliabilityReport(
        filename=filename,
        production_verdict=verdict,
        production_fusion_score=fusion_score,
        ground_truth=ground_truth,
        domain_risk={
            "state": domain_risk["state"],
            "flags": domain_risk["flags"],
        },
        modalities=modalities,
        agreement=agreement,
        evidence_summary=evidence_summary,
        explanations=explanations,
        overall_reliability=overall_reliability,
    )


# ===========================================================================
# MODALITY-LEVEL RELIABILITY  (internal helper)
# ===========================================================================

def _assign_modality_reliability(
    modality: str,
    available: bool,
    coverage: Optional[str],
    domain_risk_state: str,
    agreement: dict,
) -> str:
    """Assign reliability state to a single modality."""
    if not available:
        return ReliabilityState.UNAVAILABLE.value

    if domain_risk_state == ReliabilityState.DOMAIN_RISK.value:
        return ReliabilityState.DOMAIN_RISK.value

    if modality == "genconvit":
        if coverage == CoverageStatus.GOOD.value:
            if agreement.get("neural_agreement") == "NEURAL_AGREEMENT":
                return ReliabilityState.HIGH.value
            return ReliabilityState.MODERATE.value
        elif coverage == CoverageStatus.PARTIAL.value:
            return ReliabilityState.MODERATE.value
        elif coverage == CoverageStatus.LOW.value:
            return ReliabilityState.LOW.value
        elif coverage == CoverageStatus.NO_FACE.value:
            return ReliabilityState.UNAVAILABLE.value
        return ReliabilityState.LOW.value

    if modality == "aasist":
        # AASIST reliability depends on audio presence and domain-risk
        if agreement.get("neural_vs_heuristic_disagreement"):
            return ReliabilityState.MODERATE.value
        return ReliabilityState.MODERATE.value

    return ReliabilityState.MODERATE.value


# ===========================================================================
# HUMAN-READABLE EXPLANATIONS  (Task 9)
# ===========================================================================

def _generate_explanations(
    gc_available: bool,
    gc_label: Optional[str],
    gc_coverage: dict,
    aa_available: bool,
    aa_label: Optional[str],
    aa_temporal: dict,
    has_audio: bool,
    has_face: bool,
    domain_risk_flags: list,
    agreement: dict,
    overall_reliability: str,
) -> list[str]:
    """Generate concise forensic explanations for human review.

    Important: these never state "this video is fake/real".  They describe
    what the evidence says and how it should be interpreted.
    """
    explanations = []

    # AASIST explanations
    if aa_available and aa_label:
        pattern = aa_temporal.get("segment_pattern", "unknown")
        n = aa_temporal.get("window_count", 0)
        mean = aa_temporal.get("mean_spoof")
        if aa_label == "SPOOF":
            if pattern == "persistent_high_spoof":
                explanations.append(
                    f"AASIST produced persistent spoof-oriented audio evidence "
                    f"across all {n} analyzed windows (mean {mean:.4f}). This "
                    f"is audio-level evidence only; it does NOT by itself "
                    f"establish video-level manipulation."
                )
            else:
                explanations.append(
                    f"AASIST detected spoof-oriented audio evidence "
                    f"(pattern: {pattern}, {n} windows, mean {mean:.4f}). "
                    f"The evidence is localized and should be interpreted "
                    f"in context."
                )
        elif aa_label == "BONAFIDE":
            explanations.append(
                f"AASIST read the audio as authentic (bonafide; "
                f"mean spoof score: {mean:.4f}). This is an audio-level "
                f"statement only; it does NOT imply the video is real."
            )
    elif not has_audio:
        explanations.append(
            "No usable audio was available; audio authenticity could not "
            "be assessed."
        )

    # GenConViT explanations
    if gc_available and gc_label:
        coverage_status = gc_coverage.get("status", "unknown")
        if coverage_status in ("LOW", "NO_FACE"):
            explanations.append(
                f"GenConViT had limited visual coverage "
                f"({coverage_status}); its output should be interpreted "
                f"cautiously."
            )
        elif coverage_status == "PARTIAL":
            explanations.append(
                f"GenConViT had partial visual coverage; its output "
                f"carries moderate weight."
            )
        else:
            explanations.append(
                f"GenConViT produced a {gc_label} reading with adequate "
                f"visual coverage."
            )
    elif not has_face:
        explanations.append(
            "No usable face frames were detected; face-dependent neural "
            "detectors could not produce visual authenticity evidence."
        )

    # Cross-modal explanations
    if agreement.get("neural_vs_heuristic_disagreement"):
        explanations.append(
            "AASIST and the existing audio heuristic disagree on the "
            "authenticity of the audio track."
        )
    if agreement.get("neural_agreement") == "NEURAL_DISAGREEMENT":
        explanations.append(
            "The two neural detectors (GenConViT and AASIST) read the "
            "manipulation evidence in different directions."
        )

    # Domain-risk explanations
    for flag in domain_risk_flags:
        fname = flag["flag"]
        if fname == "LOW_BITRATE":
            explanations.append(
                "Input exhibits a domain-risk indicator associated with "
                "low-bitrate media; detector readings may be less "
                "reliable under these conditions."
            )
        elif fname == "WHATSAPP_OR_CHAT_RELINK":
            explanations.append(
                "Filename suggests WhatsApp/chat relink; platform "
                "transcoding may have introduced artefacts that affect "
                "detector reliability."
            )
        elif fname == "MISSING_AUDIO":
            explanations.append(
                "No audio stream was available for analysis."
            )
        elif fname == "MISSING_FACE":
            explanations.append(
                "No face was detected in any sampled frame."
            )
        elif fname == "NEURAL_AUDIO_HEURISTIC_DISAGREEMENT":
            explanations.append(
                "AASIST and the audio heuristic disagree; this pattern "
                "is consistent with either synthetic audio or codec "
                "artefacts."
            )
        elif fname == "NEURAL_MODALITY_DISAGREEMENT":
            explanations.append(
                "The two neural detectors disagree on the manipulation "
                "evidence direction."
            )
        elif fname == "PERSISTENT_AASIST_SPOOF_ON_RISKY_MEDIA":
            explanations.append(
                "AASIST shows persistent spoof on media with additional "
                "risk indicators; this reading may be inflated by "
                "domain-specific artefacts."
            )

    # Overall reliability
    if overall_reliability == ReliabilityState.UNAVAILABLE.value:
        explanations.append(
            "No reliable evidence source was available; this video "
            "should be re-analyzed with additional data if possible."
        )
    elif overall_reliability == ReliabilityState.DOMAIN_RISK.value:
        explanations.append(
            "The overall evidence reliability is limited by domain-risk "
            "factors; detector readings should be treated with caution."
        )

    return explanations


# ===========================================================================
# CORPUS-LEVEL RELIABILITY ANALYSIS
# ===========================================================================

def build_corpus_reliability(evals: dict, probe_data: Optional[dict] = None) -> dict:
    """Build reliability reports for every video in an evaluation.

    Args:
        evals: the full evaluation JSON (from evaluate_testing_videos.py).
        probe_data: optional per-filename ffprobe metadata dict.

    Returns dict with per_video_reports, corpus_summary, production_invariance.
    """
    probe = probe_data or {}
    reports = []
    for row in evals.get("results", []):
        if row.get("ground_truth") not in ("FAKE", "REAL"):
            continue
        # Merge probe data
        fn = row.get("filename")
        p = probe.get(fn, {})
        enriched = {**row}
        if p:
            enriched.setdefault("bitrate_kbps", p.get("bitrate_kbps"))
            enriched.setdefault("resolution", p.get("resolution"))
            enriched.setdefault("codec", p.get("codec"))
            enriched.setdefault("filesize_mb", p.get("filesize_mb"))
        report = build_reliability_report(enriched)
        reports.append(report.to_dict())

    # Corpus summary
    reliability_counts = {}
    domain_risk_counts = {}
    for r in reports:
        rs = r.get("agreement", {}).get("summary", "unknown")
        reliability_counts[rs] = reliability_counts.get(rs, 0) + 1
        dr = r.get("domain_risk", {}).get("state", "unknown")
        domain_risk_counts[dr] = domain_risk_counts.get(dr, 0) + 1

    return {
        "per_video_reports": reports,
        "corpus_summary": {
            "total_videos": len(reports),
            "agreement_patterns": reliability_counts,
            "domain_risk_distribution": domain_risk_counts,
        },
    }


# ===========================================================================
# RUNTIME RELIABILITY (Streamlit / live analysis path)
# ===========================================================================

def build_runtime_reliability(
    video_result,
    audio_result,
    filename: str = "",
    verdict: Optional[str] = None,
    fusion_score: Optional[float] = None,
    probe: Optional[dict] = None,
) -> dict:
    """Build a reliability report from LIVE detector outputs (UI path).

    This helper consumes the runtime detectors' outputs WITHOUT running any
    detector itself and WITHOUT modifying the production verdict.  It is a
    reporting-only view used by the Streamlit UI.

    AASIST is reported as unavailable here because it is evaluation-only and
    not executed in the production `main.run_pipeline` path.

    Args:
        video_result: result.video_result (ModalResult) with GenConViT
            metadata exposed under .metadata.
        audio_result: result.audio_result (ModalResult).
        filename: the video filename (used for WhatsApp/chat-relink detection).
        verdict: the production verdict string.
        fusion_score: the production fused score.
        probe: optional ffprobe metadata (bitrate_kbps, resolution, has_audio).

    Returns: dict (same shape as ReliabilityReport.to_dict()).
    """
    vmeta = getattr(video_result, "metadata", None) or {}

    # GenConViT (evaluation-only neural video detector) — exposed in metadata.
    gc_status = vmeta.get("genconvit_status")
    gc_label = vmeta.get("genconvit_predicted_label")
    gc_score = vmeta.get("genconvit_score")
    gc_available = gc_status == "ACTIVE_MODEL" and gc_label is not None

    # AASIST is not run in the production runtime path (evaluation-only).
    aa_available = False

    # Audio heuristic
    audio_heur_mean = None
    aref = getattr(audio_result, "audio_results", None)
    if aref:
        scores = [s.fake_prob for s in aref if s.fake_prob is not None]
        if scores:
            audio_heur_mean = sum(scores) / len(scores)

    # has_audio
    has_audio = bool(getattr(audio_result, "error", None) is None) or (
        (probe or {}).get("has_audio", True)
    )

    # resolution / bitrate from probe
    resolution = (probe or {}).get("resolution")
    bitrate_kbps = (probe or {}).get("bitrate_kbps")

    # texture score — from frame sub_scores if available
    texture_score = None
    tex_vals = []
    for fr in getattr(video_result, "frame_results", []):
        ss = getattr(fr, "sub_scores", None) or {}
        if isinstance(ss.get("texture"), (int, float)):
            tex_vals.append(ss["texture"])
    if tex_vals:
        texture_score = sum(tex_vals) / len(tex_vals)

    provenance_override = bool(vmeta.get("provenance_score", 0) is not None
                               and vmeta.get("provenance_score", 0) >= 0.90)

    # Build an enriched row-shaped dict and delegate to build_reliability_report
    row = {
        "filename": filename,
        "ground_truth": None,
        "has_audio": has_audio,
        "bitrate_kbps": bitrate_kbps,
        "resolution": resolution,
        "verdict": verdict or "UNKNOWN",
        "fusion_score": fusion_score,
        "texture_score": texture_score,
        "provenance_override": provenance_override,
        "genconvit_status": gc_status,
        "genconvit_predicted_label": gc_label,
        "genconvit_score": gc_score,
        "genconvit_face_frames_found": vmeta.get("genconvit_face_frames_found"),
        "genconvit_frames_requested": vmeta.get("genconvit_frames_requested"),
        "genconvit_face_coverage": vmeta.get("genconvit_face_coverage"),
        "genconvit_average_face_confidence": vmeta.get("genconvit_average_face_confidence"),
        "genconvit_min_face_confidence": vmeta.get("genconvit_min_face_confidence"),
        "genconvit_max_face_confidence": vmeta.get("genconvit_max_face_confidence"),
        "genconvit_coverage_status": vmeta.get("genconvit_coverage_status")
        or vmeta.get("genconvit_face_quality"),
        "aasist_status": None,
        "aasist_predicted_label": None,
        "aasist_spoof_score": None,
        "aasist_window_scores": [],
        "aasist_window_count": 0,
        "audio_segments": [
            {"segment_index": s.segment_index, "start": s.start_sec,
             "end": s.end_sec, "fake_prob": s.fake_prob}
            for s in aref or []
        ],
    }
    report = build_reliability_report(row)
    # Runtime path: annotate that AASIST is evaluation-only & not executed here.
    if "aasist" in report.modalities:
        report.modalities["aasist"]["available"] = False
        report.modalities["aasist"]["note"] = (
            "AASIST is evaluation-only and is not executed in the live "
            "analysis path; run the offline evaluation to obtain AASIST "
            "evidence."
        )
    return report.to_dict()
