"""Canonical ForenSight types — do not deviate."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Verdict enum  (used by Fuser / AnalysisResult)
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    REAL      = "REAL"
    FAKE      = "FAKE"
    UNCERTAIN = "UNCERTAIN"


# ---------------------------------------------------------------------------
# Pipeline types
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    frame_index: int
    timestamp_sec: float
    fake_prob: float
    face_detected: bool
    gradcam_overlay: Optional[bytes] = None  # PNG bytes or None

    # --- Video parity (Sprint 9): each frame now gets the same multi-signal
    # fusion an image gets (texture/spectral/ela/noise/+classifier), not just
    # texture+spectral. These are additive/optional so old callers (tests,
    # any code only reading fake_prob/gradcam_overlay) are unaffected.
    sub_scores: dict = field(default_factory=dict)       # {"texture":.., "spectral":.., "ela":.., "noise":.., "classifier":..}
    findings: list = field(default_factory=list)          # plain-language reasons for this frame, ranked
    is_document: bool = False                              # frame looked like a document/text page (classifier skipped)
    classifier_active: bool = False                        # whether the trained AI classifier ran on this frame


@dataclass
class AudioSegmentResult:
    segment_index: int
    start_sec: float
    end_sec: float
    fake_prob: float


@dataclass
class ModalResult:
    modality: str  # "video" | "audio"
    fake_prob: float
    frame_results: list[FrameResult] = field(default_factory=list)
    audio_results: list[AudioSegmentResult] = field(default_factory=list)
    aligned_scores: list[float] = field(default_factory=list)
    error: Optional[str] = None   # set by detectors on failure; read by main.py

    # --- Video parity (Sprint 9) ---
    # Container-level provenance (C2PA/generator-tag scan of the whole video
    # file — analogous to metadata_forensics for images) plus classifier/
    # document-detector bookkeeping. Only populated by VideoDetector; empty
    # dict for audio / anything else, so existing readers are unaffected.
    metadata: dict = field(default_factory=dict)
    # keys when modality=="video": generator_tag, has_c2pa, provenance_score,
    #   provenance_findings, classifier_backend, classifier_error,
    #   frames_analysed, documents_skipped

    # "Where" (spatial, within the most-suspicious frame) and "when"
    # (temporal, which part of the timeline looks edited) — same idea as
    # pipeline/forensics/localization.py for images, extended across time.
    # Display/explanation layers; do not feed back into fake_prob.
    spatial_localization: dict = field(default_factory=dict)
    temporal_localization: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    video_id: str
    verdict: Verdict
    fused_score: float
    video_result: ModalResult
    audio_result: ModalResult
    metadata: dict = field(default_factory=dict)
    # metadata keys: run_id, video_path, video_score, audio_score,
    #                calibrated_video_score, calibrated_audio_score,
    #                verdict_timeline  (list of (timestamp_sec, fused_score) tuples)

    @property
    def analysis_id(self) -> str:
        """Alias for video_id — main.py references result.analysis_id."""
        return self.metadata.get("run_id", self.video_id)


# ---------------------------------------------------------------------------
# Image detection type  (Sprint 7)
# ---------------------------------------------------------------------------

@dataclass
class ImageResult:
    """Result of single-image deepfake detection (multi-signal fusion)."""
    image_id: str
    verdict: str                              # "REAL" | "FAKE" | "UNCERTAIN"
    fused_score: float                        # weighted fusion of all signals
    gradcam_score: float                      # texture heuristic fake probability 0-1
    freq_score: float                         # FFT spectral-artefact fake probability 0-1
    gradcam_overlay: Optional[bytes] = None  # PNG bytes of texture heatmap blended on image
    freq_heatmap: Optional[bytes] = None     # PNG bytes of FFT magnitude heatmap
    ela_heatmap: Optional[bytes] = None      # PNG bytes of error-level-analysis heatmap
    findings: list = field(default_factory=list)      # plain-language reasons, ranked
    headline: str = ""                                 # one-line human summary
    confidence_label: str = ""                          # "High" | "Moderate" | "Low"
    sub_scores: dict = field(default_factory=dict)      # {"texture":.., "spectral":.., "ela":.., "metadata":..}
    metadata: dict = field(default_factory=dict)
    # metadata keys: run_id, image_path, width, height, face_detected,
    #                gradcam_weight, freq_weight, ela_weight, metadata_weight,
    #                generator_tag, has_exif

    # --- Localization (Sprint 8): "where" a suspicious edit sits, not just
    # a whole-image verdict. Built from noise+ela grids; does not change
    # the fused score/weights. Only meaningful for partial edits — see
    # pipeline/forensics/localization.py docstring for the fully-generated-
    # image caveat.
    localization_heatmap: Optional[bytes] = None   # PNG, red overlay on original
    localization_summary: str = ""                 # one-line plain-language description
    localization_is_localized: bool = False        # True = a concentrated region was found
    localization_regions: list = field(default_factory=list)
    # each region: {"bbox_px": (y0,y1,x0,x1), "position_desc": str, "area_fraction": float}