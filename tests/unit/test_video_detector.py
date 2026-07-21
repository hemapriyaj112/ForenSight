"""
tests/unit/test_video_detector.py
==================================
Unit tests for pipeline/video/detector.py (VideoDetector + _video_provenance).

Rewritten from scratch (session 5) — the previous version of this file
tested an abandoned MTCNN+EfficientNet design (_GradCAM, VideoDetector(device=...),
no_face_strategy, etc.) that matches nothing in the current codebase. The
real VideoDetector reuses ImageDetector's per-frame multi-signal fusion,
adds container-level provenance + temporal localization, and is invoked via
`detect(frames, timestamps, video_bytes, video_path)` or `run(frames_dir, ...)`.

No GPU, no real ML weights, no network required — everything here runs on
synthetic numpy frames and the pure-Python heuristic pipeline.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.video.detector import VideoDetector, ImageDetector, _video_provenance
from utils.types import ModalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(seed: int = 0, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """A deterministic-but-varied synthetic RGB frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (*size, 3), dtype=np.uint8)


def _make_frame_file(tmp_dir: Path, index: int, seed: int) -> Path:
    from PIL import Image
    arr = _make_frame(seed=seed)
    p = tmp_dir / f"frame_{index:06d}.jpg"
    Image.fromarray(arr).save(p, format="JPEG")
    return p


# A minimal fake JPEG-ish blob containing a C2PA marker + an AI
# digitalSourceType string, byte-scanned by metadata_forensics regardless of
# real container validity (see metadata_forensics._C2PA_MARKERS).
_C2PA_AI_VIDEO_BYTES = b"....c2pa....trainedAlgorithmicMedia....urn:uuid....some-mp4-bytes"
_PLAIN_VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64  # no provenance markers


# ---------------------------------------------------------------------------
# VideoDetector.detect() — core per-frame fusion behavior
# ---------------------------------------------------------------------------

class TestVideoDetectorDetect:
    def test_returns_modal_result_with_one_frame_result_per_frame(self):
        frames = [_make_frame(seed=i) for i in range(3)]
        result = VideoDetector().detect(frames)
        assert isinstance(result, ModalResult)
        assert result.modality == "video"
        assert len(result.frame_results) == 3

    def test_overall_fake_prob_is_mean_of_frame_scores(self):
        frames = [_make_frame(seed=i) for i in range(4)]
        result = VideoDetector().detect(frames)
        expected = float(np.mean([fr.fake_prob for fr in result.frame_results]))
        assert result.fake_prob == pytest.approx(expected, abs=1e-9)

    def test_empty_frame_list_gives_zero_prob_and_no_frame_results(self):
        result = VideoDetector().detect([])
        assert result.frame_results == []
        assert result.fake_prob == 0.0

    def test_per_frame_fusion_excludes_noise_ela_metadata_and_spectral(self):
        """
        Per the video-parity rebuild: noise-floor and ELA were confirmed to
        have ~zero (noise) or backwards (ELA) discriminative power for
        H.264 video, and per-frame EXIF/metadata is meaningless for an
        ffmpeg-extracted frame — all three are excluded from per-frame
        fusion (image pipeline keeps all five).

        Session 5: spectral was also excluded — confirmed to sit at its
        floor value (peak=0, slope=0.15 -> score=0.075) on every real AND
        AI video frame tested, creating a hard ~0.47 ceiling on the fused
        score that made a confirmed-AI clip land UNCERTAIN no matter how
        confident texture was (see include_spectral docstring on
        ImageDetector.detect). Texture is now the sole per-frame heuristic
        for video (+classifier, if wired in).
        """
        frames = [_make_frame(seed=0)]
        result = VideoDetector().detect(frames)
        sub_scores = result.frame_results[0].sub_scores
        assert "noise" not in sub_scores
        assert "ela" not in sub_scores
        assert "metadata" not in sub_scores
        assert "spectral" not in sub_scores
        assert "texture" in sub_scores

    def test_timestamps_are_passed_through_to_frame_results(self):
        frames = [_make_frame(seed=i) for i in range(3)]
        timestamps = [0.0, 2.5, 5.0]
        result = VideoDetector().detect(frames, timestamps=timestamps)
        assert [fr.timestamp_sec for fr in result.frame_results] == timestamps

    def test_missing_timestamps_default_to_frame_index(self):
        frames = [_make_frame(seed=i) for i in range(2)]
        result = VideoDetector().detect(frames)
        assert [fr.timestamp_sec for fr in result.frame_results] == [0.0, 1.0]


# ---------------------------------------------------------------------------
# Container-level provenance
# ---------------------------------------------------------------------------

class TestVideoProvenance:
    def test_no_markers_reports_no_provenance_found(self):
        out = _video_provenance(_PLAIN_VIDEO_BYTES, filename="clip.mp4")
        assert not out["generator_tag"]
        assert any("No embedded provenance" in f for f in out["findings"])

    def test_c2pa_ai_tag_detected(self):
        out = _video_provenance(_C2PA_AI_VIDEO_BYTES, filename="clip.mp4")
        assert out["generator_tag"]
        assert out["score"] >= 0.9

    def test_generic_image_failure_message_swapped_for_video_message(self):
        """_video_provenance should never surface the image-metadata module's
        'could not read image metadata' line — that's misleading for a
        video container, which was never expected to parse as an image."""
        out = _video_provenance(_PLAIN_VIDEO_BYTES, filename="clip.mp4")
        assert not any("could not read image metadata" in f.lower() for f in out["findings"])

    def test_confirmed_provenance_floors_overall_score(self):
        """A confirmed AI generator tag should floor the fused video score at
        0.9, the same override behavior ImageDetector applies for images."""
        frames = [_make_frame(seed=0)]  # near-random noise frame, low heuristic score
        result = VideoDetector().detect(frames, video_bytes=_C2PA_AI_VIDEO_BYTES,
                                         video_path="clip.mp4")
        assert result.fake_prob >= 0.9
        assert result.metadata["generator_tag"]

    def test_no_video_bytes_skips_provenance_entirely(self):
        frames = [_make_frame(seed=0)]
        result = VideoDetector().detect(frames)  # no video_bytes passed
        assert result.metadata["generator_tag"] is None
        assert result.metadata["provenance_score"] is None


# ---------------------------------------------------------------------------
# Spatial ("where") + temporal ("when") localization
# ---------------------------------------------------------------------------

class TestLocalization:
    def test_spatial_localization_present_for_nonempty_video(self):
        frames = [_make_frame(seed=i) for i in range(3)]
        result = VideoDetector().detect(frames)
        assert "frame_index" in result.spatial_localization
        assert 0 <= result.spatial_localization["frame_index"] < 3

    def test_spatial_localization_targets_hottest_frame(self):
        frames = [_make_frame(seed=i) for i in range(3)]
        result = VideoDetector().detect(frames)
        scores = [fr.fake_prob for fr in result.frame_results]
        assert result.spatial_localization["frame_index"] == int(np.argmax(scores))

    def test_temporal_localization_present_for_nonempty_video(self):
        frames = [_make_frame(seed=i) for i in range(5)]
        result = VideoDetector().detect(frames, timestamps=[0, 1, 2, 3, 4])
        assert "is_localized" in result.temporal_localization
        assert "suspicious_fraction" in result.temporal_localization


# ---------------------------------------------------------------------------
# Document-frame gating (skips classifier on document-looking frames)
# ---------------------------------------------------------------------------

class TestDocumentGating:
    def test_documents_skipped_counted_in_metadata(self):
        """
        Frames that look like flat document/text pages should have
        is_document=True on their FrameResult and be reflected in the
        container-level documents_skipped count — even with no classifier
        wired in, this bookkeeping should still work off ImageDetector's
        document_detector output.
        """
        # A flat, near-white frame is a reasonable proxy for "document-like"
        # for this gating check; we don't assert an exact classification
        # here (that's document_detector's own test surface) — only that
        # whatever ImageDetector decides is correctly threaded through to
        # the video-level documents_skipped counter.
        flat_frame = np.full((64, 64, 3), 250, dtype=np.uint8)
        detector = VideoDetector()
        result = detector.detect([flat_frame])
        fr = result.frame_results[0]
        expected_skipped = 1 if fr.is_document else 0
        assert result.metadata["documents_skipped"] == expected_skipped


# ---------------------------------------------------------------------------
# run() — directory-of-frames entry point used by main.py
# ---------------------------------------------------------------------------

class TestVideoDetectorRun:
    def test_run_missing_dir_returns_error_result(self):
        result = VideoDetector().run("/nonexistent/frames/dir/xyz")
        assert result.modality == "video"
        assert result.fake_prob == 0.5
        assert result.error is not None

    def test_run_empty_dir_returns_error_result(self):
        with tempfile.TemporaryDirectory() as d:
            result = VideoDetector().run(d)
            assert result.fake_prob == 0.5
            assert result.error is not None

    def test_run_loads_and_sorts_frames_from_directory(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            # write out of order to confirm alphabetical sort is applied
            _make_frame_file(tmp, index=2, seed=2)
            _make_frame_file(tmp, index=0, seed=0)
            _make_frame_file(tmp, index=1, seed=1)
            result = VideoDetector().run(tmp)
            assert len(result.frame_results) == 3
            assert [fr.frame_index for fr in result.frame_results] == [0, 1, 2]

    def test_run_ignores_non_frame_files(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _make_frame_file(tmp, index=0, seed=0)
            (tmp / "notes.txt").write_text("not a frame")
            result = VideoDetector().run(tmp)
            assert len(result.frame_results) == 1

    def test_run_passes_video_bytes_through_to_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _make_frame_file(tmp, index=0, seed=0)
            result = VideoDetector().run(
                tmp, video_bytes=_C2PA_AI_VIDEO_BYTES, video_path="clip.mp4",
            )
            assert result.fake_prob >= 0.9
            assert result.metadata["generator_tag"]


# ---------------------------------------------------------------------------
# ai_classifier wiring — shared instance passed through to ImageDetector
# ---------------------------------------------------------------------------

class TestClassifierWiring:
    def test_ai_classifier_shared_with_internal_image_detector(self):
        fake_classifier = MagicMock()
        fake_classifier.available = False  # not loaded — just verifying wiring
        fake_classifier.backend = "transformers"
        detector = VideoDetector(ai_classifier=fake_classifier)
        assert detector._image_detector._ai_classifier is fake_classifier

    def test_classifier_backend_surfaced_in_metadata(self):
        fake_classifier = MagicMock()
        fake_classifier.available = False
        fake_classifier.backend = "transformers"
        fake_classifier.load_error = "network unreachable"
        detector = VideoDetector(ai_classifier=fake_classifier)
        result = detector.detect([_make_frame(seed=0)])
        assert result.metadata["classifier_backend"] == "transformers"
        assert result.metadata["classifier_error"] == "network unreachable"


class TestTextureCorroborationGate:
    """
    Regression tests for a real production bug (session 5, reported by the
    user against a live video, not caught by synthetic testing alone):
    texture is the sole per-frame heuristic for video, and was gated so it
    couldn't alone push a frame into FAKE without the classifier
    corroborating — but the FIRST version of that gate only checked
    whether the classifier was *active*, not whether it *agreed*. A real
    video scored texture=0.828 / classifier=0.401 (classifier correctly
    leaning REAL) and still fused to 0.636 (FAKE) because texture's 55%
    weight overpowered a disagreeing classifier. The gate must key off the
    classifier's own reading vs fake_threshold, not merely whether it ran.
    """

    @staticmethod
    def _detect_with_fixed_texture(texture_score: float, classifier_score: float | None):
        frame = _make_frame(seed=1)
        clf = None
        if classifier_score is not None:
            clf = MagicMock()
            clf.available = True
            clf.backend = "transformers"
            clf.predict.return_value = {
                "score": classifier_score, "available": True,
                "findings": ["mock"], "backend": "transformers",
            }
        detector = ImageDetector(ai_classifier=clf)
        with patch.object(
            detector._gradcam, "score_and_overlay",
            return_value=(texture_score, np.zeros((64, 64, 3), dtype=np.uint8)),
        ):
            return detector.detect(
                frame, image_id="f", include_metadata=False, include_noise=False,
                include_ela=False, include_spectral=False,
            )

    def test_high_texture_disagreeing_classifier_does_not_reach_fake(self):
        # The exact real-world numbers reported: texture 0.828, classifier
        # 0.401 (well below fake_threshold=0.60, i.e. genuinely disagreeing,
        # not just "not yet confirming"). Must land in UNCERTAIN, not FAKE.
        result = self._detect_with_fixed_texture(0.828, 0.401)
        assert result.fused_score < 0.60
        assert result.fused_score < 0.828  # confirms the gate actually engaged

    def test_high_texture_corroborating_classifier_reaches_fake(self):
        # Classifier itself reads at/above fake_threshold -> corroborates,
        # gate should NOT suppress escalation into FAKE.
        result = self._detect_with_fixed_texture(0.95, 0.74)
        assert result.fused_score >= 0.60

    def test_high_texture_no_classifier_caps_below_fake(self):
        result = self._detect_with_fixed_texture(1.0, None)
        assert result.fused_score < 0.60
