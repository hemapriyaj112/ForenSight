"""
tests/unit/test_face_crop.py

Session 6 regression coverage: ai_noaud2.mp4 (an AI-generated video of a
hand with no face anywhere) was getting "face found" on a frame because
FaceCropper.crop() accepted any box MTCNN's .detect() proposed, without
checking the confidence it came back with.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.forensics.face_crop import FaceCropper


def _make_cropper_with_fake_mtcnn(boxes, probs, min_confidence: float = 0.90) -> FaceCropper:
    cropper = FaceCropper.__new__(FaceCropper)  # bypass __init__'s real MTCNN load
    cropper.margin = 0.4
    cropper.min_face_size = 20
    cropper.min_confidence = min_confidence
    cropper.available = True
    cropper.load_error = None

    class _FakeMTCNN:
        def detect(self, img):
            return boxes, probs

    cropper._mtcnn = _FakeMTCNN()
    return cropper


class TestFaceCropperConfidenceFiltering:
    def test_low_confidence_box_is_rejected(self):
        # Regression: a skin-tone/blob-like region (e.g. a hand) triggering
        # MTCNN's cascade at low confidence must not count as a face.
        cropper = _make_cropper_with_fake_mtcnn(
            boxes=np.array([[10.0, 10.0, 50.0, 50.0]]),
            probs=np.array([0.55]),
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cropper.crop(image)
        assert result["face_found"] is False
        assert result["crop"] is None

    def test_high_confidence_box_is_accepted(self):
        cropper = _make_cropper_with_fake_mtcnn(
            boxes=np.array([[10.0, 10.0, 50.0, 50.0]]),
            probs=np.array([0.995]),
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cropper.crop(image)
        assert result["face_found"] is True
        assert result["confidence"] == pytest.approx(0.995)

    def test_picks_largest_among_only_the_confident_boxes(self):
        # A large low-confidence box (false positive) must not win over a
        # smaller high-confidence one.
        cropper = _make_cropper_with_fake_mtcnn(
            boxes=np.array([
                [0.0, 0.0, 90.0, 90.0],    # huge but low-confidence
                [10.0, 10.0, 30.0, 30.0],  # small but confident
            ]),
            probs=np.array([0.40, 0.98]),
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cropper.crop(image)
        assert result["face_found"] is True
        assert result["confidence"] == pytest.approx(0.98)

    def test_no_boxes_at_all(self):
        cropper = _make_cropper_with_fake_mtcnn(boxes=None, probs=None)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cropper.crop(image)
        assert result["face_found"] is False

    def test_unavailable_cropper_returns_no_face_found(self):
        cropper = FaceCropper.__new__(FaceCropper)
        cropper.margin = 0.4
        cropper.min_face_size = 20
        cropper.min_confidence = 0.90
        cropper.available = False
        cropper.load_error = "torch not installed"
        cropper._mtcnn = None
        result = cropper.crop(np.zeros((10, 10, 3), dtype=np.uint8))
        assert result["face_found"] is False
        assert result["error"] == "torch not installed"
