"""
tests/unit/test_face_crop.py

Session 6 regression coverage:
- ai_noaud2.mp4 (an AI-generated video of a hand with no face anywhere)
  was getting "face found" on a frame because FaceCropper.crop() accepted
  any box MTCNN's .detect() proposed, without checking the confidence it
  came back with (fixed: min_confidence filtering).
- Even with that filter in place, the same video's hand region still
  scored >=0.90 confidence on a later run -- a genuinely hard false
  positive the score alone can't catch. Fixed with an independent
  landmark-geometry sanity check (eyes level, nose between/below eyes,
  mouth below nose) on top of the confidence filter.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.forensics.face_crop import FaceCropper

# A coherent, roughly frontal face layout for a 40x40 box at (10,10)-(50,50):
# left eye, right eye, nose, mouth-left, mouth-right (facenet-pytorch order).
_COHERENT_FACE_LANDMARKS = np.array([
    [20.0, 20.0],  # left eye
    [40.0, 20.0],  # right eye
    [30.0, 30.0],  # nose
    [22.0, 40.0],  # mouth-left
    [38.0, 40.0],  # mouth-right
])

# A skin-tone blob (e.g. a hand) that scored well on the cascade but whose
# landmark regressor output doesn't form a coherent face: "eyes" stacked
# near-vertically instead of level, and the "nose" not between them.
_INCOHERENT_LANDMARKS = np.array([
    [15.0, 15.0],  # "left eye"
    [18.0, 45.0],  # "right eye" -- far below the other eye, not level
    [45.0, 12.0],  # "nose" -- off to the side, not between the eyes
    [20.0, 20.0],  # "mouth-left" -- above the "nose", geometry inverted
    [22.0, 22.0],  # "mouth-right"
])


def _make_cropper_with_fake_mtcnn(boxes, probs, landmarks=None, min_confidence: float = 0.90) -> FaceCropper:
    cropper = FaceCropper.__new__(FaceCropper)  # bypass __init__'s real MTCNN load
    cropper.margin = 0.4
    cropper.min_face_size = 20
    cropper.min_confidence = min_confidence
    cropper.available = True
    cropper.load_error = None
    landmarks_out = landmarks

    class _FakeMTCNN:
        def detect(self, img, landmarks=True):
            return boxes, probs, landmarks_out

    cropper._mtcnn = _FakeMTCNN()
    return cropper


class TestFaceCropperConfidenceFiltering:
    def test_low_confidence_box_is_rejected(self):
        # Regression: a skin-tone/blob-like region (e.g. a hand) triggering
        # MTCNN's cascade at low confidence must not count as a face.
        cropper = _make_cropper_with_fake_mtcnn(
            boxes=np.array([[10.0, 10.0, 50.0, 50.0]]),
            probs=np.array([0.55]),
            landmarks=np.array([_COHERENT_FACE_LANDMARKS]),
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cropper.crop(image)
        assert result["face_found"] is False
        assert result["crop"] is None

    def test_high_confidence_box_is_accepted(self):
        cropper = _make_cropper_with_fake_mtcnn(
            boxes=np.array([[10.0, 10.0, 50.0, 50.0]]),
            probs=np.array([0.995]),
            landmarks=np.array([_COHERENT_FACE_LANDMARKS]),
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
            landmarks=np.array([_COHERENT_FACE_LANDMARKS, _COHERENT_FACE_LANDMARKS * 0.5]),
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cropper.crop(image)
        assert result["face_found"] is True
        assert result["confidence"] == pytest.approx(0.98)

    def test_no_boxes_at_all(self):
        cropper = _make_cropper_with_fake_mtcnn(boxes=None, probs=None, landmarks=None)
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


class TestFaceCropperGeometryFiltering:
    """
    Session 6 finding #2: on a later run, ai_noaud2.mp4's hand region
    scored >=0.90 confidence -- past the confidence filter above -- so an
    independent, geometry-based check was added on MTCNN's 5-point
    landmarks. These tests cover that check directly.
    """

    def test_high_confidence_but_incoherent_landmarks_is_rejected(self):
        # This is the exact failure mode that slipped past min_confidence:
        # a high score, but landmarks that don't form a plausible face.
        cropper = _make_cropper_with_fake_mtcnn(
            boxes=np.array([[10.0, 10.0, 50.0, 50.0]]),
            probs=np.array([0.97]),
            landmarks=np.array([_INCOHERENT_LANDMARKS]),
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cropper.crop(image)
        assert result["face_found"] is False

    def test_missing_landmarks_is_rejected(self):
        cropper = _make_cropper_with_fake_mtcnn(
            boxes=np.array([[10.0, 10.0, 50.0, 50.0]]),
            probs=np.array([0.97]),
            landmarks=None,
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cropper.crop(image)
        assert result["face_found"] is False

    def test_extremely_elongated_box_is_rejected_even_with_landmarks(self):
        # A hand/forearm-shaped box: far taller (or wider) than any real
        # face crop MTCNN would propose.
        cropper = _make_cropper_with_fake_mtcnn(
            boxes=np.array([[10.0, 10.0, 30.0, 90.0]]),  # 20 wide x 80 tall
            probs=np.array([0.97]),
            landmarks=np.array([[
                [15.0, 20.0], [25.0, 20.0], [20.0, 30.0], [16.0, 40.0], [24.0, 40.0],
            ]]),
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cropper.crop(image)
        assert result["face_found"] is False

    def test_coherent_geometry_with_slight_head_turn_still_accepted(self):
        # A plausible, slightly turned real face shouldn't be punished by
        # the geometry check -- only genuinely incoherent layouts should.
        turned_landmarks = np.array([
            [18.0, 19.0],  # left eye
            [39.0, 21.0],  # right eye (slightly lower -- head tilt)
            [32.0, 29.0],  # nose (slightly off-center -- head turn)
            [21.0, 39.0],  # mouth-left
            [37.0, 41.0],  # mouth-right
        ])
        cropper = _make_cropper_with_fake_mtcnn(
            boxes=np.array([[10.0, 10.0, 50.0, 50.0]]),
            probs=np.array([0.96]),
            landmarks=np.array([turned_landmarks]),
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = cropper.crop(image)
        assert result["face_found"] is True
