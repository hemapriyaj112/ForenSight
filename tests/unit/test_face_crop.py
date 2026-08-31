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

import sys

import numpy as np
import pytest
from unittest.mock import patch

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


def _flat_image_with_patches(size: int, patches: list[tuple[float, float, int]], seed: int = 0) -> np.ndarray:
    """A uniform mid-grey image (near-zero Laplacian variance everywhere)
    with small random-noise squares stamped at given (cx, cy, amplitude)
    points -- amplitude 0 leaves that spot flat. Used to give the bilateral
    texture-symmetry check something real to measure, since it inspects
    actual pixel content rather than just the landmark coordinates."""
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 128, dtype=np.uint8)
    patch = 7
    half = patch // 2
    for cx, cy, amplitude in patches:
        y0, x0 = int(cy) - half, int(cx) - half
        if amplitude <= 0:
            continue
        noise = rng.integers(128 - amplitude, 128 + amplitude + 1, size=(patch, patch, 3))
        image[y0 : y0 + patch, x0 : x0 + patch, :] = noise
    return image


def _make_cropper_with_fake_mtcnn(boxes, probs, landmarks=None, min_confidence: float = 0.90) -> FaceCropper:
    cropper = FaceCropper.__new__(FaceCropper)  # bypass __init__'s real MTCNN load
    cropper.margin = 0.4
    cropper.min_face_size = 20
    cropper.min_confidence = min_confidence
    cropper.available = True
    cropper.load_error = None
    cropper.backend = "mtcnn"
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
        image = _flat_image_with_patches(100, [
            (20.0, 20.0, 40), (40.0, 20.0, 40),  # eyes, matching amplitude
            (22.0, 40.0, 40), (38.0, 40.0, 40),  # mouth corners, matching
        ])
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
        image = _flat_image_with_patches(100, [
            (10.0, 10.0, 40), (20.0, 10.0, 40),  # scaled-down eyes (*0.5 box)
            (11.0, 20.0, 40), (19.0, 20.0, 40),  # scaled-down mouth corners
        ])
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
        cropper.backend = None
        cropper._mtcnn = None
        result = cropper.crop(np.zeros((10, 10, 3), dtype=np.uint8))
        assert result["face_found"] is False
        assert result["error"] == "torch not installed"
        assert result["backend"] is None


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
        image = _flat_image_with_patches(100, [
            (18.0, 19.0, 40), (39.0, 21.0, 35),
            (21.0, 39.0, 38), (37.0, 41.0, 42),
        ])
        result = cropper.crop(image)
        assert result["face_found"] is True


class TestFaceCropperTextureSymmetry:
    """
    Session 6 finding #3: the geometry check wasn't enough either -- real
    diagnostic data pulled from the actual false-positive frame in
    ai_noaud2.mp4 (a hand, no face) showed MTCNN's landmark regressor had
    produced a geometrically plausible-looking layout anyway (a known
    failure mode: it was only ever trained on real faces, so it defaults
    to something face-shaped rather than garbage on out-of-distribution
    input). The real discriminator found in that data: a genuine face's
    two eyes -- and two mouth corners -- have local contrast/edge energy
    within the same rough order of magnitude of each other. On the hand,
    one landmark point happened to land on a knuckle/finger-gap edge while
    its mirror point sat on smooth skin, producing 13x (eyes) and 617x
    (mouth) left/right gaps no real face would show.
    """

    def test_real_ai_noaud2_false_positive_data_is_rejected(self):
        # Exact numbers from the user's diagnose_face_false_positive.py run
        # against ai_noaud2.mp4 frame_000006.jpg.
        cropper = FaceCropper.__new__(FaceCropper)
        points = np.array([
            [466.2, 181.1],  # left_eye
            [486.2, 190.6],  # right_eye
            [472.6, 197.2],  # nose
            [452.2, 207.1],  # mouth_left
            [467.3, 215.0],  # mouth_right
        ])
        real_vars = {
            (466.2, 181.1): 197.99395751953125,
            (486.2, 190.6): 2626.44287109375,
            (472.6, 197.2): 126.7009048461914,
            (452.2, 207.1): 5.667226314544678,
            (467.3, 215.0): 3497.695556640625,
        }
        cropper._local_variance = lambda gray, cx, cy, patch=7: real_vars[(cx, cy)]
        # Geometry check passes on this real data (confirmed via the
        # diagnostic script) -- this test isolates that the NEW check is
        # what catches it.
        assert cropper._has_coherent_face_geometry(
            points, np.array([434.7, 147.9, 501.1, 234.5])
        ) is True
        assert cropper._has_bilateral_texture_symmetry(
            np.zeros((300, 600, 3), dtype=np.uint8), points
        ) is False

    def test_symmetric_texture_is_accepted(self):
        cropper = FaceCropper.__new__(FaceCropper)
        image = _flat_image_with_patches(100, [
            (20.0, 20.0, 40), (40.0, 20.0, 45),
            (22.0, 40.0, 38), (38.0, 40.0, 42),
        ])
        assert cropper._has_bilateral_texture_symmetry(image, _COHERENT_FACE_LANDMARKS) is True

    def test_asymmetric_texture_is_rejected_despite_valid_geometry(self):
        cropper = FaceCropper.__new__(FaceCropper)
        image = _flat_image_with_patches(100, [
            (20.0, 20.0, 5),    # left eye: near-flat, low contrast
            (40.0, 20.0, 50),   # right eye: strong edge -- big asymmetry
            (22.0, 40.0, 38), (38.0, 40.0, 40),  # mouth corners fine
        ])
        assert cropper._has_bilateral_texture_symmetry(image, _COHERENT_FACE_LANDMARKS) is False

    def test_fully_flat_patch_is_rejected_even_if_symmetric(self):
        cropper = FaceCropper.__new__(FaceCropper)
        image = np.full((100, 100, 3), 128, dtype=np.uint8)  # perfectly uniform
        assert cropper._has_bilateral_texture_symmetry(image, _COHERENT_FACE_LANDMARKS) is False

    def test_landmark_near_frame_edge_does_not_reject_on_this_check_alone(self):
        cropper = FaceCropper.__new__(FaceCropper)
        image = np.full((30, 30, 3), 128, dtype=np.uint8)
        # _COHERENT_FACE_LANDMARKS' points (up to x=40, y=40) fall outside
        # this tiny 30x30 image -- local_variance patches run off the edge.
        assert cropper._has_bilateral_texture_symmetry(image, _COHERENT_FACE_LANDMARKS) is True


class TestSmoothAiFaceRecall:
    """
    Session 7 regression: the bilateral-texture floor was originally 8.0,
    which rejected smooth AI-GENERATED faces wholesale -- synthetic skin
    renders with very low local edge energy on BOTH sides (symmetric,
    ratio ~1), so an entire multi-face AI video produced 0/46 detected
    faces, the trained classifier never ran on a single frame, and the
    video was called REAL from texture alone. The floor now only has to
    stop two trivially-equal near-zero values; these tests pin that
    behaviour on both sides.
    """

    @staticmethod
    def _cropper_with_fixed_variances(variances: dict) -> FaceCropper:
        cropper = FaceCropper.__new__(FaceCropper)
        cropper._local_variance = lambda gray, cx, cy, patch=7: variances[(cx, cy)]
        return cropper

    def test_smooth_but_symmetric_ai_face_is_accepted(self):
        # Both eye patches below the OLD 8.0 floor yet structured (>2.0)
        # and symmetric -- exactly what a rendered AI face looks like.
        vars_map = {(20.0, 20.0): 3.1, (40.0, 20.0): 3.4,
                    (22.0, 40.0): 2.6, (38.0, 40.0): 2.9}
        cropper = self._cropper_with_fixed_variances(vars_map)
        image = np.full((100, 100, 3), 128, dtype=np.uint8)  # content unused (variance mocked)
        assert cropper._has_bilateral_texture_symmetry(image, _COHERENT_FACE_LANDMARKS) is True

    def test_truly_blank_patches_still_rejected_below_new_floor(self):
        vars_map = {(20.0, 20.0): 0.5, (40.0, 20.0): 0.6,
                    (22.0, 40.0): 0.5, (38.0, 40.0): 0.6}
        cropper = self._cropper_with_fixed_variances(vars_map)
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert cropper._has_bilateral_texture_symmetry(image, _COHERENT_FACE_LANDMARKS) is False


class TestOpenCvHaarFallback:
    """
    Session 7 regression: facenet_pytorch wasn't importable in the user's
    active environment, and the cropper used to go silently unavailable --
    every frame of every video reported "no face", the trained classifier
    never ran anywhere, and AI videos were labelled REAL from texture
    alone with no hint anything was wrong. The cropper must now degrade to
    an OpenCV Haar cascade instead, and say that it did.
    """

    @staticmethod
    def _haar_cropper(boxes) -> FaceCropper:
        cropper = FaceCropper.__new__(FaceCropper)
        cropper.margin = 0.4
        cropper.min_face_size = 20
        cropper.min_confidence = 0.80
        cropper.available = True
        cropper.load_error = None
        cropper.backend = "opencv_haar"
        cropper._mtcnn = None

        class _FakeHaar:
            def __init__(self, out):
                self._out = np.array(out, dtype=np.float32).reshape(-1, 4)

            def detectMultiScale(self, gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)):
                return self._out if len(self._out) else []

        cropper._haar = _FakeHaar(boxes)
        return cropper

    def test_fallback_detects_box_without_landmarks_or_confidence(self):
        cropper = self._haar_cropper([[10, 10, 40, 48]])
        result = cropper.crop(np.zeros((120, 120, 3), dtype=np.uint8))
        assert result["face_found"] is True
        assert result["backend"] == "opencv_haar"
        assert result["confidence"] is None   # Haar gives no score
        assert result["landmarks"] is None    # landmark checks skipped, not failed
        assert result["crop"] is not None

    def test_fallback_rejects_elongated_limb_like_boxes(self):
        cropper = self._haar_cropper([[10, 10, 20, 80]])  # aspect ~4:1 -- a limb
        result = cropper.crop(np.zeros((200, 120, 3), dtype=np.uint8))
        assert result["face_found"] is False

    def test_fallback_no_detections_returns_clean_no_face(self):
        cropper = self._haar_cropper([])
        result = cropper.crop(np.zeros((120, 120, 3), dtype=np.uint8))
        assert result["face_found"] is False
        assert result["error"] is None
        assert result["backend"] == "opencv_haar"

    def test_load_falls_back_to_opencv_when_facenet_missing(self):
        import sys
        with patch.dict(sys.modules, {"facenet_pytorch": None}):
            cropper = FaceCropper()
        assert cropper.available is True
        assert cropper.backend == "opencv_haar"
        assert "facenet_pytorch" in (cropper.load_error or "")

    def test_unavailable_when_both_backends_missing(self):
        import sys
        with patch.dict(sys.modules, {"facenet_pytorch": None, "cv2": None}):
            cropper = FaceCropper()
        assert cropper.available is False
        assert cropper.load_error
        assert "facenet_pytorch" in cropper.load_error
        assert "opencv" in cropper.load_error
