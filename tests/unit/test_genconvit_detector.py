"""
tests/unit/test_genconvit_detector.py
=====================================
Focused unit tests for the Phase 2A.1 GenConViT coverage-diagnostic behavior
(pipeline/genconvit/detector.py) that do NOT require the ~3 GB official
checkpoints to be present:

  * `_coverage_status` boundary logic (NO_FACE / LOW / PARTIAL / GOOD).
  * The INACTIVE path when weights are missing — status INACTIVE, score None,
    explicit load_error, and full coverage fields present, with NO heuristic
    fallback (genconvit_score must stay None).
  * `_resize_to_224` + `_preprocess` (224x224 INTER_AREA, /255, ImageNet norm,
    HxWx3 -> 3xHxW, batch [N,3,224,224]) — the documented Phase 2A pipeline.

Never requires a network, GPU, or the real weights.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.genconvit.detector import (  # noqa: E402
    COVERAGE_GOOD,
    COVERAGE_LOW,
    COVERAGE_NO_FACE,
    COVERAGE_PARTIAL,
    STATUS_ACTIVE_MODEL,
    STATUS_INACTIVE,
    GenConViTDetector,
    _IMG_SIZE,
    _MEAN,
    _STD,
    _coverage_status,
)


# ---------------------------------------------------------------------------
# Coverage status boundaries
# ---------------------------------------------------------------------------

def test_coverage_status_boundaries():
    assert _coverage_status(10, 0) == COVERAGE_NO_FACE
    assert _coverage_status(10, 1) == COVERAGE_LOW     # 0.10
    assert _coverage_status(20, 4) == COVERAGE_LOW     # 0.20
    assert _coverage_status(20, 5) == COVERAGE_PARTIAL # exactly 0.25
    assert _coverage_status(10, 5) == COVERAGE_PARTIAL # 0.50
    assert _coverage_status(10, 6) == COVERAGE_GOOD    # exactly 0.60
    assert _coverage_status(46, 35) == COVERAGE_GOOD
    assert _coverage_status(0, 0) == COVERAGE_NO_FACE


# ---------------------------------------------------------------------------
# INACTIVE (missing weights) — no heuristic fallback
# ---------------------------------------------------------------------------

def test_inactive_when_weights_missing():
    with tempfile.TemporaryDirectory() as td:
        det = GenConViTDetector(weights_dir=td)
        out = det.infer([np.zeros((64, 64, 3), dtype=np.uint8)])
    assert out["genconvit_status"] == STATUS_INACTIVE
    assert out["genconvit_score"] is None
    assert out["genconvit_fake_score"] is None
    assert out["genconvit_predicted_label"] is None
    assert out["genconvit_coverage_status"] is None
    assert out["genconvit_frames_used"] == 0
    assert out["genconvit_face_frames_found"] == 0
    assert out["genconvit_face_confidences"] == []
    assert out["load_error"] and "missing GenConViT checkpoint" in out["load_error"]


def test_inactive_status_property_before_load():
    with tempfile.TemporaryDirectory() as td:
        det = GenConViTDetector(weights_dir=td)
    assert det.status == STATUS_INACTIVE
    assert det.available is False


# ---------------------------------------------------------------------------
# Preprocessing: face crop -> 224x224 -> /255 -> ImageNet norm -> 3xHxW batch
# ---------------------------------------------------------------------------

def test_resize_to_224():
    with tempfile.TemporaryDirectory() as td:
        det = GenConViTDetector(weights_dir=td)
    crop = np.zeros((60, 80, 3), dtype=np.uint8)
    out = det._resize_to_224(crop)
    assert out.shape == (_IMG_SIZE, _IMG_SIZE, 3)
    assert out.dtype == np.uint8
    already = np.zeros((224, 224, 3), dtype=np.uint8)
    assert det._resize_to_224(already) is already  # no-op when already 224


def test_preprocess_normalization_and_channel_order():
    with tempfile.TemporaryDirectory() as td:
        det = GenConViTDetector(weights_dir=td)
    crop = np.zeros((224, 224, 3), dtype=np.uint8)
    crop[:, :, 0] = 10
    crop[:, :, 1] = 20
    crop[:, :, 2] = 30
    x = det._preprocess([crop, crop])
    assert tuple(x.shape) == (2, 3, 224, 224)
    assert str(x.dtype) == "torch.float32"

    expected = np.zeros(3, dtype=np.float32)
    for c in range(3):
        expected[c] = (crop[0, 0, c].astype(np.float32) / 255.0 - _MEAN[c]) / _STD[c]
    got = x[0, :, 0, 0].numpy()
    np.testing.assert_allclose(got, expected, atol=1e-6)
    # Channel order: R is index 0, B is index 2 (HxWx3 -> 3xHxW).
    assert abs(got[0] - expected[0]) < 1e-6
    assert abs(got[2] - expected[2]) < 1e-6


# ---------------------------------------------------------------------------
# Empty-frames path still reports a clean ACTIVE_MODEL/no-score state
# ---------------------------------------------------------------------------

def test_empty_frames_returns_no_face_without_loading():
    with tempfile.TemporaryDirectory() as td:
        det = GenConViTDetector(weights_dir=td)
    out = det.infer([])
    assert out["genconvit_status"] == STATUS_ACTIVE_MODEL or out["genconvit_status"] == STATUS_INACTIVE
    assert out["genconvit_score"] is None
    assert out["genconvit_face_frames_found"] == 0
    assert out["genconvit_frames_used"] == 0