"""
tests/unit/test_aasist_detector.py
==================================
Focused unit tests for the Phase 2B AASIST wrapper
(pipeline/aasist/detector.py) that do NOT require the official checkpoint
(~1.2 MB) to be present on disk:

  * `_pad_window` — upstream data_utils.pad() semantics (truncate / tile).
  * `_window_waveform` — full-audio deterministic windowing boundaries.
  * `_label_for_spoof_score` — class-0=SPOOF / class-1=BONAFIDE mapping.
  * The INACTIVE paths — no audio track (None), missing wav file, and missing
    weights — all yield status INACTIVE + score None + an explicit reason,
    with NO heuristic fallback and NO fake 0.5.
  * Lazy-load status property.

Never requires a network, GPU, or the real weights.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.aasist.detector import (  # noqa: E402
    STATUS_ACTIVE_MODEL,
    STATUS_INACTIVE,
    _NB_SAMP,
    AASISTDetector,
    _label_for_spoof_score,
    _pad_window,
    _window_waveform,
)


# ---------------------------------------------------------------------------
# _pad_window : upstream data_utils.pad() clone
# ---------------------------------------------------------------------------

def test_pad_window_truncates_long_enough_input():
    x = np.ones(_NB_SAMP + 100, dtype=np.float32)
    out = _pad_window(x, _NB_SAMP)
    assert out.shape == (_NB_SAMP,)
    assert out.dtype == np.float32


def test_pad_window_tiles_short_input():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(16_000).astype(np.float32)  # exactly 1 s
    out = _pad_window(x, _NB_SAMP)
    assert out.shape == (_NB_SAMP,)
    one_repeat = np.tile(x, int(_NB_SAMP / 16_000) + 1)[:_NB_SAMP]
    np.testing.assert_array_equal(out, one_repeat)
    # Every tiled repeat is identical to the source window (deterministic).
    assert bool((out[:16_000] == x).all())


def test_pad_window_exact_length_is_passthrough():
    x = np.random.default_rng(3).standard_normal(_NB_SAMP).astype(np.float32)
    out = _pad_window(x, _NB_SAMP)
    assert out.shape == (_NB_SAMP,)
    np.testing.assert_array_equal(out, x)


# ---------------------------------------------------------------------------
# _window_waveform : deterministic full-audio windowing
# ---------------------------------------------------------------------------

def test_window_waveform_short_clip_yields_one_tiled_window():
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(500).astype(np.float32)  # < one window
    wins = _window_waveform(wav, _NB_SAMP)
    assert len(wins) == 1
    assert wins[0].shape == (_NB_SAMP,)


def test_window_waveform_exact_multiple():
    rng = np.random.default_rng(1)
    n = 3
    wav = rng.standard_normal(n * _NB_SAMP).astype(np.float32)
    wins = _window_waveform(wav, _NB_SAMP)
    assert len(wins) == n
    assert all(w.shape == (_NB_SAMP,) for w in wins)
    # Non-overlapping: windows tile the timeline exactly.
    np.testing.assert_array_equal(wins[1], wav[_NB_SAMP:2 * _NB_SAMP])


def test_window_waveform_tail_is_tile_padded():
    rng = np.random.default_rng(2)
    tail_len = 100
    wav = rng.standard_normal(2 * _NB_SAMP + tail_len).astype(np.float32)
    wins = _window_waveform(wav, _NB_SAMP)
    assert len(wins) == 3
    np.testing.assert_array_equal(wins[2][:tail_len], wav[2 * _NB_SAMP:])
    assert wins[2].shape == (_NB_SAMP,)


def test_window_waveform_empty():
    assert _window_waveform(np.array([], dtype=np.float32), _NB_SAMP) == []


# ---------------------------------------------------------------------------
# Label mapping : class 0 = SPOOF, class 1 = BONAFIDE
# ---------------------------------------------------------------------------

def test_label_mapping():
    assert _label_for_spoof_score(0.999) == "SPOOF"
    assert _label_for_spoof_score(0.5) == "SPOOF"      # argmax tie -> class 0
    assert _label_for_spoof_score(0.499) == "BONAFIDE"
    assert _label_for_spoof_score(0.0) == "BONAFIDE"


# ---------------------------------------------------------------------------
# Lazy-load + INACTIVE paths (no weights, no audio) — no fallback, no 0.5
# ---------------------------------------------------------------------------

def test_inactive_status_property_before_load():
    with tempfile.TemporaryDirectory() as td:
        det = AASISTDetector(weights_dir=td)
    assert det.status == STATUS_INACTIVE
    assert det.available is False


def test_inactive_when_weights_missing():
    with tempfile.TemporaryDirectory() as td:
        det = AASISTDetector(weights_dir=td)
        wav = np.zeros(_NB_SAMP, dtype=np.float32)
        out = det.infer_waveform(wav, sample_rate=16_000)
    assert out["aasist_status"] == STATUS_INACTIVE
    assert out["aasist_spoof_score"] is None
    assert out["aasist_bonafide_score"] is None
    assert out["aasist_predicted_label"] is None
    assert out["aasist_window_count"] == 0
    assert out["aasist_load_error"] and "missing AASIST checkpoint" in out["aasist_load_error"]
    # No heuristic fallback: the spoof score must stay None.
    assert out["aasist_note"] is not None


def test_inactive_when_no_audio_track():
    with tempfile.TemporaryDirectory() as td:
        det = AASISTDetector(weights_dir=td)
        out = det.infer_wav(None)
    assert out["aasist_status"] == STATUS_INACTIVE
    assert out["aasist_spoof_score"] is None
    assert out["aasist_predicted_label"] is None
    assert out["aasist_audio_readable"] is False
    assert out["aasist_note"] and "No audio track" in out["aasist_note"]


def test_inactive_when_wav_missing():
    with tempfile.TemporaryDirectory() as td:
        det = AASISTDetector(weights_dir=td)
        out = det.infer_wav(Path(td) / "does_not_exist.wav")
    assert out["aasist_status"] == STATUS_INACTIVE
    assert out["aasist_spoof_score"] is None
    assert "not found" in (out["aasist_note"] or "")


def test_lazy_load_flag_set_after_failed_attempt():
    # ensure_loaded() failure is cached (no retry churn in a long eval loop).
    with tempfile.TemporaryDirectory() as td:
        det = AASISTDetector(weights_dir=td)
        assert det.ensure_loaded() is False
        assert det._loaded is True
        assert det.status == STATUS_INACTIVE