"""
tests/unit/test_config_and_types.py
─────────────────────────────────────
Fast smoke tests — no model weights or GPU needed.
"""

import sys
from pathlib import Path

import pytest

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.config import CFG
from utils.types import AnalysisResult, FrameResult, AudioSegmentResult, ModalResult, Verdict


class TestConfig:
    def test_config_loads(self):
        assert CFG is not None

    def test_video_fps_range(self):
        # config.yaml uses `frame_rate`, not `fps` (see pipeline/video/detector.py's
        # `getattr(cfg.video, "frame_rate", 1.0)` usage) — the video-sampling
        # rate, not a display framerate, so it's intentionally low (1-5 fps).
        fps = int(CFG.video.frame_rate)
        assert 1 <= fps <= 5

    def test_fusion_weights_sum_to_one(self):
        w_v = float(CFG.fusion.video_weight)
        w_a = float(CFG.fusion.audio_weight)
        assert abs((w_v + w_a) - 1.0) < 1e-6

    def test_verdict_thresholds_in_range_and_ordered(self):
        # config.yaml splits the verdict boundary into two thresholds
        # (real_threshold / fake_threshold) with an UNCERTAIN band between
        # them, rather than a single verdict_threshold (see Fuser._assign_verdict).
        real_t = float(CFG.fusion.real_threshold)
        fake_t = float(CFG.fusion.fake_threshold)
        assert 0.0 < real_t < fake_t < 1.0


class TestTypes:
    def test_frame_result_creation(self):
        fr = FrameResult(frame_index=0, timestamp_sec=0.0, fake_prob=0.8, face_detected=True)
        assert fr.fake_prob == 0.8
        assert fr.face_detected is True   # ← was has_face

    def test_audio_segment_result_creation(self):
        ar = AudioSegmentResult(segment_index=0, start_sec=0.0, end_sec=4.0, fake_prob=0.3)
        assert ar.end_sec - ar.start_sec == 4.0

    def test_analysis_result_requires_core_fields(self):
        # AnalysisResult has no defaults for verdict/fused_score/video_result/
        # audio_result — a result is meant to always represent a completed
        # (or explicitly errored) analysis, never a partially-built stub, so
        # omitting them is a TypeError by design rather than silently giving
        # None. Only `metadata` has a default (empty dict).
        with pytest.raises(TypeError):
            AnalysisResult(video_id="test-123")

        result = AnalysisResult(
            video_id="test-123", verdict=Verdict.REAL, fused_score=0.1,
            video_result=ModalResult(modality="video", fake_prob=0.1),
            audio_result=ModalResult(modality="audio", fake_prob=0.1),
        )
        assert result.metadata == {}

    def test_verdict_enum_values(self):
        assert Verdict.REAL.value == "REAL"
        assert Verdict.FAKE.value == "FAKE"
        assert Verdict.UNCERTAIN.value == "UNCERTAIN"
