"""
tests/unit/test_main.py — Sprint 5 unit tests for main.py

Coverage targets (from Sprint 5 spec):
  ✓ CLI parsing — required --video, optional --output choices, --analysis-id
  ✓ Happy-path FAKE verdict  (fused_score in FAKE zone)
  ✓ Happy-path REAL verdict  (fused_score in REAL zone)
  ✓ Happy-path UNCERTAIN verdict
  ✓ Missing video file → exit code 255, no pipeline called
  ✓ Demux failure (DemuxError) → exit code 255
  ✓ DB write called when output=db
  ✓ DB write called when output=both
  ✓ DB write NOT called when output=json
  ✓ Exit code 0 for REAL
  ✓ Exit code 1 for FAKE
  ✓ Exit code 2 for UNCERTAIN
  ✓ JSON output structure / required keys
  ✓ JSON output contains verdict and fused_score
  ✓ --output json prints valid JSON to stdout
"""
from __future__ import annotations

import json
import sys
import uuid
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make the project root importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.types import AnalysisResult, ModalResult, Verdict


# ---------------------------------------------------------------------------
# Fixtures helpers
# ---------------------------------------------------------------------------

def _modal(modality: str, score: float) -> ModalResult:
    return ModalResult(modality=modality, fake_prob=score)


def _analysis(verdict: Verdict, fused: float, vid_path: str = "test.mp4") -> AnalysisResult:
    aid = str(uuid.uuid4())
    return AnalysisResult(
        video_id=vid_path,
        verdict=verdict,
        fused_score=fused,
        video_result=_modal("video", fused),
        audio_result=_modal("audio", fused),
        metadata={
            "run_id": aid,
            "video_path": vid_path,
            "video_score": fused,
            "audio_score": fused,
            "calibrated_video_score": fused,
            "calibrated_audio_score": fused,
            "verdict_timeline": [],
        },
    )


# ---------------------------------------------------------------------------
# 1. CLI parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def test_video_arg_required(self):
        from main import parse_args
        with pytest.raises(SystemExit):
            parse_args([])

    def test_video_arg_captured(self):
        from main import parse_args
        args = parse_args(["--video", "clip.mp4"])
        assert args.video == "clip.mp4"

    def test_output_default_is_json(self):
        from main import parse_args
        args = parse_args(["--video", "clip.mp4"])
        assert args.output == "json"

    def test_output_choices(self):
        from main import parse_args
        for choice in ("json", "db", "both"):
            args = parse_args(["--video", "clip.mp4", "--output", choice])
            assert args.output == choice

    def test_output_invalid_choice_raises(self):
        from main import parse_args
        with pytest.raises(SystemExit):
            parse_args(["--video", "clip.mp4", "--output", "csv"])

    def test_analysis_id_passthrough(self):
        from main import parse_args
        aid = "my-custom-id"
        args = parse_args(["--video", "clip.mp4", "--analysis-id", aid])
        assert args.analysis_id == aid


# ---------------------------------------------------------------------------
# 2. Exit codes
# ---------------------------------------------------------------------------

class TestExitCodes:
    def _run_with_result(self, result: AnalysisResult, output: str = "json") -> int:
        from main import main
        video = result.metadata["video_path"]
        with (
            patch("main.Path.exists", return_value=True),
            patch("main.run_pipeline", return_value=result),
            patch("main.print_summary"),
            patch("main.print_json"),
            patch("main._db_module.save_result"),
        ):
            return main(["--video", video, "--output", output])

    def test_exit_0_for_real(self):
        result = _analysis(Verdict.REAL, fused=0.20)
        assert self._run_with_result(result) == 0

    def test_exit_1_for_fake(self):
        result = _analysis(Verdict.FAKE, fused=0.80)
        assert self._run_with_result(result) == 1

    def test_exit_2_for_uncertain(self):
        result = _analysis(Verdict.UNCERTAIN, fused=0.50)
        assert self._run_with_result(result) == 2


# ---------------------------------------------------------------------------
# 3. Missing video file
# ---------------------------------------------------------------------------

class TestMissingVideo:
    def test_missing_file_returns_error_code(self, tmp_path):
        from main import main
        nonexistent = str(tmp_path / "ghost.mp4")
        code = main(["--video", nonexistent])
        assert code == 255

    def test_missing_file_does_not_call_pipeline(self, tmp_path):
        from main import main
        nonexistent = str(tmp_path / "ghost.mp4")
        with patch("main.run_pipeline") as mock_pipeline:
            main(["--video", nonexistent])
        mock_pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Demux failure
# ---------------------------------------------------------------------------

class TestDemuxFailure:
    def test_demux_error_returns_error_code(self, tmp_path):
        from main import main
        from utils.demux import DemuxError

        fake_vid = tmp_path / "bad.mp4"
        fake_vid.write_bytes(b"\x00" * 16)

        with patch("main.run_pipeline", side_effect=DemuxError("ffmpeg crashed")):
            code = main(["--video", str(fake_vid)])

        assert code == 255

    def test_demux_error_prints_to_stderr(self, tmp_path, capsys):
        from main import main
        from utils.demux import DemuxError

        fake_vid = tmp_path / "bad.mp4"
        fake_vid.write_bytes(b"\x00" * 16)

        with patch("main.run_pipeline", side_effect=DemuxError("ffmpeg crashed")):
            main(["--video", str(fake_vid)])

        captured = capsys.readouterr()
        assert "demux" in captured.err.lower() or "error" in captured.err.lower()


# ---------------------------------------------------------------------------
# 5. DB write behaviour
# ---------------------------------------------------------------------------

class TestDBWrite:
    def _run(self, output: str, result: AnalysisResult) -> MagicMock:
        from main import main
        with (
            patch("main.Path.exists", return_value=True),
            patch("main.run_pipeline", return_value=result),
            patch("main.print_summary"),
            patch("main.print_json"),
            patch("main._db_module.save_result") as mock_save,
        ):
            main(["--video", "test.mp4", "--output", output])
        return mock_save

    def test_db_write_called_for_output_db(self):
        result = _analysis(Verdict.FAKE, fused=0.85)
        mock_save = self._run("db", result)
        mock_save.assert_called_once_with(result)

    def test_db_write_called_for_output_both(self):
        result = _analysis(Verdict.REAL, fused=0.10)
        mock_save = self._run("both", result)
        mock_save.assert_called_once_with(result)

    def test_db_write_not_called_for_output_json(self):
        result = _analysis(Verdict.UNCERTAIN, fused=0.50)
        mock_save = self._run("json", result)
        mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# 6. JSON output format
# ---------------------------------------------------------------------------

class TestJSONOutput:
    _REQUIRED_KEYS = {
        "analysis_id",
        "verdict",
        "fused_score",
        "video_score",
        "audio_score",
        "calibrated_video_score",
        "calibrated_audio_score",
        "video_frames_analysed",
        "audio_segments_analysed",
        "metadata",
    }

    def test_json_contains_required_keys(self):
        result = _analysis(Verdict.FAKE, fused=0.75)
        from main import _result_to_dict
        d = _result_to_dict(result)
        assert self._REQUIRED_KEYS.issubset(d.keys())

    def test_json_verdict_matches_result(self):
        for verdict, fused in [(Verdict.REAL, 0.2), (Verdict.FAKE, 0.8), (Verdict.UNCERTAIN, 0.5)]:
            result = _analysis(verdict, fused)
            from main import _result_to_dict
            d = _result_to_dict(result)
            assert d["verdict"] == verdict.value

    def test_json_fused_score_is_float(self):
        result = _analysis(Verdict.FAKE, fused=0.91)
        from main import _result_to_dict
        d = _result_to_dict(result)
        assert isinstance(d["fused_score"], float)

    def test_print_json_produces_valid_json(self, capsys):
        result = _analysis(Verdict.REAL, fused=0.15)
        from main import print_json
        print_json(result)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["verdict"] == "REAL"

    def test_json_analysis_id_matches(self):
        result = _analysis(Verdict.UNCERTAIN, fused=0.49)
        from main import _result_to_dict
        d = _result_to_dict(result)
        assert d["analysis_id"] == result.analysis_id


# ---------------------------------------------------------------------------
# 7. Happy-path integration
# ---------------------------------------------------------------------------

class TestHappyPath:
    def _run(self, verdict: Verdict, fused: float) -> tuple[int, AnalysisResult]:
        from main import main
        result = _analysis(verdict, fused)
        with (
            patch("main.Path.exists", return_value=True),
            patch("main.run_pipeline", return_value=result),
            patch("main.print_summary"),
            patch("main.print_json"),
            patch("main._db_module.save_result"),
        ):
            code = main(["--video", "test.mp4"])
        return code, result

    def test_fake_verdict_exit_code(self):
        code, _ = self._run(Verdict.FAKE, 0.88)
        assert code == 1

    def test_real_verdict_exit_code(self):
        code, _ = self._run(Verdict.REAL, 0.05)
        assert code == 0

    def test_uncertain_verdict_exit_code(self):
        code, _ = self._run(Verdict.UNCERTAIN, 0.50)
        assert code == 2