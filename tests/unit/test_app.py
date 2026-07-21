"""tests/unit/test_app.py — Sprint 6 unit tests for app.py.

Run with:
    py -m pytest tests/unit/test_app.py -v

Strategy
--------
* Import only the pure helper functions from app (no Streamlit runtime needed).
* For sections that call st.* we patch streamlit wholesale so no server is required.
* main.run_pipeline and main._result_to_dict are always mocked.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Stub out streamlit before app.py is imported so we never need a real server
# ---------------------------------------------------------------------------

def _make_streamlit_stub() -> types.ModuleType:
    """Return a minimal streamlit stub with trackable mock methods."""
    st = types.ModuleType("streamlit")

    # Simple attribute stubs
    for attr in (
        "set_page_config", "title", "caption", "divider", "subheader",
        "header", "info", "error", "metric", "image", "plotly_chart",
        "download_button", "file_uploader", "markdown", "table", "json",
    ):
        setattr(st, attr, MagicMock())

    # expander context manager
    _expander_mock = MagicMock()
    _expander_mock.__enter__ = MagicMock(return_value=_expander_mock)
    _expander_mock.__exit__  = MagicMock(return_value=False)
    st.expander = MagicMock(return_value=_expander_mock)

    # Progress bar
    progress_mock = MagicMock()
    progress_mock.progress = MagicMock()
    st.progress = MagicMock(return_value=progress_mock)

    # columns returns a list of context-manager mocks
    def _columns(n, *args, **kwargs):
        mocks = []
        for _ in range(n if isinstance(n, int) else len(n)):
            m = MagicMock()
            m.__enter__ = MagicMock(return_value=m)
            m.__exit__  = MagicMock(return_value=False)
            mocks.append(m)
        return mocks

    st.columns = MagicMock(side_effect=_columns)

    # cache_resource / cache_data decorators (session 4: used on
    # _load_ai_classifier) — just return the function unmodified so caching
    # semantics don't matter in unit tests.
    def _passthrough_decorator(fn=None, **kw):
        if fn is not None:
            return fn
        return lambda f: f

    st.cache_resource = MagicMock(side_effect=_passthrough_decorator)
    st.cache_data     = MagicMock(side_effect=_passthrough_decorator)
    st.warning        = MagicMock()
    return st


_ST_STUB = _make_streamlit_stub()
sys.modules["streamlit"] = _ST_STUB


# ---------------------------------------------------------------------------
# Stub out main so importing app.py does not need the real pipeline
# ---------------------------------------------------------------------------

import main as _real_main_module  # noqa: E402  (captured so it can be restored below)

_MAIN_STUB = types.ModuleType("main")
_MAIN_STUB.run_pipeline   = MagicMock()
_MAIN_STUB._result_to_dict = MagicMock(return_value={"stub": True})
sys.modules["main"] = _MAIN_STUB

# Stub plotly as well (we test our wrappers, not plotly itself)
_PLOTLY_GO = types.ModuleType("plotly.graph_objects")


class _FakeFigure:
    def __init__(self, *a, **kw): pass
    def update_layout(self, **kw): return self
    def add_hline(self, **kw):    return self
    def add_trace(self, *a, **kw): return self


class _FakeIndicator:
    def __init__(self, *a, **kw): pass


class _FakeBar:
    def __init__(self, *a, **kw): pass


class _FakeScatter:
    def __init__(self, *a, **kw): pass


_PLOTLY_GO.Figure    = _FakeFigure
_PLOTLY_GO.Indicator = _FakeIndicator
_PLOTLY_GO.Bar       = _FakeBar
_PLOTLY_GO.Scatter   = _FakeScatter

sys.modules["plotly"]              = types.ModuleType("plotly")
sys.modules["plotly.graph_objects"] = _PLOTLY_GO

# ---------------------------------------------------------------------------
# Stub pipeline.video.detector.ImageDetector before app.py imports it.
#
# NOTE: app.py now also imports `pipeline.forensics.explain` and
# `pipeline.forensics.ai_classifier.AIClassifier` directly (session 4). Those
# are lightweight, pure-Python modules (no heavy ML deps at import time), so
# we let them import for real rather than stubbing the whole `pipeline`
# package wholesale — replacing the top-level `pipeline` module previously
# broke every submodule import that wasn't explicitly re-created here.
# Only `pipeline.video.detector` (which pulls in numpy-heavy per-frame
# fusion) is stubbed, and only its `ImageDetector` symbol, which is what
# app.py actually imports (the old `detect_image` module function this test
# file previously patched no longer exists in app.py's current API).
# ---------------------------------------------------------------------------

import pipeline  # noqa: E402  (real package, so pipeline.forensics etc. resolve)
import pipeline.video  # noqa: E402
import pipeline.video.detector as _real_video_detector_module  # noqa: E402

_DETECTOR_STUB                = types.ModuleType("pipeline.video.detector")
_DETECTOR_STUB.ImageDetector  = MagicMock()
sys.modules["pipeline.video.detector"] = _DETECTOR_STUB
pipeline.video.detector = _DETECTOR_STUB

# Also stub st.tabs (new in Sprint 7)
def _make_tab_mock(label: str):
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__  = MagicMock(return_value=False)
    return m

_ST_STUB.tabs = MagicMock(
    side_effect=lambda labels: [_make_tab_mock(l) for l in labels]
)

# ---------------------------------------------------------------------------
# Now import app — all external deps are stubbed
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import app  # noqa: E402  (must come after stubs)

# ---------------------------------------------------------------------------
# Restore the real pipeline.video.detector module in sys.modules now that
# app.py's own `from pipeline.video.detector import ImageDetector` has
# already bound the *stubbed* ImageDetector into app's namespace. Leaving
# the stub in sys.modules would otherwise leak into every other test module
# that imports pipeline.video.detector for real in the same pytest process
# (e.g. test_video_detector.py), since sys.modules is shared/global.
# ---------------------------------------------------------------------------
sys.modules["pipeline.video.detector"] = _real_video_detector_module
pipeline.video.detector = _real_video_detector_module
sys.modules["main"] = _real_main_module

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

from utils.types import (  # noqa: E402
    FrameResult,
    AudioSegmentResult,
    ModalResult,
    AnalysisResult,
    ImageResult,
    Verdict,
)


def _make_frame(i: int, ts: float, prob: float, overlay: bytes | None = None) -> FrameResult:
    return FrameResult(
        frame_index=i,
        timestamp_sec=ts,
        fake_prob=prob,
        face_detected=True,
        gradcam_overlay=overlay,
    )


def _make_audio_seg(i: int, start: float, end: float, prob: float) -> AudioSegmentResult:
    return AudioSegmentResult(segment_index=i, start_sec=start, end_sec=end, fake_prob=prob)


@pytest.fixture()
def fake_video_result() -> ModalResult:
    frames = [
        _make_frame(0, 0.0, 0.8, b"\x89PNG"),
        _make_frame(1, 0.5, 0.7, b"\x89PNG"),
        _make_frame(2, 1.0, 0.3, None),          # no overlay
        _make_frame(3, 1.5, 0.9, b"\x89PNG"),
    ]
    return ModalResult(modality="video", fake_prob=0.75, frame_results=frames)


@pytest.fixture()
def fake_audio_result() -> ModalResult:
    segs = [
        _make_audio_seg(0, 0.0,  1.0, 0.2),
        _make_audio_seg(1, 1.0,  2.0, 0.8),
        _make_audio_seg(2, 2.0,  3.0, 0.5),
    ]
    return ModalResult(modality="audio", fake_prob=0.5, audio_results=segs)


@pytest.fixture()
def fake_result(fake_video_result, fake_audio_result) -> AnalysisResult:
    return AnalysisResult(
        video_id="vid_001",
        verdict=Verdict.FAKE,
        fused_score=0.82,
        video_result=fake_video_result,
        audio_result=fake_audio_result,
        metadata={
            "run_id": "run_001",
            "video_path": "/tmp/test.mp4",
            "video_score": 0.75,
            "audio_score": 0.50,
            "calibrated_video_score": 0.77,
            "calibrated_audio_score": 0.51,
            "verdict_timeline": [(0.0, 0.3), (1.0, 0.6), (2.0, 0.82)],
        },
    )


# ---------------------------------------------------------------------------
# 1. Verdict colour logic
# ---------------------------------------------------------------------------

class TestVerdictColor:
    def test_real_is_green(self):
        assert app._verdict_color("REAL") == "green"

    def test_fake_is_red(self):
        assert app._verdict_color("FAKE") == "red"

    def test_uncertain_is_orange(self):
        assert app._verdict_color("UNCERTAIN") == "orange"

    def test_unknown_verdict_falls_back_to_gray(self):
        assert app._verdict_color("UNKNOWN_LABEL") == "gray"


# ---------------------------------------------------------------------------
# 2. Verdict icon logic
# ---------------------------------------------------------------------------

class TestVerdictIcon:
    def test_real_icon(self):
        assert app._verdict_icon("REAL") == "✅"

    def test_fake_icon(self):
        assert app._verdict_icon("FAKE") == "🚨"

    def test_uncertain_icon(self):
        assert app._verdict_icon("UNCERTAIN") == "⚠️"


# ---------------------------------------------------------------------------
# 3. Gauge helper
# ---------------------------------------------------------------------------

class TestBuildGauge:
    def test_returns_figure(self):
        fig = app._build_gauge(0.5)
        assert isinstance(fig, _FakeFigure)

    def test_accepts_zero(self):
        app._build_gauge(0.0)   # must not raise

    def test_accepts_one(self):
        app._build_gauge(1.0)   # must not raise


# ---------------------------------------------------------------------------
# 4. render_verdict_badge — calls st.markdown and st.metric
# ---------------------------------------------------------------------------

class TestRenderVerdictBadge:
    def setup_method(self):
        _ST_STUB.markdown.reset_mock()
        _ST_STUB.metric.reset_mock()

    def test_calls_markdown_with_verdict(self):
        app.render_verdict_badge("FAKE", 0.82)
        html = _ST_STUB.markdown.call_args[0][0]
        assert "FAKE" in html

    def test_calls_metric_with_score(self):
        app.render_verdict_badge("REAL", 0.12)
        _ST_STUB.metric.assert_called_once()
        kw = _ST_STUB.metric.call_args[1]
        assert "12" in kw.get("value", "")       # "12.00%" contains "12"

    def test_badge_colour_for_real(self):
        app.render_verdict_badge("REAL", 0.1)
        html = _ST_STUB.markdown.call_args[0][0]
        assert "green" in html

    def test_badge_colour_for_fake(self):
        app.render_verdict_badge("FAKE", 0.9)
        html = _ST_STUB.markdown.call_args[0][0]
        assert "red" in html


# ---------------------------------------------------------------------------
# 5. render_gauge — calls st.plotly_chart
# ---------------------------------------------------------------------------

class TestRenderGauge:
    def setup_method(self):
        _ST_STUB.plotly_chart.reset_mock()

    def test_calls_plotly_chart(self):
        app.render_gauge(0.7)
        _ST_STUB.plotly_chart.assert_called_once()


# ---------------------------------------------------------------------------
# 6. render_gradcam_grid — skips frames with None overlay
# ---------------------------------------------------------------------------

class TestRenderGradcamGrid:
    def setup_method(self):
        _ST_STUB.image.reset_mock()
        _ST_STUB.info.reset_mock()

    def test_skips_none_overlays(self, fake_video_result):
        # fixture has 3 overlays, 1 None
        app.render_gradcam_grid(fake_video_result)
        assert _ST_STUB.image.call_count == 3

    def test_empty_overlays_shows_info(self):
        modal = ModalResult(modality="video", fake_prob=0.5, frame_results=[
            _make_frame(0, 0.0, 0.5, None),
        ])
        app.render_gradcam_grid(modal)
        _ST_STUB.info.assert_called_once()
        _ST_STUB.image.assert_not_called()

    def test_no_frames_shows_info(self):
        modal = ModalResult(modality="video", fake_prob=0.5)
        app.render_gradcam_grid(modal)
        _ST_STUB.info.assert_called_once()


# ---------------------------------------------------------------------------
# 7. render_audio_section — calls plotly_chart twice (bar + waveform)
# ---------------------------------------------------------------------------

class TestRenderAudioSection:
    def setup_method(self):
        _ST_STUB.plotly_chart.reset_mock()
        _ST_STUB.info.reset_mock()

    def test_renders_two_charts(self, fake_audio_result):
        app.render_audio_section(fake_audio_result)
        assert _ST_STUB.plotly_chart.call_count == 2

    def test_empty_segments_shows_info(self):
        modal = ModalResult(modality="audio", fake_prob=0.5)
        app.render_audio_section(modal)
        _ST_STUB.info.assert_called_once()
        _ST_STUB.plotly_chart.assert_not_called()


# ---------------------------------------------------------------------------
# 8. render_per_frame_chart
# ---------------------------------------------------------------------------

class TestRenderPerFrameChart:
    def setup_method(self):
        _ST_STUB.plotly_chart.reset_mock()
        _ST_STUB.info.reset_mock()

    def test_renders_chart(self, fake_video_result):
        app.render_per_frame_chart(fake_video_result)
        _ST_STUB.plotly_chart.assert_called_once()

    def test_empty_frames_shows_info(self):
        modal = ModalResult(modality="video", fake_prob=0.5)
        app.render_per_frame_chart(modal)
        _ST_STUB.info.assert_called_once()
        _ST_STUB.plotly_chart.assert_not_called()


# ---------------------------------------------------------------------------
# 9. render_verdict_timeline
# ---------------------------------------------------------------------------

class TestRenderVerdictTimeline:
    def setup_method(self):
        _ST_STUB.plotly_chart.reset_mock()
        _ST_STUB.info.reset_mock()

    def test_renders_chart_when_data_present(self, fake_result):
        app.render_verdict_timeline(fake_result.metadata)
        _ST_STUB.plotly_chart.assert_called_once()

    def test_empty_timeline_shows_info(self):
        app.render_verdict_timeline({})
        _ST_STUB.info.assert_called_once()
        _ST_STUB.plotly_chart.assert_not_called()

    def test_empty_list_shows_info(self):
        app.render_verdict_timeline({"verdict_timeline": []})
        _ST_STUB.info.assert_called_once()


# ---------------------------------------------------------------------------
# 10. render_download_button — correct file name and MIME type
# ---------------------------------------------------------------------------

class TestRenderDownloadButton:
    def setup_method(self):
        _ST_STUB.download_button.reset_mock()
        _MAIN_STUB._result_to_dict.reset_mock()
        _MAIN_STUB._result_to_dict.return_value = {"stub": True}

    def test_download_button_called(self, fake_result):
        app.render_download_button(fake_result)
        _ST_STUB.download_button.assert_called_once()

    def test_download_mime_type(self, fake_result):
        app.render_download_button(fake_result)
        kw = _ST_STUB.download_button.call_args[1]
        assert kw.get("mime") == "application/json"

    def test_download_filename_contains_video_id(self, fake_result):
        # AnalysisResult.analysis_id prefers metadata["run_id"] over video_id
        # (see utils/types.py) — the fixture sets run_id="run_001", so that's
        # what should appear in the filename, not the raw video_id.
        app.render_download_button(fake_result)
        kw = _ST_STUB.download_button.call_args[1]
        assert "run_001" in kw.get("file_name", "")

    def test_download_data_is_valid_json(self, fake_result):
        app.render_download_button(fake_result)
        kw  = _ST_STUB.download_button.call_args[1]
        raw = kw.get("data", b"")
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_result_to_dict_called_once(self, fake_result):
        app.render_download_button(fake_result)
        _MAIN_STUB._result_to_dict.assert_called_once_with(fake_result)


# ---------------------------------------------------------------------------
# 11. Pipeline called exactly once per upload
# ---------------------------------------------------------------------------

class TestPipelineCalledOnce:
    def setup_method(self):
        _MAIN_STUB.run_pipeline.reset_mock()

    def test_run_pipeline_called_once(self, fake_result):
        _MAIN_STUB.run_pipeline.return_value = fake_result
        _ST_STUB.progress.return_value = MagicMock()
        app.run_pipeline_with_progress("/tmp/test.mp4")
        # app.py now calls main.run_pipeline(video_path=Path(...)) (session 4:
        # main.run_pipeline needs a Path, and the arg is passed by keyword).
        _MAIN_STUB.run_pipeline.assert_called_once_with(video_path=Path("/tmp/test.mp4"))


# ---------------------------------------------------------------------------
# 12. Progress bar updates through stages
# ---------------------------------------------------------------------------

class TestProgressUpdates:
    def test_progress_called_multiple_times(self, fake_result):
        prog_mock = MagicMock()
        _ST_STUB.progress.return_value = prog_mock
        _MAIN_STUB.run_pipeline.return_value = fake_result

        app.run_pipeline_with_progress("/tmp/test.mp4")

        # Should be called at least 4 times (stages + final 1.0)
        assert prog_mock.progress.call_count >= 4

    def test_progress_reaches_100_percent(self, fake_result):
        prog_mock = MagicMock()
        _ST_STUB.progress.return_value = prog_mock
        _MAIN_STUB.run_pipeline.return_value = fake_result

        app.run_pipeline_with_progress("/tmp/test.mp4")

        # Last call should set value to 1.0
        last_val = prog_mock.progress.call_args_list[-1][0][0]
        assert last_val == 1.0


# ---------------------------------------------------------------------------
# 13. Error state when pipeline raises
# ---------------------------------------------------------------------------

class TestPipelineError:
    def setup_method(self):
        _ST_STUB.error.reset_mock()
        _ST_STUB.file_uploader.reset_mock()
        _MAIN_STUB.run_pipeline.reset_mock()

    def test_error_shown_when_pipeline_fails(self):
        _MAIN_STUB.run_pipeline.side_effect = RuntimeError("model load failed")
        prog_mock = MagicMock()
        _ST_STUB.progress.return_value = prog_mock

        with pytest.raises(RuntimeError):
            # run_pipeline_with_progress propagates after pipeline call
            app.run_pipeline_with_progress("/tmp/bad.mp4")


# ---------------------------------------------------------------------------
# 14. File upload handling — no upload returns early (no pipeline call)
# ---------------------------------------------------------------------------

class TestFileUploadHandling:
    def setup_method(self):
        _MAIN_STUB.run_pipeline.reset_mock()
        _ST_STUB.info.reset_mock()
        _ST_STUB.file_uploader.reset_mock()

    def test_no_upload_does_not_call_pipeline(self):
        _ST_STUB.file_uploader.return_value = None
        app.main_app()
        _MAIN_STUB.run_pipeline.assert_not_called()

    def test_no_upload_shows_info_prompt(self):
        _ST_STUB.file_uploader.return_value = None
        app.main_app()
        _ST_STUB.info.assert_called()


# ---------------------------------------------------------------------------
# 15. Gauge value is passed through correctly
# ---------------------------------------------------------------------------

class TestGaugeValue:
    def test_gauge_built_with_correct_score(self, fake_result):
        # _build_gauge is a pure helper; verify it receives the exact score
        with patch.object(app, "_build_gauge", wraps=app._build_gauge) as spy:
            _ST_STUB.plotly_chart.reset_mock()
            app.render_gauge(0.77)
            spy.assert_called_once_with(0.77)


# ===========================================================================
# Sprint 7 — Image tab tests
# ===========================================================================

@pytest.fixture()
def fake_image_result() -> ImageResult:
    return ImageResult(
        image_id="portrait",
        verdict=Verdict.FAKE,
        fused_score=0.74,
        gradcam_score=0.82,
        freq_score=0.61,
        gradcam_overlay=b"\x89PNG\r\ngradcam",
        freq_heatmap=b"\x89PNG\r\nfft",
        metadata={
            "run_id":         "img-run-001",
            "image_path":     "/tmp/portrait.jpg",
            "width":          512,
            "height":         512,
            "face_detected":  True,
            "gradcam_weight": 0.6,
            "freq_weight":    0.4,
        },
    )


# ---------------------------------------------------------------------------
# 16. render_image_score_breakdown — one plotly_chart call
# ---------------------------------------------------------------------------

class TestRenderImageScoreBreakdown:
    # render_image_score_breakdown now renders 4 st.metric cards (via
    # st.columns(4)) rather than a plotly chart — render_image_score_chart
    # is the separate plotly-based view. Test the current metric-card
    # behavior instead of the stale plotly_chart expectation.
    def setup_method(self):
        _ST_STUB.columns.reset_mock()

    def test_renders_four_metric_columns(self, fake_image_result):
        app.render_image_score_breakdown(fake_image_result)
        _ST_STUB.columns.assert_called_once_with(4)


# ---------------------------------------------------------------------------
# 17. render_image_overlays — two st.image calls when both overlays present
# ---------------------------------------------------------------------------

class TestRenderImageOverlays:
    def setup_method(self):
        _ST_STUB.image.reset_mock()
        _ST_STUB.info.reset_mock()

    def test_shows_both_overlays(self, fake_image_result):
        app.render_image_overlays(fake_image_result)
        assert _ST_STUB.image.call_count == 2

    def test_info_shown_when_gradcam_overlay_missing(self, fake_image_result):
        fake_image_result.gradcam_overlay = None
        app.render_image_overlays(fake_image_result)
        _ST_STUB.info.assert_called()

    def test_info_shown_when_freq_heatmap_missing(self, fake_image_result):
        fake_image_result.freq_heatmap = None
        app.render_image_overlays(fake_image_result)
        _ST_STUB.info.assert_called()


# ---------------------------------------------------------------------------
# 18. render_image_metadata — calls st.json (inside an st.expander)
# ---------------------------------------------------------------------------

class TestRenderImageMetadata:
    def setup_method(self):
        _ST_STUB.json.reset_mock()

    def test_calls_st_json(self, fake_image_result):
        app.render_image_metadata(fake_image_result)
        _ST_STUB.json.assert_called_once()

    def test_json_contains_image_id(self, fake_image_result):
        app.render_image_metadata(fake_image_result)
        payload = _ST_STUB.json.call_args[0][0]
        assert payload["image_id"] == "portrait"

    def test_json_contains_dimensions(self, fake_image_result):
        app.render_image_metadata(fake_image_result)
        payload = _ST_STUB.json.call_args[0][0]
        assert payload["width"] == 512 and payload["height"] == 512


# ---------------------------------------------------------------------------
# 19. render_image_download_button — mime type, filename, valid JSON
# ---------------------------------------------------------------------------

class TestRenderImageDownloadButton:
    def setup_method(self):
        _ST_STUB.download_button.reset_mock()

    def test_download_button_called(self, fake_image_result):
        app.render_image_download_button(fake_image_result)
        _ST_STUB.download_button.assert_called_once()

    def test_mime_type_is_json(self, fake_image_result):
        app.render_image_download_button(fake_image_result)
        kw = _ST_STUB.download_button.call_args[1]
        assert kw.get("mime") == "application/json"

    def test_filename_contains_image_id(self, fake_image_result):
        app.render_image_download_button(fake_image_result)
        kw = _ST_STUB.download_button.call_args[1]
        assert "portrait" in kw.get("file_name", "")

    def test_data_is_valid_json(self, fake_image_result):
        app.render_image_download_button(fake_image_result)
        kw  = _ST_STUB.download_button.call_args[1]
        raw = kw.get("data", b"")
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_json_contains_all_score_fields(self, fake_image_result):
        app.render_image_download_button(fake_image_result)
        kw     = _ST_STUB.download_button.call_args[1]
        parsed = json.loads(kw["data"])
        for field in ("verdict", "fused_score", "gradcam_score", "freq_score"):
            assert field in parsed, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# 20. run_image_pipeline — ImageDetector.detect_from_bytes called once
#
# (Old API: `run_image_detection_with_progress` + a bare module-level
# `detect_image` function no longer exist — session 4 introduced
# `run_image_pipeline`, which builds an ImageDetector instance and calls
# `.detect_from_bytes(...)` on it, plus an optional cached AI classifier
# via `_load_ai_classifier()`. That loader is patched out here since it
# would otherwise try to reach huggingface.co, which isn't reachable/
# desired in a unit test.)
# ---------------------------------------------------------------------------

class TestRunImagePipeline:
    def setup_method(self):
        _DETECTOR_STUB.ImageDetector.reset_mock()

    def test_detect_from_bytes_called_once(self, fake_image_result):
        _DETECTOR_STUB.ImageDetector.return_value.detect_from_bytes.return_value = fake_image_result
        fake_upload = MagicMock()
        fake_upload.name = "portrait.jpg"
        fake_upload.getvalue.return_value = b"fake-jpeg-bytes"

        with patch("app._load_ai_classifier", return_value=None):
            result = app.run_image_pipeline(fake_upload)

        _DETECTOR_STUB.ImageDetector.return_value.detect_from_bytes.assert_called_once_with(
            b"fake-jpeg-bytes", image_id="portrait", image_path="portrait.jpg",
        )
        assert result is fake_image_result


# ---------------------------------------------------------------------------
# 21. _image_tab — no upload → no pipeline run, info prompt shown
# ---------------------------------------------------------------------------

class TestImageTabNoUpload:
    def setup_method(self):
        _DETECTOR_STUB.ImageDetector.reset_mock()
        _ST_STUB.info.reset_mock()
        _ST_STUB.file_uploader.reset_mock()

    def test_no_upload_does_not_run_detector(self):
        _ST_STUB.file_uploader.return_value = None
        app._image_tab()
        _DETECTOR_STUB.ImageDetector.return_value.detect_from_bytes.assert_not_called()

    def test_no_upload_shows_info_prompt(self):
        _ST_STUB.file_uploader.return_value = None
        app._image_tab()
        _ST_STUB.info.assert_called()


# ---------------------------------------------------------------------------
# 22. _image_tab — error state when the detector raises
# ---------------------------------------------------------------------------

class TestImageTabError:
    def setup_method(self):
        _ST_STUB.error.reset_mock()
        _DETECTOR_STUB.ImageDetector.reset_mock()

    def test_error_shown_when_detection_fails(self, tmp_path):
        _DETECTOR_STUB.ImageDetector.return_value.detect_from_bytes.side_effect = RuntimeError("model crash")
        prog = MagicMock()
        _ST_STUB.progress.return_value = prog

        fake_upload = MagicMock()
        fake_upload.name = "photo.jpg"
        fake_upload.getvalue.return_value = b"fake-jpeg"
        _ST_STUB.file_uploader.return_value = fake_upload

        with patch("app._load_ai_classifier", return_value=None):
            app._image_tab()

        _ST_STUB.error.assert_called_once()
        err_msg = str(_ST_STUB.error.call_args)
        assert "failed" in err_msg.lower() or "crash" in err_msg.lower()


# ---------------------------------------------------------------------------
# 23. main_app — both tabs created
# ---------------------------------------------------------------------------

class TestMainAppTabs:
    def setup_method(self):
        _ST_STUB.tabs.reset_mock()
        _ST_STUB.file_uploader.return_value = None  # keep both tabs idle

    def test_two_tabs_created(self):
        app.main_app()
        _ST_STUB.tabs.assert_called_once()
        labels = _ST_STUB.tabs.call_args[0][0]
        assert len(labels) == 2

    def test_video_tab_label_present(self):
        app.main_app()
        labels = _ST_STUB.tabs.call_args[0][0]
        assert any("Video" in l or "🎬" in l for l in labels)

    def test_image_tab_label_present(self):
        app.main_app()
        labels = _ST_STUB.tabs.call_args[0][0]
        assert any("Image" in l or "🖼" in l for l in labels)