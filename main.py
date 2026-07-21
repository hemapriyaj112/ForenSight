"""
main.py — ForenSight CLI orchestrator  (Sprint 5)
==================================================
Usage
-----
    python main.py --video path/to/clip.mp4 [--output json|db|both]

Exit codes
----------
    0  →  REAL
    1  →  FAKE
    2  →  UNCERTAIN
   -1  →  error (non-zero, maps to 255 on POSIX shells)

Pipeline (in order)
-------------------
1.  Parse CLI args.
2.  Validate the video file exists.
3.  Demux via utils/demux.py → frames_dir + wav_path.
4.  VideoDetector(CFG).run(frames_dir) → ModalResult.
5.  AudioDetector(CFG).run(wav_path)   → ModalResult.
6.  Fuser().fuse(…)                    → AnalysisResult.
7.  Optionally persist to SQLite DB.
8.  Print verdict summary to stdout (always).
9.  Optionally print JSON to stdout.
10. Exit with code matching verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports — adjust sys.path so the package root is always resolvable
# whether the script is run from the project root or any subdirectory.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from database import db as _db_module
from pipeline.audio.detector import AudioDetector
from pipeline.forensics.ai_classifier import AIClassifier
from pipeline.fusion.fuser import Fuser
from pipeline.video.detector import VideoDetector
from utils.config import CFG
from utils.demux import DemuxError, demux
from utils.logger import get_logger
from utils.types import AnalysisResult, Verdict

log = get_logger("forensight.main")

# Exit codes
_EXIT_REAL = 0
_EXIT_FAKE = 1
_EXIT_UNCERTAIN = 2
_EXIT_ERROR = 255          # sys.exit accepts 0-255; we use 255 for errors

_VERDICT_EXIT: dict[Verdict, int] = {
    Verdict.REAL: _EXIT_REAL,
    Verdict.FAKE: _EXIT_FAKE,
    Verdict.UNCERTAIN: _EXIT_UNCERTAIN,
}

_OUTPUT_CHOICES = ("json", "db", "both")


def _load_ai_classifier() -> AIClassifier | None:
    """
    Loads the optional trained AI-vs-real image classifier from config, if
    enabled — same config block (image_ai_classifier) and same model used
    by the image tab in app.py, reused here because the classifier operates
    per-frame on ordinary RGB arrays regardless of whether the frame came
    from an uploaded photo or a sampled video frame. Returns None if
    disabled/unconfigured/failed to load; VideoDetector/ImageDetector both
    handle that gracefully and fall back to the heuristic-only pipeline.
    """
    cfg = getattr(CFG, "image_ai_classifier", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        return None
    model_path = getattr(cfg, "model_path", "") or None
    if not model_path:
        return None
    clf = AIClassifier(
        model_path=model_path,
        backend=getattr(cfg, "backend", "auto"),
        input_size=getattr(cfg, "input_size", 224),
        ai_class_index=getattr(cfg, "ai_class_index", 0),
    )
    if not clf.available:
        log.warning("AI classifier enabled in config but failed to load: %s", clf.load_error)
    return clf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forensight",
        description="ForenSight — deepfake detection pipeline",
    )
    p.add_argument(
        "--video",
        required=True,
        metavar="FILE",
        help="Path to the MP4 video file to analyse.",
    )
    p.add_argument(
        "--output",
        choices=_OUTPUT_CHOICES,
        default="json",
        help=(
            "Output mode: 'json' prints a JSON summary to stdout, "
            "'db' saves to SQLite, 'both' does both. (default: json)"
        ),
    )
    p.add_argument(
        "--analysis-id",
        default=None,
        metavar="UUID",
        help="Override the auto-generated analysis UUID (useful for testing).",
    )
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    video_path: Path,
    analysis_id: str | None = None,
    output_dir: Path | None = None,
) -> AnalysisResult:
    """
    Execute the full ForenSight pipeline for *video_path*.

    Parameters
    ----------
    video_path:   Path to the input MP4.
    analysis_id:  Pre-assigned UUID; auto-generated if None.
    output_dir:   Where to write frames/audio; a temp dir is used if None.

    Returns
    -------
    AnalysisResult — fully populated, verdict set.

    Raises
    ------
    FileNotFoundError  if video_path does not exist.
    DemuxError         if ffmpeg fails.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if analysis_id is None:
        analysis_id = str(uuid.uuid4())

    log.info("Starting analysis %s for %s", analysis_id, video_path)

    # Step 3 — demux
    _use_tmp = output_dir is None
    if _use_tmp:
        output_dir = Path(tempfile.mkdtemp(prefix="forensight_"))

    try:
        frames_dir, wav_path = demux(video_path, output_dir=output_dir)
    except DemuxError:
        raise  # propagate; callers handle display

    # Step 4 — video
    ai_classifier = _load_ai_classifier()
    video_detector = VideoDetector(CFG, ai_classifier=ai_classifier)
    video_bytes = video_path.read_bytes()
    video_result = video_detector.run(frames_dir, video_bytes=video_bytes, video_path=str(video_path))

    # Step 5 — audio
    audio_detector = AudioDetector(CFG)
    audio_result = audio_detector.run(wav_path)

    # Step 6 — fusion
    fuser = Fuser(
        video_weight=CFG.fusion.video_weight,
        audio_weight=CFG.fusion.audio_weight,
        real_threshold=CFG.fusion.real_threshold,
        fake_threshold=CFG.fusion.fake_threshold,
    )
    result = fuser.fuse(
        video_result=video_result,
        audio_result=audio_result,
        video_path=video_path,
        analysis_id=analysis_id,
    )

    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _result_to_dict(result: AnalysisResult) -> dict:
    """Serialise AnalysisResult to a JSON-safe dict."""
    return {
        "analysis_id": result.analysis_id,   # compat property on AnalysisResult
        "verdict": result.verdict.value,
        "fused_score": round(result.fused_score, 6),
        "video_score": round(result.metadata.get("video_score", 0.0), 6),
        "audio_score": round(result.metadata.get("audio_score", 0.0), 6),
        "calibrated_video_score": round(
            result.metadata.get("calibrated_video_score", 0.0), 6
        ),
        "calibrated_audio_score": round(
            result.metadata.get("calibrated_audio_score", 0.0), 6
        ),
        "video_frames_analysed": len(result.video_result.frame_results),
        "audio_segments_analysed": len(result.audio_result.audio_results),
        "video_error": result.video_result.error,
        "audio_error": result.audio_result.error,
        "metadata": result.metadata,
    }


def print_summary(result: AnalysisResult) -> None:
    """Print a human-readable one-liner verdict to stdout."""
    score_pct = result.fused_score * 100
    frames = len(result.video_result.frame_results)
    segments = len(result.audio_result.audio_results)
    video_path = result.metadata.get("video_path", "?")

    icon = {"REAL": "✅", "FAKE": "⚠️ ", "UNCERTAIN": "❓"}.get(result.verdict.value, "")
    print(
        f"\n{icon} ForenSight verdict: {result.verdict.value}\n"
        f"   File          : {video_path}\n"
        f"   Fused score   : {score_pct:.1f}% fake probability\n"
        f"   Video frames  : {frames}\n"
        f"   Audio segments: {segments}\n"
        f"   Analysis ID   : {result.analysis_id}\n"
    )


def print_json(result: AnalysisResult) -> None:
    """Dump JSON summary to stdout."""
    print(json.dumps(_result_to_dict(result), indent=2))


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """
    Entry point. Returns an integer exit code.

    Separated from the ``if __name__ == "__main__"`` block so tests can call
    ``main(["--video", "..."]) `` without spawning a subprocess.
    """
    args = parse_args(argv)
    video_path = Path(args.video)

    # ── Validate before doing any work ──────────────────────────────────────
    if not video_path.exists():
        print(f"ERROR: video file not found: {video_path}", file=sys.stderr)
        return _EXIT_ERROR

    # ── Run pipeline ─────────────────────────────────────────────────────────
    try:
        result = run_pipeline(
            video_path=video_path,
            analysis_id=args.analysis_id,
        )
    except DemuxError as exc:
        print(f"ERROR: demux failed — {exc}", file=sys.stderr)
        return _EXIT_ERROR
    except Exception as exc:  # pylint: disable=broad-except
        log.error("Unexpected error: %s", exc)
        traceback.print_exc(file=sys.stderr)
        return _EXIT_ERROR

    # ── Output ───────────────────────────────────────────────────────────────
    output_mode = args.output

    # Always print human-readable summary
    print_summary(result)

    if output_mode in ("json", "both"):
        print_json(result)

    if output_mode in ("db", "both"):
        try:
            _db_module.save_result(result)
            print(f"[db] Saved analysis {result.analysis_id} to {CFG.database.path}",
                  file=sys.stderr)
        except Exception as exc:  # pylint: disable=broad-except
            log.error("DB write failed: %s", exc)
            # Non-fatal — verdict is still correct; exit code reflects verdict

    return _VERDICT_EXIT[result.verdict]


if __name__ == "__main__":
    sys.exit(main())