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
from utils.types import AnalysisResult, ModalResult, Verdict

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


def _load_ai_classifier():
    """
    Loads the optional trained AI-vs-real image classifier(s) from config.

    Supports two modes:
      1. Ensemble mode (recommended): loads multiple free HuggingFace models
         and averages their predictions. Enabled when config.ensemble.enabled
         is true.
      2. Single model mode (legacy): loads one model. Enabled when
         config.image_ai_classifier.enabled is true.

    Returns None if disabled/unconfigured/failed to load; VideoDetector/
    ImageDetector both handle that gracefully and fall back to the
    heuristic-only pipeline.
    """
    ensemble_cfg = getattr(CFG, "ensemble", None)
    if ensemble_cfg is not None and getattr(ensemble_cfg, "enabled", False):
        models = getattr(ensemble_cfg, "models", []) or []
        if not models:
            log.warning("Ensemble mode enabled but no models configured")
            return None
        from pipeline.forensics.ensemble_classifier import EnsembleClassifier
        clf = EnsembleClassifier(models=models)
        # TASK 3: models load lazily on first predict(); the constructor alone
        # does NOT load them, so status must be evaluated AFTER the load attempt.
        clf._ensure_loaded()
        if not clf.available:
            log.warning(
                "Ensemble classifier unavailable after load attempt: %s",
                clf.load_error,
            )
        else:
            log.info(
                "Ensemble classifier loaded (%d model(s)): %s",
                len(clf.models),
                [m.get("path") for m in clf.models],
            )
        return clf

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
# TASK 3 — honest, structured detector status (ACTIVE_MODEL / ACTIVE_HEURISTIC
# / INACTIVE). Never call a stub ACTIVE. Intended for startup reporting so the
# UI/operators can see exactly what is and isn't actually running.
# ---------------------------------------------------------------------------
STATUS_ACTIVE_MODEL = "ACTIVE_MODEL"
STATUS_ACTIVE_HEURISTIC = "ACTIVE_HEURISTIC"
STATUS_INACTIVE = "INACTIVE"


def get_detector_status(cfg=None, classifier=None) -> list[dict]:
    """Return an honest per-detector status report.

    Returns a list of dicts {name, category, status, detail}. `classifier` is
    an already-loaded EnsembleClassifier (optional); if None it is loaded once
    here so the report reflects the real post-load availability.
    """
    cfg = cfg or CFG
    statuses: list[dict] = []

    eff_w = getattr(getattr(cfg, "video", None), "weights", "models/efficientnet_v2s.pt")
    raw_w = getattr(getattr(cfg, "audio", None), "weights", "models/rawnet2.pt")

    # --- Configured neural nets that are NOT actually loaded (stubs/heuristics)
    statuses.append({
        "name": "EfficientNetV2 (video)",
        "category": STATUS_INACTIVE, "status": STATUS_INACTIVE,
        "detail": f"config.video.weights={eff_w} never loaded; 'models/' absent. "
                  "Video 'texture' score is a Laplacian-variance heuristic, not a network.",
    })
    statuses.append({
        "name": "RawNet2 (audio)",
        "category": STATUS_INACTIVE, "status": STATUS_INACTIVE,
        "detail": f"config.audio.weights={raw_w} absent; RawNet2Classifier is a stub. "
                  "Audio uses heuristic forensics only.",
    })
    statuses.append({
        "name": "GradCAM (neural)",
        "category": STATUS_INACTIVE, "status": STATUS_INACTIVE,
        "detail": "VideoDetector._GradCAM runs _stub_forward (model=None); not a network.",
    })
    statuses.append({
        "name": "ELA / Noise-floor / Spectral-FFT",
        "category": STATUS_INACTIVE, "status": "INACTIVE(for video)",
        "detail": "excluded from the video pipeline; only active on the image tab.",
    })

    # --- Real (non-ML) heuristic signals that DO run
    statuses.append({"name": "Audio heuristic forensics", "category": STATUS_ACTIVE_HEURISTIC,
                     "status": "ACTIVE",
                     "detail": "spectral stability / phase incoherence / MFCC / flux / ZCR."})
    statuses.append({"name": "Video texture heuristic", "category": STATUS_ACTIVE_HEURISTIC,
                     "status": "ACTIVE", "detail": "per-frame Laplacian variance."})
    statuses.append({"name": "Metadata / C2PA provenance", "category": STATUS_ACTIVE_HEURISTIC,
                     "status": "ACTIVE", "detail": "EXIF/C2PA/generator signature scan."})
    statuses.append({"name": "Temporal localization", "category": STATUS_ACTIVE_HEURISTIC,
                     "status": "ACTIVE", "detail": "frame-to-frame consistency."})
    statuses.append({"name": "Document detector", "category": STATUS_ACTIVE_HEURISTIC,
                     "status": "ACTIVE", "detail": "colour/edge heuristic gating the classifier."})

    # --- Real loaded model
    statuses.append({"name": "Face cropper (MTCNN)", "category": STATUS_ACTIVE_MODEL,
                     "status": "ACTIVE", "detail": "facenet_pytorch face detection."})

    # --- AI image classifier ensemble (evaluated AFTER load attempt)
    if classifier is None:
        classifier = _load_ai_classifier()
    if classifier is not None and getattr(classifier, "available", False):
        loaded = [m.get("path") for m in getattr(classifier, "models", [])]
        statuses.append({
            "name": "AI image classifier (ensemble)",
            "category": STATUS_ACTIVE_MODEL, "status": "ACTIVE",
            "detail": f"loaded from HuggingFace Hub at runtime: {loaded}. "
                      "Validated as reading BACKWARDS on video -> disabled for video "
                      "verdicts (Task 2); still used for IMAGE analysis.",
        })
    else:
        statuses.append({
            "name": "AI image classifier (ensemble)",
            "category": STATUS_INACTIVE, "status": STATUS_INACTIVE,
            "detail": f"unavailable after load attempt: {getattr(classifier, 'load_error', 'no model')}.",
        })

    # --- Video image-classifier contribution (post Task 2): explicitly INACTIVE
    statuses.append({
        "name": "video_image_classifier",
        "category": STATUS_INACTIVE, "status": STATUS_INACTIVE,
        "detail": "Classifier behavior is reversed/invalid for current video preprocessing "
                  "and requires independent validation; disabled for video verdicts (Task 2).",
    })

    # --- GenConViT (Phase 2A): real pretrained video deepfake detector.
    # Report ACTIVE_MODEL only if the weights actually loaded and inference is
    # possible. Any load failure yields INACTIVE with the exact reason — never
    # a silent texture-heuristic fallback.
    gc_cfg = getattr(cfg, "genconvit", None)
    if gc_cfg is not None and getattr(gc_cfg, "enabled", False):
        from pipeline.genconvit.detector import GenConViTDetector
        import os
        from pathlib import Path

        wdir = getattr(gc_cfg, "weights_dir", "models/genconvit")
        if not os.path.isabs(wdir):
            wdir = str(_HERE / wdir)
        gc = GenConViTDetector(
            weights_dir=wdir,
            num_frames=getattr(gc_cfg, "num_frames", 15) or 15,
            use_vae=bool(getattr(gc_cfg, "use_vae", True)),
            fp16=bool(getattr(gc_cfg, "fp16", True)),
        )
        if gc.ensure_loaded():
            statuses.append({
                "name": "GenConViT (video)",
                "category": STATUS_ACTIVE_MODEL, "status": "ACTIVE",
                "detail": (
                    f"branches={gc.branches} device={gc.device}; weights in "
                    f"{gc._weights_dir}. Kept OUT of final fusion until evaluation "
                    "(config.genconvit.use_in_final_fusion=false)."
                    + (f" note={gc.load_error}" if gc.load_error else "")
                ),
            })
        else:
            statuses.append({
                "name": "GenConViT (video)",
                "category": STATUS_INACTIVE, "status": STATUS_INACTIVE,
                "detail": f"INACTIVE: {gc.load_error}",
            })

    # --- AASIST (Phase 2B): real pretrained audio anti-spoofing detector.
    # Report ACTIVE_MODEL only if the official weights actually loaded. Any
    # failure yields INACTIVE with the exact reason — never a heuristic
    # fallback. Kept OUT of the audio verdict (evaluation-only, Phase 2B).
    a_config = getattr(cfg, "aasist", None)
    if a_config is not None and getattr(a_config, "enabled", False):
        from pipeline.aasist.detector import AASISTDetector

        wdir = getattr(a_config, "weights_dir", "models/aasist")
        if not os.path.isabs(wdir):
            wdir = str(_HERE / wdir)
        aa = AASISTDetector(
            weights_dir=wdir,
            window_samples=getattr(a_config, "window_samples", 64600) or 64600,
        )
        if aa.ensure_loaded():
            statuses.append({
                "name": "AASIST (audio anti-spoofing)",
                "category": STATUS_ACTIVE_MODEL, "status": "ACTIVE",
                "detail": (
                    f"official checkpoint loaded (params={aa.model_params}, "
                    f"device={aa.device}); weights in {aa._weights_dir}. "
                    "Kept OUT of the audio verdict until evaluation "
                    "(config.aasist.use_in_final_fusion=false)."
                ),
            })
        else:
            statuses.append({
                "name": "AASIST (audio anti-spoofing)",
                "category": STATUS_INACTIVE, "status": STATUS_INACTIVE,
                "detail": f"INACTIVE: {aa.load_error}",
            })
    return statuses


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

    fast_mode = getattr(getattr(CFG, "fast_mode", None), "enabled", False)

    # Step 4 — video
    # TASK 2: the image classifier is DISABLED for the VIDEO pipeline. It was
    # validated as reversed/invalid on video frames (reports/FULL_BACKEND_AUDIT.md),
    # so it must not influence video verdicts until independently validated. It
    # stays available for IMAGE analysis (app.run_image_pipeline). Passing None
    # keeps VideoDetector purely heuristic + provenance for video.
    video_detector = VideoDetector(CFG, ai_classifier=None, fast_mode=fast_mode)
    video_bytes = video_path.read_bytes()
    video_result = video_detector.run(frames_dir, video_bytes=video_bytes, video_path=str(video_path))

    # Step 5 — audio
    audio_detector = AudioDetector(CFG)
    audio_segments = audio_detector.run(wav_path)
    if audio_segments:
        audio_fake_prob = sum(s.fake_prob for s in audio_segments) / len(audio_segments)
    else:
        audio_fake_prob = 0.5
    audio_result = ModalResult(
        modality="audio",
        fake_prob=audio_fake_prob,
        audio_results=audio_segments,
        error=None if audio_segments else "Empty audio file",
    )

    # Step 6 — fusion
    fusion_cfg = getattr(CFG, "fusion", None)
    fuser = Fuser(
        video_weight=getattr(fusion_cfg, "video_weight", 0.6),
        audio_weight=getattr(fusion_cfg, "audio_weight", 0.4),
        real_threshold=getattr(fusion_cfg, "real_threshold", 0.40),
        fake_threshold=getattr(fusion_cfg, "fake_threshold", 0.60),
        aggregation=getattr(fusion_cfg, "aggregation", "mean"),
        percentile=getattr(fusion_cfg, "percentile", 90.0),
        confidence_weighting=getattr(fusion_cfg, "confidence_weighting", False),
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