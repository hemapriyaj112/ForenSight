"""
validate_classifier_on_video.py
================================
Run this on YOUR machine (not the sandbox) — this sandbox can't reach
huggingface.co, so the classifier never actually loads here. This script
reuses your real pipeline and the real, already-loaded classifier to check
one specific thing: does prithivMLmods/deepfake-detector-model-v1 read
REAL video frames as mostly low-probability and AI video frames as mostly
high-probability? It was only ever validated against still PHOTOS in an
earlier session — never against video frames specifically, which look
meaningfully different (H.264 compression, motion blur, ffmpeg-extraction
artifacts). That's exactly the kind of domain gap that already silently
broke noise-floor, ELA, and spectral for video.

USAGE
-----
1. Edit the VIDEOS list below: point each entry at a real file path on
   your machine, and set the label to "real" or "ai" based on ground
   truth (use the same test videos we've already been using this
   session if you still have them — that gives a clean before/after
   comparison against everything we already know about those clips).
2. From the repo root:
       python validate_classifier_on_video.py
3. Share the printed summary table back (or the saved
   classifier_validation_results.json) — that's what tells us whether
   the classifier itself needs the same "exclude it / down-weight it"
   treatment noise/ELA/spectral got, or whether it's actually solid on
   video and the corroboration gate is trustworthy as-is.

This does NOT modify any pipeline file — it only reads results and
prints/saves them, so it's safe to run without touching anything you've
already applied via the patches.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# Point this at your ForenSight repo root if this script isn't already
# sitting inside it.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import main as main_mod  # noqa: E402


# ---------------------------------------------------------------------------
# EDIT THIS: (video_path, ground_truth_label) pairs.
# label must be exactly "real" or "ai".
# ---------------------------------------------------------------------------
VIDEOS: list[tuple[str, str]] = [
    (r"C:\Users\Hemapriya\Downloads\testing_videos\ai_vid_without_audio.mp4", "ai"),
    (r"C:\Users\Hemapriya\Downloads\testing_videos\girl_speaking_video_gemini.mp4", "ai"),
    (r"C:\Users\Hemapriya\Downloads\testing_videos\ai_noaud2.mp4", "ai"),
    (r"C:\Users\Hemapriya\Downloads\testing_videos\real_vid_to_test_withaudio.mp4", "real"),
    (r"C:\Users\Hemapriya\Downloads\testing_videos\vid_with_noaudio.mp4", "real"),
    # Add the new real video that scored texture=82.8%/classifier=40.1% here:
    # ("path/to/that_video.mp4", "real"),
]


def main() -> None:
    all_results = {}

    print(f"{'video':40s} {'label':6s} {'frames':7s} {'clf avg':9s} {'clf min':9s} {'clf max':9s} "
          f"{'texture avg':12s} {'documents_skipped':18s}")
    print("-" * 130)

    for video_name, label in VIDEOS:
        path = Path(video_name)
        if not path.exists():
            print(f"{video_name:40s}  SKIPPED (file not found — edit the path in VIDEOS)")
            continue

        result = main_mod.run_pipeline(video_path=path)
        vr = result.video_result

        classifier_vals = [
            fr.sub_scores.get("classifier") for fr in vr.frame_results
            if fr.sub_scores.get("classifier") is not None
        ]
        texture_vals = [
            fr.sub_scores.get("texture") for fr in vr.frame_results
            if fr.sub_scores.get("texture") is not None
        ]

        clf_avg = statistics.mean(classifier_vals) if classifier_vals else None
        clf_min = min(classifier_vals) if classifier_vals else None
        clf_max = max(classifier_vals) if classifier_vals else None
        tex_avg = statistics.mean(texture_vals) if texture_vals else None

        def fmt(x):
            return f"{x:.3f}" if x is not None else "n/a"

        print(f"{video_name:40s} {label:6s} {len(vr.frame_results):<7d} "
              f"{fmt(clf_avg):9s} {fmt(clf_min):9s} {fmt(clf_max):9s} "
              f"{fmt(tex_avg):12s} {vr.metadata.get('documents_skipped')}/{vr.metadata.get('frames_analysed')}")

        all_results[video_name] = {
            "label": label,
            "classifier_active": any(fr.classifier_active for fr in vr.frame_results),
            "classifier_backend": vr.metadata.get("classifier_backend"),
            "classifier_error": vr.metadata.get("classifier_error"),
            "classifier_avg": clf_avg,
            "classifier_min": clf_min,
            "classifier_max": clf_max,
            "classifier_per_frame": classifier_vals,
            "texture_avg": tex_avg,
            "texture_per_frame": texture_vals,
            "fused_score": result.fused_score,
            "verdict": result.verdict.value,
            "generator_tag": vr.metadata.get("generator_tag"),
            "documents_skipped": vr.metadata.get("documents_skipped"),
            "frames_analysed": vr.metadata.get("frames_analysed"),
        }

    out_path = REPO_ROOT / "classifier_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nFull per-frame details saved to {out_path}")
    print("\n--- Quick read ---")
    print("A well-calibrated classifier should show 'ai' rows averaging high")
    print("(closer to 1.0) and 'real' rows averaging low (closer to 0.0), with")
    print("min/max not swinging wildly across frames of the SAME video. If")
    print("real videos average high, or the range within a single video is")
    print("wide, that's the same video-specific domain gap that already broke")
    print("noise-floor/ELA/spectral -- the classifier would need the same")
    print("'exclude or down-weight for video' treatment.")


if __name__ == "__main__":
    main()
