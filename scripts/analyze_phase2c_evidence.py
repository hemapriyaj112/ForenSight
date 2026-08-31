"""Phase 2C — build the unified multimodal evidence table.

PURE ANALYSIS SCRIPT. It does NOT run, load, or modify any detector, model,
fuser, threshold, or configuration. Inputs:
  * reports/testing_videos_evaluation.json  (output of the unchanged pipeline)
  * read-only ffprobe metadata per video    (media container only, no detectors)
Outputs:
  * reports/phase2c_multimodal_evidence.json
  * reports/phase2c_multimodal_evidence.csv
  * prints a production-invariance verification block

Usage: py scripts/analyze_phase2c_evidence.py [--eval PATH] [--no-probe]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline.analysis import evidence  # noqa: E402

DEFAULT_EVAL = os.path.join(ROOT, "reports", "testing_videos_evaluation.json")
OUT_JSON = os.path.join(ROOT, "reports", "phase2c_multimodal_evidence.json")
OUT_CSV = os.path.join(ROOT, "reports", "phase2c_multimodal_evidence.csv")

CSV_COLUMNS = [
    "filename", "ground_truth", "duration_seconds", "fps", "has_audio",
    "codec", "resolution", "filesize_mb", "bitrate_kbps",
    "verdict", "fusion_score", "video_score", "audio_score",
    "texture_score", "frequency_score", "temporal_score",
    "provenance_score", "provenance_override", "uncorroborated_video_evidence",
    "genconvit_status", "genconvit_score", "genconvit_predicted_label",
    "genconvit_face_coverage", "genconvit_frames_used",
    "genconvit_average_face_confidence", "genconvit_coverage_status",
    "aasist_status", "aasist_spoof_score", "aasist_predicted_label",
    "aasist_window_count", "aasist_pattern",
    "audio_heuristic_status", "audio_heuristic_mean", "audio_heuristic_min",
    "audio_heuristic_max", "audio_segment_count",
    "agreement_pattern", "evidence_categories", "disagreement_flags",
]


def build_probe_map(evals: dict, do_probe: bool) -> dict:
    """Best-effort read-only ffprobe metadata keyed by filename."""
    probe = {}
    if not do_probe:
        return probe
    import shutil
    if not shutil.which("ffprobe"):
        print("[phase2c] ffprobe not found; skipping media probe", file=sys.stderr)
        return probe
    for row in evals.get("results", []):
        path = row.get("path")
        if not path or not os.path.exists(path):
            probe[row["filename"]] = {"probe_ok": False}
            continue
        try:
            probe[row["filename"]] = evidence.probe_media_readonly(path)
        except Exception as e:
            probe[row["filename"]] = {"probe_ok": False,
                                      "probe_error": f"{type(e).__name__}: {e}"}
    return probe


def flat_row(ev: dict, disagreement_cases: list) -> dict:
    id_, prod, gc, aa, ah = (ev["identity"], ev["production"],
                             ev["genconvit"], ev["aasist"], ev["audio_heuristic"])
    cats = ";".join(c["category"] for c in ev["evidence_categories"])
    flags = ";".join(c["type"] for c in disagreement_cases if c["filename"] == ev["filename"])
    return {
        "filename": id_["filename"], "ground_truth": id_["ground_truth"],
        "duration_seconds": id_["duration_seconds"], "fps": id_["fps"],
        "has_audio": id_["has_audio"], "codec": id_["codec"],
        "resolution": id_["resolution"], "filesize_mb": id_["filesize_mb"],
        "bitrate_kbps": id_["bitrate_kbps"],
        "verdict": prod["verdict"], "fusion_score": prod["fusion_score"],
        "video_score": prod["video_score"], "audio_score": prod["audio_score"],
        "texture_score": prod["texture_score"],
        "frequency_score": prod["frequency_score"],
        "temporal_score": prod["temporal_score"],
        "provenance_score": prod["provenance_score"],
        "provenance_override": prod["provenance_override"],
        "uncorroborated_video_evidence": prod["uncorroborated_video_evidence"],
        "genconvit_status": gc["status"], "genconvit_score": gc["score"],
        "genconvit_predicted_label": gc["predicted_label"],
        "genconvit_face_coverage": gc["face_coverage"],
        "genconvit_frames_used": gc["frames_used"],
        "genconvit_average_face_confidence": gc["average_face_confidence"],
        "genconvit_coverage_status": gc["coverage_status"],
        "aasist_status": aa["status"], "aasist_spoof_score": aa["spoof_score"],
        "aasist_predicted_label": aa["predicted_label"],
        "aasist_window_count": aa["window_count"],
        "aasist_pattern": aa["segment_pattern"]["pattern"],
        "audio_heuristic_status": ah["status"],
        "audio_heuristic_mean": ah["shape"]["mean"],
        "audio_heuristic_min": ah["shape"]["min"],
        "audio_heuristic_max": ah["shape"]["max"],
        "audio_segment_count": ah["shape"]["segments"],
        "agreement_pattern": ev["agreement"]["pattern"],
        "evidence_categories": cats, "disagreement_flags": flags,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=DEFAULT_EVAL)
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the read-only ffprobe media probe")
    args = ap.parse_args()

    evals = evidence.load_eval_json(args.eval)
    probe = build_probe_map(evals, do_probe=not args.no_probe)
    videos = evidence.build_corpus(evals, probe)

    disagreement_cases = evidence.disagreement_cases(videos)
    rows_for_csv = [flat_row(ev, disagreement_cases) for ev in videos]

    invariance = evidence.verify_production_invariance(evals["results"])
    summary = evidence.corpus_summary(videos)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": args.eval,
        "phase": "2C",
        "analysis_only_note": ("Evidence analysis only. No detector, fusion, "
                               "threshold, or configuration was modified; "
                               "genconvit/aasist use_in_final_fusion stay false."),
        "production_invariance": invariance,
        "corpus_summary": summary,
        "disagreement_cases": disagreement_cases,
        "videos": videos,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows_for_csv:
            w.writerow(r)

    print("=== PHASE 2C PRODUCTION INVARIANCE CHECK ===", flush=True)
    print(json.dumps(invariance["computed"], indent=2), flush=True)
    print("matches_phase2b_baseline:", invariance["matches_phase2b_baseline"], flush=True)
    per_video = [(r["filename"][:30], r["ground_truth"], r["verdict"])
                 for r in invariance["per_video_verdicts"]]
    print("per-video verdicts:", json.dumps(per_video), flush=True)
    print("\n=== CORPUS SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("\n[done] wrote", OUT_JSON, "and", OUT_CSV, flush=True)


if __name__ == "__main__":
    main()