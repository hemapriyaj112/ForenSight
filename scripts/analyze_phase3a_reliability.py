"""Phase 3A — build the evidence reliability and domain-risk reporting layer.

PURE ANALYSIS / REPORTING SCRIPT. It does NOT run, load, or modify any
detector, model, fuser, threshold, or configuration.  It consumes already
produced detector evidence:
  * reports/testing_videos_evaluation.json  (output of the unchanged pipeline)

and produces reliability/domain-risk metadata:
  * reports/phase3a_reliability_evidence.json
  * reports/phase3a_reliability_evidence.csv

It verifies production invariance against the frozen Phase 2B/2C baseline.

Usage: py scripts/analyze_phase3a_reliability.py [--eval PATH] [--no-probe]
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
from pipeline.analysis import reliability  # noqa: E402

DEFAULT_EVAL = os.path.join(ROOT, "reports", "testing_videos_evaluation.json")
OUT_JSON = os.path.join(ROOT, "reports", "phase3a_reliability_evidence.json")
OUT_CSV = os.path.join(ROOT, "reports", "phase3a_reliability_evidence.csv")

CSV_COLUMNS = [
    "filename", "ground_truth", "production_verdict", "fusion_score",
    "genconvit_label", "genconvit_score", "genconvit_coverage",
    "aasist_label", "aasist_spoof", "aasist_pattern", "aasist_window_count",
    "audio_heuristic_mean", "domain_risk_state", "domain_risk_flags",
    "gc_reliability", "aasist_reliability", "overall_reliability",
    "agreement_summary", "neural_agreement", "evidence_summary",
    "explanations",
]


def build_probe_map(evals: dict, do_probe: bool) -> dict:
    """Best-effort read-only ffprobe metadata keyed by filename."""
    probe = {}
    if not do_probe:
        return probe
    import shutil
    if not shutil.which("ffprobe"):
        print("[phase3a] ffprobe not found; skipping media probe", file=sys.stderr)
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


def flat_row(report: dict, probe: dict) -> dict:
    gc = report["modalities"].get("genconvit", {}) or {}
    aa = report["modalities"].get("aasist", {}) or {}
    cov = gc.get("coverage") or {}
    aa_t = aa.get("segment_pattern") or {}
    dr = report.get("domain_risk") or {}
    agr = report.get("agreement") or {}
    return {
        "filename": report["filename"],
        "ground_truth": report.get("ground_truth"),
        "production_verdict": report["production_verdict"],
        "fusion_score": report.get("production_fusion_score"),
        "genconvit_label": (gc.get("evidence_direction") or {}).get("evidence_direction"),
        "genconvit_score": gc.get("score"),
        "genconvit_coverage": cov.get("status"),
        "aasist_label": (aa.get("evidence_direction") or {}).get("evidence_direction"),
        "aasist_spoof": aa.get("score"),
        "aasist_pattern": aa_t.get("segment_pattern"),
        "aasist_window_count": aa_t.get("window_count"),
        "audio_heuristic_mean": None,
        "domain_risk_state": dr.get("state"),
        "domain_risk_flags": ";".join(f["flag"] for f in dr.get("flags", [])),
        "gc_reliability": gc.get("reliability"),
        "aasist_reliability": aa.get("reliability"),
        "overall_reliability": report.get("overall_reliability"),
        "agreement_summary": agr.get("summary"),
        "neural_agreement": agr.get("neural_agreement"),
        "evidence_summary": ";".join(report.get("evidence_summary", [])),
        "explanations": " || ".join(report.get("explanations", [])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=DEFAULT_EVAL)
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the read-only ffprobe media probe")
    args = ap.parse_args()

    evals = evidence.load_eval_json(args.eval)
    probe = build_probe_map(evals, do_probe=not args.no_probe)

    # Build per-video reliability reports from the unchanged eval output.
    corpus = reliability.build_corpus_reliability(evals, probe)

    # Production invariance — read-only verification of the frozen baseline.
    invariance = evidence.verify_production_invariance(evals["results"])

    # Attach overall reliability state to each report for CSV export.
    reports = []
    for rep in corpus["per_video_reports"]:
        # overall reliability derived in build_reliability_report
        rep.setdefault("overall_reliability", None)
        reports.append(rep)

    rows_for_csv = [flat_row(rep, probe.get(rep["filename"], {})) for rep in reports]

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": args.eval,
        "phase": "3A",
        "reporting_only_note": (
            "Evidence reliability & domain-risk reporting only. No detector, "
            "fusion formulation, threshold, or configuration was modified; "
            "genconvit/aasist use_in_final_fusion stay false; the production "
            "verdict is unchanged."
        ),
        "calibration_status": "NOT READY",
        "production_invariance": invariance,
        "corpus_summary": corpus["corpus_summary"],
        "per_video_reports": reports,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows_for_csv:
            w.writerow(r)

    print("=== PHASE 3A PRODUCTION INVARIANCE CHECK ===", flush=True)
    print(json.dumps(invariance["computed"], indent=2), flush=True)
    print("matches_phase2b_baseline:", invariance["matches_phase2b_baseline"], flush=True)
    print(json.dumps(invariance.get("metric_deltas", {}), indent=2), flush=True)

    print("\n=== RELIABILITY SUMMARY ===", flush=True)
    print(json.dumps(corpus["corpus_summary"], indent=2), flush=True)

    print("\n=== DOMAIN-RISK STATE DISTRIBUTION ===", flush=True)
    state_counts = {}
    for rep in reports:
        st = rep["domain_risk"]["state"]
        state_counts[st] = state_counts.get(st, 0) + 1
    print(json.dumps(state_counts, indent=2), flush=True)

    print("\n=== PER-VIDEO RELIABILITY TABLE ===", flush=True)
    for rep in reports:
        dr_flags = ";".join(f["flag"] for f in rep["domain_risk"]["flags"])
        gc = rep["modalities"].get("genconvit", {})
        aa = rep["modalities"].get("aasist", {})
        print(
            f"{rep['filename'][:32]:<34} | gt={rep.get('ground_truth'):<5} "
            f"verdict={rep['production_verdict']:<9} "
            f"dr={rep['domain_risk']['state']:<10} "
            f"reliab={rep.get('overall_reliability') or '?':<10} "
            f"gc={gc.get('reliability','?'):<10} aa={aa.get('reliability','?'):<10} "
            f"flags=[{dr_flags}]",
            flush=True,
        )

    print("\n[done] wrote", OUT_JSON, "and", OUT_CSV, flush=True)


if __name__ == "__main__":
    main()
