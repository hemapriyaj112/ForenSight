"""pipeline/analysis/evidence.py

Phase 2C — multimodal evidence assembly and domain-robustness analysis.

PURE ANALYSIS UTILITIES ONLY.

* Does NOT import, instantiate, or change any detector, model, fuser,
  threshold, or configuration.
* Reads the existing evaluation output `reports/testing_videos_evaluation.json`
  (+ optional read-only ffprobe enrichment) and derives:
    - a unified per-video evidence record,
    - deterministic evidence-taxonomy categories,
    - AASIST segment-level pattern summaries,
    - an agreement / disagreement matrix,
    - a production-invariance verification against the frozen baseline.

Nothing returned by this module may be treated as a "final verdict": every
derived category states whether it is an observation, a hypothesis, or a
validated conclusion, and the reasoning cites the exact source fields.

Reproducible: pure functions on stored JSON (no model inference, no RNG).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# AASIST fixed window geometry (pipeline/aasist/detector.py; upstream config).
AASIST_SAMPLE_RATE = 16_000
AASIST_WINDOW_SAMPLES = 64_600
AASIST_WINDOW_SECONDS = AASIST_WINDOW_SAMPLES / AASIST_SAMPLE_RATE  # 4.0375

# Production baseline frozen at the end of Phase 2B (must NOT change in 2C).
PRODUCTION_BASELINE = {
    "confusion": {"TP": 6, "TN": 4, "FP": 0, "FN": 5},
    "accuracy": 0.6666666666666666,
    "precision_fake": 1.0,
    "recall_fake": 0.5454545454545454,
    "FAKE_RECALL": 0.5454545454545454,
    "f1_fake": 0.7058823529411764,
    "false_negative_rate": 0.45454545454545453,
    "false_positive_rate": 0.0,
    "fake_verdict_rate": 0.4,
    "uncertain_rate": 0.26666666666666666,
}

_CATEGORIES = (
    "VIDEO_MANIPULATION_EVIDENCE",
    "AUDIO_SPOOF_EVIDENCE",
    "AUTHENTIC_AUDIO_EVIDENCE",
    "COMPRESSION_OR_CODEC_RISK",
    "LOW_VISUAL_COVERAGE",
    "LOW_FACE_COVERAGE",
    "NO_FACE",
    "NO_AUDIO",
    "MODALITY_DISAGREEMENT",
    "MULTIMODAL_AGREEMENT",
    "PROVENANCE_EVIDENCE",
    "WEAK_OR_AMBIGUOUS_EVIDENCE",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _round(x: Any, n: int = 4) -> Any:
    return round(float(x), n) if isinstance(x, (int, float)) else x


def _flat(score, lo: float = 0.4, hi: float = 0.6):
    """Classify a 0-1 suspicion score into REAL-LIKE / WEAK / FAKE-LIKE."""
    if score is None:
        return None
    if score >= hi:
        return "FAKE-LIKE"
    if score <= lo:
        return "REAL-LIKE"
    return "WEAK"


def load_eval_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def probe_media_readonly(path) -> dict:
    """Read-only ffprobe metadata (no detectors touched). Best-effort."""
    import os
    import shutil
    import subprocess

    out = {"filesize_mb": None, "resolution": None, "codec": None,
           "bitrate_kbps": None, "probe_ok": False}
    try:
        out["filesize_mb"] = round(os.path.getsize(path) / (1024 * 1024), 2)
    except Exception:
        pass
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return out
    try:
        r = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=60,
        )
        data = json.loads(r.stdout or "{}")
        fmt = data.get("format", {})
        br = fmt.get("bit_rate")
        out["bitrate_kbps"] = round(int(br) / 1000, 1) if br else None
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                w, h = s.get("width"), s.get("height")
                out["resolution"] = f"{w}x{h}" if w and h else None
                out["codec"] = s.get("codec_name")
        out["probe_ok"] = True
    except Exception:
        out["probe_ok"] = False
    return out


def aasist_window_timestamps(window_count: int,
                             window_seconds: float = AASIST_WINDOW_SECONDS) -> list[float]:
    """Deterministic window start-seconds (detector uses i*window_seconds)."""
    return [round(i * window_seconds, 3) for i in range(int(window_count or 0))]


def aasist_segment_pattern(window_scores) -> dict:
    """Classify the temporal pattern of per-window AASIST spoof scores.

    Deterministic rules over stored per-window scores:
      - no scores            -> "no_windows"
      - >=80% of windows high (>=0.8) and mean >= 0.8 -> "persistent_high_spoof"
      - >=80% of windows low  (<0.2)  and mean <= 0.2 -> "persistent_low_spoof"
      - otherwise            -> "mixed_or_localized"
    This is a pattern summary only — NOT a new threshold decision.
    """
    if not window_scores:
        return {"pattern": "no_windows", "n": 0, "mean": None,
                "min": None, "max": None, "pct_high_ge_0_8": None,
                "pct_low_lt_0_2": None, "high_windows": [], "low_windows": []}
    xs = [float(s) for s in window_scores]
    n = len(xs)
    hi = [i for i, s in enumerate(xs) if s >= 0.8]
    lo = [i for i, s in enumerate(xs) if s < 0.2]
    hi_pct = len(hi) / n
    lo_pct = len(lo) / n
    mean = sum(xs) / n
    if hi_pct >= 0.8 and mean >= 0.8:
        pattern = "persistent_high_spoof"
    elif lo_pct >= 0.8 and mean <= 0.2:
        pattern = "persistent_low_spoof"
    else:
        pattern = "mixed_or_localized"
    return {
        "pattern": pattern, "n": n,
        "mean": _round(mean, 6), "min": _round(min(xs), 6), "max": _round(max(xs), 6),
        "pct_high_ge_0_8": _round(hi_pct, 4), "pct_low_lt_0_2": _round(lo_pct, 4),
        "high_windows": hi, "low_windows": lo,
    }


# --------------------------------------------------------------------------
# Unified evidence assembly
# --------------------------------------------------------------------------

def assemble_video_evidence(row: dict, probe: Optional[dict] = None) -> dict:
    """Build one unified evidence record from one eval result row."""
    prov = row.get("provenance") or {}
    temporal = prov.get("temporal_analysis") or {}

    aw_scores = row.get("aasist_window_scores") or []
    aasist_pat = aasist_segment_pattern(aw_scores)

    heur_segs = row.get("audio_segments") or []
    heur_scores = [float(s.get("fake_prob")) for s in heur_segs]
    heuristic_shape = {
        "segments": len(heur_scores),
        "mean": _round(sum(heur_scores) / len(heur_scores), 4) if heur_scores else None,
        "min": _round(min(heur_scores), 4) if heur_scores else None,
        "max": _round(max(heur_scores), 4) if heur_scores else None,
        "pct_high_ge_0_6": _round(sum(1 for s in heur_scores if s >= 0.6) / len(heur_scores), 4) if heur_scores else None,
        "pct_low_le_0_4": _round(sum(1 for s in heur_scores if s <= 0.4) / len(heur_scores), 4) if heur_scores else None,
    }

    gc_found = row.get("genconvit_face_frames_found")
    gc_req = row.get("genconvit_frames_requested")

    identity = {
        "filename": row.get("filename"),
        "ground_truth": row.get("ground_truth"),
        "duration_seconds": row.get("duration_seconds"),
        "fps": row.get("fps"),
        "has_audio": row.get("has_audio"),
        "codec": (probe or {}).get("codec"),
        "resolution": (probe or {}).get("resolution"),
        "filesize_mb": (probe or {}).get("filesize_mb"),
        "bitrate_kbps": (probe or {}).get("bitrate_kbps"),
        "probe_ok": bool((probe or {}).get("probe_ok")),
    }

    production = {
        "verdict": row.get("verdict"),
        "fusion_score": _round(row.get("fusion_score")),
        "video_score": _round(row.get("video_score")),
        "audio_score": _round(row.get("audio_score")),
        "audio_status": row.get("audio_status"),
        "texture_score": _round(row.get("texture_score")),
        "frequency_score": _round(row.get("frequency_score")),
        "temporal_score": _round(temporal.get("score")),
        "provenance_score": _round(row.get("provenance_score")),
        "provenance_override": bool(row.get("provenance_override")),
        "provenance_reason": row.get("provenance_reason"),
        "uncorroborated_video_evidence": bool(prov.get("uncorroborated_video_evidence")),
        "classifier_status": row.get("classifier_status"),
        "classifier_score": row.get("classifier_score"),
        "classifier_reason": row.get("classifier_reason"),
        "error": row.get("error"),
    }

    genconvit = {
        "status": row.get("genconvit_status"),
        "score": _round(row.get("genconvit_score")),
        "predicted_label": row.get("genconvit_predicted_label"),
        "frames_requested": gc_req,
        "frames_sampled": row.get("genconvit_frames_sampled"),
        "face_frames_found": gc_found,
        "face_coverage": _round(row.get("genconvit_face_coverage")),
        "average_face_confidence": _round(row.get("genconvit_average_face_confidence")),
        "min_face_confidence": _round(row.get("genconvit_min_face_confidence")),
        "max_face_confidence": _round(row.get("genconvit_max_face_confidence")),
        "frames_used": row.get("genconvit_frames_used"),
        "coverage_status": row.get("genconvit_coverage_status") or row.get("genconvit_face_quality"),
        "frame_indices": row.get("genconvit_frame_indices"),
        "timestamps": row.get("genconvit_timestamps"),
        "face_sizes": row.get("genconvit_face_sizes"),
        "face_confidences": row.get("genconvit_face_confidences"),
        "frame_scores": row.get("genconvit_frame_scores"),
        "branches": row.get("genconvit_branches"),
        "load_error": row.get("genconvit_load_error"),
    }

    aasist = {
        "status": row.get("aasist_status"),
        "spoof_score": _round(row.get("aasist_spoof_score")),
        "bonafide_score": _round(row.get("aasist_bonafide_score")),
        "predicted_label": row.get("aasist_predicted_label"),
        "window_count": row.get("aasist_window_count"),
        "window_scores": row.get("aasist_window_scores") or [],
        "window_timestamps_sec": aasist_window_timestamps(row.get("aasist_window_count") or 0),
        "note": row.get("aasist_note"),
        "load_error": row.get("aasist_load_error"),
        "segment_pattern": aasist_pat,
    }

    audio_heuristic = {
        "status": row.get("audio_status"),
        "aggregate_score": _round(row.get("audio_score")),
        "segments": [
            {"segment_index": s.get("segment_index"),
             "start": _round(s.get("start")),
             "end": _round(s.get("end")),
             "fake_prob": _round(s.get("fake_prob"))}
            for s in heur_segs
        ],
        "shape": heuristic_shape,
    }

    temporal_texture = {
        "frame_count": row.get("frame_count"),
        "frame_scores": row.get("frame_scores") or [],
        "texture_score": production["texture_score"],
        "frequency_score": production["frequency_score"],
        "temporal_analysis": {
            "score": _round(temporal.get("score")),
            "findings": temporal.get("findings") or [],
        },
        "face_segment": prov.get("face_segment"),
        "non_face_segment": prov.get("non_face_segment"),
    }

    provenance = {
        "generator_tag": prov.get("generator_tag"),
        "has_c2pa": bool(prov.get("has_c2pa")),
        "provenance_score": _round(prov.get("provenance_score")),
        "provenance_findings": prov.get("provenance_findings") or [],
        "classifier_backend": prov.get("classifier_backend"),
        "face_detector_backend": prov.get("face_detector_backend"),
        "frames_analysed": prov.get("frames_analysed"),
        "documents_skipped": prov.get("documents_skipped"),
        "uncorroborated_video_evidence": bool(prov.get("uncorroborated_video_evidence")),
    }

    return {
        "filename": row.get("filename"),
        "identity": identity,
        "production": production,
        "genconvit": genconvit,
        "aasist": aasist,
        "audio_heuristic": audio_heuristic,
        "temporal_texture": temporal_texture,
        "provenance": provenance,
    }


# --------------------------------------------------------------------------
# Evidence taxonomy (Task 2) + agreement analysis (Tasks 3, 5, 8)
# --------------------------------------------------------------------------

def classify_evidence(ev: dict) -> list[dict]:
    """Deterministic category assignment with explicit reasons.

    Every entry declares its epistemic status:
      'observation'  — directly read from stored evidence / dataset label
      'hypothesis'   — inferred pattern that requires more data to validate
      'conclusion'   — a fact about the evidence itself (not about truth)
    """
    cats: list[dict] = []
    id_ = ev["identity"]
    prod = ev["production"]
    gc = ev["genconvit"]
    aa = ev["aasist"]
    tex = prod["texture_score"]
    prov = ev["provenance"]

    if not id_["has_audio"] or "No audio track" in (aa.get("note") or ""):
        cats.append({
            "category": "NO_AUDIO", "status": "observation",
            "reason": "no audio stream in container / AASIST note 'No audio track'",
        })

    if aa["status"] == "ACTIVE_MODEL" and aa["predicted_label"] == "SPOOF":
        cats.append({
            "category": "AUDIO_SPOOF_EVIDENCE", "status": "observation",
            "reason": (f"AASIST ACTIVE_MODEL; label=SPOOF, spoof_score="
                       f"{aa['spoof_score']}, segment_pattern="
                       f"{aa['segment_pattern']['pattern']} (n={aa['window_count']})"),
        })
    if aa["status"] == "ACTIVE_MODEL" and aa["predicted_label"] == "BONAFIDE":
        cats.append({
            "category": "AUTHENTIC_AUDIO_EVIDENCE", "status": "observation",
            "reason": (f"AASIST ACTIVE_MODEL; label=BONAFIDE, spoof_score="
                       f"{aa['spoof_score']}. This is an AUDIO-level statement; "
                       "it does NOT imply the video is real."),
        })

    cov = gc["coverage_status"]
    found = gc["face_frames_found"] or 0
    req = gc["frames_requested"]
    if gc["status"] == "ACTIVE_MODEL" and found == 0:
        cats.append({
            "category": "NO_FACE", "status": "observation",
            "reason": f"GenConViT ACTIVE but 0 usable face frames in {req} sampled frames",
        })
    if gc["status"] == "ACTIVE_MODEL" and cov in ("LOW_FACE_COVERAGE", "PARTIAL_FACE_COVERAGE"):
        cats.append({
            "category": "LOW_FACE_COVERAGE", "status": "observation",
            "reason": (f"coverage_status={cov}; face_frames_found={found}/{req} "
                       f"(coverage={gc['face_coverage']}), frames_used="
                       f"{gc['frames_used']}"),
        })
    if prod["video_score"] is not None and not gc["status"]:
        cats.append({
            "category": "LOW_VISUAL_COVERAGE", "status": "observation",
            "reason": "neural video model unavailable; texture heuristic only",
        })

    # GenConViT vs AASIST cross-modal disagreement (modality level).
    if gc["predicted_label"] in ("FAKE", "REAL") and aa["predicted_label"] in ("SPOOF", "BONAFIDE"):
        aa_video = "FAKE" if aa["predicted_label"] == "SPOOF" else "REAL"
        if gc["predicted_label"] != aa_video:
            cats.append({
                "category": "MODALITY_DISAGREEMENT", "status": "observation",
                "reason": (f"GenConViT={gc['predicted_label']} vs AASIST-mapped="
                           f"{aa_video} (audio modality); independent neural "
                           "modalities read manipulation differently"),
            })

    # Neural-vs-heuristic audio disagreement (Task 3C). The compression / codec
    # / telephony hypothesis is only asserted for videos the PROVIDED label
    # says are REAL, where a low-quality-domain cause is the lead reading; for
    # FAKE-labelled videos the AASIST-vs-heuristic split is also recorded as a
    # NEURAL_VS_HEURISTIC disagreement case (see disagreement_cases()) but is
    # NOT auto-labelled "compression risk".
    hs = ev["audio_heuristic"]["shape"]
    if (aa["status"] == "ACTIVE_MODEL" and aa["predicted_label"] == "SPOOF"
            and hs["mean"] is not None and hs["mean"] <= 0.4):
        reason_detail = (f"AASIST={aa['predicted_label']} (spoof_score="
                         f"{aa['spoof_score']}) while the production audio "
                         f"heuristic reads natural (mean={hs['mean']})")
        if id_["ground_truth"] == "REAL":
            extra = (f"; provided label is REAL — if the label is correct, the "
                     f"high spoof reading is a FALSE positive whose pattern is "
                     f"consistent with codec/telephony artifacts. Hypothesis, "
                     f"NOT proven.")
            if gc["predicted_label"] == "FAKE":
                extra += (f" Note: GenConViT ALSO reads FAKE here → two "
                          f"independent neural detectors converge toward "
                          f"manipulation on a REAL-labelled video (shared/"
                          f"correlated domain effect suspected).")
            cats.append({
                "category": "COMPRESSION_OR_CODEC_RISK", "status": "hypothesis",
                "reason": reason_detail + extra,
            })
        else:
            cats.append({
                "category": "MODALITY_DISAGREEMENT", "status": "observation",
                "reason": (reason_detail + "; on a FAKE-labelled video this is "
                           "an AASIST vs heuristic disagreement — the high "
                           "spoof reading is plausibly genuine synthetic audio "
                           "(observation), compression being only one hypothesis"),
            })
    if (aa["status"] == "ACTIVE_MODEL" and aa["predicted_label"] == "BONAFIDE"
            and hs["mean"] is not None and hs["mean"] >= 0.4):
        cats.append({
            "category": "MODALITY_DISAGREEMENT", "status": "observation",
            "reason": (f"AASIST=BONAFIDE but audio heuristic mean={hs['mean']} >= 0.4; "
                       "neural vs heuristic audio disagreement"),
        })

    # Provenance evidence.
    if prod["provenance_override"] or prov["generator_tag"] or prov["has_c2pa"]:
        cats.append({
            "category": "PROVENANCE_EVIDENCE", "status": "observation",
            "reason": (f"provenance_override={prod['provenance_override']}, "
                       f"generator_tag={prov['generator_tag']}, has_c2pa="
                       f"{prov['has_c2pa']}, provenance_score="
                       f"{prov['provenance_score']}"),
        })

    # Weak / ambiguous.
    if prod["verdict"] == "UNCERTAIN":
        cats.append({
            "category": "WEAK_OR_AMBIGUOUS_EVIDENCE", "status": "observation",
            "reason": f"production verdict UNCERTAIN (fusion_score={prod['fusion_score']})",
        })

    # Multimodal agreement (>=2 independent manipulation-oriented signals).
    manip = []
    if gc["predicted_label"] == "FAKE":
        manip.append("GenConViT=FAKE")
    if aa["predicted_label"] == "SPOOF":
        manip.append("AASIST=SPOOF")
    if tex is not None and tex >= 0.6:
        manip.append(f"texture={tex}")
    if prod["provenance_override"]:
        manip.append("provenance_override")
    if len(manip) >= 2:
        reason = (f"{len(manip)} independent manipulation-oriented signals "
                  f"co-occur: {', '.join(manip)}. This is an observation about "
                  f"signal agreement, not a truth claim.")
        if id_["ground_truth"] == "REAL":
            reason += (f" Provided label is REAL — the agreement CONFLICTS with "
                       f"the label, i.e. converging false positives; strongly "
                       f"suggests a shared/correlated error source (see "
                       f"COMPRESSION_OR_CODEC_RISK).")
        cats.append({
            "category": "MULTIMODAL_AGREEMENT", "status": "observation",
            "reason": reason,
        })

    return cats


def agreement_pattern(ev: dict) -> dict:
    """Compact per-video agreement summary for the Task 3 evidence matrix."""
    prod = ev["production"]
    gc = ev["genconvit"]
    aa = ev["aasist"]
    tex = prod["texture_score"]
    hs = ev["audio_heuristic"]["shape"]
    prov = ev["provenance"]
    cats = [c["category"] for c in ev.get("evidence_categories", [])]

    signals = {
        "production": prod["verdict"],
        "GenConViT": gc["predicted_label"] or ("unavailable" if gc["status"] != "ACTIVE_MODEL" else "weak"),
        "AASIST": aa["predicted_label"] or ("unavailable" if aa["status"] != "ACTIVE_MODEL" else "weak"),
        "texture": _flat(tex) if tex is not None else None,
        "audio_heuristic": _flat(hs["mean"]) if hs["mean"] is not None else None,
        "temporal": _flat(prod["temporal_score"]) if prod["temporal_score"] is not None else None,
        "provenance": "OVERRIDE" if prod["provenance_override"] else (
            "neutral" if prov["generator_tag"] is None and not prov["has_c2pa"] else "flagged"),
    }

    if "MULTIMODAL_AGREEMENT" in cats:
        label = "STRONG_MULTIMODAL_AGREEMENT"
    elif "MODALITY_DISAGREEMENT" in cats:
        label = "MODALITY_DISAGREEMENT"
    elif "COMPRESSION_OR_CODEC_RISK" in cats:
        label = "DOMAIN_RISK_DISAGREEMENT"
    elif "NO_AUDIO" in cats and "NO_FACE" in cats:
        label = "SINGLE_WEAK_MODALITY"
    elif "WEAK_OR_AMBIGUOUS_EVIDENCE" in cats:
        label = "AMBI"
    else:
        label = "MIXED"

    return {"pattern": label, "signals": signals, "categories": cats}


# --------------------------------------------------------------------------
# Corpus-level analyses (Tasks 3, 8) + production invariance (Task 14)
# --------------------------------------------------------------------------

def build_corpus(evals: dict, probe_existing: Optional[dict] = None) -> list[dict]:
    """Assemble unified + derived evidence for every labeled video."""
    probe = probe_existing or {}
    videos = []
    for row in evals.get("results", []):
        if row.get("ground_truth") not in ("FAKE", "REAL"):
            continue
        fn = row.get("filename")
        ev = assemble_video_evidence(row, probe.get(fn))
        cats = classify_evidence(ev)
        ev["evidence_categories"] = cats
        ev["agreement"] = agreement_pattern(ev)
        ev["probe_metadata"] = probe.get(fn) or {"probe_ok": False}
        videos.append(ev)
    return sorted(videos, key=lambda v: v["filename"].lower())


def verify_production_invariance(rows: list, baseline: Optional[dict] = None) -> dict:
    """Recompute the production confusion from stored verdicts and compare.

    Provenance: verdict / fusion_score fields come from the eval JSON produced
    by the UNCHANGED production pipeline. This function only reads them.
    """
    base = baseline or PRODUCTION_BASELINE
    labeled = [r for r in rows if r.get("ground_truth") in ("FAKE", "REAL")]
    n = len(labeled)
    tp = sum(1 for r in labeled if r["ground_truth"] == "FAKE" and r["verdict"] == "FAKE")
    tn = sum(1 for r in labeled if r["ground_truth"] == "REAL" and r["verdict"] in ("REAL", "UNCERTAIN"))
    fp = sum(1 for r in labeled if r["ground_truth"] == "REAL" and r["verdict"] == "FAKE")
    fn = sum(1 for r in labeled if r["ground_truth"] == "FAKE" and r["verdict"] != "FAKE")

    acc = (tp + tn) / n if n else None
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and (prec + rec)) else None
    fnr = fn / (tp + fn) if (tp + fn) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    fake_rate = sum(1 for r in labeled if r["verdict"] == "FAKE") / n if n else None
    unc_rate = sum(1 for r in labeled if r["verdict"] == "UNCERTAIN") / n if n else None

    computed = {
        "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "accuracy": acc,
        "precision_fake": prec,
        "recall_fake": rec,
        "FAKE_RECALL": rec,
        "f1_fake": f1,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "fake_verdict_rate": fake_rate,
        "uncertain_rate": unc_rate,
    }
    matches = computed["confusion"] == base.get("confusion")
    metric_keys = ["accuracy", "precision_fake", "recall_fake", "FAKE_RECALL",
                   "f1_fake", "false_negative_rate", "false_positive_rate",
                   "fake_verdict_rate", "uncertain_rate"]
    metric_deltas = {
        k: {"baseline": base.get(k), "computed": computed.get(k)}
        for k in metric_keys
        if base.get(k) is not None and computed.get(k) is not None
        and abs(float(base[k]) - float(computed[k])) > 1e-9
    }
    per_video = [{"filename": r["filename"], "ground_truth": r["ground_truth"],
                  "verdict": r["verdict"], "fusion_score": r.get("fusion_score")}
                 for r in labeled]
    return {
        "matches_phase2b_baseline": matches,
        "metric_deltas": metric_deltas,
        "baseline": base.get("confusion"),
        "computed": computed,
        "invariance_note": "Phase 2C performs analysis only; no production "
                           "detector, fusion, or threshold was modified.",
        "per_video_verdicts": per_video,
    }


def corpus_summary(videos: list[dict]) -> dict:
    """Count evidence categories + agreement patterns across the corpus."""
    cat_counts: dict[str, int] = {}
    pat_counts: dict[str, int] = {}
    for v in videos:
        for c in v["evidence_categories"]:
            cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1
        pat = v["agreement"]["pattern"]
        pat_counts[pat] = pat_counts.get(pat, 0) + 1
    return {
        "videos": len(videos),
        "by_ground_truth": {
            g: sum(1 for v in videos if v["identity"]["ground_truth"] == g)
            for g in ("FAKE", "REAL")
        },
        "evidence_categories": dict(sorted(cat_counts.items())),
        "agreement_patterns": dict(sorted(pat_counts.items())),
    }


def disagreement_cases(videos: list[dict]) -> list[dict]:
    """Rows for Tasks 3B/3C/3D (explicit disagreement callouts)."""
    cases = []
    for v in videos:
        gc = v["genconvit"]["predicted_label"]
        aa = v["aasist"]["predicted_label"]
        aa_video = "FAKE" if aa == "SPOOF" else ("REAL" if aa == "BONAFIDE" else None)
        heur = v["audio_heuristic"]["shape"]["mean"]
        if gc in ("FAKE", "REAL") and aa_video and gc != aa_video:
            cases.append({
                "filename": v["filename"], "ground_truth": v["identity"]["ground_truth"],
                "type": "NEURAL_VS_NEURAL",
                "GenConViT": gc, "AASIST": aa,
                "genconvit_score": v["genconvit"]["score"],
                "aasist_spoof_score": v["aasist"]["spoof_score"],
            })
        if aa == "SPOOF" and heur is not None and heur <= 0.4:
            cases.append({
                "filename": v["filename"], "ground_truth": v["identity"]["ground_truth"],
                "type": "NEURAL_VS_HEURISTIC",
                "AASIST": aa, "aasist_spoof_score": v["aasist"]["spoof_score"],
                "audio_heuristic_mean": heur,
            })
    return cases