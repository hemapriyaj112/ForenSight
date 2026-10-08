"""
v2_1/features.py
================
Builds the V2.1 second-stage feature table from ALREADY PERSISTED evidence.

Hard constraints
----------------
* No detector is re-run. No model is loaded. No media is decoded. Every
  number below is either read verbatim from a persisted Day-2..Day-11 CSV or
  deterministically derived from persisted per-frame / per-window scores
  using the SAME formulas the historical Day-9/Day-11 code used
  (``run_day9_experiment.build_feature_table`` and
  ``run_day11_xception_temporal.compute_temporal_descriptors`` are imported
  and reused, never re-implemented or re-tuned).
* The frozen 25-sample rank reference is used through
  ``day6_candidate`` only. It is never recomputed.
* Closed populations are refused by ``populations.assert_not_closed``.

Feature-set policy
------------------
``CORE``      - 14 raw-evidence features, rank-free, computable for BOTH
                development pools. Rank-free on purpose: the percentile rank
                is a function of the whole reference population, so using it
                as a model input makes a sample's feature depend on other
                samples' scores. On P4B_DEV25 the reference *is* the
                evaluation population, which makes rank features
                transductive. Rank features are therefore only used inside
                the frozen baseline and inside the D7-only ``EXTENDED`` set,
                where the reference is a fixed external population.
``MIN``       - 5-feature parsimony variant of CORE.
``EXTENDED``  - D7-only; CORE plus rank/disagreement and the already
                persisted Day-9/Day-10 window + identity statistics. Reported
                as a high-risk diagnostic only (n = 16).

Explicitly EXCLUDED from every feature set (label leakage):
``subtype``, ``source_group``, ``source``, ``acquisition``,
``exposure_tier``, ``partition``, ``filename``, ``path``, ``md5``,
``duration_s``, and every one-hot of them. In P4B_DEV25 ``source_group`` is
100% label-pure (13/13 groups), so it is a perfect label proxy.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from . import populations as pops

# --------------------------------------------------------------------------- #
# Artifact paths (read-only)
# --------------------------------------------------------------------------- #

ARTIFACTS = {
    # P4B frozen-reference population
    "p4b_frame": "reports/xception_devval_per_frame.csv",
    "p4b_video": "reports/xception_devval_predictions.csv",
    "p4b_audio": "reports/aasist_day4_predictions.csv",
    "p4b_split_manifest": "phase4b/manifests/phase4b_split_manifest.csv",
    "p4b_dataset_manifest": "phase4b/manifests/phase4b_combined_dataset_manifest.csv",
    "p4b_fusion_inputs": "reports/DAY5_FUSION_INPUTS.csv",
    "p4b_fusion_predictions": "reports/day5_fusion_predictions.csv",
    # D7 development population
    "d7_manifest": "reports/DAY7_FRESH_VALIDATION_MANIFEST.csv",
    "d7_frame": "reports/DAY9_RAW_FRAME_EVIDENCE.csv",
    "d7_window": "reports/DAY9_AASIST_WINDOW_EVIDENCE.csv",
    "d7_x_agg": "reports/DAY9_XCEPTION_VIDEO_AGGREGATES.csv",
    "d7_a_agg": "reports/DAY9_AASIST_VIDEO_SUMMARIES.csv",
    "d7_audio_robustness": "reports/DAY9_AUDIO_ROBUSTNESS.csv",
    "d7_identity": "reports/DAY10_VISUAL_IDENTITY_SUMMARY.csv",
    # closed / diagnostic (audit + baselines only, never fitted)
    "locked_raw": "reports/V2_LOCKED_RAW_PREDICTIONS.csv",
    "locked_inputs": "reports/V2_LOCKED_EVALUATION_INPUTS.csv",
    "d7_fresh_predictions": "reports/DAY7_FRESH_PREDICTIONS.csv",
    "d7_baseline_predictions": "reports/DAY7_BASELINE_PREDICTIONS.csv",
    "d8_failure_mode_table": "reports/DAY8_FAILURE_MODE_TABLE.csv",
    "d9_dev_predictions": "reports/DAY9_DEVELOPMENT_PREDICTIONS.csv",
    "d9_holdout_predictions": "reports/DAY9_HONEST_HOLDOUT_PREDICTIONS.csv",
    "d10_dev_predictions": "reports/DAY10_DEVELOPMENT_PREDICTIONS.csv",
    "d10_holdout_predictions": "reports/DAY10_HONEST_HOLDOUT_PREDICTIONS.csv",
    "d11_descriptors": "reports/DAY11_XCEPTION_TEMPORAL_DESCRIPTORS.csv",
    "d11_dev_predictions": "reports/DAY11_DEVELOPMENT_PREDICTIONS.csv",
    "d11_holdout_predictions": "reports/DAY11_HONEST_HOLDOUT_PREDICTIONS.csv",
    "d5_crossvalidation": "reports/DAY5_CROSSVALIDATION.json",
    "candidate_freeze": "reports/V2_CANDIDATE_FREEZE.json",
    "final_summary": "reports/FORENSIGHT_V2_FINAL_SUMMARY.json",
}


def artifact_path(key: str, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else pops.REPO_ROOT
    return base / ARTIFACTS[key]


# --------------------------------------------------------------------------- #
# Small CSV helpers (no pandas dependency in the experiment core)
# --------------------------------------------------------------------------- #

def read_csv_rows(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def fnum(value, default=None):
    """Day-9/Day-11 numeric coercion: '' / 'nan' / None -> ``default``."""
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return None if isinstance(value, float) and math.isnan(value) else value
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    if math.isnan(out):
        return default
    return out


def by_id(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        sid = str(row.get("sample_id", "")).strip()
        if sid:
            out[sid] = row
    return out


# --------------------------------------------------------------------------- #
# Predeclared feature sets
# --------------------------------------------------------------------------- #

CORE_FEATURES = [
    # visual - video level
    "x_mean",
    "x_std",
    "x_q90",
    "x_max",
    # visual - temporal shape / persistence (Day-11 descriptors)
    "x_frac_ge_060",
    "x_longest_run_060",
    "x_burst_coverage_050",
    "x_half_delta_abs",
    # visual - evidence availability
    "x_face_fraction",
    # audio - persisted Day-4 / Day-9 aggregates
    "a_mean",
    "a_iqr",
    "a_max",
    "a_n_windows",
    # cross-modal disagreement proxy (raw, rank-free)
    "xa_abs_diff",
]

MIN_FEATURES = [
    "x_mean",
    "x_q90",
    "x_frac_ge_060",
    "a_mean",
    "a_max",
]

#: D7-only. Rank features are legitimate here (fixed external reference) but
#: the set is deliberately small and is reported as a high-risk diagnostic.
EXTENDED_FEATURES = CORE_FEATURES + [
    "rank_x",
    "rank_a",
    "fused_v2",
    "rank_abs_gap",
    "a_std",
    "a_q90",
    "a_high_frac",
    "a_low_frac",
    "x_very_high_frac",
    "x_temporal_consistency",
]

FEATURE_SETS = {
    "CORE": CORE_FEATURES,
    "MIN": MIN_FEATURES,
    "EXTENDED": EXTENDED_FEATURES,
}

LEAKAGE_COLUMNS = [
    "subtype", "source_group", "source", "acquisition", "exposure_tier",
    "partition", "filename", "path", "md5", "duration_s", "true_label",
]


# --------------------------------------------------------------------------- #
# P4B_DEV25 - temporal descriptors from persisted per-frame Xception evidence
# --------------------------------------------------------------------------- #

def _p4b_frame_rows_as_day11_rows(rows: list[dict]) -> list[dict]:
    """Adapt ``xception_devval_per_frame.csv`` rows to the Day-11 evidence
    shape so the historical descriptor code can be reused verbatim.

    Xception samples at 1 FPS (requested_max_frames=16, frame_pool_fps=1), so
    ``timestamp_s`` is exactly ``frame_index`` seconds. Verified against
    DAY9_RAW_FRAME_EVIDENCE.csv where both exist.
    """
    out = []
    for r in rows:
        frame_index = fnum(r.get("frame_index"), 0.0)
        out.append({
            "sample_id": str(r.get("sample_id")),
            "frame_index": frame_index,
            "timestamp_s": frame_index,
            "face_detected": r.get("face_detected"),
            "fake_probability": r.get("fake_probability"),
        })
    return out


def build_p4b_pool(root: Path | None = None) -> list[dict]:
    """One feature row per P4B frozen-reference sample (n = 25)."""
    import day6_candidate as cand
    import run_day11_xception_temporal as d11

    ids = pops.frozen_reference_ids()
    pops.assert_not_closed(ids, "build_p4b_pool")

    video = by_id(read_csv_rows(artifact_path("p4b_video", root)))
    audio = by_id(read_csv_rows(artifact_path("p4b_audio", root)))
    split = by_id(read_csv_rows(artifact_path("p4b_split_manifest", root)))
    dataset = by_id(read_csv_rows(artifact_path("p4b_dataset_manifest", root)))

    frames_by: dict[str, list[dict]] = {}
    for r in _p4b_frame_rows_as_day11_rows(
            read_csv_rows(artifact_path("p4b_frame", root))):
        frames_by.setdefault(r["sample_id"], []).append(r)

    rows: list[dict] = []
    for sid in ids:
        vrow, arow = video[sid], audio[sid]
        n_sampled = int(fnum(vrow.get("sampled_frames_count"), 0) or 0)
        seed_row = {
            "frames_sampled": n_sampled,
            "x_mean": fnum(vrow.get("video_score_mean")),
            "exposure_tier": "frozen-reference-devval",
        }
        desc = d11.compute_temporal_descriptors(
            sid, str(vrow.get("true_label")).strip(), seed_row,
            frames_by.get(sid, []))

        a_mean = fnum(arow.get("audio_fake_score"))
        x_mean = fnum(vrow.get("video_score_mean"))
        rank_x = cand.percentile_rank_against_reference(x_mean, cand.XCEPTION_REF)
        rank_a = cand.percentile_rank_against_reference(a_mean, cand.AASIST_REF)
        fused = (pops.FROZEN_X_WEIGHT * rank_x
                 + pops.FROZEN_A_WEIGHT * rank_a)

        n_usable = desc.get("n_usable") or 0
        a_iqr = fnum(arow.get("audio_fake_score_iqr"))
        if a_iqr is None:
            a_med = fnum(arow.get("audio_fake_score_median"))
            lo, hi = fnum(arow.get("audio_fake_score_min")), fnum(arow.get("audio_fake_score_max"))
            a_iqr = (hi - lo) if (a_med is not None and lo is not None and hi is not None) else None

        row = {
            "pool": "P4B_DEV25",
            "sample_id": sid,
            "true_label": str(vrow.get("true_label")).strip(),
            "y": 1 if str(vrow.get("true_label")).strip() == "FAKE" else 0,
            "n_per_frame_rows": len(frames_by.get(sid, [])),
            "n_usable": n_usable,
            "frames_sampled": n_sampled,
            "x_n_usable": n_usable,
            "x_face_fraction": desc.get("face_fraction"),
            "evidence_status": desc.get("evidence_status"),
            "x_mean": desc.get("mean") if desc.get("mean") is not None else x_mean,
            "x_median": desc.get("median"),
            "x_std": desc.get("std"),
            "x_min": desc.get("min"),
            "x_max": desc.get("max"),
            "x_iqr": desc.get("iqr"),
            "x_q75": desc.get("q75"),
            "x_q90": desc.get("q90"),
            "x_q95": desc.get("q95"),
            "x_frac_ge_040": desc.get("frac_ge_0.40"),
            "x_frac_ge_050": desc.get("frac_ge_0.50"),
            "x_frac_ge_060": desc.get("frac_ge_0.60"),
            "x_frac_ge_070": desc.get("frac_ge_0.70"),
            "x_longest_run_050": desc.get("longest_run_0.50"),
            "x_longest_run_060": desc.get("longest_run_0.60"),
            "x_longest_run_070": desc.get("longest_run_0.70"),
            "x_n_transitions_050": desc.get("n_transitions_0.50"),
            "x_n_transitions_060": desc.get("n_transitions_0.60"),
            "x_burst_coverage_050": desc.get("burst_coverage_0.50"),
            "x_first_half_mean": desc.get("first_half_mean"),
            "x_second_half_mean": desc.get("second_half_mean"),
            "x_half_delta_abs": desc.get("half_delta_abs"),
            "a_mean": a_mean,
            "a_median": fnum(arow.get("audio_fake_score_median")),
            "a_min": fnum(arow.get("audio_fake_score_min")),
            "a_max": fnum(arow.get("audio_fake_score_max")),
            "a_iqr": a_iqr,
            "a_bonafide": fnum(arow.get("audio_bonafide_score")),
            "a_n_windows": fnum(arow.get("num_windows")),
            "rank_x": rank_x,
            "rank_a": rank_a,
            "fused_v2": fused,
            "rank_abs_gap": abs(rank_x - rank_a),
            "xa_abs_diff": (abs(x_mean - a_mean)
                            if (x_mean is not None and a_mean is not None) else None),
            "x_available": x_mean is not None,
            "a_available": a_mean is not None,
            # grouping only - never a model input
            "group_source_group": split.get(sid, {}).get("source_group"),
            "split": split.get(sid, {}).get("split"),
            "md5": dataset.get(sid, {}).get("md5"),
        }
        rows.append(row)

    pops.assert_not_closed([r["sample_id"] for r in rows], "build_p4b_pool result")
    return rows


# --------------------------------------------------------------------------- #
# D7_DEV16 - reuse the Day-9 / Day-10 / Day-11 feature builders verbatim
# --------------------------------------------------------------------------- #

def build_d7_pool(root: Path | None = None) -> list[dict]:
    """One feature row per Day-9/10/11 DEV sample (n = 16)."""
    import day6_candidate as cand
    import run_day9_experiment as d9
    import run_day11_xception_temporal as d11

    dev_ids = sorted(pops.D7_DEV_IDS)
    pops.assert_not_closed(dev_ids, "build_d7_pool")

    feats = {r["sample_id"]: r for r in d9.build_feature_table()}
    frames_by: dict[str, list[dict]] = {}
    for r in d11.load_frame_rows():
        frames_by.setdefault(r["sample_id"], []).append(r)

    identity = by_id(read_csv_rows(artifact_path("d7_identity", root)))
    manifest = by_id(read_csv_rows(artifact_path("d7_manifest", root)))

    a_agg = by_id(read_csv_rows(artifact_path("d7_a_agg", root)))
    x_agg = by_id(read_csv_rows(artifact_path("d7_x_agg", root)))

    rows: list[dict] = []
    for sid in dev_ids:
        src = feats[sid]
        desc = d11.compute_temporal_descriptors(
            sid, src["true_label"], src, frames_by.get(sid, []))
        arow = a_agg.get(sid, {})
        xrow = x_agg.get(sid, {})
        idrow = identity.get(sid, {})
        mrow = manifest.get(sid, {})

        x_mean, a_mean = src["x_mean"], src["a_mean"]
        rank_x = cand.percentile_rank_against_reference(x_mean, cand.XCEPTION_REF)
        rank_a = cand.percentile_rank_against_reference(a_mean, cand.AASIST_REF)
        fused = (pops.FROZEN_X_WEIGHT * rank_x
                 + pops.FROZEN_A_WEIGHT * rank_a)

        row = {
            "pool": "D7_DEV16",
            "sample_id": sid,
            "true_label": src["true_label"],
            "y": 1 if src["true_label"] == "FAKE" else 0,
            "n_per_frame_rows": src.get("n_per_frame_rows"),
            "n_per_window_rows": src.get("n_per_window_rows"),
            "n_usable": desc.get("n_usable"),
            "frames_sampled": desc.get("frames_sampled"),
            "x_n_usable": desc.get("n_usable"),
            "x_face_fraction": desc.get("face_fraction"),
            "evidence_status": desc.get("evidence_status"),
            "x_mean": desc.get("mean") if desc.get("mean") is not None else x_mean,
            "x_median": desc.get("median"),
            "x_std": desc.get("std") if desc.get("std") is not None
                     else fnum(xrow.get("x_std")),
            "x_min": desc.get("min"),
            "x_max": desc.get("max"),
            "x_iqr": desc.get("iqr"),
            "x_q75": desc.get("q75"),
            "x_q90": desc.get("q90"),
            "x_q95": desc.get("q95"),
            "x_frac_ge_040": desc.get("frac_ge_0.40"),
            "x_frac_ge_050": desc.get("frac_ge_0.50"),
            "x_frac_ge_060": desc.get("frac_ge_0.60"),
            "x_frac_ge_070": desc.get("frac_ge_0.70"),
            "x_longest_run_050": desc.get("longest_run_0.50"),
            "x_longest_run_060": desc.get("longest_run_0.60"),
            "x_longest_run_070": desc.get("longest_run_0.70"),
            "x_n_transitions_050": desc.get("n_transitions_0.50"),
            "x_n_transitions_060": desc.get("n_transitions_0.60"),
            "x_burst_coverage_050": desc.get("burst_coverage_0.50"),
            "x_first_half_mean": desc.get("first_half_mean"),
            "x_second_half_mean": desc.get("second_half_mean"),
            "x_half_delta_abs": desc.get("half_delta_abs"),
            "x_high_frac": fnum(xrow.get("x_high_frac")),
            "x_very_high_frac": fnum(xrow.get("x_very_high_frac")),
            "x_temporal_consistency": fnum(xrow.get("temporal_consistency")),
            "a_mean": a_mean,
            "a_median": fnum(arow.get("a_median")),
            "a_min": fnum(arow.get("a_min")),
            "a_max": fnum(arow.get("a_max")),
            "a_iqr": fnum(arow.get("a_iqr")),
            "a_std": fnum(arow.get("a_std")),
            "a_q90": fnum(arow.get("a_q90")),
            "a_high_frac": fnum(arow.get("a_high_frac")),
            "a_low_frac": fnum(arow.get("a_low_frac")),
            "a_pattern": arow.get("a_pattern"),
            "a_n_windows": fnum(arow.get("n_windows")),
            "rank_x": rank_x,
            "rank_a": rank_a,
            "fused_v2": fused,
            "rank_abs_gap": abs(rank_x - rank_a),
            "xa_abs_diff": (abs(x_mean - a_mean)
                            if (x_mean is not None and a_mean is not None) else None),
            "x_available": x_mean is not None,
            "a_available": a_mean is not None,
            "audio_risk": src.get("audio_risk"),
            "domain_risk_state": src.get("domain_risk_state"),
            "id_cos_mean": fnum(idrow.get("id_cos_mean")),
            "id_low_frac": fnum(idrow.get("id_low_frac")),
            "face_persistence": fnum(idrow.get("face_persistence")),
            # grouping only - never a model input
            "group_source_group": src.get("subtype"),
            "split": src.get("source"),
            "md5": mrow.get("md5"),
        }
        rows.append(row)

    pops.assert_not_closed([r["sample_id"] for r in rows], "build_d7_pool result")
    return rows


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def build_feature_table(root: Path | None = None) -> dict[str, list[dict]]:
    """Return ``{"P4B_DEV25": [...], "D7_DEV16": [...]}``."""
    table = {
        "P4B_DEV25": build_p4b_pool(root),
        "D7_DEV16": build_d7_pool(root),
    }
    for name, rows in table.items():
        pops.assert_not_closed([r["sample_id"] for r in rows],
                                f"feature table {name}")
    return table


def pooled_rows(table: dict[str, list[dict]]) -> list[dict]:
    """Explicit concatenation of the two development pools.

    SENSITIVITY USE ONLY. Callers must report a per-pool breakdown; the two
    pools come from different source collections and are never presented as
    a single homogeneous population.
    """
    return list(table["P4B_DEV25"]) + list(table["D7_DEV16"])


def feature_matrix(rows: list[dict], names: list[str]) -> list[list]:
    """Rows -> design matrix with ``None`` for missing (imputed IN-FOLD)."""
    out = []
    for r in rows:
        out.append([fnum(r.get(name)) for name in names])
    return out


def label_vector(rows: list[dict]) -> list[int]:
    return [int(r["y"]) for r in rows]


def available_features(rows: list[dict], names: list[str]) -> list[str]:
    """Subset of ``names`` that is present (non-null) for every row."""
    keep = []
    for name in names:
        if all(fnum(r.get(name)) is not None for r in rows):
            keep.append(name)
    return keep
