"""
v2_1/run_v2_1_accuracy_experiment.py
====================================
Driver for the V2.1 ACCURACY IMPROVEMENT EXPERIMENT.

Run:
    py -3.13 -m phase4b.experimental_detectors.v2_1.run_v2_1_accuracy_experiment

or, from the repo root:
    py -3.13 phase4b/experimental_detectors/v2_1/run_v2_1_accuracy_experiment.py

What it does
------------
1. Audits the persisted Day-2..Day-15 artifacts (read-only).
2. Builds the V2.1 second-stage feature table from persisted evidence only.
3. Verifies that the frozen V2 candidate still reproduces its historical
   Day-5 OOF confusion matrix on the P4B development pool.
4. Evaluates the pre-declared baseline + learned candidates with overlap-safe
   out-of-fold cross-validation, in-fold preprocessing only.
5. Writes NEW artifacts under ``reports/v2_1_accuracy_experiment/``.

What it does NOT do
-------------------
* No detector, model or media is executed. No download. No retraining.
* V1, the frozen V2 candidate, the frozen rank reference, the locked
  evaluation, and every Day-5..Day-15 report are read-only inputs.
* The production inference path and the production UI are untouched.
* Nothing here is a promotion. The project remains
  RESEARCH / EXPERIMENTAL / UNVALIDATED.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):  # allow direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from phase4b.experimental_detectors.v2_1 import (  # type: ignore
        artifact_audit, candidates, evaluation, features, populations,
        reporting)
else:
    from . import (artifact_audit, candidates, evaluation, features,
                   populations, reporting)

OUT = populations.V2_1_REPORTS
STATUS = "RESEARCH / EXPERIMENTAL / UNVALIDATED"
TARGET_ACCURACY = 0.80
TARGET_RECALL = 0.70
SCHEMES = (evaluation.PRIMARY_SCHEME, "LOO", "CV_SGK5")

METRIC_CSV_COLUMNS = [
    "pool", "candidate_id", "family", "feature_set", "n_features",
    "cv_scheme", "threshold_mode", "n", "n_fake", "n_real",
    "oof_accuracy", "oof_accuracy_std", "oof_accuracy_min", "oof_accuracy_max",
    "oof_precision", "oof_fake_recall", "oof_fake_recall_std",
    "oof_real_specificity", "oof_fpr", "oof_fpr_std", "oof_f1", "oof_f1_std",
    "oof_roc_auc", "oof_roc_auc_std", "oof_pr_auc",
    "tp", "tn", "fp", "fn", "predicted_fake_rate",
    "n_folds", "degenerate_folds", "degenerate_fold_fraction",
    "meets_accuracy_target", "meets_recall_target", "not_dominant_class_only",
    "exploratory", "optimistic_baseline", "description",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _r(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, digits)
    return value


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# baseline reproduction check
# --------------------------------------------------------------------------- #

HISTORICAL_DAY5_OOF = {
    "population": "P4B_DEV25",
    "candidate": "Xception+AASIST_rank_OOF",
    "tp": 13, "tn": 6, "fp": 2, "fn": 4,
    "accuracy": 0.76, "fake_recall": 0.7647, "fpr": 0.25, "f1": 0.8125,
    "source": "reports/DAY5_CROSSVALIDATION.json",
}


def verify_frozen_baseline(p4b_rows: list[dict]) -> dict:
    spec = next(c for c in candidates.BASELINES
                if c.candidate_id == "BASE_FROZEN_FUSION_V2")
    res = evaluation.evaluate_baseline("P4B_DEV25", spec, p4b_rows)
    a = res.aggregate
    got = {"tp": a["tp"], "tn": a["tn"], "fp": a["fp"], "fn": a["fn"],
           "accuracy": round(a["accuracy"], 4),
           "fake_recall": round(a["fake_recall"], 4),
           "fpr": round(a["fpr"], 4), "f1": round(a["f1"], 4)}
    want = {k: HISTORICAL_DAY5_OOF[k] for k in got}
    return {
        "historical": {**want, "source": HISTORICAL_DAY5_OOF["source"],
                       "candidate": HISTORICAL_DAY5_OOF["candidate"]},
        "recomputed": got,
        "matches": got == want,
        "note": ("confirms the frozen 25-sample rank reference was NOT "
                 "recomputed and the frozen V2 candidate is unchanged"),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/5] auditing persisted artifacts ...", flush=True)
    audit = artifact_audit.audit()

    print("[2/5] building V2.1 feature table from persisted evidence ...",
          flush=True)
    table = features.build_feature_table()
    frozen_check = verify_frozen_baseline(table["P4B_DEV25"])
    if not frozen_check["matches"]:
        raise SystemExit(
            "frozen V2 baseline reproduction FAILED - the rank reference or the "
            f"frozen candidate changed: {frozen_check}")

    print("[3/5] running overlap-safe OOF cross-validation ...", flush=True)
    results = []
    for pool, rows in table.items():
        print(f"      pool {pool} (n={len(rows)}) ...", flush=True)
        results.extend(evaluation.evaluate_pool(pool, rows, SCHEMES))
    # explicit pooled sensitivity view
    pooled_rows = features.pooled_rows(table)
    pooled = evaluation.evaluate_pool("POOLED41_SENSITIVITY", pooled_rows,
                                      (evaluation.PRIMARY_SCHEME,))
    results.extend(pooled)

    # ---------------- metrics CSV ---------------------------------------- #
    label_counts = {}
    for pool, rows in list(table.items()) + [("POOLED41_SENSITIVITY", pooled_rows)]:
        label_counts[pool] = {
            "n": len(rows),
            "n_fake": sum(1 for r in rows if r["y"] == 1),
            "n_real": sum(1 for r in rows if r["y"] == 0),
        }

    metric_rows = []
    for r in results:
        a = r.aggregate
        lc = label_counts[r.pool]
        degen = a.get("degenerate_folds", 0)
        n_folds = max(a.get("n_folds", 1), 1)
        metric_rows.append({
            "pool": r.pool,
            "candidate_id": r.candidate_id,
            "family": r.family,
            "feature_set": r.feature_set,
            "n_features": r.n_features,
            "cv_scheme": r.scheme,
            "threshold_mode": r.threshold_mode,
            "n": lc["n"], "n_fake": lc["n_fake"], "n_real": lc["n_real"],
            "oof_accuracy": _r(a["accuracy"]),
            "oof_accuracy_std": _r(a["accuracy_agg"]["std"]),
            "oof_accuracy_min": _r(a["accuracy_agg"]["min"]),
            "oof_accuracy_max": _r(a["accuracy_agg"]["max"]),
            "oof_precision": _r(a["precision"]),
            "oof_fake_recall": _r(a["fake_recall"]),
            "oof_fake_recall_std": _r(a["fake_recall_agg"]["std"]),
            "oof_real_specificity": _r(a["real_specificity"]),
            "oof_fpr": _r(a["fpr"]),
            "oof_fpr_std": _r(a["fpr_agg"]["std"]),
            "oof_f1": _r(a["f1"]),
            "oof_f1_std": _r(a["f1_agg"]["std"]),
            "oof_roc_auc": _r(a.get("roc_auc")),
            "oof_roc_auc_std": _r(a.get("roc_auc_std")),
            "oof_pr_auc": _r(a.get("pr_auc")),
            "tp": _r(a["tp"], 3), "tn": _r(a["tn"], 3),
            "fp": _r(a["fp"], 3), "fn": _r(a["fn"], 3),
            "predicted_fake_rate": _r(a["predicted_fake_rate"]),
            "n_folds": n_folds,
            "degenerate_folds": degen,
            "degenerate_fold_fraction": _r(degen / n_folds, 4),
            "meets_accuracy_target": bool(a["accuracy"] >= TARGET_ACCURACY),
            "meets_recall_target": bool(a["fake_recall"] >= TARGET_RECALL),
            "not_dominant_class_only": bool(
                a["predicted_fake_rate"] < 0.95
                and a["real_specificity"] > 0.0),
            "exploratory": bool(r.notes.get("exploratory", False)),
            "optimistic_baseline": bool(r.notes.get("optimistic_baseline", False)),
            "description": r.notes.get("description", r.notes.get("rule", "")),
        })
    _write_csv(OUT / "V2_1_CANDIDATE_OOF_METRICS.csv", METRIC_CSV_COLUMNS,
               metric_rows)

    # ---------------- best candidate selection -------------------------- #
    selection = select_candidate(results, table, label_counts)
    best = selection["selected"]
    best_role = ("SELECTED_FOR_IMPLEMENTATION"
                 if selection["selected_for_implementation"]
                 else "STRONGEST_NOT_ADMISSIBLE_REPORTED_FOR_TRANSPARENCY")
    best_pred_rows = []
    if best is not None:
        pool_rows = (table.get(best.pool)
                     or (pooled_rows if best.pool.startswith("POOLED") else None))
        by_id = {r["sample_id"]: r for r in (pool_rows or [])}
        for sid, score, pred in zip(best.sample_ids, best.oof_score,
                                    best.oof_pred):
            row = by_id.get(sid, {})
            best_pred_rows.append({
                "role": best_role,
                "candidate_id": best.candidate_id,
                "threshold_mode": best.threshold_mode,
                "cv_scheme": best.scheme,
                "pool": best.pool,
                "sample_id": sid,
                "true_label": row.get("true_label", ""),
                "y": row.get("y", ""),
                "oof_score": _r(score),
                "oof_pred_fake": pred,
                "correct": (int(pred) == int(row.get("y", -1))
                            if row.get("y") is not None else ""),
                "evidence_status": row.get("evidence_status", ""),
                "x_mean": _r(features.fnum(row.get("x_mean"))),
                "a_mean": _r(features.fnum(row.get("a_mean"))),
                "fused_v2": _r(features.fnum(row.get("fused_v2"))),
            })
    _write_csv(OUT / "V2_1_BEST_CANDIDATE_OOF_PREDICTIONS.csv",
               ["role", "candidate_id", "threshold_mode", "cv_scheme", "pool",
                "sample_id", "true_label", "y", "oof_score",
                "oof_pred_fake", "correct", "evidence_status", "x_mean",
                "a_mean", "fused_v2"],
               best_pred_rows)

    # ---------------- feature table CSV ---------------------------------- #
    feat_cols = (["pool", "sample_id", "true_label", "y", "evidence_status",
                  "group_source_group"]
                + features.CORE_FEATURES
                + [c for c in features.EXTENDED_FEATURES
                   if c not in features.CORE_FEATURES])
    all_rows = list(table["P4B_DEV25"]) + list(table["D7_DEV16"])
    _write_csv(OUT / "V2_1_FEATURE_TABLE.csv", feat_cols, [
        {**{k: _r(features.fnum(r.get(k))) for k in feat_cols},
         "pool": r["pool"], "sample_id": r["sample_id"],
         "true_label": r["true_label"], "y": r["y"],
         "evidence_status": r.get("evidence_status", ""),
         "group_source_group": r.get("group_source_group", "")}
        for r in all_rows])

    # ---------------- fold-level CSV ------------------------------------ #
    fold_rows = []
    for r in results:
        for f in r.fold_metrics:
            fold_rows.append({
                "pool": r.pool, "candidate_id": r.candidate_id,
                "cv_scheme": r.scheme, "threshold_mode": r.threshold_mode,
                "fold": f.get("fold"), "repeat": f.get("repeat", 0),
                "n_test": f.get("n_test", r.aggregate.get("n")),
                "accuracy": _r(f.get("accuracy")),
                "precision": _r(f.get("precision")),
                "fake_recall": _r(f.get("fake_recall")),
                "fpr": _r(f.get("fpr")),
                "f1": _r(f.get("f1")),
                "degenerate": f.get("degenerate", False),
            })
    _write_csv(OUT / "V2_1_FOLD_METRICS.csv",
               ["pool", "candidate_id", "cv_scheme", "threshold_mode", "fold",
                "repeat", "n_test", "accuracy", "precision", "fake_recall",
                "fpr", "f1", "degenerate"], fold_rows)

    # ---------------- JSON result ---------------------------------------- #
    print("[4/5] writing machine-readable outputs ...", flush=True)
    result = {
        "artifact": "V2_1_ACCURACY_EXPERIMENT_RESULT",
        "version": "V2.1",
        "status": STATUS,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": {
            "type": "experimental model-selection / analysis pass",
            "detector_retraining": False,
            "inference_rerun": False,
            "new_data_collected": False,
            "production_path_modified": False,
            "frozen_v2_candidate_modified": False,
            "frozen_rank_reference_recomputed": False,
        },
        "frozen_v2": {
            "candidate": "Xception+AASIST_rank_fusion_v1",
            "xception_weight": populations.FROZEN_X_WEIGHT,
            "aasist_weight": populations.FROZEN_A_WEIGHT,
            "threshold": populations.FROZEN_THRESHOLD,
            "rank_reference_count": populations.FROZEN_REFERENCE_COUNT,
            "rank_reference_recomputed": False,
        },
        "frozen_baseline_reproduction_check": frozen_check,
        "data_integrity": {
            "closed_populations": populations.CLOSED_POPULATIONS,
            "development_pools": populations.pool_definitions(),
            "pooled_view_is_sensitivity_only": True,
            "leakage_controls": [
                "Day-6 locked 8 excluded from every fit/selection/threshold step",
                "Day-9/10/11 honest holdout 5 excluded (already consumed)",
                "D7 visual-unavailable 8 and P4B visual-unavailable 3 not scored",
                "frozen 25-sample rank reference read through day6_candidate only",
                "imputation median, scaler, model and threshold all fitted "
                "inside the training fold",
                "no training-set accuracy computed or reported",
                "label-proxy columns (subtype, source_group, source, acquisition, "
                "exposure_tier, partition, filename, path, md5) excluded from "
                "every feature set; P4B source_group is 100% label-pure",
                "no sample appears in two development pools (disjoint id "
                "namespaces, zero shared md5)",
            ],
        },
        "feature_sets": {k: list(v) for k, v in features.FEATURE_SETS.items()},
        "excluded_leakage_columns": features.LEAKAGE_COLUMNS,
        "candidates": {
            "baselines": [
                {"candidate_id": c.candidate_id, "family": c.family,
                 "rule": c.rule, "description": c.description,
                 "optimistic": c.optimistic} for c in candidates.BASELINES],
            "learned": [
                {"candidate_id": c.candidate_id, "family": c.family,
                 "feature_set": c.feature_set, "description": c.description,
                 "exploratory": c.exploratory} for c in candidates.LEARNED],
            "threshold_modes": list(candidates.THRESHOLD_MODES),
            "predeclared": True,
        },
        "validation": {
            "primary_scheme": evaluation.PRIMARY_SCHEME,
            "primary_scheme_detail": (
                "RepeatedStratifiedKFold(n_splits=5, n_repeats=10, "
                f"random_state={evaluation.CV_SEED})"),
            "sensitivity_schemes": {
                "LOO": "LeaveOneOut, in-fold preprocessing",
                "CV_SGK5": ("StratifiedGroupKFold(5) grouped by source family "
                            "(P4B source_group / D7 subtype). Both groupings are "
                            "label-pure, so this is a SOURCE-TRANSFER STRESS TEST, "
                            "not an independence-corrected estimate"),
            },
            "fpr_constraint": candidates.FPR_CONSTRAINT,
            "primary_probability_threshold": candidates.PRIMARY_PROB_THRESHOLD,
        },
        "populations": label_counts,
        "results": [serialise_result(r) for r in results],
        "selection": serialise_selection(selection),
        "target": {
            "primary_oof_accuracy": TARGET_ACCURACY,
            "secondary_fake_recall": TARGET_RECALL,
            "reached_80_percent": selection["reached_80_percent"],
            "reached_80_percent_definition":
                selection["reached_80_percent_definition"],
            "max_observed_accuracy_by_pool":
                selection["max_observed_accuracy_by_pool"],
            "interpretation_rule": (
                "any >=0.80 figure is a 'V2.1 development OOF result' only; it is "
                "NOT validated accuracy, NOT real-world accuracy and NOT "
                "production accuracy"),
        },
        "artifact_audit_summary": {
            "n_artifacts": len(audit["artifacts"]),
            "missing": [a["artifact_key"] for a in audit["artifacts"]
                        if not a["exists"]],
            "cross_pool_byte_identical_overlap":
                audit["cross_pool_content_hash"]["byte_identical_overlap"],
        },
    }
    (OUT / "V2_1_EXPERIMENT_RESULT.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")

    print("[5/5] writing markdown reports ...", flush=True)
    _write_text(OUT / "V2_1_EXPERIMENT_REPORT.md",
                reporting.render_report(result, metric_rows, results, table))
    _write_text(OUT / "V2_1_ARTIFACT_AUDIT.md",
                reporting.render_audit(audit))
    _write_text(OUT / "V2_1_PROVENANCE_OVERLAP.md",
                reporting.render_provenance(audit, table))

    print(f"\noutputs written to {OUT}")
    print(f"reached >= {TARGET_ACCURACY:.2f} OOF accuracy under the "
          f"pre-declared cross-pool rule: {selection['reached_80_percent']}")
    print(f"selected for implementation: "
          f"{selection['selected_for_implementation']}")
    if best is not None:
        def base_acc(pool):
            return next((b.aggregate["accuracy"] for b in results
                         if b.candidate_id == "BASE_FROZEN_FUSION_V2"
                         and b.pool == pool), None)

        print(f"strongest honestly-evaluated learned candidate (reported for "
              f"transparency, NOT a selection): {selection['selected_id']}")
        for r in sorted((x for x in results
                         if x.scheme == evaluation.PRIMARY_SCHEME
                         and x.candidate_id == best.candidate_id
                         and x.threshold_mode == best.threshold_mode),
                        key=lambda x: x.pool):
            base = base_acc(r.pool)
            delta = (f"  (frozen V2 baseline {base:.4f} -> "
                     f"{r.aggregate['accuracy'] - base:+.4f})"
                     if base is not None else "")
            print(f"  on {r.pool}: acc={_r(r.aggregate['accuracy'])} "
                  f"fake_recall={_r(r.aggregate['fake_recall'])} "
                  f"fpr={_r(r.aggregate['fpr'])}{delta}")
    print("\nfrozen V2 backbone unchanged. RESEARCH / EXPERIMENTAL / UNVALIDATED.")
    return 0


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

def serialise_result(r) -> dict:
    return {
        "pool": r.pool, "candidate_id": r.candidate_id, "family": r.family,
        "feature_set": r.feature_set, "n_features": r.n_features,
        "cv_scheme": r.scheme, "threshold_mode": r.threshold_mode,
        "aggregate": r.aggregate, "notes": r.notes,
        "repeat_metrics": r.repeat_metrics,
    }


def serialise_selection(selection: dict) -> dict:
    """JSON-safe copy of the selection verdict.

    ``selection["selected"]`` is an ``OofResult`` dataclass, which
    ``json.dumps(default=str)`` would flatten into a repr blob; replace it
    with the same explicit structure every other result gets.
    """
    out = {k: v for k, v in selection.items() if k != "selected"}
    best = selection.get("selected")
    out["selected"] = serialise_result(best) if best is not None else None
    return out


def select_candidate(results, table, label_counts) -> dict:
    """Pre-declared selection rule, applied to the PRIMARY scheme only.

    The two eligible development populations are independent source
    collections (P4B n=25, D7 n=16) with zero content overlap. A learned
    decision layer is therefore judged on BOTH, not on the pool that happens
    to flatter it. A learned candidate is ELIGIBLE only if ALL of:

      * P4B_DEV25 (primary development population, the frozen rank-reference
        population, and the only pool with n >= 20) primary-scheme OOF
        accuracy >= 0.80
      * P4B_DEV25 primary-scheme OOF FAKE recall >= 0.70
      * D7_DEV16 primary-scheme OOF accuracy >= the frozen V2 baseline on
        D7_DEV16 (no regression on the second source collection)
      * on BOTH pools: REAL specificity > 0.0 and predicted-FAKE rate < 0.95
        (i.e. not "predict the dominant class")
      * on BOTH pools: degenerate-fold fraction <= 0.20
      * not flagged exploratory
      * not worse than the frozen V2 baseline on P4B_DEV25 either

    The P4B/D7 cross-pool requirement is deliberately strict: a rule that
    gains accuracy on one collection while regressing on the other is
    collection-specific fitting, not an improvement to the research
    backbone. POOLED41_SENSITIVITY and every exploratory candidate are
    excluded from the headline claim by construction.

    Among eligible candidates prefer, in order: lowest mean FPR across both
    pools, then highest worst-pool accuracy, then fewest features, then
    candidate_id for determinism.
    """
    primary = [r for r in results if r.scheme == evaluation.PRIMARY_SCHEME]
    baseline_by_pool = {
        r.pool: r.aggregate for r in primary
        if r.candidate_id == "BASE_FROZEN_FUSION_V2"}

    # group every learned primary-scheme row by (candidate_id, threshold_mode)
    by_arm: dict = {}
    for r in primary:
        if r.family.startswith("baseline"):
            continue
        by_arm.setdefault((r.candidate_id, r.threshold_mode), {})[r.pool] = r

    eligible, rejected = [], []
    for (cid, mode), arms in sorted(by_arm.items()):
        any_r = next(iter(arms.values()))
        p4b, d7 = arms.get("P4B_DEV25"), arms.get("D7_DEV16")
        p4b_a = p4b.aggregate if p4b else {}
        d7_a = d7.aggregate if d7 else {}

        def sane(a):
            return (a.get("real_specificity", 0.0) > 0.0
                    and a.get("predicted_fake_rate", 1.0) < 0.95
                    and (a.get("degenerate_folds", 1)
                         / max(a.get("n_folds", 1), 1)) <= 0.20)

        def base_acc(pool):
            return baseline_by_pool.get(pool, {}).get("accuracy")

        checks = {
            "p4b_accuracy_ge_0.80":
                p4b_a.get("accuracy", 0.0) >= TARGET_ACCURACY,
            "p4b_fake_recall_ge_0.70":
                p4b_a.get("fake_recall", 0.0) >= TARGET_RECALL,
            "d7_no_regression_vs_frozen_v2": (
                base_acc("D7_DEV16") is not None
                and d7_a.get("accuracy", 0.0) >= base_acc("D7_DEV16")),
            "p4b_no_regression_vs_frozen_v2": (
                base_acc("P4B_DEV25") is not None
                and p4b_a.get("accuracy", 0.0) >= base_acc("P4B_DEV25")),
            "both_pools_non_degenerate": sane(p4b_a) and sane(d7_a),
            "evaluated_on_both_pools": p4b is not None and d7 is not None,
            "not_exploratory": not any_r.notes.get("exploratory", False),
        }
        rec = {
            "candidate": cid,
            "threshold_mode": mode,
            "p4b_accuracy": p4b_a.get("accuracy"),
            "p4b_fake_recall": p4b_a.get("fake_recall"),
            "p4b_fpr": p4b_a.get("fpr"),
            "d7_accuracy": d7_a.get("accuracy"),
            "d7_fake_recall": d7_a.get("fake_recall"),
            "d7_fpr": d7_a.get("fpr"),
        }
        (eligible if all(checks.values()) else rejected).append(
            (any_r, checks, rec))

    def sort_key(item):
        r, _, rec = item
        fprs = [v for v in (rec["p4b_fpr"], rec["d7_fpr"]) if v is not None]
        accs = [v for v in (rec["p4b_accuracy"], rec["d7_accuracy"])
                if v is not None]
        return (round(max(fprs) if fprs else 1.0, 4),
                -round(min(accs) if accs else 0.0, 4),
                r.n_features,
                r.candidate_id)

    eligible.sort(key=sort_key)
    chosen = eligible[0][0] if eligible else None

    # "reached the target" is only claimed for an ADMISSIBLE candidate, not
    # for whichever single row happened to print the largest number.
    reached = bool(eligible)

    if chosen is None:
        pool_primary = [r for r in primary
                        if not r.family.startswith("baseline")
                        and r.pool in table
                        and r.candidate_id != "LR_EXT_C01"]
        strongest = max(
            pool_primary,
            key=lambda r: (round(r.aggregate.get("accuracy", 0.0), 4),
                           round(r.aggregate.get("f1", 0.0), 4)), default=None)
        reason = ("no learned candidate satisfied every pre-declared "
                  "admissibility check on BOTH development populations; the "
                  "strongest honestly-evaluated learned candidate is "
                  "reported for transparency and NOTHING is selected for "
                  "implementation")
    else:
        strongest = chosen
        reason = ("best worst-pool FPR among candidates that passed every "
                  "pre-declared admissibility check on both development "
                  "populations")

    max_seen = {}
    for r in primary:
        if r.family.startswith("baseline"):
            continue
        cur = max_seen.get(r.pool)
        if cur is None or r.aggregate["accuracy"] > cur[1]:
            max_seen[r.pool] = (
                f"{r.candidate_id} / {r.threshold_mode}",
                r.aggregate["accuracy"], r.aggregate["fake_recall"],
                r.aggregate["fpr"])

    return {
        "rule": ("primary scheme only, judged on BOTH development "
                 "populations; P4B_DEV25 accuracy>=0.80 AND P4B_DEV25 FAKE "
                 "recall>=0.70 AND D7_DEV16 accuracy >= frozen V2 on D7 AND "
                 "P4B_DEV25 accuracy >= frozen V2 on P4B AND on both pools "
                 "REAL specificity>0 AND predicted-FAKE rate<0.95 AND "
                 "degenerate-fold fraction<=0.20 AND not exploratory"),
        "rule_note": ("POOLED41_SENSITIVITY and LR_EXT_C01 are excluded "
                      "from the headline claim by construction"),
        "reached_80_percent": reached,
        "reached_80_percent_definition": (
            "true only if a candidate passed every pre-declared "
            "admissibility check; the largest single accuracy printed "
            "anywhere in this experiment is NOT sufficient"),
        "max_observed_accuracy_by_pool": {
            pool: {"candidate_and_mode": v[0], "accuracy": v[1],
                   "fake_recall": v[2], "fpr": v[3]}
            for pool, v in sorted(max_seen.items())},
        "selected": strongest,
        "selected_id": (f"{strongest.candidate_id} [{strongest.pool} / "
                        f"{strongest.scheme} / {strongest.threshold_mode}]"
                        if strongest else "NONE"),
        "selected_for_implementation": chosen is not None,
        "reason": reason,
        "eligible": [rec for _, _, rec in eligible],
        "rejected": [dict(rec, failed=[k for k, v in checks.items() if not v])
                     for _, checks, rec in rejected],
    }


if __name__ == "__main__":
    raise SystemExit(main())
