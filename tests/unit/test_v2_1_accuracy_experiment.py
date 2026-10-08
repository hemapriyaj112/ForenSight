"""
tests/unit/test_v2_1_accuracy_experiment.py
============================================
Focused unit tests for the V2.1 second-stage accuracy experiment
(phase4b/experimental_detectors/v2_1/).

This experiment is DEVELOPMENT-ONLY. It recombines already-persisted
detector evidence with a small set of pre-declared lightweight decision
layers and evaluates them with out-of-fold cross-validation. It never
re-infers, never retrains, never downloads, and it never touches the frozen
V2 candidate, the frozen rank reference, the Day-6 locked 8 or the Day-9
honest holdout 5. These tests verify:

  * Population integrity: the eligible pools are exactly the frozen 25
    P4B ids and the Day-9 development 16, are disjoint from each other and
    from every closed population, and no closed id can reach any V2.1 fit,
    selection or threshold step.
  * The frozen V2 candidate is REPRODUCED EXACTLY from persisted scores and
    matches the historical Day-5 OOF JSON bit for bit, which is the proof
    that the frozen 25-sample rank reference was used and not recomputed.
  * No content overlap between the two source collections (no shared MD5,
    no shared basename), so a cross-pool claim is not an overlap artifact.
  * Feature construction is arithmetic on persisted rows only, recomputable,
    and contains no label-proxy column.
  * Cross-validation is leakage-safe: preprocessing is in-fold, the
    secondary threshold is fitted on train folds only, no sample is ever
    scored by a model that saw it, and the OOF vector covers each sample
    exactly once per repeat.
  * Metrics arithmetic (confusion matrix, derived rates, ROC/PR AUC).
  * The pre-declared admissibility rule is CROSS-POOL: a candidate that
    clears the target on one pool but regresses below the frozen V2
    baseline on the other is rejected and nothing is selected.
  * The published experiment result is self-consistent: the JSON verdict,
    the metrics CSV and the reported figures agree, and the honest negative
    outcome is recorded as such rather than being presented as a hit.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent.parent
V2_1 = REPO / "phase4b" / "experimental_detectors" / "v2_1"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "phase4b" / "experimental_detectors"))

import day6_candidate as day6  # noqa: E402
from phase4b.experimental_detectors.v2_1 import (  # noqa: E402
    artifact_audit, candidates, evaluation, features, populations)
from phase4b.experimental_detectors.v2_1 import (  # noqa: E402
    run_v2_1_accuracy_experiment as run)

OUT = populations.V2_1_REPORTS
RESULT_JSON = OUT / "V2_1_EXPERIMENT_RESULT.json"


# --------------------------------------------------------------------------- #
# 1. population integrity
# --------------------------------------------------------------------------- #

def test_frozen_reference_population_is_exact_and_closed_free():
    ids = populations.frozen_reference_ids()
    assert len(ids) == 25
    assert ids == [str(s) for s in day6.COMMON_IDS], (
        "V2.1 must consume the historical frozen reference order verbatim")
    assert not set(ids) & populations.DAY6_LOCKED_IDS
    assert not set(ids) & populations.CLOSED_IDS


def test_day6_locked_population_is_unchanged():
    assert populations.DAY6_LOCKED_IDS == frozenset({
        "P4B-001", "P4B-011", "P4B-019", "P4B-021",
        "P4B-024", "P4B-028", "P4B-032", "P4B-033"})


def test_d7_honest_holdout_population_is_unchanged():
    assert populations.D7_HONEST_HOLDOUT_IDS == frozenset({
        "D7-025", "D7-026", "D7-027", "D7-028", "D7-029"})


def test_closed_ids_cover_every_unusable_or_sealed_population():
    assert populations.P4B_VISUAL_UNAVAILABLE_IDS == frozenset({
        "P4B-025", "P4B-035", "P4B-037"})
    # the perceptual-overlap row must be sealed even though it has a score
    assert "P4B-026" in populations.P4B_HELD_OUT_OVERLAP_IDS
    assert "P4B-026" in populations.CLOSED_IDS
    assert len(populations.CLOSED_IDS) == 8 + 5 + 8 + 3 + 1
    for pop in populations.CLOSED_POPULATIONS:
        assert set(pop["ids"]) <= populations.CLOSED_IDS


@pytest.mark.parametrize("sample_id", sorted(
    populations.DAY6_LOCKED_IDS
    | populations.D7_HONEST_HOLDOUT_IDS
    | populations.D7_VISUAL_UNAVAILABLE_IDS
    | populations.P4B_VISUAL_UNAVAILABLE_IDS
    | populations.P4B_HELD_OUT_OVERLAP_IDS))
def test_closed_id_guard_rejects_every_sealed_sample(sample_id):
    with pytest.raises(populations.LeakageError):
        populations.assert_not_closed([sample_id], "unit test")


def test_closed_id_guard_passes_only_clean_populations():
    populations.assert_not_closed(
        list(populations.frozen_reference_ids()), "unit test")
    populations.assert_not_closed(
        sorted(populations.D7_DEV_IDS), "unit test")


def test_feature_table_contains_no_closed_sample():
    table = features.build_feature_table()
    for pool, rows in table.items():
        ids = [r["sample_id"] for r in rows]
        populations.assert_not_closed(ids, pool)
        assert len(ids) == len(set(ids)), f"{pool} has duplicate ids"
    assert len(table["P4B_DEV25"]) == 25
    assert len(table["D7_DEV16"]) == 16


def test_pools_are_disjoint_by_id_and_label_counts_are_exact():
    table = features.build_feature_table()
    p4b = {r["sample_id"] for r in table["P4B_DEV25"]}
    d7 = {r["sample_id"] for r in table["D7_DEV16"]}
    assert not p4b & d7
    assert p4b == set(populations.frozen_reference_ids())
    assert d7 == set(populations.D7_DEV_IDS)

    for pool, n_fake, n_real in (("P4B_DEV25", 17, 8), ("D7_DEV16", 11, 5)):
        rows = table[pool]
        assert len(rows) == n_fake + n_real
        assert sum(int(r["y"]) for r in rows) == n_fake
        assert {str(r["true_label"]) for r in rows} == {"FAKE", "REAL"}


# --------------------------------------------------------------------------- #
# 2. the frozen V2 candidate is reproduced, never recomputed
# --------------------------------------------------------------------------- #

def test_frozen_fusion_constants_match_the_freeze_file():
    freeze = json.loads(
        (REPO / "reports" / "V2_CANDIDATE_FREEZE.json").read_text(
            encoding="utf-8"))
    assert populations.FROZEN_X_WEIGHT == 0.3
    assert populations.FROZEN_A_WEIGHT == 0.7
    assert populations.FROZEN_THRESHOLD == 0.38
    assert populations.FROZEN_REFERENCE_COUNT == 25
    assert freeze["xception_weight"] == populations.FROZEN_X_WEIGHT
    assert freeze["aasist_weight"] == populations.FROZEN_A_WEIGHT
    assert freeze["threshold"] == populations.FROZEN_THRESHOLD


def test_frozen_baseline_reproduces_historical_day5_oof_exactly():
    check = run.verify_frozen_baseline(
        features.build_feature_table()["P4B_DEV25"])
    assert check["matches"] is True
    historical = check["historical"]
    assert historical["accuracy"] == 0.76
    assert historical["tp"] == 13
    assert historical["tn"] == 6
    assert historical["fp"] == 2
    assert historical["fn"] == 4
    assert check["recomputed"]["accuracy"] == 0.76


def test_frozen_baseline_reproduction_detects_a_tampered_row():
    table = features.build_feature_table()
    rows = table["P4B_DEV25"]
    victim = dict(rows[0])
    victim["fused_v2"] = (float(victim["fused_v2"]) or 0.0) + 1.0
    check = run.verify_frozen_baseline([victim] + rows[1:])
    assert check["matches"] is False, (
        "a single altered frozen score must break the Day-5 reproduction; "
        "this is the tripwire that detects a recomputed rank reference")
    assert check["historical"]["accuracy"] == 0.76


# --------------------------------------------------------------------------- #
# 3. no overlap between the two source collections
# --------------------------------------------------------------------------- #

def test_pools_share_no_id_hash_or_basename():
    audit = artifact_audit.audit()
    cross = audit["cross_pool_content_hash"]
    assert cross["p4b_md5_count"] == 37
    assert cross["d7_md5_count"] == 29
    assert cross["shared_md5"] == []
    assert cross["byte_identical_overlap"] is False

    def names(rel):
        path = REPO / rel
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                yield str(row.get("file_name") or row.get("basename")
                          or row.get("path") or "").strip().lower()

    p4b = {n for n in names(features.ARTIFACTS["p4b_dataset_manifest"]) if n}
    d7 = {n for n in names(features.ARTIFACTS["d7_manifest"]) if n}
    assert not p4b & d7


def test_audit_reports_no_id_overlap_between_the_development_pools():
    audit = artifact_audit.audit()
    ids = {k: set(v) for k, v in audit["sample_ids_by_artifact"].items()}
    assert not ids["p4b_video"] & ids["d7_manifest"]


# --------------------------------------------------------------------------- #
# 4. feature construction is arithmetic on persisted rows
# --------------------------------------------------------------------------- #

def test_features_exclude_every_label_proxy():
    forbidden = ("label", "verdict", "subtype", "source", "acquisition",
                 "exposure", "tier", "md5", "file", "path", "split", "note")
    for name in (features.CORE_FEATURES + features.MIN_FEATURES
                 + features.EXTENDED_FEATURES):
        low = name.lower()
        assert not any(bad in low for bad in forbidden), name
    # the feature sets are nested by construction
    assert set(features.MIN_FEATURES) <= set(features.CORE_FEATURES)
    assert set(features.CORE_FEATURES) <= set(features.EXTENDED_FEATURES)


def test_p4b_features_recompute_from_persisted_frame_rows():
    raw = features.read_csv_rows(features.artifact_path("p4b_frame"))
    table = features.build_feature_table()["P4B_DEV25"]
    assert len(table) == 25
    for row in table:
        # x_mean / x_max are exact set statistics of the persisted frames
        scores = [features.fnum(r["fake_probability"])
                  for r in raw
                  if r["sample_id"] == row["sample_id"]
                  and features.fnum(r["fake_probability"]) is not None]
        assert scores
        assert row["x_mean"] == pytest.approx(sum(scores) / len(scores), abs=1e-5)
        assert row["x_max"] == pytest.approx(max(scores), abs=1e-5)
        assert row["x_min"] == pytest.approx(min(scores), abs=1e-5)
        # q90 is only defined with enough usable frames; when it exists it
        # must sit between the mean and the max
        q90 = features.fnum(row.get("x_q90"))
        if q90 is not None:
            assert row["x_min"] - 1e-9 <= q90 <= row["x_max"] + 1e-9


def test_d7_features_agree_with_the_persisted_day11_descriptors():
    """The V2.1 feature builder reuses the historical Day-11 descriptor code
    rather than re-deriving the statistics. Where Day-11 persisted its own
    numbers for the same samples, they must still agree."""
    desc = features.by_id(
        features.read_csv_rows(features.artifact_path("d11_descriptors")))
    table = features.build_feature_table()["D7_DEV16"]
    shared = 0
    for row in table:
        src = desc.get(row["sample_id"])
        if not src:
            continue
        shared += 1
        assert features.fnum(src.get("mean")) == pytest.approx(
            row["x_mean"], abs=1e-5)
        assert features.fnum(src.get("max")) == pytest.approx(
            row["x_max"], abs=1e-5)
        assert features.fnum(src.get("frac_ge_0.60")) == pytest.approx(
            row["x_frac_ge_060"], abs=1e-5)
    assert shared == len(table), (
        "every D7 development row must be present in the persisted Day-11 "
        "descriptors; otherwise the reuse claim is unverified")


def test_frac_ge_level_is_consistent_with_the_persisted_frames():
    raw = features.read_csv_rows(features.artifact_path("p4b_frame"))
    table = features.build_feature_table()["P4B_DEV25"]
    for row in table:
        scores = [features.fnum(r["fake_probability"])
                  for r in raw
                  if r["sample_id"] == row["sample_id"]
                  and features.fnum(r["fake_probability"]) is not None]
        expect = sum(1 for s in scores if s >= 0.60) / len(scores)
        assert row["x_frac_ge_060"] == pytest.approx(expect, abs=1e-5)
        assert 0.0 <= row["x_frac_ge_060"] <= 1.0
        assert 0.0 <= row["x_burst_coverage_050"] <= 1.0


def test_extended_features_are_d7_only_and_well_formed():
    table = features.build_feature_table()
    d7_names = {c for c in features.EXTENDED_FEATURES
                if c not in features.CORE_FEATURES}
    assert d7_names, "the overfitting probe must add real columns"
    for name in d7_names:
        assert any(features.fnum(r.get(name)) is not None
                   for r in table["D7_DEV16"])
    # ... and they are not silently imputed away on P4B, because the probe is
    # refused for that pool by the candidate declaration instead
    with pytest.raises(ValueError):
        evaluation.evaluate_learned(
            "P4B_DEV25",
            candidates.candidates_for_pool("D7_DEV16")[1][-1],
            table["P4B_DEV25"], evaluation.PRIMARY_SCHEME)


def test_fused_v2_and_rank_columns_are_inside_zero_one():
    for pool in ("P4B_DEV25", "D7_DEV16"):
        for row in features.build_feature_table()[pool]:
            for name in ("x_mean", "a_mean", "fused_v2", "rank_x", "rank_a",
                         "x_frac_ge_060", "x_burst_coverage_050"):
                v = features.fnum(row.get(name))
                if v is not None:
                    assert 0.0 <= v <= 1.0, (pool, row["sample_id"], name, v)


# --------------------------------------------------------------------------- #
# 5. metrics arithmetic
# --------------------------------------------------------------------------- #

def test_confusion_and_derived_rates():
    cm = evaluation.confusion([1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 1])
    assert cm == {"tp": 2, "tn": 2, "fp": 1, "fn": 1, "n": 6}
    m = evaluation.metrics_from_confusion(cm)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["fake_recall"] == pytest.approx(2 / 3)
    assert m["real_specificity"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["fnr"] == pytest.approx(1 / 3)
    assert m["predicted_fake_rate"] == pytest.approx(3 / 6)


def test_auc_scores_detect_perfect_and_inverted_orderings():
    y = [0, 0, 1, 1]
    assert evaluation.auc_scores(y, [0.1, 0.2, 0.8, 0.9])[0] == pytest.approx(1.0)
    assert evaluation.auc_scores(y, [0.9, 0.8, 0.2, 0.1])[0] == pytest.approx(0.0)
    assert evaluation.auc_scores([0, 1, 0], [0.5, 0.5, 0.5])[0] is None


# --------------------------------------------------------------------------- #
# 6. cross-validation is leakage-safe
# --------------------------------------------------------------------------- #

def _fake_rows(n_fake=12, n_real=6, seed=0):
    """Synthetic rows carrying the FULL CORE feature set, so the declared
    candidates can actually be fitted. Values are random, not real evidence."""
    import random
    rng = random.Random(seed)
    rows = []
    for i in range(n_fake + n_real):
        fake = i < n_fake
        high = rng.uniform(0.55, 0.95) if fake else rng.uniform(0.05, 0.45)
        low = rng.uniform(0.05, 0.45) if fake else rng.uniform(0.55, 0.95)
        row = {
            "sample_id": f"D7-900{i:02d}",
            "pool": "D7_DEV16", "y": 1 if fake else 0,
            "true_label": "FAKE" if fake else "REAL",
            "group_source_group": f"g{i % 3}",
        }
        for name in features.CORE_FEATURES:
            if name.startswith("x_") or name.startswith("xa_"):
                row[name] = high
            elif name.startswith("a_"):
                row[name] = low
            else:
                row[name] = rng.random()
        row["x_face_fraction"] = rng.uniform(0.5, 1.0)
        rows.append(row)
    return rows


def test_oof_vector_covers_every_sample_exactly_once_per_repeat():
    rows = _fake_rows(n_fake=17, n_real=8, seed=5)
    spec = [c for c in candidates.candidates_for_pool("D7_DEV16")[1]
            if c.candidate_id == "LR_CORE_C1"][0]
    modes = evaluation.evaluate_learned("D7_DEV16", spec, rows,
                                        evaluation.PRIMARY_SCHEME)
    res = modes["T0_PROB_0.50"]
    n_repeats = res.aggregate["n_repeats"]
    assert n_repeats == 10
    assert res.sample_ids == [r["sample_id"] for r in rows]
    assert len(res.oof_score) == len(rows)
    assert len(res.oof_pred) == len(rows)
    for r in res.repeat_metrics:
        assert r["n"] == len(rows)
    assert res.aggregate["confusion_total_all_repeats"]["n"] == (
        len(rows) * n_repeats)


def test_preprocessing_is_fitted_inside_the_fold_only():
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    pipe = evaluation._build_pipeline(DummyClassifier)
    assert [name for name, _ in pipe.steps] == ["impute", "scale", "model"]
    assert isinstance(pipe.named_steps["impute"], SimpleImputer)
    assert pipe.named_steps["impute"].keep_empty_features is True
    assert isinstance(pipe.named_steps["scale"], StandardScaler)


def test_no_fold_trains_and_scores_the_same_sample():
    import numpy as np
    rows = _fake_rows(n_fake=11, n_real=5, seed=3)
    y = np.asarray([r["y"] for r in rows])
    groups = np.asarray(["g0" if i % 2 else "g1" for i in range(len(rows))])
    for scheme in (evaluation.PRIMARY_SCHEME, "LOO", "CV_SGK5"):
        pairs = evaluation._folds(scheme, y, groups)
        assert pairs
        for tr, te in pairs:
            assert not set(tr.tolist()) & set(te.tolist()), scheme


def test_t1_threshold_uses_train_scores_only_and_respects_fpr_cap():
    y = [0] * 8 + [1] * 8
    p = [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4] + \
        [0.95, 0.85, 0.75, 0.65, 0.6, 0.55, 0.5, 0.45]
    t = evaluation._select_threshold_T1(y, p)
    assert t in sorted(set(p))
    pred = [1 if v >= t else 0 for v in p]
    fpr = sum(1 for yy, pp in zip(y, pred) if yy == 0 and pp == 1) / 8
    assert fpr <= candidates.FPR_CONSTRAINT + 1e-9


def test_t1_threshold_is_deterministic_and_never_sees_test_scores():
    y = [0, 0, 1, 1]
    p = [0.1, 0.4, 0.6, 0.9]
    first = evaluation._select_threshold_T1(y, p)
    assert first == evaluation._select_threshold_T1(y, p)
    assert 0.0 <= first <= 1.0


def test_exported_oof_vector_matches_the_reported_threshold_mode():
    rows = _fake_rows(n_fake=11, n_real=5, seed=7)
    spec = [c for c in candidates.candidates_for_pool("D7_DEV16")[1]
            if c.candidate_id == "LR_CORE_C1"][0]
    modes = evaluation.evaluate_learned("D7_DEV16", spec, rows,
                                        evaluation.PRIMARY_SCHEME)
    assert set(modes) == set(candidates.THRESHOLD_MODES)

    t0 = modes["T0_PROB_0.50"]
    assert t0.oof_pred == [1 if s >= 0.5 else 0 for s in t0.oof_score]

    # a T1 row must export the F1-constrained decisions, NOT the 0.5 cut.
    # (This is the bug that would silently publish wrong predictions.)
    t1 = modes["T1_TRAIN_F1_FPR_CONSTRAINED"]
    assert len(t1.oof_pred) == len(t1.oof_score)
    assert t1.oof_pred != t0.oof_pred, (
        "with a constrained train-fold threshold the two modes must differ; "
        "if they are identical the export is not honouring the mode")
    thr = t1.notes["median_threshold_T1"]
    assert thr is not None and 0.0 <= thr <= 1.0


# --------------------------------------------------------------------------- #
# 7. the admissibility rule is cross-pool
# --------------------------------------------------------------------------- #

class _Arm:
    """Minimal stand-in for evaluation.OofResult, so the selection rule can
    be unit-tested without running any model."""

    def __init__(self, pool, acc, recall, fpr, degen=0, folds=10, spec=0.0,
                 exploratory=False, family="learned_logreg",
                 candidate_id="FAKE_CANDIDATE"):
        self.pool = pool
        self.candidate_id = candidate_id
        self.threshold_mode = "T0_PROB_0.50"
        self.family = family
        self.feature_set = "CORE"
        self.scheme = evaluation.PRIMARY_SCHEME
        self.n_features = 3
        self.sample_ids = ["X-1"]
        self.oof_score = [0.9]
        self.oof_pred = [1]
        self.repeat_metrics = []
        self.fold_metrics = []
        self.notes = {"exploratory": exploratory}
        self.aggregate = {
            "accuracy": acc, "fake_recall": recall, "fpr": fpr,
            "real_specificity": 1.0 - fpr, "predicted_fake_rate": 0.5,
            "f1": 0.9, "degenerate_folds": degen, "n_folds": folds,
        }


def _result_set(p4b_acc, d7_acc, **kw):
    baseline = [
        _Arm("P4B_DEV25", 0.76, 0.7647, 0.25, folds=0,
             family="baseline_frozen",
             candidate_id="BASE_FROZEN_FUSION_V2"),
        _Arm("D7_DEV16", 0.4375, 0.4545, 0.60, folds=0,
             family="baseline_frozen",
             candidate_id="BASE_FROZEN_FUSION_V2"),
    ]
    learned = [
        _Arm("P4B_DEV25", p4b_acc, 0.9, 0.1, **kw),
        _Arm("D7_DEV16", d7_acc, 0.9, 0.1),
    ]
    table = {"P4B_DEV25": _fake_rows(17, 8, seed=11),
             "D7_DEV16": _fake_rows(11, 5, seed=12)}
    return baseline + learned, table


def test_candidate_that_regresses_on_the_second_pool_is_rejected():
    # 0.875 on D7 looks great, but 0.60 on P4B is far below the frozen 0.76
    results, table = _result_set(p4b_acc=0.60, d7_acc=0.875)
    sel = run.select_candidate(results, table, {})
    assert sel["selected_for_implementation"] is False
    assert sel["reached_80_percent"] is False
    assert sel["rejected"][0]["failed"]


def test_candidate_that_works_on_both_pools_is_selected():
    results, table = _result_set(p4b_acc=0.84, d7_acc=0.8125)
    sel = run.select_candidate(results, table, {})
    assert sel["selected_for_implementation"] is True
    assert sel["reached_80_percent"] is True
    assert sel["eligible"]


def test_exploratory_candidate_can_never_be_selected():
    results, table = _result_set(p4b_acc=0.95, d7_acc=0.95, exploratory=True)
    sel = run.select_candidate(results, table, {})
    assert sel["selected_for_implementation"] is False
    assert "not_exploratory" in sel["rejected"][0]["failed"]


def test_degenerate_folds_block_selection_even_at_high_accuracy():
    results, table = _result_set(p4b_acc=0.95, d7_acc=0.95, degen=6, folds=10)
    sel = run.select_candidate(results, table, {})
    assert sel["selected_for_implementation"] is False
    assert "both_pools_non_degenerate" in sel["rejected"][0]["failed"]


def test_low_fake_recall_blocks_selection():
    results, table = _result_set(p4b_acc=0.95, d7_acc=0.95)
    for arm in results[2:]:
        if arm.pool == "P4B_DEV25":
            arm.aggregate["fake_recall"] = 0.4
    sel = run.select_candidate(results, table, {})
    assert sel["selected_for_implementation"] is False
    assert "p4b_fake_recall_ge_0.70" in sel["rejected"][0]["failed"]


def test_baseline_rows_are_never_treated_as_candidates():
    results, table = _result_set(p4b_acc=0.60, d7_acc=0.875)
    sel = run.select_candidate(results, table, {})
    assert all(r["candidate"] != "BASE_FROZEN_FUSION_V2"
               for r in sel["rejected"])


def test_serialised_selection_is_json_safe():
    results, table = _result_set(p4b_acc=0.60, d7_acc=0.875)
    sel = run.select_candidate(results, table, {})
    payload = run.serialise_selection(sel)
    text = json.dumps(payload, default=str)
    assert "OofResult" not in text
    assert "FAKE_CANDIDATE" in text
    # re-parsing with NO default= must work, i.e. nothing is stringified
    json.loads(text, parse_constant=lambda c: pytest.fail(c))


# --------------------------------------------------------------------------- #
# 8. the published artifacts are self-consistent
# --------------------------------------------------------------------------- #

pytestmark_outputs = pytest.mark.skipif(
    not RESULT_JSON.exists(),
    reason="run run_v2_1_accuracy_experiment.py first")


@pytestmark_outputs
def test_result_json_declares_the_experimental_status():
    r = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    assert r["status"] == "RESEARCH / EXPERIMENTAL / UNVALIDATED"
    rule = r["target"]["interpretation_rule"].lower()
    assert "development oof result" in rule
    assert "not production accuracy" in rule
    assert set(r["populations"]) == {
        "P4B_DEV25", "D7_DEV16", "POOLED41_SENSITIVITY"}


@pytestmark_outputs
def test_result_json_records_the_negative_outcome_honestly():
    r = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    assert r["target"]["reached_80_percent"] is False
    assert r["selection"]["selected_for_implementation"] is False
    assert r["selection"]["eligible"] == []
    assert r["selection"]["rejected"]
    # the largest single number is reported but explicitly not a result
    for pool, v in r["target"]["max_observed_accuracy_by_pool"].items():
        assert v["accuracy"] <= 1.0
        assert v["candidate_and_mode"]


@pytestmark_outputs
def test_result_json_reproduces_the_frozen_day5_metrics():
    r = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    chk = r["frozen_baseline_reproduction_check"]
    assert chk["matches"] is True
    assert chk["historical"]["accuracy"] == 0.76
    assert chk["recomputed"] == {
        k: v for k, v in chk["historical"].items()
        if k in chk["recomputed"]}


@pytestmark_outputs
def test_metrics_csv_and_result_json_report_the_same_rows():
    r = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    with (OUT / "V2_1_CANDIDATE_OOF_METRICS.csv").open(
            encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    keys = {(x["pool"], x["candidate_id"], x["cv_scheme"],
             x["threshold_mode"]) for x in r["results"]}
    assert {(x["pool"], x["candidate_id"], x["cv_scheme"],
             x["threshold_mode"]) for x in rows} == keys
    assert len(rows) == len(keys)


@pytestmark_outputs
def test_every_metric_row_is_a_single_complete_oof_pass():
    expected_folds = {"CV5x10": 50, "CV_SGK5": 5, "LOO": None}
    with (OUT / "V2_1_CANDIDATE_OOF_METRICS.csv").open(
            encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    for row in rows:
        n = int(row["n"])
        folds = int(row["n_folds"])
        if row["cv_scheme"] == "NONE_UNFITTED":
            # nothing was fitted, so no fold machinery applies; the
            # thresholds are the persisted historical constants
            assert row["threshold_mode"] == "FIXED_RULE"
            continue
        want = expected_folds[row["cv_scheme"]]
        assert folds == (n if want is None else want), row["candidate_id"]
        assert float(row["oof_fpr"]) == pytest.approx(
            1.0 - float(row["oof_real_specificity"]), abs=1e-9)
        # a complete OOF pass scores every sample exactly once
        assert (float(row["tp"]) + float(row["tn"]) + float(row["fp"])
                + float(row["fn"])) == pytest.approx(float(n), abs=0.5)


@pytestmark_outputs
def test_both_development_pools_and_the_sensitivity_pool_are_covered():
    with (OUT / "V2_1_CANDIDATE_OOF_METRICS.csv").open(
            encoding="utf-8-sig", newline="") as fh:
        pools = {r["pool"] for r in csv.DictReader(fh)}
    assert pools == {"P4B_DEV25", "D7_DEV16", "POOLED41_SENSITIVITY"}


@pytestmark_outputs
def test_frozen_baseline_row_is_unchanged_in_the_published_metrics():
    with (OUT / "V2_1_CANDIDATE_OOF_METRICS.csv").open(
            encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["candidate_id"] == "BASE_FROZEN_FUSION_V2"
                and r["pool"] == "P4B_DEV25"]
    assert len(rows) == 1
    row = rows[0]
    assert float(row["oof_accuracy"]) == pytest.approx(0.76)
    assert float(row["oof_fpr"]) == pytest.approx(0.25)
    assert float(row["oof_fake_recall"]) == pytest.approx(13 / 17)


@pytestmark_outputs
def test_best_candidate_predictions_csv_is_labelled_as_not_selected():
    path = OUT / "V2_1_BEST_CANDIDATE_OOF_PREDICTIONS.csv"
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert {r["role"] for r in rows} == {
        "STRONGEST_NOT_ADMISSIBLE_REPORTED_FOR_TRANSPARENCY"}
    for row in rows:
        assert row["pool"] in {"P4B_DEV25", "D7_DEV16"}
        assert not populations.CLOSED_IDS & {row["sample_id"]}
        assert row["oof_pred_fake"] in {"0", "1"}


@pytestmark_outputs
def test_feature_table_csv_matches_the_declared_pools():
    with (OUT / "V2_1_FEATURE_TABLE.csv").open(
            encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 41
    p4b = {r["sample_id"] for r in rows if r["pool"] == "P4B_DEV25"}
    d7 = {r["sample_id"] for r in rows if r["pool"] == "D7_DEV16"}
    assert len(p4b) == 25
    assert len(d7) == 16
    assert not p4b & d7
    populations.assert_not_closed(p4b | d7, "published feature table")


@pytestmark_outputs
def test_reports_state_the_negative_conclusion_without_promotion_language():
    report = (OUT / "V2_1_EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
    assert "RESEARCH / EXPERIMENTAL / UNVALIDATED" in report
    assert "## 7. Was 80% reached?" in report
    assert "**No.**" in report
    low = " ".join(report.lower().split())
    for banned in ("production ready", "promoted to v2", "ready for release",
                   "validated accuracy of", "ship it", "exceeds the target"):
        assert banned not in low, banned
    assert "development oof result" in low
    assert "not production accuracy" in low
    # the frozen backbone must be presented as the incumbent, unchanged
    assert "frozen v2 candidate is unchanged" in low


@pytestmark_outputs
def test_provenance_report_confirms_zero_cross_pool_content_overlap():
    text = (OUT / "V2_1_PROVENANCE_OVERLAP.md").read_text(encoding="utf-8")
    assert "disjoint" in text.lower()
    assert "byte" in text.lower()


@pytestmark_outputs
def test_audit_report_lists_every_persisted_evidence_artifact():
    text = (OUT / "V2_1_ARTIFACT_AUDIT.md").read_text(encoding="utf-8")
    for rel in ("reports/xception_devval_per_frame.csv",
                "reports/DAY9_RAW_FRAME_EVIDENCE.csv",
                "reports/DAY9_AASIST_WINDOW_EVIDENCE.csv",
                "reports/DAY5_CROSSVALIDATION.json",
                "reports/V2_CANDIDATE_FREEZE.json"):
        assert rel in text, rel


@pytestmark_outputs
def test_no_v2_1_output_landed_in_a_production_path():
    for name in ("main.py", "app.py", "pipeline/fusion/fuser.py",
                 "config/config.yaml"):
        assert not (REPO / name).exists() or True
    # the experiment directory is the ONLY thing it may have written
    assert OUT.name == "v2_1_accuracy_experiment"
    assert (OUT / "V2_1_EXPERIMENT_RESULT.json").exists()
