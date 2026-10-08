"""
v2_1/evaluation.py
==================
Overlap-safe out-of-fold evaluation for the V2.1 second-stage decision layer.

Guarantees
----------
* Every parameter that learns from data (imputation median, scaler mean/std,
  model coefficients, decision threshold) is fitted on the TRAIN FOLD ONLY.
* No closed sample id can enter any fold: ``populations.assert_not_closed`` is
  called on the whole population before splitting and on each fold.
* Grouping is available for stratified-group cross-validation, so folds can be
  made independent by source family when the repository has grouping info.
* Training-set accuracy is never computed or reported.
* The frozen V2 rank reference is never touched here; the frozen-fusion
  baseline is scored through the pre-computed ``fused_v2`` column.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import candidates as cand
from . import features as feat
from . import populations as pops

CV_SEED = 20260927
PRIMARY_SCHEME = "CV5x10"
SENSITIVITY_SCHEMES = ("LOO", "CV_SGK5")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def confusion(y_true, y_pred) -> dict:
    tp = int(sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1))
    tn = int(sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0))
    fp = int(sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1))
    fn = int(sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": len(y_true)}


def metrics_from_confusion(cm: dict) -> dict:
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    n = cm["n"]
    pos, neg = tp + fn, tn + fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / pos if pos else 0.0
    specificity = tn / neg if neg else 0.0
    fpr = 1.0 - specificity
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "fake_recall": recall,
        "real_specificity": specificity,
        "fpr": fpr,
        "fnr": 1.0 - recall if pos else 0.0,
        "f1": f1,
        "predicted_fake_rate": (tp + fp) / n if n else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def auc_scores(y_true, scores) -> tuple[float | None, float | None]:
    y = list(y_true)
    if len(set(y)) < 2:
        return None, None
    s = [(-1e18 if v is None else float(v)) for v in scores]
    if len(set(s)) < 2:
        return None, None
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(statistics.fmean(vals)),
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n": len(vals),
    }


# --------------------------------------------------------------------------- #
# Fold generation
# --------------------------------------------------------------------------- #

def _folds(scheme: str, y: np.ndarray, groups: np.ndarray | None):
    n = len(y)
    if scheme == PRIMARY_SCHEME:
        splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                                           random_state=CV_SEED)
        return [(tr, te) for tr, te in splitter.split(np.zeros(n), y)]
    if scheme == "LOO":
        return [(tr, te) for tr, te in LeaveOneOut().split(np.zeros(n))]
    if scheme == "CV_SGK5":
        if groups is None:
            return None
        g = np.asarray([str(v) for v in groups])
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                       random_state=CV_SEED)
        return [(tr, te) for tr, te in splitter.split(np.zeros(n), y, g)]
    raise ValueError(f"unknown CV scheme {scheme!r}")


def _repeat_index(fold_pairs):
    """Map flat fold index -> repeat id (RepeatedStratifiedKFold layout)."""
    if not fold_pairs:
        return []
    sizes = {len(te) for _, te in fold_pairs}
    if len(sizes) == 1 and len(fold_pairs) % 5 == 0 and len(fold_pairs) >= 5:
        return [i // 5 for i in range(len(fold_pairs))]
    return [0] * len(fold_pairs)


# --------------------------------------------------------------------------- #
# Threshold procedures (both pre-declared, both in-fold)
# --------------------------------------------------------------------------- #

def _select_threshold_T1(y_train, p_train) -> float:
    """argmax train-fold F1 subject to train-fold FPR <= 0.375.
    Candidates are the sorted unique TRAIN scores only. Ties -> smallest
    threshold. Never sees an evaluation fold."""
    order = sorted({float(p) for p in p_train})
    if not order:
        return cand.PRIMARY_PROB_THRESHOLD
    grid = [order[0] - 1e-9] + order
    best_t, best_f1, best_fpr = grid[0], -1.0, 1.0
    for t in grid:
        pred = [1 if p >= t else 0 for p in p_train]
        cm = confusion(y_train, pred)
        m = metrics_from_confusion(cm)
        if m["fpr"] > cand.FPR_CONSTRAINT + 1e-12:
            continue
        if m["f1"] > best_f1 + 1e-12:
            best_t, best_f1, best_fpr = t, m["f1"], m["fpr"]
    if best_f1 < 0:
        return cand.PRIMARY_PROB_THRESHOLD
    return float(best_t)


# --------------------------------------------------------------------------- #
# Core OOF runner
# --------------------------------------------------------------------------- #

@dataclass
class OofResult:
    scheme: str
    pool: str
    candidate_id: str
    family: str
    feature_set: str
    threshold_mode: str
    n_features: int
    sample_ids: list
    oof_score: list
    oof_pred: list
    repeat_metrics: list
    fold_metrics: list
    aggregate: dict
    notes: dict


def _build_pipeline(factory) -> Pipeline:
    """Imputation + scaling + model. Every step is fitted inside the fold.

    ``keep_empty_features=True`` makes an all-missing training column impute
    to 0.0 instead of propagating NaN; the median itself is still taken from
    the training fold only.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", factory()),
    ])


def evaluate_baseline(pool: str, spec, rows: list[dict]) -> OofResult:
    pops.assert_not_closed([r["sample_id"] for r in rows],
                           f"baseline {spec.candidate_id} / {pool}")
    y = [int(r["y"]) for r in rows]
    pred, score = [], []
    for r in rows:
        v = spec._impl(r)
        if v is None:
            v = 0  # frozen policy: unscored is never routed FAKE
        pred.append(int(v))
        score.append(float(v))
    cm = confusion(y, pred)
    m = metrics_from_confusion(cm)
    auc, ap = auc_scores(y, score) if len(set(score)) > 1 else (None, None)
    single = len(set(pred)) == 1
    return OofResult(
        scheme="NONE_UNFITTED", pool=pool,
        candidate_id=spec.candidate_id, family=spec.family,
        feature_set=spec.feature_set, threshold_mode="FIXED_RULE",
        n_features=0,
        sample_ids=[r["sample_id"] for r in rows],
        oof_score=score, oof_pred=pred,
        repeat_metrics=[m],
        fold_metrics=[dict(m, fold="all", degenerate=single)],
        aggregate={
            **m,
            "roc_auc": auc, "pr_auc": ap,
            "accuracy_agg": _agg([m["accuracy"]]),
            "fake_recall_agg": _agg([m["fake_recall"]]),
            "fpr_agg": _agg([m["fpr"]]),
            "f1_agg": _agg([m["f1"]]),
            "degenerate_folds": 1 if single else 0,
            "n_folds": 1,
        },
        notes={"optimistic_baseline": bool(spec.optimistic),
               "rule": spec.rule, "description": spec.description},
    )


def evaluate_learned(pool: str, spec, rows: list[dict], scheme: str,
                     group_key: str = "group_source_group") -> OofResult | None:
    ids = [r["sample_id"] for r in rows]
    pops.assert_not_closed(ids, f"learned {spec.candidate_id} / {pool}")

    allowed = {c.candidate_id for c in cand.candidates_for_pool(pool)[1]}
    if spec.candidate_id not in allowed:
        raise ValueError(
            f"candidate {spec.candidate_id} is not declared for pool {pool}; "
            "its feature set is not available for that population"
        )

    names = cand.feature_names(spec, pool)
    if not all(any(feat.fnum(r.get(nm)) is not None for r in rows) for nm in names):
        raise ValueError(
            f"candidate {spec.candidate_id}: feature set {spec.feature_set} has "
            f"at least one feature with no observed value in pool {pool}"
        )
    X = np.asarray(feat.feature_matrix(rows, names), dtype=float)
    y = np.asarray(feat.label_vector(rows), dtype=int)
    groups = np.asarray([str(r.get(group_key)) for r in rows]) if any(
        r.get(group_key) for r in rows) else None

    fold_pairs = _folds(scheme, y, groups)
    if not fold_pairs:
        return None

    n = len(y)
    n_repeats = max(_repeat_index(fold_pairs)) + 1
    oof_score = [[None] * n for _ in range(n_repeats)]
    oof_pred = [[None] * n for _ in range(n_repeats)]
    thresholds = [[None] * n for _ in range(n_repeats)]
    fold_rows = []
    repeat_of = _repeat_index(fold_pairs)

    for fi, (tr, te) in enumerate(fold_pairs):
        r = repeat_of[fi]
        pops.assert_not_closed([ids[i] for i in tr],
                               f"train fold {spec.candidate_id}/{pool}")
        pops.assert_not_closed([ids[i] for i in te],
                               f"test fold {spec.candidate_id}/{pool}")
        Xtr, ytr = X[tr], y[tr]
        if len(set(ytr.tolist())) < 2:
            continue
        pipe = _build_pipeline(spec.make_estimator)
        pipe.fit(Xtr, ytr)
        p_te = pipe.predict_proba(X[te])[:, 1]
        p_tr = pipe.predict_proba(Xtr)[:, 1]
        t1 = _select_threshold_T1(ytr.tolist(), p_tr.tolist())
        for j, idx in enumerate(te):
            oof_score[r][idx] = float(p_te[j])
            thresholds[r][idx] = t1
            oof_pred[r][idx] = 1 if p_te[j] >= cand.PRIMARY_PROB_THRESHOLD else 0
        fold_rows.append({
            "fold": fi, "repeat": r, "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_train_fake": int(ytr.sum()), "n_test_fake": int(y[te].sum()),
            "threshold_T1": t1,
            "t1_test_accuracy": metrics_from_confusion(
                confusion(y[te].tolist(),
                          [1 if p >= t1 else 0 for p in p_te.tolist()]))["accuracy"],
        })

    per_mode: dict = {}
    results = {}
    for mode in cand.THRESHOLD_MODES:
        preds = []
        for r in range(n_repeats):
            if mode == "T0_PROB_0.50":
                row_pred = list(oof_pred[r])
            else:
                row_pred = []
                for i in range(n):
                    t = thresholds[r][i]
                    s = oof_score[r][i]
                    row_pred.append(0 if (s is None or t is None) else int(s >= t))
            if any(p is None for p in row_pred):
                continue
            preds.append((r, row_pred))

        repeat_metrics, cms, aucs, aps = [], [], [], []
        fold_detail = []
        for r, row_pred in preds:
            cm = confusion(y.tolist(), row_pred)
            m = metrics_from_confusion(cm)
            auc, ap = auc_scores(y.tolist(), oof_score[r])
            m["roc_auc"], m["pr_auc"] = auc, ap
            repeat_metrics.append(m)
            cms.append(cm)
            if auc is not None:
                aucs.append(auc)
            if ap is not None:
                aps.append(ap)
        if not repeat_metrics:
            continue

        # per-fold metrics at the reported threshold mode
        for fr in fold_rows:
            tr_idx, te_idx = fold_pairs[fr["fold"]]
            r = fr["repeat"]
            if any(oof_pred[r][i] is None for i in te_idx):
                continue
            if mode == "T0_PROB_0.50":
                sub = [oof_pred[r][i] for i in te_idx]
            else:
                t = thresholds[r][te_idx[0]]
                sub = [int(oof_score[r][i] >= t) for i in te_idx]
            fm = metrics_from_confusion(confusion(y[te_idx].tolist(), sub))
            fold_detail.append(dict(fm, fold=fr["fold"], repeat=fr["repeat"],
                                    n_test=fr["n_test"],
                                    degenerate=len(set(sub)) == 1))

        tot = {"tp": sum(c["tp"] for c in cms), "tn": sum(c["tn"] for c in cms),
               "fp": sum(c["fp"] for c in cms), "fn": sum(c["fn"] for c in cms),
               "n": sum(c["n"] for c in cms)}
        k = len(cms)
        mean_cm = {key: tot[key] / k for key in ("tp", "tn", "fp", "fn", "n")}
        agg = metrics_from_confusion(mean_cm)
        first_rep = preds[0][0]
        first_pred = dict(preds)[first_rep]
        res = OofResult(
            scheme=scheme, pool=pool, candidate_id=spec.candidate_id,
            family=spec.family, feature_set=spec.feature_set,
            threshold_mode=mode, n_features=len(names),
            sample_ids=ids,
            oof_score=list(oof_score[first_rep]),
            oof_pred=list(first_pred),
            repeat_metrics=repeat_metrics, fold_metrics=fold_detail,
            aggregate={
                **agg,
                "confusion_mean_per_repeat": mean_cm,
                "confusion_total_all_repeats": tot,
                "roc_auc": statistics.fmean(aucs) if aucs else None,
                "roc_auc_std": statistics.pstdev(aucs) if len(aucs) > 1 else 0.0,
                "pr_auc": statistics.fmean(aps) if aps else None,
                "accuracy_agg": _agg([m["accuracy"] for m in repeat_metrics]),
                "precision_agg": _agg([m["precision"] for m in repeat_metrics]),
                "fake_recall_agg": _agg([m["fake_recall"] for m in repeat_metrics]),
                "fpr_agg": _agg([m["fpr"] for m in repeat_metrics]),
                "f1_agg": _agg([m["f1"] for m in repeat_metrics]),
                "degenerate_folds": sum(1 for f in fold_detail if f["degenerate"]),
                "n_folds": len(fold_detail),
                "n_repeats": len(repeat_metrics),
            },
            notes={
                "features": names,
                "oof_predictions_are": (
                    "first primary-scheme repeat only; the aggregate is the "
                    "mean over all repeats"),
                "missing_per_feature": {
                    nm: sum(1 for r in rows if feat.fnum(r.get(nm)) is None)
                    for nm in names},
                "exploratory": bool(spec.exploratory),
                "description": spec.description,
                "median_threshold_T1": (
                    statistics.median([f["threshold_T1"] for f in fold_rows])
                    if fold_rows else None),
            },
        )
        results[mode] = res
        per_mode[mode] = res
    return per_mode


def evaluate_pool(pool: str, rows: list[dict], schemes) -> list[OofResult]:
    pops.assert_not_closed([r["sample_id"] for r in rows], f"pool {pool}")
    baselines, learned = cand.candidates_for_pool(pool)
    out: list[OofResult] = []
    for spec in baselines:
        out.append(evaluate_baseline(pool, spec, rows))
    for spec in learned:
        for scheme in schemes:
            got = evaluate_learned(pool, spec, rows, scheme)
            if got:
                out.extend(got.values())
    return out
