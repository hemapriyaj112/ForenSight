"""
v2_1/candidates.py
==================
The PRE-DECLARED, SMALL candidate set for the V2.1 second-stage decision
layer. Nothing is added after seeing results.

Three baseline comparators (no parameters are fitted, so they are reported
once, not per fold):

  BASE_FROZEN_FUSION_V2     frozen V2 candidate, 0.3*rank_x+0.7*rank_a >= 0.38
  BASE_XCEPTION_ONLY_050    Xception video mean >= 0.50
  BASE_XCEPTION_ONLY_R070   Xception frozen-reference rank >= 0.70
                            (the historical Day-7/Day-9 xception-only reference)
  BASE_XCEPTION_PERSIST060  single temporal-persistence reference:
                            x_frac_ge_0.60 >= 0.25 -> FAKE
                            This is the ONE additive gate of the Day-11
                            "frac060_ge_0.25" configuration applied in
                            isolation. It is NOT a replay of that Day-11
                            3-state route (which also uses x_mean, a_mean,
                            a_std, a_high_frac, face_fraction, audio_risk
                            and can return UNCERTAIN); that route lives in
                            run_day11_xception_temporal.py and is not
                            reproducible as a binary rule here.

Five learned candidates (all parameters fitted INSIDE the fold):

  LR_CORE_C1      L2 logistic regression, C=1.0,  CORE features
  LR_CORE_C01     L2 logistic regression, C=0.1,  CORE features
  LR_MIN_C1       L2 logistic regression, C=1.0,  MIN  features
  TREE_CORE_D3    decision tree, depth 3, min_samples_leaf 3, CORE
  RF_CORE_D3      random forest, 200 trees, depth 3, min_samples_leaf 2, CORE
                  (the ONLY ensemble candidate; exploratory)

Two PRE-DECLARED threshold procedures, applied to every learned candidate:

  T0  p >= 0.5                                   (primary)
  T1  per-fold threshold = argmax train-fold F1 subject to train-fold
      FPR <= 0.375, tie-break smallest threshold; candidate thresholds are
      the sorted unique TRAIN-fold scores only. This mirrors the objective
      recorded in reports/V2_CANDIDATE_FREEZE.json. No threshold grid is
      searched on any evaluation fold.

Random Forest is included only because the two linear candidates and the
tree cannot capture an interaction between visual persistence and audio
window count; it is reported as exploratory and is never allowed to be the
sole selected candidate without the linear candidate agreeing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import populations as pops

# Frozen V2 constants (mirrors of reports/V2_CANDIDATE_FREEZE.json)
FROZEN_FUSION_THRESHOLD = pops.FROZEN_THRESHOLD
XCEPTION_ONLY_RAW_THRESHOLD = 0.50
XCEPTION_ONLY_RANK_THRESHOLD = 0.70
TEMPORAL_FRAC_THRESHOLD = 0.25
TEMPORAL_FRAC_LEVEL = 0.60
FPR_CONSTRAINT = 0.375
PRIMARY_PROB_THRESHOLD = 0.50


@dataclass(frozen=True)
class BaselineCandidate:
    candidate_id: str
    family: str
    feature_set: str
    description: str
    optimistic: bool = False
    rule: str = ""
    _impl: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class LearnedCandidate:
    candidate_id: str
    family: str
    feature_set: str
    description: str
    make_estimator: object = field(repr=False, compare=False)
    exploratory: bool = False


def _rule_frozen_fusion(row):
    if row.get("fused_v2") is None:
        return None
    return 1 if row["fused_v2"] >= FROZEN_FUSION_THRESHOLD else 0


def _rule_xception_raw(row):
    if row.get("x_mean") is None:
        return None
    return 1 if row["x_mean"] >= XCEPTION_ONLY_RAW_THRESHOLD else 0


def _rule_xception_rank(row):
    if row.get("rank_x") is None:
        return None
    return 1 if row["rank_x"] >= XCEPTION_ONLY_RANK_THRESHOLD else 0


def _rule_temporal(row):
    if row.get("evidence_status") != "SUFFICIENT":
        return 0
    frac = row.get("x_frac_ge_060")
    if frac is None:
        return 0
    return 1 if frac >= TEMPORAL_FRAC_THRESHOLD else 0


def _lr(c_value: float):
    def factory():
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            penalty="l2", C=c_value, solver="lbfgs", max_iter=5000,
            random_state=0)
    return factory


def _tree():
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=3, random_state=0)


def _rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=200, max_depth=3, min_samples_leaf=2,
        random_state=0, n_jobs=1)


BASELINES: list[BaselineCandidate] = [
    BaselineCandidate(
        candidate_id="BASE_FROZEN_FUSION_V2",
        family="baseline_frozen_v2",
        feature_set="FROZEN",
        description=("frozen V2 candidate: 0.3*rank_x + 0.7*rank_a >= 0.38, "
                     "percentile rank vs the frozen 25-sample reference, "
                     "average ties, reference NOT recomputed"),
        rule="fused_v2 >= 0.38",
        _impl=_rule_frozen_fusion,
    ),
    BaselineCandidate(
        candidate_id="BASE_XCEPTION_ONLY_050",
        family="baseline_xception_only",
        feature_set="RAW",
        description="Xception video mean P(FAKE) >= 0.50, rank-free",
        rule="x_mean >= 0.50",
        _impl=_rule_xception_raw,
    ),
    BaselineCandidate(
        candidate_id="BASE_XCEPTION_ONLY_R070",
        family="baseline_xception_only",
        feature_set="FROZEN_RANK",
        description=("historical Day-7/Day-9 xception-only reference: "
                     "frozen-reference rank_x >= 0.70"),
        rule="rank_x >= 0.70",
        _impl=_rule_xception_rank,
    ),
    BaselineCandidate(
        candidate_id="BASE_XCEPTION_PERSIST060",
        family="baseline_xception_temporal",
        feature_set="RAW",
        description=("single temporal-persistence reference: "
                     "x_frac_ge_0.60 >= 0.25 with evidence_status == "
                     "SUFFICIENT. This is the additive gate of the Day-11 "
                     "'frac060_ge_0.25' configuration applied IN ISOLATION, "
                     "not a replay of that 3-state route; it was chosen "
                     "post-hoc on Day-11 development, therefore OPTIMISTIC."),
        optimistic=True,
        rule="evidence_status == SUFFICIENT and x_frac_ge_060 >= 0.25",
        _impl=_rule_temporal,
    ),
]

LEARNED: list[LearnedCandidate] = [
    LearnedCandidate(
        candidate_id="LR_CORE_C1", family="logistic_regression",
        feature_set="CORE",
        description="L2 logistic regression C=1.0 on the CORE feature set",
        make_estimator=_lr(1.0),
    ),
    LearnedCandidate(
        candidate_id="LR_CORE_C01", family="logistic_regression",
        feature_set="CORE",
        description="L2 logistic regression C=0.1 (stronger shrinkage) on CORE",
        make_estimator=_lr(0.1),
    ),
    LearnedCandidate(
        candidate_id="LR_MIN_C1", family="logistic_regression",
        feature_set="MIN",
        description="L2 logistic regression C=1.0 on the 5-feature MIN set",
        make_estimator=_lr(1.0),
    ),
    LearnedCandidate(
        candidate_id="TREE_CORE_D3", family="decision_tree",
        feature_set="CORE",
        description="decision tree depth 3, min_samples_leaf 3, on CORE",
        make_estimator=_tree,
    ),
    LearnedCandidate(
        candidate_id="RF_CORE_D3", family="random_forest",
        feature_set="CORE",
        description=("random forest 200 trees, depth 3, min_samples_leaf 2, "
                     "on CORE (exploratory; only ensemble candidate)"),
        make_estimator=_rf,
        exploratory=True,
    ),
    # D7-only high-risk diagnostic. 24 features on n = 16 is deliberately
    # over-parameterised: it exists to measure how much a richer feature set
    # can inflate development OOF accuracy, NOT as a promotion candidate.
    LearnedCandidate(
        candidate_id="LR_EXT_C01", family="logistic_regression",
        feature_set="EXTENDED",
        description=("L2 logistic regression C=0.1 on the 24-feature EXTENDED "
                     "set (rank + disagreement + aasist window stats); "
                     "D7-only, n=16, deliberately over-parameterised "
                     "overfitting probe, NOT a promotion candidate"),
        make_estimator=_lr(0.1),
        exploratory=True,
    ),
]

THRESHOLD_MODES = ("T0_PROB_0.50", "T1_TRAIN_F1_FPR_CONSTRAINED")


def all_candidates() -> list[dict]:
    return ([{"kind": "baseline", "spec": c} for c in BASELINES]
            + [{"kind": "learned", "spec": c} for c in LEARNED])


def candidates_for_pool(pool: str) -> tuple[list[BaselineCandidate], list[LearnedCandidate]]:
    """The EXTENDED feature set only exists for the D7 pool: AASIST per-window
    statistics and the Day-10 identity statistics were never persisted for
    P4B, so the EXTENDED candidate is D7-only."""
    if pool == "D7_DEV16":
        return BASELINES, LEARNED
    return BASELINES, [c for c in LEARNED if c.feature_set != "EXTENDED"]


def feature_names(candidate, pool: str) -> list[str]:
    from . import features as feat
    return list(feat.FEATURE_SETS[candidate.feature_set])
