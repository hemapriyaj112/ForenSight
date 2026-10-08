"""
v2_1/populations.py
===================
Data-integrity layer for the V2.1 accuracy experiment.

Declares, in ONE place:

  * the CLOSED populations that must never be used for tuning, feature
    selection, threshold selection, model selection or meta-classifier
    training (Day-6 locked 8, Day-9 honest holdout 5, Day-15 intake);
  * the DEVELOPMENT-ELIGIBLE populations, each kept as a separate, named
    pool (never silently merged);
  * the overlap / content-hash controls between every persisted artifact;
  * the hard refusal guards used by the feature builder and the evaluator.

Design rules enforced here
--------------------------
1. P4B locked 8 (``P4B-001 ... P4B-033``) are CLOSED. They are the Day-6
   locked evaluation. Never used for anything but descriptive reporting.
2. D7 honest holdout 5 (``D7-025 .. D7-029``) were already consumed by the
   Day-9/10/11 single-pass evaluations. CLOSED descriptive evidence.
3. D7 ``OTHER`` 8 rows have ``evidence_status`` in
   {NO_FACE, NO_USABLE_CROP, INSUFFICIENT} -> visual evidence unavailable.
   The frozen V2 ``VISUAL_UNAVAILABLE`` policy says these are NOT scored, so
   they are not eligible for a both-modality second-stage model either.
4. P4B frozen-reference population = the 25 ids in
   ``day6_candidate.COMMON_IDS``. The 3 extra DEV/VAL rows
   (``P4B-025/035/037``) have no usable Xception face score and were already
   excluded from the frozen reference.
5. Development pools are reported SEPARATELY. An explicit pooled view exists
   only as a clearly-labelled SENSITIVITY analysis with a mandatory
   per-pool breakdown.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTAL_DETECTORS = REPO_ROOT / "phase4b" / "experimental_detectors"
REPORTS = REPO_ROOT / "reports"
PHASE4B_MANIFESTS = REPO_ROOT / "phase4b" / "manifests"
V2_1_REPORTS = REPORTS / "v2_1_accuracy_experiment"

if str(EXPERIMENTAL_DETECTORS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTAL_DETECTORS))

# --------------------------------------------------------------------------- #
# Frozen / closed identifiers (mirrors of the historical freezes)
# --------------------------------------------------------------------------- #

#: Day-6 LOCKED evaluation - CLOSED. Mirrors reports/V2_CANDIDATE_FREEZE.json
#: "locked_sample_ids".
DAY6_LOCKED_IDS = frozenset({
    "P4B-001", "P4B-011", "P4B-019", "P4B-021",
    "P4B-024", "P4B-028", "P4B-032", "P4B-033",
})

#: Day-9/10/11 honest holdout - already consumed, CLOSED.
#: Mirrors reports/DAY9_RULE_CONFIG.json "honest_holdout_universe".
D7_HONEST_HOLDOUT_IDS = frozenset({
    "D7-025", "D7-026", "D7-027", "D7-028", "D7-029",
})

#: Day-7/8 diagnostic population with no usable Xception evidence.
D7_VISUAL_UNAVAILABLE_IDS = frozenset({
    "D7-001", "D7-005", "D7-009", "D7-010",
    "D7-011", "D7-022", "D7-023", "D7-024",
})

#: Day-9/10/11 development partition (the only D7 tier usable for
#: model/threshold/feature selection).
D7_DEV_IDS = frozenset({
    "D7-002", "D7-003", "D7-004", "D7-006", "D7-007", "D7-008",
    "D7-012", "D7-013", "D7-014", "D7-015", "D7-016", "D7-017",
    "D7-018", "D7-019", "D7-020", "D7-021",
})

#: P4B DEV/VAL rows dropped from the frozen reference (no usable face score).
P4B_VISUAL_UNAVAILABLE_IDS = frozenset({"P4B-025", "P4B-035", "P4B-037"})

#: P4B held-out row excluded because it perceptually overlaps a prior locked
#: file (EXCLUDE_OVERLAP in the combined manifest). Never eligible.
P4B_HELD_OUT_OVERLAP_IDS = frozenset({"P4B-026"})

#: Everything that must never enter a V2.1 fit / selection / threshold step.
CLOSED_IDS = frozenset(DAY6_LOCKED_IDS | D7_HONEST_HOLDOUT_IDS
                       | D7_VISUAL_UNAVAILABLE_IDS
                       | P4B_VISUAL_UNAVAILABLE_IDS
                       | P4B_HELD_OUT_OVERLAP_IDS)

#: Frozen V2 fusion definition (read-only mirror of
#: reports/V2_CANDIDATE_FREEZE.json - asserted against it at runtime).
FROZEN_X_WEIGHT = 0.3
FROZEN_A_WEIGHT = 0.7
FROZEN_THRESHOLD = 0.38
FROZEN_REFERENCE_COUNT = 25


class LeakageError(RuntimeError):
    """Raised when a closed sample id would enter a V2.1 development step."""


def assert_not_closed(sample_ids, context: str) -> None:
    """Hard guard: refuse to proceed if any closed id is present.

    This is the single choke point used by the feature builder, the
    cross-validator and the reporting layer.
    """
    bad = sorted(set(map(str, sample_ids)) & CLOSED_IDS)
    if bad:
        raise LeakageError(
            f"{context}: closed/unsampleable sample ids present: {bad}. "
            "Day-6 locked, Day-9 honest holdout, visual-unavailable, "
            "perceptual-overlap and no-reference P4B rows are CLOSED for "
            "V2.1 tuning."
        )


def frozen_reference_ids() -> list[str]:
    """The frozen 25-sample DEV/VAL rank-reference population.

    Read from ``day6_candidate`` (the historical module) so the V2.1 code can
    never re-derive or re-compute the frozen reference. Order is the
    historical order and is not modified.
    """
    import day6_candidate as cand  # noqa: WPS433 (runtime import by design)

    ids = [str(s) for s in cand.COMMON_IDS]
    if len(ids) != FROZEN_REFERENCE_COUNT:
        raise LeakageError(
            f"frozen reference population has {len(ids)} ids, expected "
            f"{FROZEN_REFERENCE_COUNT}; the frozen reference must not change"
        )
    overlap = sorted(set(ids) & DAY6_LOCKED_IDS)
    if overlap:
        raise LeakageError(
            f"frozen reference population contains locked ids {overlap}"
        )
    return ids


def pool_definitions() -> list[dict]:
    """Named, non-overlapping development pools with their provenance tier."""
    return [
        {
            "pool": "P4B_DEV25",
            "tier": "development (frozen V2 rank-reference population)",
            "description": (
                "P4B DEV/VAL rows that carry a usable Xception face score and "
                "an AASIST score; this is exactly the 25-sample population the "
                "frozen V2 rank reference and the Day-5/Day-6 OOF were built on."
            ),
            "id_source": "day6_candidate.COMMON_IDS (frozen, never recomputed)",
            "label_available": True,
            "per_frame_xception": True,
            "per_window_aasist": False,
            "aasist_video_aggregates": True,
            "closed_ids_inside": 0,
        },
        {
            "pool": "D7_DEV16",
            "tier": "development (Day-9/10/11 DEV partition)",
            "description": (
                "D7 samples with evidence_status == SUFFICIENT; the only D7 tier "
                "that Day-9/10/11 used for selection."
            ),
            "id_source": "reports/DAY9_RULE_CONFIG.json development_population.ids",
            "label_available": True,
            "per_frame_xception": True,
            "per_window_aasist": True,
            "aasist_video_aggregates": True,
            "closed_ids_inside": 0,
        },
    ]


CLOSED_POPULATIONS = [
    {
        "population": "DAY6_LOCKED8",
        "n": 8,
        "status": "CLOSED - never used for tuning/selection/thresholds",
        "ids": sorted(DAY6_LOCKED_IDS),
    },
    {
        "population": "D9_HONEST_HOLDOUT5",
        "n": 5,
        "status": "CLOSED - already consumed, descriptive evidence only",
        "ids": sorted(D7_HONEST_HOLDOUT_IDS),
    },
    {
        "population": "D7_VISUAL_UNAVAILABLE8",
        "n": 8,
        "status": "NOT SCORED - frozen VISUAL_UNAVAILABLE policy",
        "ids": sorted(D7_VISUAL_UNAVAILABLE_IDS),
    },
    {
        "population": "P4B_VISUAL_UNAVAILABLE3",
        "n": 3,
        "status": "NOT SCORED - no usable Xception face score",
        "ids": sorted(P4B_VISUAL_UNAVAILABLE_IDS),
    },
    {
        "population": "P4B_HELD_OUT_OVERLAP1",
        "n": 1,
        "status": (
            "EXCLUDE_OVERLAP - perceptually identical (dHash 0) to a Phase-4A "
            "LOCKED test file; deliberately kept out of every Phase-4B split"
        ),
        "ids": ["P4B-026"],
    },
]
