"""
phase4b/experimental_detectors/v2_1/
====================================
V2.1 ACCURACY IMPROVEMENT EXPERIMENT (time-constrained, analysis-only).

This package is ISOLATED from the frozen V2 research backbone. It does not
modify, import-for-mutation, or replace:

  * V1
  * the frozen Xception + AASIST rank-fusion candidate
    (0.3 * rank_x + 0.7 * rank_a, threshold 0.38)
  * the frozen 25-sample DEV/VAL rank reference
  * any Day-5 .. Day-15 report or locked evaluation result
  * the production inference path

It only READS persisted artifacts under ``reports/`` and ``phase4b/`` and
writes NEW artifacts under ``reports/v2_1_accuracy_experiment/``.

Project status remains: RESEARCH / EXPERIMENTAL / UNVALIDATED.
"""

__all__ = [
    "artifact_audit",
    "candidates",
    "evaluation",
    "features",
    "populations",
    "reporting",
]
