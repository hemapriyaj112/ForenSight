# V2.1 Accuracy Improvement Experiment - Report

**Status: RESEARCH / EXPERIMENTAL / UNVALIDATED**

This is a V2.1 EXPERIMENTAL MODEL-SELECTION / ANALYSIS PASS. It does not
replace the documented V2 final result, does not modify the frozen V2
architecture, does not touch V1, and does not modify the production
inference path or UI.

- run started (UTC): `2026-09-27T11:41:38+00:00`
- run finished (UTC): `2026-09-27T11:42:55+00:00`

## 1. Exact data populations used

Two disjoint development-eligible pools. They are reported SEPARATELY and
are never presented as one homogeneous population. A pooled view exists
only as a clearly-labelled sensitivity analysis.

| pool | n | FAKE | REAL | tier |
|---|---|---|---|---|
| `P4B_DEV25` | 25 | 17 | 8 | development (frozen V2 rank-reference population) |
| `D7_DEV16` | 16 | 11 | 5 | development (Day-9/10/11 DEV partition) |
| `POOLED41_SENSITIVITY` | 41 | 28 | 13 | SENSITIVITY ONLY - explicit concatenation of the two development pools, never a headline number |

Sample ids:

- `P4B_DEV25` (25): `P4B-002`, `P4B-003`, `P4B-004`, `P4B-005`, `P4B-006`, `P4B-007`, `P4B-008`, `P4B-009`, `P4B-010`, `P4B-012`, `P4B-013`, `P4B-014`, `P4B-015`, `P4B-016`, `P4B-017`, `P4B-018`, `P4B-020`, `P4B-022`, `P4B-023`, `P4B-027`, `P4B-029`, `P4B-030`, `P4B-031`, `P4B-034`, `P4B-036`
- `D7_DEV16` (16): `D7-002`, `D7-003`, `D7-004`, `D7-006`, `D7-007`, `D7-008`, `D7-012`, `D7-013`, `D7-014`, `D7-015`, `D7-016`, `D7-017`, `D7-018`, `D7-019`, `D7-020`, `D7-021`

### Populations deliberately NOT used

| population | n | status | ids |
|---|---|---|---|
| DAY6_LOCKED8 | 8 | CLOSED - never used for tuning/selection/thresholds | `P4B-001`, `P4B-011`, `P4B-019`, `P4B-021`, `P4B-024`, `P4B-028`, `P4B-032`, `P4B-033` |
| D9_HONEST_HOLDOUT5 | 5 | CLOSED - already consumed, descriptive evidence only | `D7-025`, `D7-026`, `D7-027`, `D7-028`, `D7-029` |
| D7_VISUAL_UNAVAILABLE8 | 8 | NOT SCORED - frozen VISUAL_UNAVAILABLE policy | `D7-001`, `D7-005`, `D7-009`, `D7-010`, `D7-011`, `D7-022`, `D7-023`, `D7-024` |
| P4B_VISUAL_UNAVAILABLE3 | 3 | NOT SCORED - no usable Xception face score | `P4B-025`, `P4B-035`, `P4B-037` |
| P4B_HELD_OUT_OVERLAP1 | 1 | EXCLUDE_OVERLAP - perceptually identical (dHash 0) to a Phase-4A LOCKED test file; deliberately kept out of every Phase-4B split | `P4B-026` |

## 2. Exact feature set

Built ONLY from already-persisted Day-2..Day-11 evidence. No detector was
re-run, no model was downloaded, nothing was retrained, no media was
decoded. Temporal descriptors reuse the historical Day-11 code verbatim.

### `CORE` (14 features)

`x_mean`, `x_std`, `x_q90`, `x_max`, `x_frac_ge_060`, `x_longest_run_060`, `x_burst_coverage_050`, `x_half_delta_abs`, `x_face_fraction`, `a_mean`, `a_iqr`, `a_max`, `a_n_windows`, `xa_abs_diff`

### `MIN` (5 features)

`x_mean`, `x_q90`, `x_frac_ge_060`, `a_mean`, `a_max`

### `EXTENDED` (24 features)

D7-only: AASIST per-window statistics and the Day-10 identity
statistics were never persisted for P4B. `rank_x` / `rank_a` /
`fused_v2` are legitimate here (fixed external reference) but the
set is deliberately over-parameterised for n=16 and is reported as
an overfitting probe, not a promotion candidate.

`x_mean`, `x_std`, `x_q90`, `x_max`, `x_frac_ge_060`, `x_longest_run_060`, `x_burst_coverage_050`, `x_half_delta_abs`, `x_face_fraction`, `a_mean`, `a_iqr`, `a_max`, `a_n_windows`, `xa_abs_diff`, `rank_x`, `rank_a`, `fused_v2`, `rank_abs_gap`, `a_std`, `a_q90`, `a_high_frac`, `a_low_frac`, `x_very_high_frac`, `x_temporal_consistency`

Explicitly excluded as label proxies: `subtype`, `source_group`, `source`, `acquisition`, `exposure_tier`, `partition`, `filename`, `path`, `md5`, `duration_s`, `true_label`

In `P4B_DEV25` the grouping column `source_group` is **100% label-pure**
(13/13 groups single-label), so using it - or any one-hot of it - would be
direct label leakage. It is used only as a CV grouping key.

## 3. Exact model candidates (pre-declared)

| baseline | family | rule | caveat |
|---|---|---|---|
| BASE_FROZEN_FUSION_V2 | baseline_frozen_v2 | fused_v2 >= 0.38 |  |
| BASE_XCEPTION_ONLY_050 | baseline_xception_only | x_mean >= 0.50 |  |
| BASE_XCEPTION_ONLY_R070 | baseline_xception_only | rank_x >= 0.70 |  |
| BASE_XCEPTION_PERSIST060 | baseline_xception_temporal | evidence_status == SUFFICIENT and x_frac_ge_060 >= 0.25 | OPTIMISTIC (post-hoc on Day-11 dev) |

| learned candidate | family | feature set | flag |
|---|---|---|---|
| LR_CORE_C1 | logistic_regression | CORE |  |
| LR_CORE_C01 | logistic_regression | CORE |  |
| LR_MIN_C1 | logistic_regression | MIN |  |
| TREE_CORE_D3 | decision_tree | CORE |  |
| RF_CORE_D3 | random_forest | CORE | exploratory |
| LR_EXT_C01 | logistic_regression | EXTENDED | exploratory |

Pre-declared threshold procedures (applied to every learned candidate):

- `T0_PROB_0.50` - predict FAKE when p >= 0.50 (primary).
- `T1_TRAIN_F1_FPR_CONSTRAINED` - per-fold threshold = argmax TRAIN-fold
  F1 subject to TRAIN-fold FPR <= 0.375, candidates are
  the sorted unique TRAIN-fold scores only, ties -> smallest threshold.
  This mirrors the objective recorded in `reports/V2_CANDIDATE_FREEZE.json`.
  No threshold is ever chosen using an evaluation fold.

## 4. Cross-validation method and leakage controls

- Primary: `CV5x10` = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=20260927)
- Sensitivity `LOO`: LeaveOneOut, in-fold preprocessing
- Sensitivity `CV_SGK5`: StratifiedGroupKFold(5) grouped by source family (P4B source_group / D7 subtype). Both groupings are label-pure, so this is a SOURCE-TRANSFER STRESS TEST, not an independence-corrected estimate
- Training-set accuracy is never computed or reported.

Leakage controls actually enforced in code:

- Day-6 locked 8 excluded from every fit/selection/threshold step
- Day-9/10/11 honest holdout 5 excluded (already consumed)
- D7 visual-unavailable 8 and P4B visual-unavailable 3 not scored
- frozen 25-sample rank reference read through day6_candidate only
- imputation median, scaler, model and threshold all fitted inside the training fold
- no training-set accuracy computed or reported
- label-proxy columns (subtype, source_group, source, acquisition, exposure_tier, partition, filename, path, md5) excluded from every feature set; P4B source_group is 100% label-pure
- no sample appears in two development pools (disjoint id namespaces, zero shared md5)

### Frozen-baseline reproduction check

| metric | historical (Day-5 OOF) | V2.1 recomputation | status |
|---|---|---|---|
| tp | 13 | 13 | MATCH |
| tn | 6 | 6 | MATCH |
| fp | 2 | 2 | MATCH |
| fn | 4 | 4 | MATCH |
| accuracy | 0.76 | 0.76 | MATCH |
| fake_recall | 0.7647 | 0.7647 | MATCH |
| fpr | 0.25 | 0.25 | MATCH |
| f1 | 0.8125 | 0.8125 | MATCH |

Historical source: `reports/DAY5_CROSSVALIDATION.json`, candidate `Xception+AASIST_rank_OOF`.

confirms the frozen 25-sample rank reference was NOT recomputed and the frozen V2 candidate is unchanged.

## 5. Baseline and candidate metrics

### P4B_DEV25  (n=25, FAKE=17, REAL=8)

**CV scheme `NONE_UNFITTED`**

| candidate | threshold | features | #f | acc | acc sd | prec | FAKE rec | REAL spec | FPR | F1 | ROC-AUC | TP/TN/FP/FN | degen folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE_FROZEN_FUSION_V2 | FIXED_RULE | FROZEN | 0 | 0.7600 | 0.0000 | 0.8667 | 0.7647 | 0.7500 | 0.2500 | 0.8125 | 0.7574 | 13/6/2/4 | 0.00 |
| BASE_XCEPTION_ONLY_050 | FIXED_RULE | RAW | 0 | 0.5200 | 0.0000 | 0.8571 | 0.3529 | 0.8750 | 0.1250 | 0.5000 | 0.6140 | 6/7/1/11 | 0.00 |
| BASE_XCEPTION_ONLY_R070 | FIXED_RULE | FROZEN_RANK | 0 | 0.4800 | 0.0000 | 0.7500 | 0.3529 | 0.7500 | 0.2500 | 0.4800 | 0.5515 | 6/6/2/11 | 0.00 |
| BASE_XCEPTION_PERSIST060 | FIXED_RULE | RAW | 0 | 0.6000 | 0.0000 | 0.8182 | 0.5294 | 0.7500 | 0.2500 | 0.6429 | 0.6397 | 9/6/2/8 | 0.00 |

**CV scheme `CV5x10`**

| candidate | threshold | features | #f | acc | acc sd | prec | FAKE rec | REAL spec | FPR | F1 | ROC-AUC | TP/TN/FP/FN | degen folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LR_CORE_C1 | T0_PROB_0.50 | CORE | 14 | 0.6720 | 0.0349 | 0.7366 | 0.8059 | 0.3875 | 0.6125 | 0.7697 | 0.6794 | 13.7/3.1/4.9/3.3 | 0.28 |
| LR_CORE_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.6520 | 0.0440 | 0.7485 | 0.7353 | 0.4750 | 0.5250 | 0.7418 | 0.6794 | 12.5/3.8/4.2/4.5 | 0.12 |
| LR_CORE_C01 | T0_PROB_0.50 | CORE | 14 | 0.7040 | 0.0196 | 0.7243 | 0.9118 | 0.2625 | 0.7375 | 0.8073 | 0.6809 | 15.5/2.1/5.9/1.5 | 0.50 |
| LR_CORE_C01 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.6800 | 0.0645 | 0.7500 | 0.7941 | 0.4375 | 0.5625 | 0.7714 | 0.6809 | 13.5/3.5/4.5/3.5 | 0.26 |
| LR_MIN_C1 | T0_PROB_0.50 | MIN | 5 | 0.6760 | 0.0332 | 0.7129 | 0.8765 | 0.2500 | 0.7500 | 0.7863 | 0.5647 | 14.9/2.0/6.0/2.1 | 0.44 |
| LR_MIN_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | MIN | 5 | 0.5480 | 0.0508 | 0.6966 | 0.5941 | 0.4500 | 0.5500 | 0.6413 | 0.5647 | 10.1/3.6/4.4/6.9 | 0.20 |
| TREE_CORE_D3 | T0_PROB_0.50 | CORE | 14 | 0.6600 | 0.0514 | 0.7429 | 0.7647 | 0.4375 | 0.5625 | 0.7536 | 0.6088 | 13.0/3.5/4.5/4.0 | 0.16 |
| TREE_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.6480 | 0.0560 | 0.7470 | 0.7294 | 0.4750 | 0.5250 | 0.7381 | 0.6088 | 12.4/3.8/4.2/4.6 | 0.12 |
| RF_CORE_D3 | T0_PROB_0.50 | CORE | 14 | 0.7480 | 0.0440 | 0.7399 | 0.9706 | 0.2750 | 0.7250 | 0.8397 | 0.6346 | 16.5/2.2/5.8/0.5 | 0.54 |
| RF_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.5800 | 0.0514 | 0.7444 | 0.5824 | 0.5750 | 0.4250 | 0.6535 | 0.6346 | 9.9/4.6/3.4/7.1 | 0.14 |

**CV scheme `LOO`**

| candidate | threshold | features | #f | acc | acc sd | prec | FAKE rec | REAL spec | FPR | F1 | ROC-AUC | TP/TN/FP/FN | degen folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LR_CORE_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.4000 | 0.1043 | 0.7778 | 0.1647 | 0.9000 | 0.1000 | 0.2718 | 0.4985 | 2.8/7.2/0.8/14.2 | 1.00 |
| LR_CORE_C01 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.3840 | 0.1120 | 0.7222 | 0.1529 | 0.8750 | 0.1250 | 0.2524 | 0.5000 | 2.6/7.0/1.0/14.4 | 1.00 |
| LR_MIN_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | MIN | 5 | 0.3680 | 0.1085 | 0.6875 | 0.1294 | 0.8750 | 0.1250 | 0.2178 | 0.4956 | 2.2/7.0/1.0/14.8 | 1.00 |
| TREE_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.3920 | 0.1143 | 0.7368 | 0.1647 | 0.8750 | 0.1250 | 0.2692 | 0.5015 | 2.8/7.0/1.0/14.2 | 1.00 |
| RF_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.3840 | 0.0999 | 0.7500 | 0.1412 | 0.9000 | 0.1000 | 0.2376 | 0.5015 | 2.4/7.2/0.8/14.6 | 1.00 |

**CV scheme `CV_SGK5`**

| candidate | threshold | features | #f | acc | acc sd | prec | FAKE rec | REAL spec | FPR | F1 | ROC-AUC | TP/TN/FP/FN | degen folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LR_CORE_C1 | T0_PROB_0.50 | CORE | 14 | 0.6800 | 0.0000 | 0.7368 | 0.8235 | 0.3750 | 0.6250 | 0.7778 | 0.5294 | 14.0/3.0/5.0/3.0 | 0.40 |
| LR_CORE_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.5200 | 0.0000 | 0.6923 | 0.5294 | 0.5000 | 0.5000 | 0.6000 | 0.5294 | 9.0/4.0/4.0/8.0 | 0.40 |
| LR_CORE_C01 | T0_PROB_0.50 | CORE | 14 | 0.6400 | 0.0000 | 0.6818 | 0.8824 | 0.1250 | 0.8750 | 0.7692 | 0.4044 | 15.0/1.0/7.0/2.0 | 0.40 |
| LR_CORE_C01 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.5200 | 0.0000 | 0.6923 | 0.5294 | 0.5000 | 0.5000 | 0.6000 | 0.4044 | 9.0/4.0/4.0/8.0 | 0.20 |
| LR_MIN_C1 | T0_PROB_0.50 | MIN | 5 | 0.6000 | 0.0000 | 0.6842 | 0.7647 | 0.2500 | 0.7500 | 0.7222 | 0.4559 | 13.0/2.0/6.0/4.0 | 0.60 |
| LR_MIN_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | MIN | 5 | 0.4800 | 0.0000 | 0.6429 | 0.5294 | 0.3750 | 0.6250 | 0.5806 | 0.4559 | 9.0/3.0/5.0/8.0 | 0.60 |
| TREE_CORE_D3 | T0_PROB_0.50 | CORE | 14 | 0.5600 | 0.0000 | 0.6875 | 0.6471 | 0.3750 | 0.6250 | 0.6667 | 0.5772 | 11.0/3.0/5.0/6.0 | 0.20 |
| TREE_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.6000 | 0.0000 | 0.7692 | 0.5882 | 0.6250 | 0.3750 | 0.6667 | 0.5772 | 10.0/5.0/3.0/7.0 | 0.00 |
| RF_CORE_D3 | T0_PROB_0.50 | CORE | 14 | 0.6800 | 0.0000 | 0.7143 | 0.8824 | 0.2500 | 0.7500 | 0.7895 | 0.3529 | 15.0/2.0/6.0/2.0 | 0.60 |
| RF_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.3600 | 0.0000 | 0.5455 | 0.3529 | 0.3750 | 0.6250 | 0.4286 | 0.3529 | 6.0/3.0/5.0/11.0 | 0.20 |

### D7_DEV16  (n=16, FAKE=11, REAL=5)

**CV scheme `NONE_UNFITTED`**

| candidate | threshold | features | #f | acc | acc sd | prec | FAKE rec | REAL spec | FPR | F1 | ROC-AUC | TP/TN/FP/FN | degen folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE_FROZEN_FUSION_V2 | FIXED_RULE | FROZEN | 0 | 0.4375 | 0.0000 | 0.6250 | 0.4545 | 0.4000 | 0.6000 | 0.5263 | 0.4273 | 5/2/3/6 | 0.00 |
| BASE_XCEPTION_ONLY_050 | FIXED_RULE | RAW | 0 | 0.7500 | 0.0000 | 0.8889 | 0.7273 | 0.8000 | 0.2000 | 0.8000 | 0.7636 | 8/4/1/3 | 0.00 |
| BASE_XCEPTION_ONLY_R070 | FIXED_RULE | FROZEN_RANK | 0 | 0.7500 | 0.0000 | 0.8889 | 0.7273 | 0.8000 | 0.2000 | 0.8000 | 0.7636 | 8/4/1/3 | 0.00 |
| BASE_XCEPTION_PERSIST060 | FIXED_RULE | RAW | 0 | 0.7500 | 0.0000 | 0.8182 | 0.8182 | 0.6000 | 0.4000 | 0.8182 | 0.7091 | 9/3/2/2 | 0.00 |

**CV scheme `CV5x10`**

| candidate | threshold | features | #f | acc | acc sd | prec | FAKE rec | REAL spec | FPR | F1 | ROC-AUC | TP/TN/FP/FN | degen folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LR_CORE_C1 | T0_PROB_0.50 | CORE | 14 | 0.7500 | 0.0000 | 0.8889 | 0.7273 | 0.8000 | 0.2000 | 0.8000 | 0.8727 | 8.0/4.0/1.0/3.0 | 0.16 |
| LR_CORE_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.7500 | 0.0000 | 0.8889 | 0.7273 | 0.8000 | 0.2000 | 0.8000 | 0.8727 | 8.0/4.0/1.0/3.0 | 0.16 |
| LR_CORE_C01 | T0_PROB_0.50 | CORE | 14 | 0.8125 | 0.0000 | 0.8333 | 0.9091 | 0.6000 | 0.4000 | 0.8696 | 0.8909 | 10.0/3.0/2.0/1.0 | 0.32 |
| LR_CORE_C01 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.8750 | 0.0000 | 0.9091 | 0.9091 | 0.8000 | 0.2000 | 0.9091 | 0.8909 | 10.0/4.0/1.0/1.0 | 0.20 |
| LR_MIN_C1 | T0_PROB_0.50 | MIN | 5 | 0.7500 | 0.0000 | 0.7692 | 0.9091 | 0.4000 | 0.6000 | 0.8333 | 0.8364 | 10.0/2.0/3.0/1.0 | 0.48 |
| LR_MIN_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | MIN | 5 | 0.7500 | 0.0000 | 0.8182 | 0.8182 | 0.6000 | 0.4000 | 0.8182 | 0.8364 | 9.0/3.0/2.0/2.0 | 0.20 |
| TREE_CORE_D3 | T0_PROB_0.50 | CORE | 14 | 0.6250 | 0.0000 | 0.7273 | 0.7273 | 0.4000 | 0.6000 | 0.7273 | 0.6818 | 8.0/2.0/3.0/3.0 | 0.40 |
| TREE_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.6250 | 0.0000 | 0.7273 | 0.7273 | 0.4000 | 0.6000 | 0.7273 | 0.6818 | 8.0/2.0/3.0/3.0 | 0.28 |
| RF_CORE_D3 | T0_PROB_0.50 | CORE | 14 | 0.7500 | 0.0000 | 0.8182 | 0.8182 | 0.6000 | 0.4000 | 0.8182 | 0.7636 | 9.0/3.0/2.0/2.0 | 0.28 |
| RF_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.7500 | 0.0000 | 0.8889 | 0.7273 | 0.8000 | 0.2000 | 0.8000 | 0.7636 | 8.0/4.0/1.0/3.0 | 0.16 |
| LR_EXT_C01 | T0_PROB_0.50 | EXTENDED | 24 | 0.8125 | 0.0000 | 0.8333 | 0.9091 | 0.6000 | 0.4000 | 0.8696 | 0.8727 | 10.0/3.0/2.0/1.0 | 0.32 |
| LR_EXT_C01 | T1_TRAIN_F1_FPR_CONSTRAINED | EXTENDED | 24 | 0.8125 | 0.0000 | 0.9000 | 0.8182 | 0.8000 | 0.2000 | 0.8571 | 0.8727 | 9.0/4.0/1.0/2.0 | 0.16 |

**CV scheme `LOO`**

| candidate | threshold | features | #f | acc | acc sd | prec | FAKE rec | REAL spec | FPR | F1 | ROC-AUC | TP/TN/FP/FN | degen folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LR_CORE_C1 | T0_PROB_0.50 | CORE | 14 | 0.8125 | 0.0000 | 0.9000 | 0.8182 | 0.8000 | 0.2000 | 0.8571 | 0.8364 | 9.0/4.0/1.0/2.0 | 1.00 |
| LR_CORE_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.8125 | 0.0000 | 0.9000 | 0.8182 | 0.8000 | 0.2000 | 0.8571 | 0.8364 | 9.0/4.0/1.0/2.0 | 1.00 |
| LR_CORE_C01 | T0_PROB_0.50 | CORE | 14 | 0.7500 | 0.0000 | 0.7692 | 0.9091 | 0.4000 | 0.6000 | 0.8333 | 0.8000 | 10.0/2.0/3.0/1.0 | 1.00 |
| LR_CORE_C01 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.8125 | 0.0000 | 0.9000 | 0.8182 | 0.8000 | 0.2000 | 0.8571 | 0.8000 | 9.0/4.0/1.0/2.0 | 1.00 |
| LR_MIN_C1 | T0_PROB_0.50 | MIN | 5 | 0.7500 | 0.0000 | 0.7692 | 0.9091 | 0.4000 | 0.6000 | 0.8333 | 0.7636 | 10.0/2.0/3.0/1.0 | 1.00 |
| LR_MIN_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | MIN | 5 | 0.8125 | 0.0000 | 0.9000 | 0.8182 | 0.8000 | 0.2000 | 0.8571 | 0.7636 | 9.0/4.0/1.0/2.0 | 1.00 |
| TREE_CORE_D3 | T0_PROB_0.50 | CORE | 14 | 0.6875 | 0.0000 | 0.8000 | 0.7273 | 0.6000 | 0.4000 | 0.7619 | 0.5273 | 8.0/3.0/2.0/3.0 | 1.00 |
| TREE_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.6875 | 0.0000 | 0.8000 | 0.7273 | 0.6000 | 0.4000 | 0.7619 | 0.5273 | 8.0/3.0/2.0/3.0 | 1.00 |
| RF_CORE_D3 | T0_PROB_0.50 | CORE | 14 | 0.6250 | 0.0000 | 0.6923 | 0.8182 | 0.2000 | 0.8000 | 0.7500 | 0.6000 | 9.0/1.0/4.0/2.0 | 1.00 |
| RF_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.6875 | 0.0000 | 0.8000 | 0.7273 | 0.6000 | 0.4000 | 0.7619 | 0.6000 | 8.0/3.0/2.0/3.0 | 1.00 |
| LR_EXT_C01 | T0_PROB_0.50 | EXTENDED | 24 | 0.7500 | 0.0000 | 0.7692 | 0.9091 | 0.4000 | 0.6000 | 0.8333 | 0.7455 | 10.0/2.0/3.0/1.0 | 1.00 |
| LR_EXT_C01 | T1_TRAIN_F1_FPR_CONSTRAINED | EXTENDED | 24 | 0.7500 | 0.0000 | 0.8182 | 0.8182 | 0.6000 | 0.4000 | 0.8182 | 0.7455 | 9.0/3.0/2.0/2.0 | 1.00 |

**CV scheme `CV_SGK5`**

| candidate | threshold | features | #f | acc | acc sd | prec | FAKE rec | REAL spec | FPR | F1 | ROC-AUC | TP/TN/FP/FN | degen folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LR_CORE_C1 | T0_PROB_0.50 | CORE | 14 | 0.5000 | 0.0000 | 0.7143 | 0.4545 | 0.6000 | 0.4000 | 0.5556 | 0.4909 | 5.0/3.0/2.0/6.0 | 0.60 |
| LR_CORE_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.5000 | 0.0000 | 0.8000 | 0.3636 | 0.8000 | 0.2000 | 0.5000 | 0.4909 | 4.0/4.0/1.0/7.0 | 0.60 |
| LR_CORE_C01 | T0_PROB_0.50 | CORE | 14 | 0.3125 | 0.0000 | 0.5000 | 0.3636 | 0.2000 | 0.8000 | 0.4211 | 0.3091 | 4.0/1.0/4.0/7.0 | 0.80 |
| LR_CORE_C01 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.4375 | 0.0000 | 0.6667 | 0.3636 | 0.6000 | 0.4000 | 0.4706 | 0.3091 | 4.0/3.0/2.0/7.0 | 0.60 |
| LR_MIN_C1 | T0_PROB_0.50 | MIN | 5 | 0.5000 | 0.0000 | 0.6364 | 0.6364 | 0.2000 | 0.8000 | 0.6364 | 0.6182 | 7.0/1.0/4.0/4.0 | 0.80 |
| LR_MIN_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | MIN | 5 | 0.6875 | 0.0000 | 0.8750 | 0.6364 | 0.8000 | 0.2000 | 0.7368 | 0.6182 | 7.0/4.0/1.0/4.0 | 0.60 |
| TREE_CORE_D3 | T0_PROB_0.50 | CORE | 14 | 0.2500 | 0.0000 | 0.3333 | 0.0909 | 0.6000 | 0.4000 | 0.1429 | 0.3091 | 1.0/3.0/2.0/10.0 | 0.80 |
| TREE_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.2500 | 0.0000 | 0.3333 | 0.0909 | 0.6000 | 0.4000 | 0.1429 | 0.3091 | 1.0/3.0/2.0/10.0 | 0.80 |
| RF_CORE_D3 | T0_PROB_0.50 | CORE | 14 | 0.0625 | 0.0000 | 0.1667 | 0.0909 | 0.0000 | 1.0000 | 0.1176 | 0.0545 | 1.0/0.0/5.0/10.0 | 1.00 |
| RF_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.1875 | 0.0000 | 0.0000 | 0.0000 | 0.6000 | 0.4000 | 0.0000 | 0.0545 | 0.0/3.0/2.0/11.0 | 0.80 |
| LR_EXT_C01 | T0_PROB_0.50 | EXTENDED | 24 | 0.3750 | 0.0000 | 0.5556 | 0.4545 | 0.2000 | 0.8000 | 0.5000 | 0.4364 | 5.0/1.0/4.0/6.0 | 0.80 |
| LR_EXT_C01 | T1_TRAIN_F1_FPR_CONSTRAINED | EXTENDED | 24 | 0.5000 | 0.0000 | 0.6667 | 0.5455 | 0.4000 | 0.6000 | 0.6000 | 0.4364 | 6.0/2.0/3.0/5.0 | 0.60 |

### POOLED41_SENSITIVITY  (n=41, FAKE=28, REAL=13)

> SENSITIVITY ONLY. Explicit concatenation of the two development
> pools. They come from different source collections, so a pooled
> number is not a population estimate; the per-pool rows above are
> the real result.

**CV scheme `NONE_UNFITTED`**

| candidate | threshold | features | #f | acc | acc sd | prec | FAKE rec | REAL spec | FPR | F1 | ROC-AUC | TP/TN/FP/FN | degen folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE_FROZEN_FUSION_V2 | FIXED_RULE | FROZEN | 0 | 0.6341 | 0.0000 | 0.7826 | 0.6429 | 0.6154 | 0.3846 | 0.7059 | 0.6291 | 18/8/5/10 | 0.00 |
| BASE_XCEPTION_ONLY_050 | FIXED_RULE | RAW | 0 | 0.6098 | 0.0000 | 0.8750 | 0.5000 | 0.8462 | 0.1538 | 0.6364 | 0.6731 | 14/11/2/14 | 0.00 |
| BASE_XCEPTION_ONLY_R070 | FIXED_RULE | FROZEN_RANK | 0 | 0.5854 | 0.0000 | 0.8235 | 0.5000 | 0.7692 | 0.2308 | 0.6222 | 0.6346 | 14/10/3/14 | 0.00 |
| BASE_XCEPTION_PERSIST060 | FIXED_RULE | RAW | 0 | 0.6585 | 0.0000 | 0.8182 | 0.6429 | 0.6923 | 0.3077 | 0.7200 | 0.6676 | 18/9/4/10 | 0.00 |

**CV scheme `CV5x10`**

| candidate | threshold | features | #f | acc | acc sd | prec | FAKE rec | REAL spec | FPR | F1 | ROC-AUC | TP/TN/FP/FN | degen folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LR_CORE_C1 | T0_PROB_0.50 | CORE | 14 | 0.7073 | 0.0000 | 0.7857 | 0.7857 | 0.5385 | 0.4615 | 0.7857 | 0.7500 | 22.0/7.0/6.0/6.0 | 0.06 |
| LR_CORE_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.7073 | 0.0000 | 0.8077 | 0.7500 | 0.6154 | 0.3846 | 0.7778 | 0.7500 | 21.0/8.0/5.0/7.0 | 0.04 |
| LR_CORE_C01 | T0_PROB_0.50 | CORE | 14 | 0.7317 | 0.0000 | 0.7429 | 0.9286 | 0.3077 | 0.6923 | 0.8254 | 0.7747 | 26.0/4.0/9.0/2.0 | 0.22 |
| LR_CORE_C01 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.7317 | 0.0000 | 0.8400 | 0.7500 | 0.6923 | 0.3077 | 0.7925 | 0.7747 | 21.0/9.0/4.0/7.0 | 0.04 |
| LR_MIN_C1 | T0_PROB_0.50 | MIN | 5 | 0.6341 | 0.0000 | 0.6857 | 0.8571 | 0.1538 | 0.8462 | 0.7619 | 0.6566 | 24.0/2.0/11.0/4.0 | 0.26 |
| LR_MIN_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | MIN | 5 | 0.6098 | 0.0000 | 0.8000 | 0.5714 | 0.6923 | 0.3077 | 0.6667 | 0.6566 | 16.0/9.0/4.0/12.0 | 0.00 |
| TREE_CORE_D3 | T0_PROB_0.50 | CORE | 14 | 0.7073 | 0.0000 | 0.8077 | 0.7500 | 0.6154 | 0.3846 | 0.7778 | 0.6799 | 21.0/8.0/5.0/7.0 | 0.04 |
| TREE_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.7073 | 0.0000 | 0.8077 | 0.7500 | 0.6154 | 0.3846 | 0.7778 | 0.6799 | 21.0/8.0/5.0/7.0 | 0.04 |
| RF_CORE_D3 | T0_PROB_0.50 | CORE | 14 | 0.7073 | 0.0000 | 0.7353 | 0.8929 | 0.3077 | 0.6923 | 0.8065 | 0.7555 | 25.0/4.0/9.0/3.0 | 0.22 |
| RF_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | CORE | 14 | 0.6829 | 0.0000 | 0.7778 | 0.7500 | 0.5385 | 0.4615 | 0.7636 | 0.7555 | 21.0/7.0/6.0/7.0 | 0.06 |

`acc sd`, `FAKE rec`, `FPR` and `F1` for the primary scheme are means over
the 10 repeats; TP/TN/FP/FN are means per repeat (so they are fractional).
`degen folds` is the fraction of folds whose test predictions collapsed to
a single class - the fold-collapse diagnostic.

## 6. Candidate admissibility and the 0.80 target

**Selection rule (pre-declared, judged on BOTH development pools):**

> primary scheme only, judged on BOTH development populations; P4B_DEV25 accuracy>=0.80 AND P4B_DEV25 FAKE recall>=0.70 AND D7_DEV16 accuracy >= frozen V2 on D7 AND P4B_DEV25 accuracy >= frozen V2 on P4B AND on both pools REAL specificity>0 AND predicted-FAKE rate<0.95 AND degenerate-fold fraction<=0.20 AND not exploratory

_POOLED41_SENSITIVITY and LR_EXT_C01 are excluded from the headline claim by construction._

The two eligible populations are independent source collections with zero
content overlap. Requiring a candidate to clear the target on `P4B_DEV25`
*and* to not regress below the frozen V2 baseline on `D7_DEV16` is what
separates a real decision layer from fitting one collection's quirks.
Without that requirement a rule that simply mis-routes `D7_DEV16` can print
an impressive number on a pool where the frozen backbone scores 0.4375.

- selected / strongest reported: `LR_CORE_C01 [D7_DEV16 / CV5x10 / T1_TRAIN_F1_FPR_CONSTRAINED]`
- selected for implementation: **False**
- reason: no learned candidate satisfied every pre-declared admissibility check on BOTH development populations; the strongest honestly-evaluated learned candidate is reported for transparency and NOTHING is selected for implementation

**Was the 0.80 target reached?**

**False**

_true only if a candidate passed every pre-declared admissibility check; the largest single accuracy printed anywhere in this experiment is NOT sufficient._

For transparency, the single largest accuracy printed anywhere in this
experiment, per pool, on the primary scheme:

| pool | candidate / threshold | acc | FAKE rec | FPR |
|---|---|---|---|---|
| D7_DEV16 | LR_CORE_C01 / T1_TRAIN_F1_FPR_CONSTRAINED | 0.8750 | 0.9091 | 0.2000 |
| P4B_DEV25 | RF_CORE_D3 / T0_PROB_0.50 | 0.7480 | 0.9706 | 0.7250 |
| POOLED41_SENSITIVITY | LR_CORE_C01 / T0_PROB_0.50 | 0.7317 | 0.9286 | 0.6923 |

Each of these is a *single arm that won on one pool*, and every one of
them collapses when the same configuration is applied to the other pool or
under the source-transfer stress test. They are shown so the failure is
visible, not because any of them is a result.

Admissibility rejections (primary scheme, learned candidates, both
pools required):

| candidate | threshold | P4B acc | P4B FPR | D7 acc | D7 FPR | failed checks |
|---|---|---|---|---|---|---|
| LR_CORE_C01 | T0_PROB_0.50 | 0.7040 | 0.7375 | 0.8125 | 0.4000 | p4b_accuracy_ge_0.80, d7_no_regression_vs_frozen_v2, p4b_no_regression_vs_frozen_v2, both_pools_non_degenerate |
| LR_CORE_C01 | T1_TRAIN_F1_FPR_CONSTRAINED | 0.6800 | 0.5625 | 0.8750 | 0.2000 | p4b_accuracy_ge_0.80, d7_no_regression_vs_frozen_v2, p4b_no_regression_vs_frozen_v2, both_pools_non_degenerate |
| LR_CORE_C1 | T0_PROB_0.50 | 0.6720 | 0.6125 | 0.7500 | 0.2000 | p4b_accuracy_ge_0.80, d7_no_regression_vs_frozen_v2, p4b_no_regression_vs_frozen_v2, both_pools_non_degenerate |
| LR_CORE_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | 0.6520 | 0.5250 | 0.7500 | 0.2000 | p4b_accuracy_ge_0.80, d7_no_regression_vs_frozen_v2, p4b_no_regression_vs_frozen_v2 |
| LR_EXT_C01 | T0_PROB_0.50 | n/a | n/a | 0.8125 | 0.4000 | p4b_accuracy_ge_0.80, p4b_fake_recall_ge_0.70, d7_no_regression_vs_frozen_v2, p4b_no_regression_vs_frozen_v2, both_pools_non_degenerate, evaluated_on_both_pools, not_exploratory |
| LR_EXT_C01 | T1_TRAIN_F1_FPR_CONSTRAINED | n/a | n/a | 0.8125 | 0.2000 | p4b_accuracy_ge_0.80, p4b_fake_recall_ge_0.70, d7_no_regression_vs_frozen_v2, p4b_no_regression_vs_frozen_v2, both_pools_non_degenerate, evaluated_on_both_pools, not_exploratory |
| LR_MIN_C1 | T0_PROB_0.50 | 0.6760 | 0.7500 | 0.7500 | 0.6000 | p4b_accuracy_ge_0.80, d7_no_regression_vs_frozen_v2, p4b_no_regression_vs_frozen_v2, both_pools_non_degenerate |
| LR_MIN_C1 | T1_TRAIN_F1_FPR_CONSTRAINED | 0.5480 | 0.5500 | 0.7500 | 0.4000 | p4b_accuracy_ge_0.80, p4b_fake_recall_ge_0.70, d7_no_regression_vs_frozen_v2, p4b_no_regression_vs_frozen_v2 |
| RF_CORE_D3 | T0_PROB_0.50 | 0.7480 | 0.7250 | 0.7500 | 0.4000 | p4b_accuracy_ge_0.80, d7_no_regression_vs_frozen_v2, p4b_no_regression_vs_frozen_v2, both_pools_non_degenerate, not_exploratory |
| RF_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | 0.5800 | 0.4250 | 0.7500 | 0.2000 | p4b_accuracy_ge_0.80, p4b_fake_recall_ge_0.70, d7_no_regression_vs_frozen_v2, p4b_no_regression_vs_frozen_v2, not_exploratory |
| TREE_CORE_D3 | T0_PROB_0.50 | 0.6600 | 0.5625 | 0.6250 | 0.6000 | p4b_accuracy_ge_0.80, d7_no_regression_vs_frozen_v2, p4b_no_regression_vs_frozen_v2, both_pools_non_degenerate |
| TREE_CORE_D3 | T1_TRAIN_F1_FPR_CONSTRAINED | 0.6480 | 0.5250 | 0.6250 | 0.6000 | p4b_accuracy_ge_0.80, d7_no_regression_vs_frozen_v2, p4b_no_regression_vs_frozen_v2, both_pools_non_degenerate |

## 7. Was 80% reached?

**Interpretation rule that applies to every number in this report:** any
figure at or above 0.80 would be reported ONLY as a **"V2.1 development
OOF result"**. It is NOT validated accuracy, NOT real-world accuracy and
NOT production accuracy, and it does not unseal the closed populations.

**No.** No learned candidate cleared the pre-declared admissibility
checks on both development populations.

The honest answer is that the existing persisted evidence, recombined
by a lightweight second-stage decision layer, does **not** reach the
0.80 target on the eligible development data. Every arm that prints a
number at or above 0.80 does so on one pool only, and every such arm
fails at least one of:

- **accuracy on the other pool** (typically 0.31-0.75, i.e. worse than
  the frozen backbone it was supposed to improve),
- **FAKE recall / FPR balance** (many arms reach 0.80+ accuracy by
  predicting FAKE far too often, e.g. `RF_CORE_D3` on `P4B_DEV25`:
  0.971 recall at 0.725 FPR),
- **fold stability** (degenerate-fold fractions of 0.2-0.6, and 1.00
  for every LOO arm, which indicates the decision surface is decided
  by one or two samples),
- **source transfer** (`CV_SGK5` collapses to 0.06-0.50 once a source
  family is held out entirely).

This is a negative result about the *evidence*, not about the
candidate family. It should be read as: 41 samples of persisted
detector scores do not contain a stable second decision layer that
improves on the frozen V2 fusion. The correct next step is not more
threshold tuning or more features on the same 41 samples.

## 8. Limitations

- **Sample size.** The eligible development data is 25 + 16 = 41 samples
  (28 FAKE / 13 REAL). One sample is worth 4.0 points of accuracy on
  `P4B_DEV25` and 6.25 points on `D7_DEV16`. Every number here has a very
  wide confidence interval; the 10-repeat spread is reported precisely
  because of this.
- **No independent promotion set exists.** The Day-6 locked 8 and the
  Day-9 honest holdout 5 are CLOSED. There is no untouched population left,
  so nothing here can be promoted regardless of the number.
- **`D7_DEV16` is near-chance for AASIST.** On that pool REAL samples have a
  HIGHER mean AASIST fake score than FAKE samples, which is why the frozen
  0.7 audio weight produces a 0.60 FPR there. Any candidate that leans on
  audio will look bad on `D7_DEV16` for reasons unrelated to its quality.
- **`a_n_windows` is a duration/codec proxy.** It is a legitimately
  persisted AASIST statistic and one of the few features that separates the
  classes in the same direction in both pools, but it encodes clip length,
  not manipulation. Treat any gain that depends on it as suspect.
- **Grouping is label-pure in both pools.** `CV_SGK5` therefore measures
  transfer to an unseen source family, which is a much harder problem than
  the primary stratified estimate. It is reported as a stress test.
- **`BASE_XCEPTION_PERSIST060` is optimistic by construction.** Its single
  gate (`x_frac_ge_0.60 >= 0.25`) is the additive condition of the Day-11
  `frac060_ge_0.25` configuration, which was itself chosen on the same 16
  samples. It is an in-sample upper bound, not a competing honest
  estimate, and it is NOT a replay of the full Day-11 3-state route.
- **`LR_EXT_C01` is an overfitting probe, not a candidate.** 24 features on
  16 samples; its score is reported only to quantify how much a richer
  feature set inflates development OOF accuracy.
- **No threshold search, no architecture change, no new data.** If the
  answer is negative, the correct next step is not more threshold tuning.

---

ForenSight remains **RESEARCH / EXPERIMENTAL / UNVALIDATED**. The frozen V2 candidate is
unchanged and remains the research scoring backbone.
