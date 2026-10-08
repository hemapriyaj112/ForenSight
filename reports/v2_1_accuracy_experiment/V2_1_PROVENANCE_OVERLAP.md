# V2.1 Data Provenance and Overlap Audit

## Provenance chain

| pool | ids | labels from | Xception evidence | AASIST evidence |
|---|---|---|---|---|
| `P4B_DEV25` | `day6_candidate.COMMON_IDS` (frozen, 25) | `reports/xception_devval_predictions.csv` | `reports/xception_devval_per_frame.csv` (427 rows) -> Day-11 descriptors | `reports/aasist_day4_predictions.csv` (video aggregates only; per-window evidence was never persisted for P4B) |
| `D7_DEV16` | `reports/DAY9_RULE_CONFIG.json` development_population | `reports/DAY9_RAW_FRAME_EVIDENCE.csv` (394 rows) | `reports/DAY9_RAW_FRAME_EVIDENCE.csv` -> Day-11 descriptors | `reports/DAY9_AASIST_WINDOW_EVIDENCE.csv` (230 rows) + `reports/DAY9_AASIST_VIDEO_SUMMARIES.csv` |

## Overlap determination

- The two pools use **disjoint id namespaces** (`P4B-###` vs `D7-###`).
- Shared id count between the pools: **0**.
- Shared md5 between `phase4b_combined_dataset_manifest.csv` and `DAY7_FRESH_VALIDATION_MANIFEST.csv`: **0** (no byte-identical media).
- Shared basename between the two manifests: **0**.
- Within P4B: **no md5 duplicates** in the 37-row combined manifest.
- `P4B-026` is `EXCLUDE_OVERLAP` in the manifest (dHash 0 to a Phase-4A
  LOCKED test file) and is in no split - correct handling, recorded here.
- `reports/DAY7_AUDIT_NEARDUPS.json` records single-frame dHash 0 matches
  between distinct files (e.g. `avh talking.mp4` vs `P4B-025`/`P4B-031`).
  These are weak single-frame perceptual matches between DIFFERENT files,
  not duplicate media, but they are recorded because three of the four
  carry different labels. A single-frame dHash is not a reliable
  identity key, so it is reported as a caveat rather than used to merge
  or split samples.
- The frozen 25-sample reference population is exactly
  `xception_devval_predictions.csv` minus the 3 rows with no usable
  Xception face score (`P4B-025`, `P4B-035`, `P4B-037`).

## Per-pool evidence coverage in the V2.1 feature table

### P4B_DEV25 (n=25)

| feature | present | missing | handling |
|---|---|---|---|
| `x_mean` | 25 | 0 | complete |
| `x_std` | 21 | 4 | in-fold median imputed |
| `x_q90` | 19 | 6 | in-fold median imputed |
| `x_max` | 25 | 0 | complete |
| `x_frac_ge_060` | 25 | 0 | complete |
| `x_longest_run_060` | 25 | 0 | complete |
| `x_burst_coverage_050` | 25 | 0 | complete |
| `x_half_delta_abs` | 21 | 4 | in-fold median imputed |
| `x_face_fraction` | 25 | 0 | complete |
| `a_mean` | 25 | 0 | complete |
| `a_iqr` | 25 | 0 | complete |
| `a_max` | 25 | 0 | complete |
| `a_n_windows` | 25 | 0 | complete |
| `xa_abs_diff` | 25 | 0 | complete |
| `rank_x` | 25 | 0 | complete |
| `rank_a` | 25 | 0 | complete |
| `fused_v2` | 25 | 0 | complete |
| `rank_abs_gap` | 25 | 0 | complete |
| `a_std` | 0 | 25 | in-fold median imputed |
| `a_q90` | 0 | 25 | in-fold median imputed |
| `a_high_frac` | 0 | 25 | in-fold median imputed |
| `a_low_frac` | 0 | 25 | in-fold median imputed |
| `x_very_high_frac` | 0 | 25 | in-fold median imputed |
| `x_temporal_consistency` | 0 | 25 | in-fold median imputed |

### D7_DEV16 (n=16)

| feature | present | missing | handling |
|---|---|---|---|
| `x_mean` | 16 | 0 | complete |
| `x_std` | 16 | 0 | complete |
| `x_q90` | 14 | 2 | in-fold median imputed |
| `x_max` | 16 | 0 | complete |
| `x_frac_ge_060` | 16 | 0 | complete |
| `x_longest_run_060` | 16 | 0 | complete |
| `x_burst_coverage_050` | 16 | 0 | complete |
| `x_half_delta_abs` | 16 | 0 | complete |
| `x_face_fraction` | 16 | 0 | complete |
| `a_mean` | 16 | 0 | complete |
| `a_iqr` | 16 | 0 | complete |
| `a_max` | 16 | 0 | complete |
| `a_n_windows` | 16 | 0 | complete |
| `xa_abs_diff` | 16 | 0 | complete |
| `rank_x` | 16 | 0 | complete |
| `rank_a` | 16 | 0 | complete |
| `fused_v2` | 16 | 0 | complete |
| `rank_abs_gap` | 16 | 0 | complete |
| `a_std` | 16 | 0 | complete |
| `a_q90` | 16 | 0 | complete |
| `a_high_frac` | 16 | 0 | complete |
| `a_low_frac` | 16 | 0 | complete |
| `x_very_high_frac` | 16 | 0 | complete |
| `x_temporal_consistency` | 16 | 0 | complete |

Missing values arise only from the documented Day-11 evidence rules:
`q90` needs >= 4 usable face frames, `q95` needs >= 8, `std` needs >= 2,
`half_delta_abs` needs >= 3. Imputation is median-based and fitted inside
the training fold only.

## Feature-level label separation (descriptive, for the record)

### P4B_DEV25

| feature | FAKE mean | REAL mean | gap |
|---|---|---|---|
| `x_mean` | +0.4429 | +0.4109 | +0.0320 |
| `x_std` | +0.1918 | +0.1826 | +0.0091 |
| `x_q90` | +0.6308 | +0.6523 | -0.0215 |
| `x_max` | +0.7073 | +0.6038 | +0.1036 |
| `x_frac_ge_060` | +0.2991 | +0.1344 | +0.1647 |
| `x_longest_run_060` | +1.7647 | +0.6250 | +1.1397 |
| `x_burst_coverage_050` | +0.4523 | +0.0714 | +0.3809 |
| `x_half_delta_abs` | +0.1120 | +0.0919 | +0.0201 |
| `x_face_fraction` | +0.7780 | +0.4297 | +0.3483 |
| `a_mean` | +0.9440 | +0.7915 | +0.1525 |
| `a_iqr` | +0.0673 | +0.2379 | -0.1706 |
| `a_max` | +0.9997 | +0.9993 | +0.0003 |
| `a_n_windows` | +8.2353 | +13.5000 | -5.2647 |
| `xa_abs_diff` | +0.5066 | +0.4186 | +0.0880 |

### D7_DEV16

| feature | FAKE mean | REAL mean | gap |
|---|---|---|---|
| `x_mean` | +0.5465 | +0.3712 | +0.1754 |
| `x_std` | +0.1915 | +0.1581 | +0.0334 |
| `x_q90` | +0.7820 | +0.5595 | +0.2225 |
| `x_max` | +0.8198 | +0.6351 | +0.1847 |
| `x_frac_ge_060` | +0.3925 | +0.1700 | +0.2225 |
| `x_longest_run_060` | +1.9091 | +0.6000 | +1.3091 |
| `x_burst_coverage_050` | +0.5052 | +0.0000 | +0.5052 |
| `x_half_delta_abs` | +0.1531 | +0.0799 | +0.0732 |
| `x_face_fraction` | +0.7260 | +0.7000 | +0.0260 |
| `a_mean` | +0.6438 | +0.9503 | -0.3065 |
| `a_iqr` | +0.1246 | +0.0490 | +0.0756 |
| `a_max` | +0.7319 | +0.9996 | -0.2676 |
| `a_n_windows` | +6.5455 | +13.6000 | -7.0545 |
| `xa_abs_diff` | +0.4364 | +0.5792 | -0.1428 |

A positive gap means FAKE scores higher. Read this as descriptive only -
no feature was selected because of this table; the feature sets were
pre-declared from the persisted-artifact inventory.
