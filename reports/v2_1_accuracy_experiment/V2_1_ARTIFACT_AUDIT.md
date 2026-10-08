# V2.1 Artifact Audit

Read-only inspection of every persisted Day-2..Day-15 artifact V2.1 was
asked to look for. No detector was re-run. No production code was modified.

- artifacts inspected: **29**
- missing: **none**

## 1-7. Per-artifact audit

### `reports/xception_devval_per_frame.csv`

1. path: `reports/xception_devval_per_frame.csv`
2. rows: **427** / 11 columns
3. available features: face (4), fake_probability (1), other (6)
   full column list: `sample_id`, `filename`, `split`, `true_label`, `subtype`, `frame_index`, `face_detected`, `fake_probability`, `faces_in_frame`, `face_box`, `face_confidence`
4. label availability: **yes (true_label) {'FAKE': 251, 'REAL': 176}**
5. sample-ID availability: **yes (28 unique)**
6. content hash available: **no**
7. population tier: **development (frozen V2 rank-reference population + 3 unscored)**

### `reports/xception_devval_predictions.csv`

1. path: `reports/xception_devval_predictions.csv`
2. rows: **28** / 28 columns
3. available features: other (22), resolution (1), video_score (5)
   full column list: `sample_id`, `filename`, `split`, `true_label`, `subtype`, `path`, `model`, `checkpoint`, `device`, `class_mapping`, `probability_semantics`, `resolution`, `normalization`, `frame_pool_fps`, `frame_pool_size`, `requested_max_frames`, `sampled_frames_count`, `frames_with_face`, `frames_without_face`, `usable_face_frames`, `total_faces_detected`, `aggregation`, `video_score_mean`, `video_score_median`, `video_score_min`, `video_score_max`, `video_score_iqr`, `runtime_seconds`
4. label availability: **yes (true_label) {'FAKE': 17, 'REAL': 11}**
5. sample-ID availability: **yes (28 unique)**
6. content hash available: **no**
7. population tier: **development (frozen V2 rank-reference population + 3 unscored)**

### `reports/aasist_day4_predictions.csv`

1. path: `reports/aasist_day4_predictions.csv`
2. rows: **28** / 27 columns
3. available features: audio_ (8), other (19)
   full column list: `sample_id`, `filename`, `split`, `true_label`, `subtype`, `path`, `model`, `checkpoint`, `device`, `class_mapping`, `score_semantics`, `status`, `audio_present`, `audio_duration`, `original_sample_rate`, `normalized_sample_rate`, `original_channels`, `num_windows`, `audio_fake_score`, `audio_fake_score_median`, `audio_fake_score_min`, `audio_fake_score_max`, `audio_fake_score_iqr`, `audio_bonafide_score`, `predicted_label`, `error`, `runtime_seconds`
4. label availability: **yes (true_label) {'FAKE': 17, 'REAL': 11}**
5. sample-ID availability: **yes (28 unique)**
6. content hash available: **no**
7. population tier: **development (frozen V2 rank-reference population + 3 unscored)**

### `phase4b/manifests/phase4b_split_manifest.csv`

1. path: `phase4b/manifests/phase4b_split_manifest.csv`
2. rows: **37** / 8 columns
3. available features: other (8)
   full column list: `sample_id`, `filename`, `label`, `subtype`, `source_group`, `split`, `seed`, `locked_guard`
4. label availability: **yes (label) {'FAKE': 21, 'REAL': 16}**
5. sample-ID availability: **yes (37 unique)**
6. content hash available: **no**
7. population tier: **manifest (all splits; LOCKED rows are closed)**

### `phase4b/manifests/phase4b_combined_dataset_manifest.csv`

1. path: `phase4b/manifests/phase4b_combined_dataset_manifest.csv`
2. rows: **37** / 22 columns
3. available features: audio_ (4), other (17), resolution (1)
   full column list: `sample_id`, `filename`, `label`, `subtype`, `provenance`, `source_group`, `composite_flag`, `status`, `duration_seconds`, `resolution`, `video_fps`, `video_codec`, `audio_codec`, `audio_sample_rate_hz`, `audio_bitrate_kbps`, `video_bitrate_kbps`, `file_size_bytes`, `container`, `video_stream_count`, `audio_stream_count`, `md5`, `notes`
4. label availability: **yes (label) {'FAKE': 21, 'REAL': 16}**
5. sample-ID availability: **yes (37 unique)**
6. content hash available: **yes (md5)**
7. population tier: **manifest (all splits; LOCKED rows are closed)**

### `reports/DAY5_FUSION_INPUTS.csv`

1. path: `reports/DAY5_FUSION_INPUTS.csv`
2. rows: **28** / 9 columns
3. available features: other (9)
   full column list: `sample_id`, `true_label`, `subtype`, `xception_fake_score`, `ftcn_fake_score`, `aasist_fake_score`, `xception_available`, `ftcn_available`, `aasist_available`
4. label availability: **yes (true_label) {'FAKE': 17, 'REAL': 11}**
5. sample-ID availability: **yes (28 unique)**
6. content hash available: **no**
7. population tier: **development (frozen V2 rank-reference population)**

### `reports/day5_fusion_predictions.csv`

1. path: `reports/day5_fusion_predictions.csv`
2. rows: **28** / 8 columns
3. available features: other (8)
   full column list: `sample_id`, `true_label`, `xception_score`, `ftcn_score`, `aasist_score`, `X+A_rank_best_w`, `X+F_rank_w0.5`, `F+A_rank_w0.5`
4. label availability: **yes (true_label) {'FAKE': 17, 'REAL': 11}**
5. sample-ID availability: **yes (28 unique)**
6. content hash available: **no**
7. population tier: **development (frozen V2 rank-reference population)**

### `reports/DAY7_FRESH_VALIDATION_MANIFEST.csv`

1. path: `reports/DAY7_FRESH_VALIDATION_MANIFEST.csv`
2. rows: **29** / 16 columns
3. available features: audio_ (1), other (15)
   full column list: `sample_id`, `path`, `true_label`, `subtype`, `source`, `acquisition`, `duration_s`, `audio_present`, `fps`, `width`, `height`, `exposure_tier`, `prior_exposure_source`, `prior_aasist_score_note`, `md5`, `notes`
4. label availability: **yes (true_label) {'FAKE': 14, 'REAL': 15}**
5. sample-ID availability: **yes (29 unique)**
6. content hash available: **yes (md5)**
7. population tier: **manifest (all 29: DEV 16 / OTHER 8 / HOLDOUT 5)**

### `reports/DAY9_RAW_FRAME_EVIDENCE.csv`

1. path: `reports/DAY9_RAW_FRAME_EVIDENCE.csv`
2. rows: **394** / 13 columns
3. available features: face (3), fake_probability (1), other (9)
   full column list: `sample_id`, `true_label`, `exposure_tier`, `filename`, `path`, `frame_index`, `timestamp_s`, `face_detected`, `face_confidence`, `faces_in_frame`, `selected_face_box`, `fake_probability`, `error`
4. label availability: **yes (true_label) {'FAKE': 165, 'REAL': 229}**
5. sample-ID availability: **yes (29 unique)**
6. content hash available: **no**
7. population tier: **mixed (DEV 16 usable, OTHER 8 unscored, HOLDOUT 5 closed)**

### `reports/DAY9_AASIST_WINDOW_EVIDENCE.csv`

1. path: `reports/DAY9_AASIST_WINDOW_EVIDENCE.csv`
2. rows: **230** / 8 columns
3. available features: fake_direction_score (1), other (7)
   full column list: `sample_id`, `true_label`, `exposure_tier`, `window_index`, `start_sec`, `end_sec`, `fake_direction_score`, `status`
4. label availability: **yes (true_label) {'FAKE': 83, 'REAL': 147}**
5. sample-ID availability: **yes (27 unique)**
6. content hash available: **no**
7. population tier: **mixed (DEV usable, HOLDOUT 5 closed)**

### `reports/DAY9_XCEPTION_VIDEO_AGGREGATES.csv`

1. path: `reports/DAY9_XCEPTION_VIDEO_AGGREGATES.csv`
2. rows: **29** / 21 columns
3. available features: face (1), other (8), x_ (12)
   full column list: `sample_id`, `true_label`, `exposure_tier`, `frames_sampled`, `frames_with_face`, `usable_face_frames`, `face_fraction`, `x_mean`, `x_median`, `x_std`, `x_min`, `x_max`, `x_q10`, `x_q25`, `x_q75`, `x_q90`, `x_high_frac`, `x_very_high_frac`, `temporal_consistency`, `x_d7_diff`, `xception_status`
4. label availability: **yes (true_label) {'FAKE': 14, 'REAL': 15}**
5. sample-ID availability: **yes (29 unique)**
6. content hash available: **no**
7. population tier: **mixed (DEV 16 usable, OTHER 8 unscored, HOLDOUT 5 closed)**

### `reports/DAY9_AASIST_VIDEO_SUMMARIES.csv`

1. path: `reports/DAY9_AASIST_VIDEO_SUMMARIES.csv`
2. rows: **29** / 19 columns
3. available features: a_ (14), n_window (1), other (4)
   full column list: `sample_id`, `true_label`, `exposure_tier`, `n_windows`, `a_mean`, `a_median`, `a_std`, `a_min`, `a_max`, `a_q10`, `a_q25`, `a_q75`, `a_q90`, `a_iqr`, `a_high_frac`, `a_low_frac`, `a_pattern`, `a_d7_diff`, `aasist_status`
4. label availability: **yes (true_label) {'FAKE': 14, 'REAL': 15}**
5. sample-ID availability: **yes (29 unique)**
6. content hash available: **no**
7. population tier: **mixed (DEV 16 usable, OTHER 8 unscored, HOLDOUT 5 closed)**

### `reports/DAY9_AUDIO_ROBUSTNESS.csv`

1. path: `reports/DAY9_AUDIO_ROBUSTNESS.csv`
2. rows: **29** / 23 columns
3. available features: audio_ (5), bitrate (1), domain_ (3), face (1), other (12), resolution (1)
   full column list: `sample_id`, `true_label`, `exposure_tier`, `audio_present`, `audio_codec`, `original_sample_rate`, `original_channels`, `audio_duration_s`, `container_format`, `bitrate_kbps`, `resolution`, `asr_available`, `audio_heuristic_mean`, `aasist_label`, `aasist_spoof_score`, `aasist_window_pattern`, `high_spoof_fraction`, `has_face`, `face_fraction`, `domain_risk_state`, `domain_risk_flags`, `audio_risk`, `domain_risk_flag_count`
4. label availability: **yes (true_label) {'FAKE': 14, 'REAL': 15}**
5. sample-ID availability: **yes (29 unique)**
6. content hash available: **no**
7. population tier: **mixed (DEV 16 usable, HOLDOUT 5 closed)**

### `reports/DAY10_VISUAL_IDENTITY_SUMMARY.csv`

1. path: `reports/DAY10_VISUAL_IDENTITY_SUMMARY.csv`
2. rows: **29** / 27 columns
3. available features: area_ (2), centroid_ (2), face (2), id_ (7), other (14)
   full column list: `sample_id`, `true_label`, `subtype`, `exposure_tier`, `partition`, `availability`, `availability_reason`, `frames_sampled`, `face_detected_frames`, `frames_with_box`, `n_embeddings`, `n_pairs`, `id_cos_mean`, `id_cos_median`, `id_cos_q25`, `id_cos_min`, `id_low_frac`, `id_first_last_cos`, `id_temporal_dim_std_mean`, `area_cv`, `area_minmax_ratio`, `centroid_disp_mean`, `centroid_disp_max`, `face_persistence`, `support`, `embedding_model`, `runtime_s`
4. label availability: **yes (true_label) {'FAKE': 14, 'REAL': 15}**
5. sample-ID availability: **yes (29 unique)**
6. content hash available: **no**
7. population tier: **mixed (DEV 16 usable, OTHER 8 unscored, HOLDOUT 5 closed)**

### `reports/V2_LOCKED_RAW_PREDICTIONS.csv`

1. path: `reports/V2_LOCKED_RAW_PREDICTIONS.csv`
2. rows: **8** / 11 columns
3. available features: other (11)
   full column list: `sample_id`, `filename`, `path`, `true_label`, `subtype`, `xception_score`, `xception_status`, `aasist_score`, `aasist_status`, `runtime_xception_s`, `runtime_aasist_s`
4. label availability: **yes (true_label) {'FAKE': 4, 'REAL': 4}**
5. sample-ID availability: **yes (8 unique)**
6. content hash available: **no**
7. population tier: **LOCKED - CLOSED**

### `reports/V2_LOCKED_EVALUATION_INPUTS.csv`

1. path: `reports/V2_LOCKED_EVALUATION_INPUTS.csv`
2. rows: **8** / 13 columns
3. available features: fused (1), other (12)
   full column list: `sample_id`, `filename`, `path`, `true_label`, `subtype`, `xception_score`, `xception_rank`, `aasist_score`, `aasist_rank`, `fused_score`, `candidate_threshold`, `candidate_verdict`, `status`
4. label availability: **yes (true_label) {'FAKE': 4, 'REAL': 4}**
5. sample-ID availability: **yes (8 unique)**
6. content hash available: **no**
7. population tier: **LOCKED - CLOSED**

### `reports/DAY7_FRESH_PREDICTIONS.csv`

1. path: `reports/DAY7_FRESH_PREDICTIONS.csv`
2. rows: **21** / 7 columns
3. available features: other (7)
   full column list: `sample_id`, `true_label`, `oof_verdict_A`, `oof_verdict_C`, `oof_verdict_D`, `oof_verdict_SEL`, `selected_config`
4. label availability: **yes (true_label) {'FAKE': 12, 'REAL': 9}**
5. sample-ID availability: **yes (21 unique)**
6. content hash available: **no**
7. population tier: **fresh diagnostic (Day-7) - descriptive only**

### `reports/DAY7_BASELINE_PREDICTIONS.csv`

1. path: `reports/DAY7_BASELINE_PREDICTIONS.csv`
2. rows: **29** / 13 columns
3. available features: fused (1), other (12)
   full column list: `sample_id`, `filename`, `path`, `true_label`, `subtype`, `xception_score`, `xception_rank`, `aasist_score`, `aasist_rank`, `fused_score`, `candidate_threshold`, `candidate_verdict`, `status`
4. label availability: **yes (true_label) {'FAKE': 14, 'REAL': 15}**
5. sample-ID availability: **yes (29 unique)**
6. content hash available: **no**
7. population tier: **fresh diagnostic (Day-7) - descriptive only**

### `reports/DAY8_FAILURE_MODE_TABLE.csv`

1. path: `reports/DAY8_FAILURE_MODE_TABLE.csv`
2. rows: **29** / 19 columns
3. available features: other (19)
   full column list: `sample_id`, `true_label`, `subtype`, `source`, `exposure_tier`, `xception_score`, `aasist_fake_score`, `day6_fusion_score`, `day6_fusion_verdict`, `strategy_A_verdict`, `strategy_B_verdict`, `strategy_C_verdict`, `strategy_D_verdict`, `strategy_E_verdict`, `xception_correct`, `aasist_correct`, `fusion_correct`, `error_category`, `modality_note`
4. label availability: **yes (true_label) {'FAKE': 14, 'REAL': 15}**
5. sample-ID availability: **yes (29 unique)**
6. content hash available: **no**
7. population tier: **fresh diagnostic (Day-8) - descriptive only**

### `reports/DAY9_DEVELOPMENT_PREDICTIONS.csv`

1. path: `reports/DAY9_DEVELOPMENT_PREDICTIONS.csv`
2. rows: **16** / 45 columns
3. available features: a_ (9), audio_ (4), bitrate (1), domain_ (2), face (1), n_window (1), other (17), resolution (1), x_ (9)
   full column list: `sample_id`, `true_label`, `subtype`, `source`, `acquisition`, `exposure_tier`, `partition`, `x_mean`, `x_rank`, `a_rank`, `a_mean`, `day6_fused_score`, `day6_verdict`, `xception_only_verdict`, `native_aasist_verdict`, `day9_verdict`, `day9_binary`, `face_fraction`, `usable_face_frames`, `frames_sampled`, `x_std`, `x_min`, `x_max`, `x_q25`, `x_q75`, `x_high_frac`, `x_very_high_frac`, `temporal_consistency`, `n_windows`, `a_std`, `a_iqr`, `a_min`, `a_max`, `a_high_frac`, `a_low_frac`, `a_pattern`, `audio_present`, `audio_codec`, `bitrate_kbps`, `resolution`, `audio_heuristic_mean`, `domain_risk_state`, `domain_risk_flags`, `audio_risk`, `aasist_label_for_route`
4. label availability: **yes (true_label) {'FAKE': 11, 'REAL': 5}**
5. sample-ID availability: **yes (16 unique)**
6. content hash available: **no**
7. population tier: **development (Day-9 DEV 16)**

### `reports/DAY9_HONEST_HOLDOUT_PREDICTIONS.csv`

1. path: `reports/DAY9_HONEST_HOLDOUT_PREDICTIONS.csv`
2. rows: **5** / 45 columns
3. available features: a_ (9), audio_ (4), bitrate (1), domain_ (2), face (1), n_window (1), other (17), resolution (1), x_ (9)
   full column list: `sample_id`, `true_label`, `subtype`, `source`, `acquisition`, `exposure_tier`, `partition`, `x_mean`, `x_rank`, `a_rank`, `a_mean`, `day6_fused_score`, `day6_verdict`, `xception_only_verdict`, `native_aasist_verdict`, `day9_verdict`, `day9_binary`, `face_fraction`, `usable_face_frames`, `frames_sampled`, `x_std`, `x_min`, `x_max`, `x_q25`, `x_q75`, `x_high_frac`, `x_very_high_frac`, `temporal_consistency`, `n_windows`, `a_std`, `a_iqr`, `a_min`, `a_max`, `a_high_frac`, `a_low_frac`, `a_pattern`, `audio_present`, `audio_codec`, `bitrate_kbps`, `resolution`, `audio_heuristic_mean`, `domain_risk_state`, `domain_risk_flags`, `audio_risk`, `aasist_label_for_route`
4. label availability: **yes (true_label) {'REAL': 4, 'FAKE': 1}**
5. sample-ID availability: **yes (5 unique)**
6. content hash available: **no**
7. population tier: **HONEST HOLDOUT - CLOSED**

### `reports/DAY10_DEVELOPMENT_PREDICTIONS.csv`

1. path: `reports/DAY10_DEVELOPMENT_PREDICTIONS.csv`
2. rows: **16** / 51 columns
3. available features: a_ (9), area_ (2), audio_ (1), centroid_ (2), domain_ (1), face (2), id_ (7), n_window (1), other (19), x_ (7)
   full column list: `sample_id`, `true_label`, `subtype`, `source`, `exposure_tier`, `partition`, `x_mean`, `x_rank`, `x_std`, `x_q25`, `x_q75`, `x_high_frac`, `x_very_high_frac`, `face_fraction`, `usable_face_frames`, `frames_sampled`, `a_mean`, `a_rank`, `a_std`, `a_iqr`, `a_min`, `a_max`, `a_high_frac`, `a_low_frac`, `a_pattern`, `n_windows`, `day6_fused_score`, `audio_risk`, `domain_risk_state`, `day6_verdict`, `xception_only_verdict`, `native_aasist_verdict`, `day9_verdict`, `day9_binary`, `availability`, `availability_reason`, `identity_support`, `n_embeddings`, `n_pairs`, `face_persistence`, `id_cos_mean`, `id_cos_median`, `id_cos_q25`, `id_cos_min`, `id_low_frac`, `id_first_last_cos`, `id_temporal_dim_std_mean`, `area_cv`, `area_minmax_ratio`, `centroid_disp_mean`, `centroid_disp_max`
4. label availability: **yes (true_label) {'FAKE': 11, 'REAL': 5}**
5. sample-ID availability: **yes (16 unique)**
6. content hash available: **no**
7. population tier: **development (Day-10 DEV 16)**

### `reports/DAY10_HONEST_HOLDOUT_PREDICTIONS.csv`

1. path: `reports/DAY10_HONEST_HOLDOUT_PREDICTIONS.csv`
2. rows: **5** / 51 columns
3. available features: a_ (9), area_ (2), audio_ (1), centroid_ (2), domain_ (1), face (2), id_ (7), n_window (1), other (19), x_ (7)
   full column list: `sample_id`, `true_label`, `subtype`, `source`, `exposure_tier`, `partition`, `x_mean`, `x_rank`, `x_std`, `x_q25`, `x_q75`, `x_high_frac`, `x_very_high_frac`, `face_fraction`, `usable_face_frames`, `frames_sampled`, `a_mean`, `a_rank`, `a_std`, `a_iqr`, `a_min`, `a_max`, `a_high_frac`, `a_low_frac`, `a_pattern`, `n_windows`, `day6_fused_score`, `audio_risk`, `domain_risk_state`, `day6_verdict`, `xception_only_verdict`, `native_aasist_verdict`, `day9_verdict`, `day9_binary`, `availability`, `availability_reason`, `identity_support`, `n_embeddings`, `n_pairs`, `face_persistence`, `id_cos_mean`, `id_cos_median`, `id_cos_q25`, `id_cos_min`, `id_low_frac`, `id_first_last_cos`, `id_temporal_dim_std_mean`, `area_cv`, `area_minmax_ratio`, `centroid_disp_mean`, `centroid_disp_max`
4. label availability: **yes (true_label) {'REAL': 4, 'FAKE': 1}**
5. sample-ID availability: **yes (5 unique)**
6. content hash available: **no**
7. population tier: **HONEST HOLDOUT - CLOSED**

### `reports/DAY11_XCEPTION_TEMPORAL_DESCRIPTORS.csv`

1. path: `reports/DAY11_XCEPTION_TEMPORAL_DESCRIPTORS.csv`
2. rows: **29** / 39 columns
3. available features: burst (1), face (1), frac (4), half_ (1), longest_run (6), n_transitions (3), other (21), x_ (2)
   full column list: `sample_id`, `true_label`, `exposure_tier`, `partition`, `frames_sampled`, `n_face_frames`, `n_usable`, `n_unusable_crops`, `face_fraction`, `evidence_status`, `aggregate_basis`, `mean`, `median`, `min`, `max`, `q75`, `q90`, `q95`, `std`, `iqr`, `frac_ge_0.40`, `frac_ge_0.50`, `frac_ge_0.60`, `frac_ge_0.70`, `longest_run_0.50`, `longest_run_0.60`, `longest_run_0.70`, `longest_run_0.50_sec`, `longest_run_0.60_sec`, `longest_run_0.70_sec`, `n_transitions_0.50`, `n_transitions_0.60`, `n_transitions_0.70`, `burst_coverage_0.50`, `first_half_mean`, `second_half_mean`, `half_delta_abs`, `x_mean_ref`, `x_mean_diff`
4. label availability: **yes (true_label) {'FAKE': 14, 'REAL': 15}**
5. sample-ID availability: **yes (29 unique)**
6. content hash available: **no**
7. population tier: **mixed (DEV 16 usable, OTHER 8 unscored, HOLDOUT 5 closed)**

### `reports/DAY11_DEVELOPMENT_PREDICTIONS.csv`

1. path: `reports/DAY11_DEVELOPMENT_PREDICTIONS.csv`
2. rows: **16** / 46 columns
3. available features: a_ (1), audio_ (1), burst (1), domain_ (1), face (1), frac (4), half_ (1), longest_run (3), n_transitions (1), other (31), x_ (1)
   full column list: `sample_id`, `true_label`, `subtype`, `source`, `exposure_tier`, `partition`, `x_mean`, `a_mean`, `day6_fused_score`, `day6_verdict`, `xception_only_verdict`, `native_aasist_verdict`, `day9_verdict`, `audio_risk`, `domain_risk_state`, `face_fraction`, `n_usable`, `evidence_status`, `mean`, `median`, `min`, `max`, `q75`, `q90`, `q95`, `std`, `frac_ge_0.40`, `frac_ge_0.50`, `frac_ge_0.60`, `frac_ge_0.70`, `longest_run_0.50`, `longest_run_0.60`, `longest_run_0.70`, `n_transitions_0.50`, `burst_coverage_0.50`, `first_half_mean`, `second_half_mean`, `half_delta_abs`, `cfg_d9_origin`, `cfg_q90_strong`, `cfg_q95_strong`, `cfg_run060_min2`, `cfg_run050_min3`, `cfg_frac060_ge_0.25`, `cfg_frac050_compact`, `cfg_peak_and_support`
4. label availability: **yes (true_label) {'FAKE': 11, 'REAL': 5}**
5. sample-ID availability: **yes (16 unique)**
6. content hash available: **no**
7. population tier: **development (Day-11 DEV 16)**

### `reports/DAY11_HONEST_HOLDOUT_PREDICTIONS.csv`

1. path: `reports/DAY11_HONEST_HOLDOUT_PREDICTIONS.csv`
2. rows: **5** / 33 columns
3. available features: a_ (1), audio_ (1), burst (1), domain_ (1), frac (3), half_ (1), longest_run (3), n_transitions (1), other (20), x_ (1)
   full column list: `sample_id`, `true_label`, `exposure_tier`, `partition`, `x_mean`, `a_mean`, `day6_verdict`, `xception_only_verdict`, `native_aasist_verdict`, `day9_verdict`, `day11_verdict`, `day11_selected`, `audio_risk`, `domain_risk_state`, `n_usable`, `evidence_status`, `mean`, `median`, `min`, `max`, `q90`, `q95`, `frac_ge_0.50`, `frac_ge_0.60`, `frac_ge_0.70`, `longest_run_0.50`, `longest_run_0.60`, `longest_run_0.70`, `n_transitions_0.50`, `burst_coverage_0.50`, `first_half_mean`, `second_half_mean`, `half_delta_abs`
4. label availability: **yes (true_label) {'REAL': 4, 'FAKE': 1}**
5. sample-ID availability: **yes (5 unique)**
6. content hash available: **no**
7. population tier: **HONEST HOLDOUT - CLOSED**

### `reports/DAY5_CROSSVALIDATION.json`

1. path: `reports/DAY5_CROSSVALIDATION.json`
2. rows: **None**
3. available features: ['FTCN+AASIST_rank_OOF', 'Xception+AASIST_rank_OOF', 'Xception+FTCN_rank_OOF']
4. label availability: **no**
5. sample-ID availability: **yes (None unique)**
6. content hash available: **no**
7. population tier: **development OOF (historical, frozen reference)**

### `reports/V2_CANDIDATE_FREEZE.json`

1. path: `reports/V2_CANDIDATE_FREEZE.json`
2. rows: **None**
3. available features: ['aasist_weight', 'candidate_name', 'detectors', 'devval_reference_sample_count', 'fpr_constraint', 'freeze_json_sha256', 'freeze_timestamp', 'input_artifact_sha256', 'locked_sample_ids', 'missing_modality_policy', 'normalization_method', 'objective', 'threshold', 'threshold_selection_method', 'weight_selection_method', 'xception_weight']
4. label availability: **no**
5. sample-ID availability: **no**
6. content hash available: **no**
7. population tier: **frozen V2 candidate definition**

### `reports/FORENSIGHT_V2_FINAL_SUMMARY.json`

1. path: `reports/FORENSIGHT_V2_FINAL_SUMMARY.json`
2. rows: **None**
3. available features: ['architecture', 'completion_statement', 'date', 'evidence', 'final_decision', 'findings', 'limitations', 'persistent_ui_label', 'project_status', 'promotion_gate', 'promotion_gate_statement', 'report', 'reproducibility', 'results_table']
4. label availability: **no**
5. sample-ID availability: **no**
6. content hash available: **no**
7. population tier: **historical V2 final summary**

## Overlap between artifacts (by sample id)

| artifact A | artifact B | shared ids | byte-identical md5 |
|---|---|---|---|
| d10_dev_predictions | d11_descriptors | 16 | False |
| d10_dev_predictions | d11_dev_predictions | 16 | False |
| d10_dev_predictions | d7_a_agg | 16 | False |
| d10_dev_predictions | d7_audio_robustness | 16 | False |
| d10_dev_predictions | d7_baseline_predictions | 16 | False |
| d10_dev_predictions | d7_frame | 16 | False |
| d10_dev_predictions | d7_fresh_predictions | 16 | False |
| d10_dev_predictions | d7_identity | 16 | False |
| d10_dev_predictions | d7_manifest | 16 | False |
| d10_dev_predictions | d7_window | 16 | False |
| d10_dev_predictions | d7_x_agg | 16 | False |
| d10_dev_predictions | d8_failure_mode_table | 16 | False |
| d10_dev_predictions | d9_dev_predictions | 16 | False |
| d10_holdout_predictions | d11_descriptors | 5 | False |
| d10_holdout_predictions | d11_holdout_predictions | 5 | False |
| d10_holdout_predictions | d7_a_agg | 5 | False |
| d10_holdout_predictions | d7_audio_robustness | 5 | False |
| d10_holdout_predictions | d7_baseline_predictions | 5 | False |
| d10_holdout_predictions | d7_frame | 5 | False |
| d10_holdout_predictions | d7_fresh_predictions | 5 | False |
| d10_holdout_predictions | d7_identity | 5 | False |
| d10_holdout_predictions | d7_manifest | 5 | False |
| d10_holdout_predictions | d7_window | 5 | False |
| d10_holdout_predictions | d7_x_agg | 5 | False |
| d10_holdout_predictions | d8_failure_mode_table | 5 | False |
| d10_holdout_predictions | d9_holdout_predictions | 5 | False |
| d11_descriptors | d11_dev_predictions | 16 | False |
| d11_descriptors | d11_holdout_predictions | 5 | False |
| d11_descriptors | d7_a_agg | 29 | False |
| d11_descriptors | d7_audio_robustness | 29 | False |
| d11_descriptors | d7_baseline_predictions | 29 | False |
| d11_descriptors | d7_frame | 29 | False |
| d11_descriptors | d7_fresh_predictions | 21 | False |
| d11_descriptors | d7_identity | 29 | False |
| d11_descriptors | d7_manifest | 29 | False |
| d11_descriptors | d7_window | 27 | False |
| d11_descriptors | d7_x_agg | 29 | False |
| d11_descriptors | d8_failure_mode_table | 29 | False |
| d11_descriptors | d9_dev_predictions | 16 | False |
| d11_descriptors | d9_holdout_predictions | 5 | False |
| d11_dev_predictions | d7_a_agg | 16 | False |
| d11_dev_predictions | d7_audio_robustness | 16 | False |
| d11_dev_predictions | d7_baseline_predictions | 16 | False |
| d11_dev_predictions | d7_frame | 16 | False |
| d11_dev_predictions | d7_fresh_predictions | 16 | False |
| d11_dev_predictions | d7_identity | 16 | False |
| d11_dev_predictions | d7_manifest | 16 | False |
| d11_dev_predictions | d7_window | 16 | False |
| d11_dev_predictions | d7_x_agg | 16 | False |
| d11_dev_predictions | d8_failure_mode_table | 16 | False |
| d11_dev_predictions | d9_dev_predictions | 16 | False |
| d11_holdout_predictions | d7_a_agg | 5 | False |
| d11_holdout_predictions | d7_audio_robustness | 5 | False |
| d11_holdout_predictions | d7_baseline_predictions | 5 | False |
| d11_holdout_predictions | d7_frame | 5 | False |
| d11_holdout_predictions | d7_fresh_predictions | 5 | False |
| d11_holdout_predictions | d7_identity | 5 | False |
| d11_holdout_predictions | d7_manifest | 5 | False |
| d11_holdout_predictions | d7_window | 5 | False |
| d11_holdout_predictions | d7_x_agg | 5 | False |
| d11_holdout_predictions | d8_failure_mode_table | 5 | False |
| d11_holdout_predictions | d9_holdout_predictions | 5 | False |
| d7_a_agg | d7_audio_robustness | 29 | False |
| d7_a_agg | d7_baseline_predictions | 29 | False |
| d7_a_agg | d7_frame | 29 | False |
| d7_a_agg | d7_fresh_predictions | 21 | False |
| d7_a_agg | d7_identity | 29 | False |
| d7_a_agg | d7_manifest | 29 | False |
| d7_a_agg | d7_window | 27 | False |
| d7_a_agg | d7_x_agg | 29 | False |
| d7_a_agg | d8_failure_mode_table | 29 | False |
| d7_a_agg | d9_dev_predictions | 16 | False |
| d7_a_agg | d9_holdout_predictions | 5 | False |
| d7_audio_robustness | d7_baseline_predictions | 29 | False |
| d7_audio_robustness | d7_frame | 29 | False |
| d7_audio_robustness | d7_fresh_predictions | 21 | False |
| d7_audio_robustness | d7_identity | 29 | False |
| d7_audio_robustness | d7_manifest | 29 | False |
| d7_audio_robustness | d7_window | 27 | False |
| d7_audio_robustness | d7_x_agg | 29 | False |
| d7_audio_robustness | d8_failure_mode_table | 29 | False |
| d7_audio_robustness | d9_dev_predictions | 16 | False |
| d7_audio_robustness | d9_holdout_predictions | 5 | False |
| d7_baseline_predictions | d7_frame | 29 | False |
| d7_baseline_predictions | d7_fresh_predictions | 21 | False |
| d7_baseline_predictions | d7_identity | 29 | False |
| d7_baseline_predictions | d7_manifest | 29 | False |
| d7_baseline_predictions | d7_window | 27 | False |
| d7_baseline_predictions | d7_x_agg | 29 | False |
| d7_baseline_predictions | d8_failure_mode_table | 29 | False |
| d7_baseline_predictions | d9_dev_predictions | 16 | False |
| d7_baseline_predictions | d9_holdout_predictions | 5 | False |
| d7_frame | d7_fresh_predictions | 21 | False |
| d7_frame | d7_identity | 29 | False |
| d7_frame | d7_manifest | 29 | False |
| d7_frame | d7_window | 27 | False |
| d7_frame | d7_x_agg | 29 | False |
| d7_frame | d8_failure_mode_table | 29 | False |
| d7_frame | d9_dev_predictions | 16 | False |
| d7_frame | d9_holdout_predictions | 5 | False |
| d7_fresh_predictions | d7_identity | 21 | False |
| d7_fresh_predictions | d7_manifest | 21 | False |
| d7_fresh_predictions | d7_window | 21 | False |
| d7_fresh_predictions | d7_x_agg | 21 | False |
| d7_fresh_predictions | d8_failure_mode_table | 21 | False |
| d7_fresh_predictions | d9_dev_predictions | 16 | False |
| d7_fresh_predictions | d9_holdout_predictions | 5 | False |
| d7_identity | d7_manifest | 29 | False |
| d7_identity | d7_window | 27 | False |
| d7_identity | d7_x_agg | 29 | False |
| d7_identity | d8_failure_mode_table | 29 | False |
| d7_identity | d9_dev_predictions | 16 | False |
| d7_identity | d9_holdout_predictions | 5 | False |
| d7_manifest | d7_window | 27 | False |
| d7_manifest | d7_x_agg | 29 | False |
| d7_manifest | d8_failure_mode_table | 29 | False |
| d7_manifest | d9_dev_predictions | 16 | False |
| d7_manifest | d9_holdout_predictions | 5 | False |
| d7_window | d7_x_agg | 27 | False |
| d7_window | d8_failure_mode_table | 27 | False |
| d7_window | d9_dev_predictions | 16 | False |
| d7_window | d9_holdout_predictions | 5 | False |
| d7_x_agg | d8_failure_mode_table | 29 | False |
| d7_x_agg | d9_dev_predictions | 16 | False |
| d7_x_agg | d9_holdout_predictions | 5 | False |
| d8_failure_mode_table | d9_dev_predictions | 16 | False |
| d8_failure_mode_table | d9_holdout_predictions | 5 | False |
| locked_inputs | locked_raw | 8 | False |
| locked_inputs | p4b_dataset_manifest | 8 | False |
| locked_inputs | p4b_split_manifest | 8 | False |
| locked_raw | p4b_dataset_manifest | 8 | False |
| locked_raw | p4b_split_manifest | 8 | False |
| p4b_audio | p4b_dataset_manifest | 28 | False |
| p4b_audio | p4b_frame | 28 | False |
| p4b_audio | p4b_fusion_inputs | 28 | False |
| p4b_audio | p4b_fusion_predictions | 28 | False |
| p4b_audio | p4b_split_manifest | 28 | False |
| p4b_audio | p4b_video | 28 | False |
| p4b_dataset_manifest | p4b_frame | 28 | False |
| p4b_dataset_manifest | p4b_fusion_inputs | 28 | False |
| p4b_dataset_manifest | p4b_fusion_predictions | 28 | False |
| p4b_dataset_manifest | p4b_split_manifest | 37 | False |
| p4b_dataset_manifest | p4b_video | 28 | False |
| p4b_frame | p4b_fusion_inputs | 28 | False |
| p4b_frame | p4b_fusion_predictions | 28 | False |
| p4b_frame | p4b_split_manifest | 28 | False |
| p4b_frame | p4b_video | 28 | False |
| p4b_fusion_inputs | p4b_fusion_predictions | 28 | False |
| p4b_fusion_inputs | p4b_split_manifest | 28 | False |
| p4b_fusion_inputs | p4b_video | 28 | False |
| p4b_fusion_predictions | p4b_split_manifest | 28 | False |
| p4b_fusion_predictions | p4b_video | 28 | False |
| p4b_split_manifest | p4b_video | 28 | False |

## Cross-pool content-hash check

- P4B manifest md5 count: 37
- D7 manifest md5 count: 29
- shared md5: **none**
- byte-identical overlap between the two development pools: **False**

## Closed populations and where they appear

| population | n | status | present in artifacts |
|---|---|---|---|
| DAY6_LOCKED8 | 8 | CLOSED - never used for tuning/selection/thresholds | locked_inputs, locked_raw, p4b_dataset_manifest, p4b_split_manifest |
| D9_HONEST_HOLDOUT5 | 5 | CLOSED - already consumed, descriptive evidence only | d10_holdout_predictions, d11_descriptors, d11_holdout_predictions, d7_a_agg, d7_audio_robustness, d7_baseline_predictions, d7_frame, d7_fresh_predictions, d7_identity, d7_manifest, d7_window, d7_x_agg, d8_failure_mode_table, d9_holdout_predictions |
| D7_VISUAL_UNAVAILABLE8 | 8 | NOT SCORED - frozen VISUAL_UNAVAILABLE policy | d11_descriptors, d7_a_agg, d7_audio_robustness, d7_baseline_predictions, d7_frame, d7_identity, d7_manifest, d7_window, d7_x_agg, d8_failure_mode_table |
| P4B_VISUAL_UNAVAILABLE3 | 3 | NOT SCORED - no usable Xception face score | p4b_audio, p4b_dataset_manifest, p4b_frame, p4b_fusion_inputs, p4b_fusion_predictions, p4b_split_manifest, p4b_video |
| P4B_HELD_OUT_OVERLAP1 | 1 | EXCLUDE_OVERLAP - perceptually identical (dHash 0) to a Phase-4A LOCKED test file; deliberately kept out of every Phase-4B split | p4b_dataset_manifest, p4b_split_manifest |
