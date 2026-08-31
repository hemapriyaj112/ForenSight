# ForenSight — Phase 1 Repair Report

**Date:** 2026-08-27
**Source of truth:** `reports/FULL_BACKEND_AUDIT.md`
**Scope:** Repair-and-baseline only. No thresholds tuned, no new detectors, no training,
no UI changes beyond honest status, all FREE/local.

---

## 1. Changes made

### TASK 1 — Provenance fusion override (explicit, not just weighted average)
`pipeline/fusion/fuser.py` (`Fuser.fuse`):
- After the weighted-average fusion and the existing REAL→UNCERTAIN downgrade, an
  **explicit provenance override** runs: if `video_result.metadata["provenance_score"] >= 0.90`
  the verdict is forced to `FAKE`.
- The returned `AnalysisResult.metadata` now carries structured evidence:
  `provenance_override` (bool), `provenance_score` (float), `provenance_reason` (str).
- The 0.40 / 0.60 thresholds are **unchanged**; the override sits on top of fusion.
- This fixes the dilution bug the audit found: a video with `provenance_score=0.97`
  produced `video_score=0.9` but fusion with a low audio reading dropped the fused
  score to 0.594 → UNCERTAIN. It is now correctly `FAKE`.

### TASK 2 — Remove the backwards image classifier from VIDEO decision-making
`pipeline/video/detector.py` is unchanged; the wiring is disabled in `main.py`
(`run_pipeline`): `VideoDetector(CFG, ai_classifier=None, ...)`.
- The image classifier is **no longer passed to the video detector**, so it cannot
  influence video verdicts. Its code is preserved and still used for IMAGE analysis
  (`app.run_image_pipeline` → `ImageDetector(ai_classifier=...)`).
- Eval script reports per video: `classifier_status=INACTIVE`,
  `classifier_score=null`, reason = "Classifier behavior is reversed/invalid for
  current video preprocessing and requires independent validation."

### TASK 3 — Honest detector status
`pipeline/fusion` unchanged; `main.py`:
- `_load_ai_classifier()` now calls `clf._ensure_loaded()` **before** checking
  availability, so it no longer logs "Ensemble classifier failed to load" at
  construction time (the misleading pre-load warning from the audit).
- Added `main.get_detector_status(cfg, classifier)` returning a structured list with
  three honest categories: `ACTIVE_MODEL`, `ACTIVE_HEURISTIC`, `INACTIVE`. A stub is
  **never** called ACTIVE. The eval script prints this at startup (FAIL-LOUD).

### TASK 4 — Audio segment config mismatch
`pipeline/audio/detector.py` (`AudioDetector.__init__`): `_segment_dur` now reads
`config.audio.segment_duration` (authoritative). `pipeline/audio/audio_forensics.py`
default constant aligned to `2.0` to match config. All segment timestamps derive from
the configured duration (no more 2.0-vs-4.0 split).

### TASK 5 — Missing audio is neutral but explicit
`main.run_pipeline` path (mirrored in the eval script): when there is no audio track,
`audio_score = null` and `audio_status = INACTIVE` with reason "No audio track".
`Fuser.fuse` skips inactive audio (it was already down-weighting it, but now treats
`fake_prob=None` safely and never folds a "fake-looking" 0.5 into the average).
Same honesty rule applies to any inactive detector (null score + explicit status).

### TASK 6 — Improved evaluation reporting
`scripts/evaluate_testing_videos.py` rewritten. Per video now records:
`filename, ground_truth, duration, fps, has_audio, video_status, video_score,
audio_status, audio_score, provenance_status, provenance_score, provenance_override,
classifier_status, classifier_score, fusion_score, verdict` (plus texture/frequency,
frame/segment counts, per-frame + per-segment detail). Computes TP/TN/FP/FN, precision,
recall, F1, FNR, FPR, `fake_recall`, **`fake_verdict_rate`**, **`uncertain_rate`**, and
for FAKE videos separately: predicted FAKE / UNCERTAIN / REAL.

### TASK 7 — Per-modality confusion analysis ("which detector catches which video")
The eval JSON now includes `per_fake_detector_breakdown`: for every FAKE video, the
provenance result, video result, audio result, image-classifier result (INACTIVE), and
final verdict — so we can see exactly which signal would need upgrading.

### TASK 8 — Tests
`pytest tests/` → **228 passed, 5 failed**. The 5 failures are the same pre-existing
environmental failures in `tests/unit/test_app.py` (`streamlit.session_state` unavailable
outside a `streamlit run` runtime) — unrelated to these changes. The evaluation script
was executed end-to-end (`reports/testing_videos_evaluation.json/.csv`,
`reports/testing_videos_manifest.csv` regenerated).

---

## 2. Before / after FAKE recall

| Metric | Before (audit baseline) | After (Phase 1) | Δ |
|---|---|---|---|
| FAKE_RECALL | 0.4545 (5/11) | **0.5455 (6/11)** | +0.0909 |
| TP | 5 | 6 | +1 |
| TN | 4 | 4 | 0 |
| FP | 0 | 0 | 0 |
| FN | 6 | 5 | −1 |
| Accuracy | 0.600 | 0.667 | +0.067 |
| Precision (fake) | 1.000 | 1.000 | 0 |
| F1 (fake) | 0.625 | 0.706 | +0.081 |
| fake_verdict_rate | — | 0.400 | — |
| uncertain_rate | — | 0.267 | — |

Thresholds (0.40 / 0.60) and weights (video 0.6 / audio 0.4) were **not** changed.

---

## 3. Before / after false negatives

False negatives (AI videos called not-FAKE):

| Video | Before | After |
|---|---|---|
| ai_noaud2.mp4 | UNCERTAIN (FN) | UNCERTAIN (FN) |
| avengers ai vid 1.mp4 | REAL (FN) | REAL (FN) |
| avengers ai vid 2.mp4 | UNCERTAIN (FN) | UNCERTAIN (FN) |
| insta vid.mp4 | REAL (FN) | REAL (FN) |
| real vid ai aud.mp4 | REAL (FN) | REAL (FN) |
| **girl walkin video.mp4** | **UNCERTAIN (FN)** | **FAKE (now TP)** |

All 5 remaining FNs are **untagged AI videos** (no C2PA/AI manifest). They are caught
only by the weak/noisy texture heuristic and the (now disabled, backwards) classifier.

---

## 4. Before / after false positives

False positives (REAL videos called FAKE): **0 before, 0 after.** Disabling the
backwards classifier and adding the provenance override did **not** increase false
positives. The four REAL videos remain correctly not-FAKE (3 REAL + 1 UNCERTAIN).

---

## 5. Which videos changed verdict

Exactly **one**: `ai/girl walkin video.mp4` — UNCERTAIN → FAKE, caused by the Task 1
provenance override (`provenance_score=0.97`, `provenance_override=True`). No video
moved toward a false positive.

---

## 6. Current ACTIVE detectors

| Detector | Category | Note |
|---|---|---|
| Face cropper (MTCNN) | ACTIVE_MODEL | facenet_pytorch, real loaded model |
| AI image classifier (ensemble) | ACTIVE_MODEL | loads from HuggingFace Hub; **IMAGE analysis only** (disabled for video, Task 2) |
| Audio heuristic forensics | ACTIVE_HEURISTIC | spectral stability / phase / MFCC / flux / ZCR |
| Video texture heuristic | ACTIVE_HEURISTIC | per-frame Laplacian variance |
| Metadata / C2PA provenance | ACTIVE_HEURISTIC | EXIF/C2PA/generator scan; strongest FAKE signal |
| Temporal localization | ACTIVE_HEURISTIC | frame-to-frame consistency |
| Document detector | ACTIVE_HEURISTIC | colour/edge gating |

---

## 7. Current INACTIVE detectors (must NOT be called ACTIVE)

| Detector | Why |
|---|---|
| EfficientNetV2 (video) | `config.video.weights=models/efficientnet_v2s.pt` never loaded; `models/` absent. Video "texture" is a heuristic, not a network. |
| RawNet2 (audio) | `config.audio.weights=models/rawnet2.pt` absent; `RawNet2Classifier` is a stub. Audio is heuristic-only. |
| GradCAM (neural) | `_GradCAM` runs `_stub_forward` (model=None); not a network. |
| ELA / Noise-floor / Spectral-FFT | excluded from the video pipeline; only active on the image tab. |
| video_image_classifier | disabled for video verdicts (Task 2) — validated backwards on video. |

---

## 8. Remaining weaknesses

1. **5 of 11 AI videos still missed (FN)** — all lack a C2PA/AI manifest, so provenance
   (the only reliable FAKE signal) cannot catch them. They fall back to the weak texture
   heuristic, which is noisy (some REAL videos have texture ≈ 0.45–0.83; some AI ≈ 0.14–0.99).
2. **No real video neural network.** EfficientNetV2 is configured but never loaded; the
   video path has no trained spatial model — only the Laplacian-variance heuristic.
3. **No real audio neural network.** RawNet2 is a stub; audio heuristics do not reliably
   flag AI/TTS speech (audio scores for AI videos were 0.07–0.45, i.e. "clean").
4. **1 fps frame sampling** may miss short, localized temporal edits.
5. **Image classifier validity on video is unverified**; it is only kept for IMAGE analysis
   and must be independently validated before re-enabling for video.
6. **No calibration fitted** — scores are raw weighted means, so they are *not* calibrated
   probabilities (consumers must not treat `fake_prob`/`fused_score` as probabilities).
7. **Provenance only covers signed AI output** (C2PA / generator tags). Unsigned AI
   generation is invisible to it.

---

## 9. Recommendation for the NEXT detector to integrate

The dominant gap is **untagged AI video** (the 5 remaining FNs). The single highest-leverage,
still-FREE/local next step is to **enable the already-configured EfficientNetV2 video
network with a locally-hosted checkpoint** (`config.video.weights`) — this is *not* a new
detector, it is the model already declared in config but never actually loaded. Validate it
on the `ai/` vs `real/` split (without tuning thresholds) and let it contribute to the
per-frame texture/video score. In parallel:

- **Re-enable ELA / noise-residual / spectral-FFT on video frames** (they are currently
  excluded from video) to add real pixel-level forensic signal beyond texture.
- **Independently validate the image classifier's direction on video**; only re-enable it
  for video once its sign is confirmed (replacing the silent flip with an explicit,
  logged correction).

Do **not** add GenConViT / AASIST / HAVIC / DeepfakeBench or any paid/cloud-only detector.
Keep everything local and permissively licensed.
