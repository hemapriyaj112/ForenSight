# ForenSight — Full Backend Audit

**Date:** 2026-08-27
**Scope:** Backend only. No UI, no threshold changes, no new detectors added.
**Method:** Static read of the full pipeline + a reproducible black-box evaluation of every
video in the local `testing_videos` directory, run through the *actual* current pipeline
(`main.run_pipeline` path), with FAIL-LOUD status reporting per detector.

---

## 1. Executive summary

- The system is **not** using the "neural" detectors it is configured for. **EfficientNetV2
  (video) and RawNet2 (audio) are never loaded** — their checkpoints (`models/…`) do not exist
  and the code paths fall back to heuristics / stubs. Per the FAIL-LOUD rule both are reported
  `INACTIVE`.
- The **only genuinely trained signal is an image classifier ensemble**
  (`prithivMLmods/deepfake-detector-model-v1` + `dima806/deepfake_vs_real_image_detection`) loaded
  dynamically from the HuggingFace Hub at runtime. It is `ACTIVE` here, but it **reads backwards on
  video frames** (AI→low, real→high) — this was already known (`classifier_validation_results.json`,
  `face_crop.py` docstring) and the pipeline applies a fragile heuristic flip.
- **Provenance (C2PA / generator-manifest scan) is the single most reliable FAKE signal**, and it
  is the only thing catching the AI videos in this test set.
- **False-negative rate is the dominant problem.** On the 15 labelled testing videos we measured
  **FAKE_RECALL = 45.5%** (6 of 11 AI videos missed) at **0 false positives**. The false negatives
  are almost entirely AI videos that lack a C2PA/AI manifest and that the backwards classifier +
  noisy texture heuristic fail to flag.

> The pipeline is currently excellent at *not* calling real videos fake, and weak at *catching*
> AI videos that were not signed by an AI tool. That is the opposite of the project's stated goal
> (minimize false negatives).

---

## 2. Scope & method

1. Read `main.py`, `app.py`, `utils/config.py`, `utils/types.py`, `utils/demux.py`,
   `config/config.yaml`, `config/config.toml`.
2. Read every pipeline module: `pipeline/video/detector.py`, `pipeline/audio/detector.py`,
   `pipeline/audio/audio_forensics.py`, `pipeline/fusion/fuser.py`,
   `pipeline/forensics/{ai_classifier,ensemble_classifier,face_crop,metadata_forensics,
   document_detector,ela,noise_residual,temporal_localization,localization}.py`.
3. Located the testing videos (see §3).
4. Ran the existing unit suite (`pytest tests/`): **228 passed, 5 failed** (all 5 failures are in
   `tests/unit/test_app.py` and fail only because `streamlit.session_state` is unavailable outside a
   `streamlit run` runtime — environmental, not a regression).
5. Ran a new black-box evaluation script `scripts/evaluate_testing_videos.py` over all 15 videos.
   It mirrors `main.run_pipeline` exactly (reusing one classifier instance), prints per-detector
   status at startup, and writes `reports/testing_videos_evaluation.json/.csv` and
   `reports/testing_videos_manifest.csv`.

---

## 3. Where the testing videos actually are

- **NOT** at `C:\Users\Hemapriya\Downloads\testing_videos` (does not exist).
- **FOUND** at `C:\Users\Hemapriya\Documents\ForenSight docs\testing_videos`
  - `ai/` → 11 videos (ground truth = **FAKE**, inferred from folder name only)
  - `real/` → 4 videos (ground truth = **REAL**, inferred from folder name only)
  - 2 loose images at top level (`photo_to_test_AI.webp`, `saree pic pongal celeb.png`) — excluded
    from the video evaluation and listed in the manifest as `type=image`, `ground_truth=UNKNOWN`.
- Ground truth was **never** inferred from content (per the audit rules). The `ai/` vs `real/`
  split is taken at face value.

---

## 4. Detector inventory (FAIL-LOUD)

| Detector | Status | Why |
|---|---|---|
| EfficientNetV2 (video) | **INACTIVE** | `config.video.weights=models/efficientnet_v2s.pt` but `models/` does not exist; `VideoDetector` never instantiates/loads it. Per-frame "texture" is a Laplacian-variance heuristic. |
| RawNet2 (audio) | **INACTIVE** | `config.audio.weights=models/rawnet2.pt` missing; `RawNet2Classifier` is a stub (`pipeline/audio/detector.py:203`) that returns 0.5. Audio scoring is heuristic only. |
| GradCAM (neural) | **INACTIVE** | `VideoDetector._GradCAM` runs `_stub_forward` because `self._model is None`. Not connected to any network. |
| Audio heuristic forensics | **ACTIVE (heuristic)** | spectral stability / phase incoherence / MFCC / flux / ZCR via `analyse_segment`. No network. |
| Video texture heuristic | **ACTIVE (heuristic)** | per-frame Laplacian variance. |
| Metadata / C2PA provenance | **ACTIVE** | EXIF/C2PA/generator-signature scan of the video container. Strongest FAKE signal. |
| Temporal localization | **ACTIVE (heuristic)** | frame-to-frame consistency. Display/explain layer. |
| Face cropper (MTCNN) | **ACTIVE** | `facenet_pytorch` face detection; gates the classifier. |
| Document detector | **ACTIVE (heuristic)** | colour/edge heuristic; gates the classifier. |
| ELA / Noise-floor / Spectral-FFT | **INACTIVE for video** | excluded from the video pipeline; only used on the image tab. |
| AI image classifier (ensemble) | **ACTIVE (but backwards on video)** | loaded from HuggingFace Hub at runtime. Validated reading AI→low, real→high on video. |

> Note on reporting hygiene: `main._load_ai_classifier` (`main.py:73`) logs
> `"Ensemble classifier failed to load"` **at construction time**, before the models are actually
> loaded (loading is lazy, on first `predict()`). So the startup warning is misleading — the
> classifier frequently *does* load a moment later. Status should be checked **after** `_ensure_loaded()`.

---

## 5. Per-modality deep dive

### 5.1 Video — `pipeline/video/detector.py`
- `VideoDetector.run(frames_dir, video_bytes, video_path)` → `ModalResult` (`detector.py:393`).
- The intended "deepfake" network **EfficientNetV2 is never constructed**; the "texture" signal is
  `image_forensics` Laplacian-variance (`_GradCAM._stub_forward`). There is no trained spatial model
  on the video path at all.
- Frame-level fusion (Sprint 9) gives each frame a multi-signal score
  `{texture, spectral, ela, noise, classifier}` where `spectral/ela/noise` are **disabled for video**
  frames, so in practice each video frame carries only `texture` (+ `classifier` when a face is
  detected). `frequency_score` in the eval is therefore `None` for every video.
- Thresholds: `_FAKE_THRESHOLD=0.6`, `_REAL_THRESHOLD=0.4`, `_GRADCAM_WEIGHT=0.6`,
  `_FREQ_WEIGHT=0.4`, `_STRONG_TEXTURE=0.5` (constants in `detector.py`).

### 5.2 Audio — `pipeline/audio/detector.py` + `audio_forensics.py`
- `AudioDetector.run(wav_path)` → `list[AudioSegmentResult]`; aggregated as a plain mean
  (`detector.py:170`).
- `RawNet2Classifier` (`detector.py:203`) is a stub. Real scoring is `analyse_segment` (spectral
  stability, phase incoherence, MFCC, spectral flux, ZCR).
- **Config mismatch:** `config.audio.segment_duration=2.0` but `audio_forensics._SEGMENT_DURATION_S=4.0`.
- Videos with no audio track default to `audio_score=0.5` (neutral), so they contribute nothing to
  a FAKE call — fine for FP control, useless for catching silent AI clips.

### 5.3 Image classifier — `pipeline/forensics/ensemble_classifier.py`
- Loads `prithivMLmods/deepfake-detector-model-v1` (weight 1.0) and
  `dima806/deepfake_vs_real_image_detection` (weight 0.8) from the Hub.
- Runs **only on detected face crops** (`face_crop.py`); frames with no detected face get no
  classifier score.
- **Backwards on video:** on the validation set, `ai_vid_without_audio.mp4` → classifier≈0.27 (reads
  AI as "real") and `real_vid_to_test_withaudio.mp4` → ≈0.78 (reads real as "AI"). `face_crop.py`
  documents this. The pipeline tries to compensate with a `classifier_flipped` heuristic, but that
  heuristic is fragile and silent.
- In this sandbox both models loaded (ensemble `available=True`); the backwards behaviour is real.

### 5.4 Fusion — `pipeline/fusion/fuser.py`
- Weighted average **video 0.6 / audio 0.4** → optional Platt/isotonic calibrator → verdict
  (`REAL` < 0.40 < `UNCERTAIN` < 0.60 < `FAKE`).
- Calibration: the calibrator is **not fitted** in this run (no training data referenced); scores
  are the raw weighted average. No leakage observed, but there is currently **no calibration at all**,
  so "score" is not a calibrated probability (per the rules it must not be called a probability).
- **Key weakness found:** a definitive AI manifest (`provenance_score=0.97`) is *averaged down* and
  can still land in `UNCERTAIN`. Example: `ai/girl walkin video.mp4` had `provenance_score=0.97` and
  `video_score=0.9` but fused to **0.587 → UNCERTAIN** (just under the 0.6 FAKE threshold). A
  high-confidence provenance hit should not be diluted by the weighted mean.

### 5.5 Forensic signals
- All real forensic techniques (ELA, noise-residual, metadata/C2PA) exist and are honest, but
  **ELA / noise / spectral-FFT are excluded from the video path** — only metadata/C2PA runs on video.
  So video benefits from none of the pixel-level forensic evidence.
- `localization.py` / `temporal_localization.py` are display/explain layers; they do not feed back
  into `fake_prob` (correct).

---

## 6. Thresholds & weights (current values, recorded — NOT changed)

| Parameter | Value | Source |
|---|---|---|
| video.weights | `models/efficientnet_v2s.pt` (missing) | `config.yaml` |
| audio.weights | `models/rawnet2.pt` (missing) | `config.yaml` |
| video.frame_rate | `1` (1 fps sampling) | `config.yaml` |
| face_confidence | `0.80` | `config.yaml` |
| fusion video/audio weight | `0.6 / 0.4` | `config.yaml` / `fuser.py` |
| fake / real threshold | `0.60 / 0.40` | `config.yaml` / `fuser.py` |
| ensemble models | `prithivMLmods/...` (1.0), `dima806/...` (0.8) | `config.yaml` |
| audio segment_duration | `2.0` (cfg) vs `4.0` (code) | mismatch |

---

## 7. Evaluation results (black-box, current pipeline)

15 videos labelled from folder structure. Verdict vs ground truth:

| File | GT | verdict | fusion | video | audio | clf | texture | meta |
|---|---|---|---|---|---|---|---|---|
| ai/ai_noaud2.mp4 | FAKE | **UNCERTAIN** (FN) | 0.551 | 0.551 | 0.500 | – | 0.970 | 0.55 |
| ai/ai_vid_without_audio.mp4 | FAKE | FAKE (TP) | 0.724 | 0.900 | 0.461 | 0.219 | 0.532 | 0.97 |
| ai/avengers ai vid 1.mp4 | FAKE | **REAL** (FN) | 0.321 | 0.450 | 0.127 | 0.623 | 0.140 | 0.55 |
| ai/avengers ai vid 2.mp4 | FAKE | **UNCERTAIN** (FN) | 0.464 | 0.590 | 0.274 | 0.409 | 0.945 | 0.55 |
| ai/crackers_withaud.mp4 | FAKE | FAKE (TP) | 0.636 | 0.900 | 0.240 | 0.331 | 0.418 | 0.97 |
| ai/gemini ai video.mp4 | FAKE | FAKE (TP) | 0.604 | 0.900 | 0.159 | – | 0.759 | 0.97 |
| ai/girl walkin video.mp4 | FAKE | **UNCERTAIN** (FN) | 0.587 | 0.900 | 0.118 | 0.742 | 0.776 | 0.97 |
| ai/girl_speaking video gemini.mp4 | FAKE | FAKE (TP) | 0.614 | 0.914 | 0.164 | 0.768 | 0.990 | 0.97 |
| ai/insta vid.mp4 | FAKE | **REAL** (FN) | 0.370 | 0.550 | 0.100 | 0.572 | 0.427 | 0.55 |
| ai/real vid ai aud.mp4 | FAKE | **REAL** (FN) | 0.369 | 0.568 | 0.071 | 0.367 | 0.175 | 0.55 |
| ai/south indian women speaking.mp4 | FAKE | FAKE (TP) | 0.674 | 0.900 | 0.335 | 0.248 | 0.995 | 0.97 |
| real/WhatsApp … 6.15.45 PM.mp4 | REAL | REAL (TN) | 0.338 | 0.508 | 0.083 | 0.484 | 0.451 | 0.55 |
| real/WhatsApp … 6.15.46 PM.mp4 | REAL | UNCERTAIN (TN) | 0.443 | 0.581 | 0.236 | 0.386 | 0.298 | 0.55 |
| real/real_vid_to_test_withaudio.mp4 | REAL | UNCERTAIN (TN) | 0.453 | 0.540 | 0.324 | – | 0.130 | 0.55 |
| real/vid_with_noaudio.mp4 | REAL | UNCERTAIN (TN) | 0.551 | 0.551 | 0.500 | – | 0.828 | 0.55 |

Metrics (`–` = classifier not run on that video, e.g. no face / no audio):

```
n_labeled = 15   (11 FAKE, 4 REAL)
confusion: TP=5  TN=4  FP=0  FN=6
accuracy        = 0.600
precision_fake  = 1.000   (no false positives)
recall_fake     = 0.455
FAKE_RECALL     = 0.455
f1_fake         = 0.625
false_negative_rate = 0.545
false_positive_rate = 0.000
fake_called_uncertain = 3
```

### 7.1 What this tells us
- **Every REAL video is correctly NOT flagged FAKE** (0 FP). FP control is strong.
- **6 of 11 AI videos are missed.** Of the 6 misses:
  - 5 have **no C2PA/AI manifest** (`meta=0.55`): `avengers1` (REAL), `avengers2` (UNC),
    `ai_noaud2` (UNC), `insta vid` (REAL), `real vid ai aud` (REAL). These rely entirely on the
    backwards classifier + noisy texture, which fail.
  - 1 (`girl walkin`) **does** have `meta=0.97` but was diluted to UNCERTAIN by fusion.
- The only AI videos reliably caught are the C2PA-signed ones (`ai_vid_without_audio`, `crackers`,
  `gemini`, `girl_speaking`, `south indian`). That is provenance doing all the work.

---

## 8. Correctness / data-flow risks found

1. **EfficientNetV2 and RawNet2 are configured but never loaded.** `models/` is absent. The
   pipeline silently runs heuristic-only. This violates the "no silent replacement with stub" rule
   — it is currently *acceptable* only because it is documented, but it must be reported as INACTIVE.
2. **Audio `segment_duration` mismatch** (config 2.0 vs code 4.0) — timestamps/coverage differ from
   intent.
3. **Misleading startup warning** for the classifier (`main.py:73` checks `available` before lazy
   load). Should check after `_ensure_loaded()`.
4. **Backwards classifier** is compensated by a silent flip heuristic. If the flip is wrong the
   system is confidently wrong; this should be logged loudly per-video.
5. **Provenance dilution in fusion.** A `provenance_score` ≥ ~0.9 (definitive AI manifest) is being
   averaged into UNCERTAIN. A high-confidence forensic hit should dominate, not be diluted.
6. **1 fps frame sampling** (`video.frame_rate=1`) may miss short localized edits; acceptable for
   this dataset but a known limitation for temporally sparse fakes.
7. **Classifier only on face crops.** Non-face AI videos (scenery, text, cartoons) get no classifier
   signal at all.
8. **No calibration fitted** — scores are raw weighted means, so calling them "probability" would be
   wrong. They are reported as `fake_prob`/`fused_score` in code; UI/consumers must treat them as
   uncalibrated.

---

## 9. Files & functions to fix (priority order) — recommendations, no code changed

| # | Location | Issue | Recommended direction (needs your approval) |
|---|---|---|---|
| P1 | `pipeline/fusion/fuser.py` | Provenance/definitive-manifest diluted to UNCERTAIN | Add a provenance override so `provenance_score ≥ 0.9` forces FAKE (or a high-floor). This directly cuts FN without hurting FP. |
| P2 | `pipeline/forensics/ensemble_classifier.py` + `face_crop.py` | Classifier reads backwards on video, flip is silent | Make the flip explicit + logged per video; reconsider whether the image classifier should contribute to video at all until validated forward. |
| P3 | `main.py:73` | Misleading "failed to load" warning before lazy load | Check `available` after `_ensure_loaded()`; report real status. |
| P4 | `config/config.yaml` + `audio_forensics.py` | `segment_duration` 2.0 vs 4.0 mismatch | Align config and code. |
| P5 | `pipeline/video/detector.py` | EfficientNetV2 never loaded (INACTIVE) | Either load the real checkpoint (and add `models/`) or formally document the video path as heuristic-only and stop referencing the weight file. |
| P6 | `pipeline/audio/detector.py:203` | RawNet2 stub | Same as P5 for audio. |
| P7 | `pipeline/video/detector.py` | ELA/noise/spectral excluded from video | Re-enable pixel-level forensics on video frames to add real signal beyond texture. |

---

## 10. Recommended fix order (to attack false negatives without touching thresholds yet)

1. **P1 (fusion provenance override)** — biggest, safest FN reduction; does not alter thresholds,
   only stops a definitive AI manifest from being averaged away.
2. **P2 (loud, validated classifier flip)** — stops the backwards classifier from *causing* FNs.
3. **P3/P4 (reporting + config hygiene)** — make INACTIVE status and weights honest.
4. **P7 (enable ELA/noise/spectral on video)** — adds discriminative signal for untagged AI.
5. **P5/P6 (real model loading or honest stub removal)** — only after the above, decide whether to
   invest in the configured neural nets or formally run heuristic-only.

---

## 11. Honest limitations of this audit

- I did **not** change any threshold, weight, or detector — only measured and reported.
- Ground truth comes solely from the `ai/` vs `real/` folder names; if any file is mis-filed, the
  metrics shift. The 2 loose images were excluded.
- The classifier was loaded live from HuggingFace Hub (network available here). On an offline
  machine it would be `INACTIVE` and FAKE_RECALL would drop further (provenance-only).
- "FAKE_RECALL" is defined as TP/(TP+FN) with UNCERTAIN counted as a miss for FAKE detection, which
  matches the stated goal of minimizing false negatives.
- 5 unit-test failures are environmental (`streamlit` runtime) and unrelated to this audit.

---

## 12. Artifacts produced

- `scripts/evaluate_testing_videos.py` — reproducible evaluator (mirrors `main.run_pipeline`).
- `reports/testing_videos_evaluation.json` — full per-frame / per-segment / per-video results + metrics.
- `reports/testing_videos_evaluation.csv` — per-video summary (the columns requested).
- `reports/testing_videos_manifest.csv` — filename, path, ground_truth, duration, fps, resolution,
  has_audio, codec, filesize_mb for every media file.
