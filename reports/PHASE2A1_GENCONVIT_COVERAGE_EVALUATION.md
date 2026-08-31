# Phase 2A.1 — GenConViT Face/Frame Coverage Evaluation

**Date:** 2026-08-28
**Scope:** Independent, controlled investigation of whether GenConViT's weak
Phase 2A results were caused by the detector itself or by **insufficient/poor
face-frame input coverage**.
**Status:** DONE. Coverage improved; GenConViT FAKE recall improved from 33.3% to
55.6%; the production verdict is **completely unchanged** (`use_in_final_fusion`
stayed `false`).

---

## 0. Question this phase tried to answer

> Is GenConViT genuinely weak on this corpus, or is its performance being
> artificially limited by insufficient/poor face-frame coverage?

**Short answer:** it was **both**. Better frame coverage clearly helped
(FAKE recall 33.3% → 55.6%, two missed fakes recovered), but it did not make
GenConViT strong: a real WhatsApp false positive persisted even after coverage
improved, and several fakes remain missed despite excellent coverage. It should
**remain a secondary, evaluation-only signal**.

---

## 1. What was audited (Task 1 — current face/frame pipeline, BEFORE changes)

| # | Question | Actual current behaviour (before this phase) |
|---|---|---|
| 1 | How are frames sampled? | `utils/demux._extract_frames` uses `ffmpeg -vf fps=1` → 1 frame/second JPEGs (`frame_%06d.jpg`). |
| 2 | How many frames requested? | One per second (duration in seconds); then GenConViT **sub-sampled** up to `num_frames: 15` evenly via `np.linspace`. |
| 3 | How is MTCNN called? | `FaceCropper.crop(frame)` → `_crop_mtcnn` → `MTCNN.detect(img, landmarks=True)`. |
| 4 | Detection confidence threshold? | `config.video.face_confidence = 0.80`, threaded into `FaceCropper(min_confidence=0.80)` and shared with GenConViT. |
| 5 | Face size/quality checked? | MTCNN `min_face_size=20`; geometry + bilateral-texture checks in `FaceCropper`. No explicit min crop size for the model input. |
| 6 | Blurry faces rejected? | Indirectly — the bilateral-texture-symmetry check rejects blank/flat patches; not a blur detector. |
| 7 | Multiple faces supported? | Yes — `keep_all=True`, largest-by-area face chosen. |
| 8 | Only the first face used? | No — the *largest* face (by box area) is chosen, not necessarily the first. |
| 9 | Frames discarded when MTCNN fails? | No — no-face frames are **skipped** for GenConViT (not sent to model), but still analysed by the texture heuristic. Not a stop condition. |
| 10 | Frame numbering/timestamps preserved? | Original frame indices preserved (`genconvit_frame_indices`); timestamps were **not** per-face-frame recorded. |
| 11 | Same frames as texture heuristic? | Same *demuxed frames*, but GenConViT sub-sampled to 15 then face-filtered; texture uses all frames. |
| 12 | Preprocessing matches Phase 2A? | Yes — negligible change (see §6). |
| 13 | Lower-confidence-but-valid face usable? | No — any candidate below 0.80 was dropped by `FaceCropper`; there was no second tier. |
| 14 | No-face frame stops detector? | No — it is skipped. |

### The core coverage limitation found (the bug this phase fixes)

GenConViT's `infer()` did:

```
n = len(frames)
count = min(num_frames=15, n)
indices = np.linspace(0, n-1, count)   # sample 15 evenly FIRST
crops   = MTCNN over ONLY those 15
usable  = crops that had a face
```

So for a video longer than ~15s, face detection ran on **only 15 fixed
positions**. If those 15 positions happened to miss most face-bearing frames, a
clip could end up scored off **1–3 face crops even though valid faces existed in
the other, unexamined frames**. This explains the Phase 2A single-frame scores
and the "0 frames" rows.

(This is a *coverage* problem, not a *model quality* problem — see §7.)

---

## 2. What was changed (Tasks 2–6) and why

**Scope isolation (STRICT RULES honoured):**
- `config.genconvit.enabled` stayed `true`; `use_in_final_fusion` stayed **`false`**.
- No fusion weights, thresholds, calibration, provenance, or REAL/FAKE/UNCERTAIN
  logic changed. `fake_prob`/final verdict untouched (verified identical, §5).
- No training / fine-tuning / checkpoint modification. No threshold tuning to the
  testing set. No new detector introduced. Aggregation label-mapping unchanged
  (index 0 = FAKE, 1 = REAL, mean of `sigmoid(logits[:,0])`).

### 2.1 `pipeline/genconvit/detector.py` — the coverage fix

Replaced the "sample 15 first, then keep whatever had a face" logic with:

1. **Scan the whole timeline** — `_collect_face_candidates` runs the shared
   `FaceCropper` over **every** demuxed frame (the same 1-fps frames the texture
   heuristic sees) and collects every frame whose face passes the **same, unchanged
   quality rules** (MTCNN `min_confidence=0.80` + geometry + bilateral-texture
   symmetry). We never lower those bars.
2. **Then sample evenly among the face-bearing frames** — up to `num_frames` (15)
   crops are chosen evenly by original frame index, preserving temporal spread and
   keeping the model's inference budget identical to upstream (it still sees ≤15
   crops, just from a larger, face-bearing pool).
3. **Sanity guard** — a crop whose smaller axis is < 24px is not upscaled into the
   224×224 input (would feed the network a blurry smear). This is a guard on the
   resized crop, not a change to face detection.
4. **Coverage metrics** emitted per video (see §4).
5. **Coverage status label** (`genconvit_coverage_status`, with
   `genconvit_face_quality` retained as an alias):
   `NO_FACE_DETECTED / LOW_FACE_COVERAGE / PARTIAL_FACE_COVERAGE /
   GOOD_FACE_COVERAGE` (engineering diagnostics, not validated science;
   boundaries: 0 / <25% / <60% / ≥60% of requested frames).
6. **Temporal preservation** — per-face-frame `frame_index`, `timestamp`, face
   `box` size, detection `confidence`, and the GenConViT score are recorded in
   the output (`genconvit_frame_indices`, `genconvit_timestamps`,
   `genconvit_face_sizes`, `genconvit_face_confidences`, `genconvit_frame_scores`).

### 2.2 `pipeline/video/detector.py` — metadata passthrough only

The GenConViT metadata block now copies the new coverage fields through to
`ModalResult.metadata`, including `genconvit_coverage_status` and the per-frame
`genconvit_face_confidences`. It is display/report data only; nothing touches
`overall`/`fake_prob`.

### 2.3 `scripts/evaluate_testing_videos.py` — reporting

Records the new per-video coverage fields in JSON/CSV, adds a corpus-level
`coverage` summary to `metrics.genconvit`, and keeps the GenConViT confusion
matrix. Includes the new coverage columns in the CSV.

---

## 3. Verification of preprocessing & aggregation (Tasks 6 & 7)

- **Preprocessing (unchanged, verified correct):** face crop → `224×224`
  `INTER_AREA` → `/255` → transpose `HxWx3`→`3xHxW` → ImageNet normalize
  (mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`) → batch `[N,3,224,224]`
  → ED and VAE branches both run and their sigmoid outputs are averaged. Channel
  order/batch dims validated. **No preprocessing bug found; nothing altered.**
- **Aggregation (unchanged):** aggregated fake score = mean over frames of
  `sigmoid(logits[:,0])`; overall label = `argmax(mean 2-vector)` with
  index 0→FAKE, 1→REAL. This matches the verified upstream orientation. **No
  aggregation bug found; left unchanged.**

---

## 4. Coverage metrics now recorded (Tasks 3, 4, 5)

Per video (all in `reports/testing_videos_evaluation.json`):

```
genconvit_frames_requested
genconvit_frames_sampled          (= frames_requested; we scan the whole timeline)
genconvit_face_frames_found
genconvit_face_coverage           = face_frames_found / frames_requested
genconvit_average_face_confidence
genconvit_min_face_confidence
genconvit_max_face_confidence
genconvit_frames_used
genconvit_face_quality            (NO_FACE_DETECTED|LOW|PARTIAL|GOOD)
genconvit_coverage_status         (same label; standardized name)
genconvit_frame_indices           (per frame)
genconvit_timestamps              (per frame, seconds)
genconvit_face_sizes              (per frame, pre-resize box [w,h])
genconvit_face_confidences        (per frame)
genconvit_score / _fake_score / _predicted_label
```

If a value cannot be produced it is `null` (e.g. confidence is null on the
OpenCV-Haar degraded backend, which gives no scores). Nothing is fabricated.

---

## 5. Results (Tasks 8–10): BEFORE vs AFTER

### A. Input-coverage comparison (per video)

| video | GT | frames req | faces found BEFORE | faces found AFTER | cov BEFORE | cov AFTER | GC score BEFORE | GC score AFTER | label BEFORE | label AFTER | prod verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WhatsApp …6.15.45 | REAL | 58 | 1 | 2 | 2% | 3% | 0.378 | 0.463 | FAKE | FAKE | REAL |
| WhatsApp …6.15.46 | REAL | 13 | 12 | 12 | 92% | 92% | 0.548 | 0.546 | REAL | REAL | REAL |
| ai_noaud2 | FAKE | 10 | 0 | 0 | 0% | 0% | — | — | — | — | UNCERTAIN |
| ai_vid_without_audio | FAKE | 10 | 1 | 1 | 10% | 10% | 0.607 | 0.602 | REAL | REAL | FAKE |
| avengers ai vid 1 | FAKE | 46 | 12 | **35** | 26% | **76%** | 0.572 | 0.544 | REAL | REAL | REAL |
| avengers ai vid 2 | FAKE | 38 | 14 | **36** | 37% | **95%** | 0.556 | 0.484 | REAL | **FAKE** | UNCERTAIN |
| crackers_withaud | FAKE | 10 | 1 | 1 | 10% | 10% | 0.327 | 0.312 | FAKE | FAKE | FAKE |
| gemini ai video | FAKE | 10 | 0 | 0 | 0% | 0% | — | — | — | — | FAKE |
| girl walkin | FAKE | 10 | 4 | 4 | 40% | 40% | 0.624 | 0.625 | REAL | REAL | FAKE |
| girl_speaking gemini | FAKE | 10 | 4 | 4 | 40% | 40% | 0.731 | 0.729 | REAL | REAL | FAKE |
| insta vid | FAKE | 41 | 3 | **11** | 7% | **27%** | 0.529 | 0.492 | REAL | **FAKE** | REAL |
| real vid ai aud | FAKE | 102 | 3 | **20** | 3% | **20%** | 0.287 | 0.497 | FAKE | FAKE | REAL |
| real_vid_to_test_withaudio | REAL | 21 | 0 | 0 | 0% | 0% | — | — | — | — | UNCERTAIN |
| south indian women | FAKE | 10 | 2 | 2 | 20% | 20% | 0.509 | 0.499 | FAKE | FAKE | FAKE |
| vid_with_noaudio | REAL | 9 | 0 | 0 | 0% | 0% | — | — | — | — | UNCERTAIN |

> Scores in this table are from the final, shipped eval run
> (`reports/testing_videos_evaluation.json`). Because the official VAE branch
> reparameterizes with `torch.randn_like` at inference (see §6.1), re-runs move
> a given score by ~±0.005 absolute; none of the labels or metrics in this phase
> changed across runs.

Notes:
- BEFORE "faces found" = Phase 2A usable face frames; denominator for both = total
  demuxed frames.
- Short clips (≤13 frames) already had all frames scanned, so their coverage is
  unchanged — those reflect **genuine MTCNN difficulty** on compressed talking-head
  footage, not a sub-sampling artefact.
- Longer clips (avengers 1/2, insta, real vid ai aud) gained the most — these are
  exactly the videos the old 15-frame sub-sample was undersampling.

### B. GenConViT metric comparison

On the 11 videos GenConViT scored (9 FAKE / 2 REAL) — same scored set both times:

| metric | BEFORE (Phase 2A) | AFTER (2A.1) |
|---|---|---|
| Videos scored | 11 | 11 |
| **FAKE recall** | **0.333** (3/9) | **0.556** (5/9) |
| Precision (FAKE) | 0.750 | **0.833** |
| F1 (FAKE) | 0.462 | **0.667** |
| False-positive rate | 0.500 | 0.500 (unchanged) |
| Real-rejection rate | 0.500 | 0.500 |
| Confusion | TP3 / TN1 / FP1 / FN6 | **TP5** / TN1 / FP1 / **FN4** |
| Fake verdict rate | 0.364 | 0.545 |

### C. Coverage analysis (corpus level)

- Videos with **0 usable face frames**: 4 (`ai_noaud2`, `gemini`, `real_vid_to_test`,
  `vid_with_noaudio`) — unchanged; these are genuinely no-face (e.g. a hand) or the
  classifier could not detect a face at all.
- Videos with **1** face frame: 2 (`ai_vid_without_audio`, `crackers`).
- Videos with **multiple**: 9.
- **Average face coverage: 28.9%** (was ~20% by the old usable-frames measure).
- **Average face confidence: 0.958** — so faces that ARE found are high-confidence;
  the issue is coverage, not confidence quality.
- **Largest coverage improvements:** `avengers ai vid 2` (37%→95%), `avengers ai
  vid 1` (26%→76%), `insta vid` (7%→27%), `real vid ai aud` (3%→20%).

### D. False-negative analysis (the five focus fakes)

| video | frames sampled | faces found | finally used | coverage improved? | GenConViT result | changed? | still FN? |
|---|---|---|---|---|---|---|---|
| ai_noaud2 | 10 | 0 | 0 | no (none to find; it's a hand, no face) | no score | no | yes (no face → cannot be judged by GenConViT) |
| avengers ai vid 1 | 46 | 35 | 15 | **yes** (26%→76%) | REAL (0.544, moved DOWN from 0.572 toward FAKE) | no | **yes** — even with excellent coverage GenConViT says REAL |
| avengers ai vid 2 | 38 | 36 | 15 | **yes** (37%→95%) | **FAKE** (0.484) | **yes** — REAL→FAKE | **no — recovered (TP)** |
| insta vid | 41 | 11 | 11 | **yes** (7%→27%) | **FAKE** (0.492) | **yes** — REAL→FAKE | **no — recovered (TP)** |
| real vid ai aud | 102 | 20 | 15 | **yes** (3%→20%) | FAKE (0.287→0.497, still FAKE) | no (still FAKE) | **no** (was already TP) |

Key insight: `avengers ai vid 2` and `insta vid` are strong evidence that **poor
coverage was suppressing two true detections** — once GenConViT saw more real
faces it read them as FAKE correctly. Conversely, `avengers ai vid 1` shows that
coverage is *not* the whole story: with 35 faces available it still reads REAL,
so that miss is a **genuine GenConViT limitation**, not a coverage artefact.

### E. False-positive analysis

| video | GT | faces found B→A | score B→A | label | conclusion |
|---|---|---|---|---|---|
| WhatsApp …6.15.45 PM | REAL | 1 → 2 | 0.378 → 0.463 | FAKE → FAKE | **FP persists.** More coverage did NOT fix it — GenConViT's false positive on this real, compressed WhatsApp re-encode is a genuine classifier error, not a single-frame fluke. |

This is important: it confirms that simply feeding GenConViT more faces does not
eliminate its real-corpus error tendency on compressed footage.

---

## 6. Engineering improvement vs detection improvement (Task 10 — kept separate)

**Engineering (input coverage):**
> Face coverage on long clips improved substantially: `avengers2` 37%→95%,
> `avengers1` 26%→76%, `insta` 7%→27%, `real vid ai aud` 3%→20%; corpus average
> face coverage ~19%→**28.9%**, with 9/15 videos now multi-face.

**Detection (GenConViT standalone):**
> FAKE recall **0.333 → 0.556** (3/9 → 5/9), precision 0.75 → 0.833, F1 0.46 →
> 0.667. FPR unchanged at **0.50**.

We do **not** claim GenConViT is "strong" merely because more faces were available.
We claim: **two false negatives were genuinely coverage-limited and recovered**,
but **one real-corpus false positive persisted and several misses remain despite
good coverage**, so GenConViT is still a secondary signal.

### 6.1 Reproducibility note (VAE reparameterization)

The official GenConViT VAE branch uses the VAE reparameterization trick at
**inference** (`torch.randn_like(std)` in `_VAEEncoder.reparameterize`; verified
against upstream `model/genconvit_vae.py`). The official
`pred_vid → max_prediction_value` path averages this stochastic branch with the
deterministic ED branch along the batch axis, exactly as this implementation does.
Consequence: two runs of the same video agree on **labels and metrics** but the raw
`genconvit_score` can differ by ~±0.005 absolute (observed), because the VAE
samples a fresh latent per forward. This is upstream behavior — not a bug in this
reconstruction — and the combined ED+VAE mean keeps the variance small enough
that no label in this corpus changed across two independent evaluation runs.
Identified for transparency; **no code change was made** (aggregation is verified
correct per Task 8).

---

## 7. Honest conclusion

- **Coverage improved** and demonstrably recovered two real detections.
- **GenConViT still remains weak/mixed on this corpus**: FAKE recall 55.6% is
  still only marginally ahead of the texture heuristic (54.5%), and it still
  false-positives a real compressed WhatsApp clip (FPR 0.5).
- The label is therefore **B** (see §8). The main remaining issue is no longer
  purely input coverage for the recovered clips, but GenConViT's genuine
  behaviour on compressed/real footage plus true no-face videos.

---

## 8. Final recommendation (Task: choose exactly one)

**B. GenConViT is useful but should remain a secondary signal.**

Evidence based:
- It is a real, free, locally-run neural detector that now uniquely/explicitly
  flags `real vid ai aud`, `avengers ai vid 2`, `insta vid`, `crackers`, and
  `south indian` — genuinely complementary to texture/provenance.
- But FAKE recall (55.6%) ≈ texture (54.5%), FPR = 0.5 (persistent WhatsApp FP),
  and it cannot score the 4 no-face videos.
- Coverage is no longer the binding constraint for the recovered clips, so further
  controlled coverage work has low additional value for these; GenConViT's own
  reliability (esp. on compressed REAL footage) is now the limit.

Per the strict rules, **no fusion, threshold, calibration, AASIST, or new-detector
work was done** in this phase. The next decision should be made after reviewing
this report and the updated `reports/testing_videos_evaluation.json`/`.csv`.

---

## 9. Code changes (list)

| File | What changed | Why |
|---|---|---|
| `pipeline/genconvit/detector.py` | Coverage fix: scan whole timeline for faces (added `_collect_face_candidates`), sample evenly among face-bearing frames up to `num_frames`; added `_coverage_status` + `_MIN_CROP_MINOR_DIM` guard; emit per-video coverage metrics (requested/sampled/found/coverage/min/avg/max confidence, frame indices, timestamps, face sizes, per-face confidences, `genconvit_face_quality` + `genconvit_coverage_status`) from `infer()`. Deterministic ENUM paths only — score path untouched. | Core Phase 2A.1 improvement — maximize valid face-frame coverage without lowering face-quality bars. |
| `pipeline/video/detector.py` | Metadata passthrough of the new GenConViT coverage fields into `ModalResult.metadata` (incl. `genconvit_coverage_status`, `genconvit_face_confidences`). | Surface coverage diagnostics for evaluation/UI; report-only, verdict untouched. |
| `scripts/evaluate_testing_videos.py` | Record new coverage fields per video (JSON/CSV); add `compute_coverage_summary` corpus stats under `metrics.genconvit.coverage`; add new CSV columns. | Provide the machine-readable coverage data required by the phase. |
| `tests/unit/test_genconvit_detector.py` | New: boundary tests for `_coverage_status`, the INACTIVE (missing-weights) path, `_resize_to_224`, and `_preprocess` normalization/channel order. | Lock in the Phase 2A.1 diagnostic behavior so regressions are caught. |

No changes to `config/config.yaml` (genconvit section untouched:
`enabled: true`, `use_in_final_fusion: false`), fusion, thresholds, provenance,
audio, image classifier, texture heuristic, or `face_crop.py`.

---

## 10. Tests (Task 14)

`py -m pytest tests/ -q` → **234 passed, 5 failed** (228 baseline + 6 new
GenConViT-detector tests). The 5 failures are the pre-existing environmental
`streamlit.session_state` errors in `tests/unit/test_app.py` (identical to the
documented baseline — unrelated to this phase). Regression scenarios implicitly
covered by the full corpus evaluation + unit tests:
- normal video with face — yes (`WhatsApp …6.15.46`, `avengers*`, etc.)
- video with no audio — yes (`ai_vid_without_audio`, `vid_with_noaudio`, `gemini`)
- video with no detectable face — yes (`ai_noaud2`, `gemini`, `vid_with_noaudio`)
- compressed video — yes (`WhatsApp …6.15.45`, the persistent FP)
- multiple-face video — yes (`avengers ai vid 2`, 36 faces)
- existing image-analysis path — unchanged (not modified; covered by test suite)
- existing texture heuristic — unchanged (not modified)
- **production final verdict unchanged with `use_in_final_fusion=false`** — proven
  by before/after metrics being byte-identical (TP6/TN4/FP0/FN5, FAKE_RECALL 0.545)
- **GenConViT unavailable/inactive behavior** — unit-tested: missing weights →
  `status=INACTIVE`, `score=None`, explicit `load_error`, `frames_used=0`, with
  **no heuristic fallback** (`genconvit_score` stays `None`).

## 11. Artifacts produced / updated

- `reports/PHASE2A1_GENCONVIT_COVERAGE_EVALUATION.md` (this report)
- `reports/testing_videos_evaluation.json` / `.csv` / `testing_videos_manifest.csv`
  (regenerated with new coverage fields incl. `genconvit_coverage_status` and
  per-frame `genconvit_face_confidences`)
- `reports/PHASE2A1_BEFORE_baseline.json` / `.csv` (frozen copy of the Phase 2A
  baseline for the before/after comparison; the original Phase 2A report and the
  `PHASE2A_GENCONVIT_REPORT.md` were NOT overwritten)
- `tests/unit/test_genconvit_detector.py` (new — 6 tests)
