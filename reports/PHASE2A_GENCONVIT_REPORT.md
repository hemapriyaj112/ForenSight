# Phase 2A — GenConViT Integration Report

**Date:** 2026-08-27
**Goal:** Replace the nonexistent `EfficientNetV2` video detector with a **real, free,
locally-runnable pretrained** video deepfake detector: GenConViT.
**Status:** DONE — ED + VAE both load and run; score surfaced separately; **not fused**
into the production verdict yet (per plan).
**Source of truth:** `pipeline/genconvit/`, `scripts/download_genconvit_weights.py`,
`reports/GENCONVIT_EVALUATION.md`, `reports/testing_videos_evaluation.json`.

---

## 1. Source, license, weights

- **Source:** https://github.com/erprogs/GenConViT — MIT license (permissive).
- **Weights:** https://huggingface.co/Deressa/GenConViT (official release)
  - `genconvit_ed_inference.pth` — 238,174,949 bytes (complete)
  - `genconvit_vae_inference.pth` — 2,782,175,601 bytes (complete)
- Stored under `models/genconvit/`. Downloaded via the reproducible
  `scripts/download_genconvit_weights.py` (exact-size skip/verify; `.part` resume).

## 2. What was built

- `scripts/download_genconvit_weights.py` — reproducible downloader (no download at
  app startup; user runs it once).
- `pipeline/genconvit/__init__.py`, `model.py`, `detector.py` — faithful
  reconstruction of the GenConViT architecture so the official weights load and run
  under the project's installed stack (timm 1.0.x, torch 2.11 cpu).
- `GenConViTDetector` — lazy-loading wrapper reporting `genconvit_status`
  `ACTIVE_MODEL | INACTIVE` with an **exact reason** when inactive. **No silent
  fallback** to the texture heuristic.
- Wired into `VideoDetector` (config-gated), status surfaced via
  `main.get_detector_status`, and the eval script.

## 3. Architecture reconstruction (verified from the checkpoints)

The two released checkpoints use **different** module prefixes, both faithfully
rebuilt:

- **ED** (`GenConViTED`): convnext_tiny feature extractor (own conv `stem`, dims
  96→192→384→768, intact ImageNet `head.fc` = 1000 classes) run on both the original
  and its autoencoder reconstruction; concat (2000) → `fc`(2000→500) → `fc2`(500→2).
  Prefixes `backbone.*`.
- **VAE** (`GenConViTVAE`): a real variational autoencoder
  (`encoder` stride-2 convs + `mu`/`var` latent 12544 → `decoder` 4× deconv → 112×112
  reconstruction) plus `convnext_backbone` features on [original, reconstruction];
  concat (2000) → `fc`(2000→500) → `fc2`(500→2). Prefixes `convnext_backbone.*`.
  Includes an unused `fc3`(1000→500) present in the checkpoint but not on the active
  path.
- Dead weights in the archives — `embedder.*` (swin_tiny) and
  `backbone.patch_embed.*` / `convnext_backbone.patch_embed.*` (an upstream
  HybridEmbed that does not match the released active graph or run on timm 1.0) —
  load as `unexpected` keys under `strict=False` and are ignored. All **active-path**
  keys load with **zero missing keys**.

**Label mapping (verified from upstream `model/pred_func.py`):** `real_or_fake(pred)`
maps output index **0 → FAKE, index 1 → REAL**. Aggregated fake score = mean over
frames of `sigmoid(logits[:,0])`; overall label = `argmax(mean sigmoid 2-vector)`.

## 4. Preprocessing & frame strategy

- Face crop via the project's existing MTCNN `FaceCropper` (substituted for upstream's
  dlib/face-recognition which is impractical on Windows) → 224×224 INTER_AREA →
  `/255` → ImageNet Normalize (mean [0.485,0.456,0.406], std [0.229,0.224,0.225]).
- Frames come from the existing demux frame-extraction path (`utils.demux`), sampled
  evenly (default `num_frames: 15`) — exactly the frames the texture heuristic uses.
- Videos with **no face** in the sampled frames produce `genconvit_score = None` with
  an explicit note (never a texture fallback).

## 5. Runtime & hardware

- **Device:** CPU (torch 2.11.0+cpu; no CUDA on this machine).
- Both branches load in ~seconds and run through a 224×224 batch of face crops.
- Full 15-video evaluation completes (~15–20 min for the corpus).

## 6. Integration points

- `config/config.yaml` → `genconvit:` section: `enabled: true`,
  `use_in_final_fusion: false`, `weights_dir`, `num_frames: 15`, `use_vae: true`,
  `fp16: true`.
- `pipeline/video/detector.py` → `VideoDetector` lazily builds the detector when
  `genconvit.enabled`; runs it on the same frames in `detect()`; results stored in
  `metadata` only (`genconvit_status`, `genconvit_score`, `genconvit_fake_score`,
  `genconvit_predicted_label`, `genconvit_aggregation_method`, `genconvit_frame_scores`,
  `genconvit_frames_used`, `genconvit_branches`, `genconvit_load_error`,
  `genconvit_note`). **`fake_prob` (verdict) is untouched.**
- `main.py` → `get_detector_status` reports `ACTIVE_MODEL` with branches & device, or
  `INACTIVE` with the exact reason.
- `scripts/evaluate_testing_videos.py` → records GenConViT per video, plus a standalone
  confusion matrix and a GenConViT-vs-texture disagreement table.

## 7. Per-video results (GenConViT), 2026-08-27

| video | GT | gc_score | gc_label | frames | texture | verdict |
|---|---|---|---|---|---|---|
| ai_noaud2.mp4 | FAKE | — | — | 0 | 0.970 | UNCERTAIN |
| ai_vid_without_audio.mp4 | FAKE | 0.607 | REAL | 1 | 0.532 | FAKE |
| avengers ai vid 1.mp4 | FAKE | 0.572 | REAL | 4 | 0.140 | REAL |
| avengers ai vid 2.mp4 | FAKE | 0.556 | REAL | ? | 0.945 | UNCERTAIN |
| crackers_withaud.mp4 | FAKE | 0.327 | FAKE | 1 | 0.418 | FAKE |
| gemini ai video.mp4 | FAKE | — | — | 0 | 0.759 | FAKE |
| girl walkin video.mp4 | FAKE | 0.624 | REAL | ? | 0.776 | FAKE |
| girl_speaking video gemini.mp4 | FAKE | 0.731 | REAL | ? | 0.990 | FAKE |
| insta vid.mp4 | FAKE | 0.529 | REAL | ? | 0.427 | REAL |
| real vid ai aud.mp4 | FAKE | 0.287 | **FAKE** | ? | 0.175 | REAL |
| south indian women speaking.mp4 | FAKE | 0.509 | **FAKE** | ? | 0.995 | FAKE |
| WhatsApp …6.15.45 PM.mp4 | REAL | 0.378 | FAKE | 1 | 0.451 | REAL |
| WhatsApp …6.15.46 PM.mp4 | REAL | 0.548 | REAL | ? | 0.298 | REAL |
| real_vid_to_test_withaudio.mp4 | REAL | — | — | 0 | 0.130 | UNCERTAIN |
| vid_with_noaudio.mp4 | REAL | — | — | 0 | 0.828 | UNCERTAIN |

`?` = used the multi-frame path (frames ≥ 1, exact count in JSON).
`—` = no face crop in sampled frames → GenConViT produced no score.

## 8. Metrics (GenConViT only)

FAKE recall **0.333** (3/9), precision 0.75, F1 0.46, false-positive rate 0.5,
real-rejection rate 0.5. Production verdict baseline unchanged (FAKE recall 0.545).

## 9. Findings & recommendation

**Recommendation: KEEP AS A SECONDARY SIGNAL** (do not fuse yet).

- GenConViT is now a genuinely-loaded neural detector (previously the video tab had
  none). It complements the texture heuristic: it uniquely flags
  `real vid ai aud.mp4`, and reinforces the provenance-override FAKE verdicts on
  `south indian women speaking.mp4`.
- It is **not stronger than texture standalone** here (recall 0.333 < 0.545) and it
  false-positives a REAL compressed WhatsApp video.
- Several scores came from a **single face frame** (MTCNN found no face in other
  sampled frames of compressed / talking-head AI video). 1-frame predictions are
  low-confidence and likely explain the REAL-corpus error.

**Path forward (before any fusion):** widen face coverage (respect
`face_confidence`/crop quality rather than requiring a detected face per sampled
frame), measure on a larger corpus, and re-evaluate before setting
`genconvit.use_in_final_fusion: true`. Until then the signal is reported but does not
influence the verdict — an honest, reversible addition.

## 10. Files changed / added

- `scripts/download_genconvit_weights.py` (new)
- `pipeline/genconvit/__init__.py`, `pipeline/genconvit/model.py`,
  `pipeline/genconvit/detector.py` (new)
- `pipeline/video/detector.py` (GenConViT wiring, metadata only)
- `main.py` (`get_detector_status`)
- `config/config.yaml` (`genconvit:` section)
- `scripts/evaluate_testing_videos.py` (GenConViT fields + metrics + disagreements)
- `reports/GENCONVIT_EVALUATION.md` (new), `reports/PHASE2A_GENCONVIT_REPORT.md` (this)
- `models/genconvit/` (downloaded weights; not committed)
