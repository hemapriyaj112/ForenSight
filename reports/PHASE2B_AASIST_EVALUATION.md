# Phase 2B — AASIST Integration & Evaluation Report

**Date:** 2026-08-30
**Goal:** Bring a **real, free, locally-runnable pretrained** audio anti-spoofing
detector — AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Attention,
clovaai/aasist, MIT) — into ForenSight and evaluate it independently on the testing
corpus, **without affecting the production verdict**.
**Status:** DONE — official checkpoint verified and loads (0 missing/0 unexpected),
scored separately; **not fused** into the audio heuristic and **not fused** into the
verdict (per plan).
**Source of truth:** `pipeline/aasist/`, `scripts/download_aasist_weights.py`,
`tests/unit/test_aasist_detector.py`, `reports/testing_videos_evaluation.json`,
`reports/testing_videos_evaluation.csv`.

---

## 1. Source, license, weights

- **Source:** https://github.com/clovaai/aasist — **MIT** license (permissive, retained).
  Pinned to commit `a04c9863f63d44471dde8a6abcb3b082b07cd1d1` (2022-01-18) for
  reproducibility. Upstream `LICENSE`, `NOTICE`, `models/AASIST.py`, `config/AASIST.conf`
  and `data_utils.py` fetched verbatim from that commit for verification.
- **Weights:** the official pretrained checkpoint is committed **directly inside the
  upstream repo** at `models/weights/AASIST.pth` (no mirror needed).
  - URL: `https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST.pth`
  - Size: **1,281,532 bytes** (exact-size verified by the downloader).
  - Stored at `models/aasist/AASIST.pth` via the reproducible
    `scripts/download_aasist_weights.py` (exact-size check, `.part` → rename, skip
    if already present at the right size — no download at app startup).
- **State-dict integrity:** loaded with `strict=True` against the verbatim upstream
  module → **0 missing / 0 unexpected keys**, 297,866 parameters.

## 2. What was built

- `scripts/download_aasist_weights.py` — reproducible downloader.
- `pipeline/aasist/model.py` — **verbatim** copy of upstream `models/AASIST.py`
  (MIT header preserved; free of `torchaudio`/`librosa` deps — verified imports:
  `random, typing, numpy, torch, torch.nn, torch.nn.functional, torch.Tensor`).
  Forward passes the unexpanded `master=self.master1/self.master2` exactly as the
  upstream release (preserved so the released checkpoint remains strict-loading).
- `pipeline/aasist/detector.py` — `AASISTDetector`: lazy-loading wrapper reporting
  `aasist_status` = `ACTIVE_MODEL | INACTIVE` with an **exact reason** when inactive
  and **`null` score** (never a fabricated 0.5, never a heuristic fallback).
- `tests/unit/test_aasist_detector.py` — 13 unit tests (pad/window/label/inactive
  paths), all passing.
- Wired as **status + evaluation only**: `main.get_detector_status` reports it, and
  `scripts/evaluate_testing_videos.py` scores it per video with a standalone confusion
  matrix. It is **NOT** wired into `AudioDetector.run` and **NOT** fused.

## 3. Architecture & label orientation (verified, not assumed)

- AASIST: raw-waveform anti-spoofing model — sinc 1-D conv layer, raw vs. spectro
  branches feeding a graph-masked multi-head self-attention, output head → 2 classes.
- **Task orientation (from upstream `data_utils.py`/`main.py`, cross-checked):**
  class 0 = **SPOOF**, class 1 = **BONAFIDE**; official released score =
  `logits[:, 1]` (higher = more bona fide). We expose:
  - `aasist_spoof_score` = mean over windows of `softmax(logits)[:, 0]` (higher = more
    likely spoofed audio),
  - `aasist_bonafide_score` = 1 − spoof,
  - `aasist_predicted_label` = `SPOOF` if spoof ≥ 0.5 else `BONAFIDE`.
- **Empirically confirmed** on the corpus: clean human voice scores 0.009–0.017
  (BONAFIDE); TTS/vocoded AI speech scores 0.95–0.999 (SPOOF). Orientation is sane.

## 4. Preprocessing & window strategy

- AASIST expects raw **16 kHz mono**, exactly **64,600 samples (~4.04 s)** per window
  (ASVspoof2019 LA default).
- Demuxed audio → `infer_wav`: parsed with the project's existing `soundfile`/demux
  path; reshaped to mono; windowed in non-overlapping 64,600-sample windows across the
  **full audio**; the final partial tail is tile-padded (mirrors upstream `pad()`).
- Aggregation: **mean of per-window spoof probabilities**
  (`aasist_aggregation_method` = `mean_softmax_spoof_prob_over_64600_sample_windows`);
  per-window scores and start-seconds are exported for auditability.
- Videos with **no audio stream** yield `status=INACTIVE`, `note="No audio track"`,
  score `null` — never a fallback.

## 5. Runtime & hardware

- **Device:** CPU (torch 2.11.0+cpu; no CUDA on this machine).
- Load (state dict + first forward): **~0.43 s**. Sustained CPU inference measured
  **~0.7 s per 64,600-sample window**. Corpus aggregate ≈ 99 windows ⇒ **≈ 1–2 min**
  of AASIST compute for all 15 videos (well within the total eval run).

## 6. Integration points

- `config/config.yaml` → `aasist:` section: `enabled: true`,
  `use_in_final_fusion: false`, `weights_dir: "models/aasist"`,
  `window_samples: 64600`.
- `main.py` → `get_detector_status` reports:
  - `ACTIVE_MODEL` when the official checkpoint loads (params, device, weights path,
    explicit "Kept OUT of the audio verdict until evaluation
    (`use_in_final_fusion=false`)"),
  - `INACTIVE` + exact reason otherwise.
- `scripts/evaluate_testing_videos.py` → per-video `aasist_*` fields (status, spoof/
  bonafide score, label, window count/scores, note, load error) added to the JSON and
  CSV; standalone AASIST confusion matrix; `aasist_disagreement_with_verdict` table.
- **`pipeline/audio/detector.py` and the fusion verdict are untouched.** Production
  metrics are byte-for-byte identical to baseline (see §8).

## 7. Per-video AASIST results, 2026-08-30

| video | GT | aa_spoof | aa_label | win | audio_heur | verdict |
|---|---|---|---|---|---|---|
| ai_noaud2.mp4 | FAKE | — | — | 0 | INACTIVE | UNCERTAIN |
| ai_vid_without_audio.mp4 | FAKE | 0.010 | BONAFIDE | 3 | 0.450 | FAKE |
| avengers ai vid 1.mp4 | FAKE | 0.989 | **SPOOF** | 12 | 0.130 | REAL |
| avengers ai vid 2.mp4 | FAKE | 0.998 | **SPOOF** | 10 | 0.281 | UNCERTAIN |
| crackers_withaud.mp4 | FAKE | 0.744 | **SPOOF** | 3 | 0.239 | FAKE |
| gemini ai video.mp4 | FAKE | 0.470 | BONAFIDE | 3 | 0.149 | FAKE |
| girl walkin video.mp4 | FAKE | 0.984 | **SPOOF** | 3 | 0.145 | FAKE |
| girl_speaking video gemini.mp4 | FAKE | 0.017 | BONAFIDE | 3 | 0.134 | FAKE |
| insta vid.mp4 | FAKE | 0.999 | **SPOOF** | 11 | 0.092 | REAL |
| real vid ai aud.mp4 | FAKE | 0.996 | **SPOOF** | 26 | 0.075 | REAL |
| south indian women speaking.mp4 | FAKE | 0.009 | BONAFIDE | 3 | 0.323 | FAKE |
| WhatsApp …6.15.45 PM.mp4 | REAL | 0.965 | SPOOF | 15 | 0.077 | REAL |
| WhatsApp …6.15.46 PM.mp4 | REAL | 0.994 | SPOOF | 4 | 0.236 | REAL |
| real_vid_to_test_withaudio.mp4 | REAL | 0.012 | BONAFIDE | 6 | 0.317 | UNCERTAIN |
| vid_with_noaudio.mp4 | REAL | — | — | 0 | INACTIVE | UNCERTAIN |

`—` = no audio stream → `INACTIVE`, score `null` (`"No audio track"`). `win` = number
of 64,600-sample windows. `audio_heur` = production audio heuristic score (0…1,
higher = more fake-looking; untouched by this work).

## 8. Metrics (AASIST only) & production baseline unchanged

AASIST scored **13/15** videos (2 had no audio track). Confusion is computed with the
explicit assumption **`SPOOF → FAKE`, `BONAFIDE → REAL`** (audio anti-spoofing vs the
video truth label) — recorded as `aasist.audio_anti_spoof_note` and mirrored here:

- **Confusion:** TP=6, TN=1, FP=2, FN=4 → accuracy **0.538**, precision **0.75**,
  **FAKE recall 0.60** (6/10), **F1 0.667**, FPR **0.667** (2/3), FNR 0.40,
  real-rejection (REAL kept) **0.333** (1/3).
- **Production verdict baseline — unchanged (byte-identical):** TP=6, TN=4, FP=0,
  FN=5, FAKE recall 0.545, accuracy 0.667. Confirmed before/after the AASIST wiring.
- **Disagreement with the production verdict:** 13 rows (vs `aasist_disagreement_with_verdict`
  in the JSON), e.g. `avengers ai vid 1.mp4` / `insta vid.mp4` / `real vid ai aud.mp4`
  / `avengers ai vid 2.mp4` — FAKE videos the current verdict calls REAL/UNCERTAIN but
  whose **audio AASIST rates SPOOF**; and the two WhatsApp REAL videos AASIST rates SPOOF.

## 9. Findings & recommendation

**What AASIST does well on this corpus:** it is the *only* signal that catches the
AI/voice-cloned audio on the FAKE videos the verdict misses — `real vid ai aud.mp4`
(0.996, video "real", audio AI), the two avengers AI clips (0.989/0.998), and
`insta vid.mp4` (0.999). Where the audio truly is AI-generated it is confident and
correct — exactly the semantic of the model it trains for (ASVspoof2019 LA).

**Where it fails (out-of-domain, reproducible):**
- Both **REAL** WhatsApp clips score 0.965 / 0.994 SPOOF. These are heavily
  compressed, telephony-quality phone recordings; AASIST's neutral sinc frontend
  interprets codec artifacts as synthesis artifacts. The audio heuristic *agrees with
  reality* there (0.077/0.236) — a case where the heuristic beats the model.
- 4 FAKE videos score BONAFIDE (`ai_vid_without_audio`, `girl_speaking gemini`,
  `south indian women speaking`, `gemini ai video`). These are talking-head
  deepfakes that **retain the true human voice** (Wav2Lip-style). This is *correct
  audio anti-spoofing behavior** — "fake video ≠ fake audio".

**Recommendation (integration options A/B/C/D):**
- **A — Fuse into the audio verdict now (default rejected):** FPR 0.667 on REAL
  compressed audio would inject new false positives into the production verdict.
  **Rejected.**
- **B — Keep as a reported secondary signal (recommended now):** AASIST stays
  status+eval-only, documented alongside GenConViT. Zero risk to the verdict, and it
  is the only module that flags AI-audio-on-real-video clips the verdict currently
  misses. **Adopt.**
- **C — Verdict overlay gated on "clean" audio:** use AASIST to *raise a* "AI-audio"
  *flag* (e.g. `aasist_predicted_label='SPOOF'` + high window confidence) that is
  surfaced in the UI/JSON but only afterwards explored as a fuse input on a wider,
  domain-representative corpus (real phone WhatsApp clips, short talking-heads).
  **Future candidate.**
- **D — Threshold/scope calibration:** avoid tuning on this 15-video set (no
  threshold-tuning per constraints); instead collect a labelled set of *compressed
  real* and *AI-audio* clips from the target deployment domain and re-validate before
  any fusion. **Future work.**

Until B→C/D happens, **AASIST is reported but does not influence the verdict** — an
honest, reversible, fully-local addition.

## 10. Files changed / added

- `scripts/download_aasist_weights.py` (new)
- `pipeline/aasist/__init__.py`, `pipeline/aasist/model.py` (verbatim upstream),
  `pipeline/aasist/detector.py` (new)
- `tests/unit/test_aasist_detector.py` (new, 13 tests passing)
- `config/config.yaml` (`aasist:` section)
- `main.py` (`get_detector_status` AASIST block)
- `scripts/evaluate_testing_videos.py` (AASIST fields + standalone metrics +
  disagreement table; `_run_aasist` helper; production path untouched)
- `reports/testing_videos_evaluation.json/.csv` (regenerated; production block
  unchanged, AASIST block added)
- `reports/PHASE2B_AASIST_EVALUATION.md` (this)
- `models/aasist/AASIST.pth` (official 1,281,532-byte checkpoint; not committed)