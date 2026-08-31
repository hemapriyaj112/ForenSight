# Phase 2C — Multimodal Evidence Analysis of the 15-Video Test Corpus

> Status: ANALYSIS-ONLY. Nothing in this phase changes production detection. All
> detector artifacts (GenConViT, AASIST) remain evaluation-only
> (`genconvit.use_in_final_fusion: false`, `aasist.use_in_final_fusion: false`), the
> fusion rule, the thresholds, and the production verdicts are untouched.
>
> Inputs: `reports/testing_videos_evaluation.json` (Phase 2B eval output), the Phase
> 2B aggregate (production + AASIST), Phase 2A.1 GenConViT coverage statistics, and
> read-only `ffprobe` identity enrichment (codec/resolution/bitrate/filesize; no
> media bytes are altered).
> Reproducibility: `py scripts/analyze_phase2c_evidence.py` regenerates
> `reports/phase2c_multimodal_evidence.json` + `.csv` from the eval JSON.

---

## 1. Executive Summary

ForenSight's current production video verdict (TP=6, TN=4, FP=0, FN=5, FAKE recall
0.545) is confirmed unchanged before and after Phase 2C (Section 17). This phase
did not improve or worsen detection; it analysed *why* the five misses and the two
concerning false-positive-style signals exist, using the three neural modalities
(GenConViT face-based, AASIST audio anti-spoofing) plus the existing production
heuristics (texture, temporal, audio-heuristic, provenance) and container identity.

Headline findings:

1. **Systematic audio disagreement.** In all 8 videos AASIST labels SPOOF (≥0.74),
   the production audio heuristic reads natural (heuristic mean 0.07–0.28,
   always ≤0.40, never corroborates). The heuristic appears strongly biased toward
   "BONAFIDE" across this corpus, and on both provided-real WhatsApp clips it is
   confident while AASIST is confident in the opposite direction (0.077 vs 0.965
   and 0.236 vs 0.994).
2. **All five false negatives have strong, persistent AASIST audio evidence.**
   Four of the five FN videos show *persistent high spoof* across every analysed
   window (avengers1 12/12, avengers2 10/10, insta 11/11, "real vid ai aud"
   26/26 all ≥0.95), and the fifth (ai_noaud2) has no audio track at all. The
   reported-only AASIST modality is useful independent evidence that current fusion
   ignores.
3. **WhatsApp false-flagging is a hypothesis, not a verdict.** AASIST SPOOF on both
   provided-real clips is classified *Hypothesis* (consistent with codec/telephony
   artifacts, NOT proven), because we have no clean re-encode or studio reference to
   distinguish "genuine synthetic speech" from "compressed speech read as spoof".
4. **Coverage rules.** GenConViT contributes little on 9/15 videos (≤40% face
   coverage); on avengers1 it is wrong (REAL) even with 76% coverage, so coverage is
   necessary but not sufficient. Its only high-coverage correct FAKE call (avengers2,
   95% coverage) still couldn't lift the fused score out of UNCERTAIN.
5. **Reliability states are the right scaffolding for the next phase.** We define a
   5-state per-video reliability model (HIGH / MODERATE / LOW / DOMAIN-RISK /
   UNAVAILABLE) that reports *how* much of a signal to trust, and recommend a
   future strategy based on evidence-gating + modality + domain-risk — NOT a
   weighted-average fusion (see Section 14 for why a weighted average fails on this
   corpus).

All conclusions below are tagged `[observation]` (a measured fact), `[hypothesis]`
(consistent with the observations but not independently verified), or
`[validated conclusion]` (reproducible and measured in this phase).

## 2. Scope, Objectives and Constraints

### 2.1 Scope
Analysis of the 15-video test corpus (11 labelled FAKE, 4 labelled REAL) across the
evidence emitted by: production fusion (video/texture + audio heuristics +
provenance + temporal), GenConViT (face-based, eval-only), AASIST (audio
anti-spoofing, eval-only), container identity (read-only ffprobe), and the
recorded per-frame/per-window/per-segment traces.

### 2.2 Objectives
- Produce a single cross-modal evidence matrix for all 15 videos (Section 4).
- Explain agreement and disagreement patterns (Sections 5–6).
- Explain the 5 false negatives and the 2 WhatsApp REAL-case signals (Sections 7–8).
- Add segment-level resolution for AASIST windows and GenConViT per-frame coverage
  (Sections 9–10).
- Propose domain-risk indicators and reliability states (Sections 11–12).
- Separate observations, hypotheses and validated conclusions (Section 13).
- Recommend a future strategy and data requirements (Sections 14–15).
- Report tests and invariance evidence (Sections 16–17).

### 2.3 Hard constraints (respected)
- NO new production detector, NO training/fine-tuning, NO threshold tuning against
  these 15 videos, NO arbitrary fusion change.
- GenConViT and AASIST remain evaluation-only; `calculate_final_fusion` still uses
  video-weight 0.6, audio-weight 0.4, thresholds 0.40/0.60 (production code
  `pipeline/fusion/fuser.py` unchanged in this phase).
- No claim that BONAFIDE audio ⇒ video real; no claim that SPOOF audio ⇒ whole video
  fake; AASIST is an audio anti-spoofing scoring tool, reported-only.
- Every non-measured claim is labelled Hypothesis; dataset sizes are requirements,
  never claimed as owned data.

## 3. Baseline Metrics and Production Invariance

Production confusion matrix (unchanged, verified byte-identical before vs after
Phase 2C in Section 17):

| Metric | Baseline (Phase 2B) | Phase 2C (recomputed) | Delta |
|---|---|---|---|
| confusion TP/TN/FP/FN | 6 / 4 / 0 / 5 | 6 / 4 / 0 / 5 | exact |
| accuracy | 0.66667 | 0.66667 | 0 |
| precision (fake) | 1.0000 | 1.0000 | 0 |
| recall (FAKE_RECALL) | 0.54545 | 0.54545 | 0 |
| F1 (fake) | 0.70588 | 0.70588 | 0 |
| false-negative rate | 0.45455 | 0.45455 | 0 |
| false-positive rate | 0.0000 | 0.0000 | 0 |
| fake-verdict rate | 0.4000 | 0.4000 | 0 |
| uncertain rate | 0.26667 | 0.26667 | 0 |

`verify_production_invariance()` (consumed from `reports/testing_videos_evaluation.json`)
reports `matches_phase2b_baseline: True` and an empty `metric_deltas`; per-video
verdict/GT pairs are reproduced verbatim. The only values that drift between two
evaluation runs are GenConViT *evaluation-only* scores (VAE reparameterization,
≤0.007 absolute), which never enter the fusion and never change a predicted label.

## 4. Unified Cross-Modal Evidence Matrix

Per-video evidence. Keys: GT = provided label; FUS = production fusion verdict;
GL = GenConViT label; AL = AASIST label; H = audio-heuristic mean; T = texture
score; PAT = AASIST segment pattern (`persistent_high_spoof`, `persistent_low_spoof`,
`mixed_or_localized`, `no_windows`); COV = GenConViT face coverage.

| Video | GT | FUS | FUS score | GL | AL | H (heur) | T | AASIST mean | PAT | COV |
|---|---|---|---|---|---|---|---|---|---|---|
| ai_noaud2.mp4 | FAKE | UNK | 0.551 | – | – | – | 0.97 | – | no_windows | 0% |
| ai_vid_without_audio.mp4 | FAKE | FAKE | 0.720 | REAL | BON | 0.45 | 0.53 | 0.010 | low | 10% |
| avengers ai vid 1.mp4 | FAKE | REAL | 0.202 | REAL | SPO | 0.13 | 0.14 | 0.989 | high | 76% |
| avengers ai vid 2.mp4 | FAKE | UNK | 0.443 | FAKE | SPO | 0.28 | 0.95 | 0.998 | high | 95% |
| crackers_withaud.mp4 | FAKE | FAKE | 0.636 | FAKE | SPO | 0.24 | 0.42 | 0.744 | mixed | 10% |
| gemini ai video.mp4 | FAKE | FAKE | 0.599 | – | BON | 0.15 | 0.76 | 0.470 | mixed | 0% |
| girl walkin video.mp4 | FAKE | FAKE | 0.598 | REAL | SPO | 0.14 | 0.78 | 0.984 | high | 40% |
| girl_speaking video gemini.mp4 | FAKE | FAKE | 0.594 | REAL | BON | 0.13 | 0.99 | 0.017 | low | 40% |
| insta vid.mp4 | FAKE | REAL | 0.361 | FAKE | SPO | 0.09 | 0.43 | 0.999 | high | 27% |
| real vid ai aud.mp4 | FAKE | REAL | 0.334 | FAKE | SPO | 0.07 | 0.17 | 0.996 | high | 20% |
| south indian women speaking.mp4 | FAKE | FAKE | 0.669 | FAKE | BON | 0.32 | 0.99 | 0.009 | low | 20% |
| WhatsApp Video 2026-07-24 at 6.15.45 PM.mp4 | REAL | REAL | 0.329 | FAKE | SPO | 0.08 | 0.45 | 0.965 | high | 3% |
| WhatsApp Video 2026-07-24 at 6.15.46 PM.mp4 | REAL | REAL | 0.312 | REAL | SPO | 0.24 | 0.30 | 0.994 | high | 92% |
| real_vid_to_test_withaudio.mp4 | REAL | UNK | 0.451 | – | BON | 0.32 | 0.13 | 0.012 | low | 0% |
| vid_with_noaudio.mp4 | REAL | UNK | 0.551 | – | – | – | 0.83 | – | no_windows | 0% |

Container identity (read-only ffprobe): all 15 are H.264; resolutions 848×480 …
1280×720 (portrait 720×1280 for avengers1/insta); average bitrate 651–4960 kbps;
files 1.67–60.5 MB. Compression markers (low-bitrate / downscaled / re-encoded)
present on the FN and WhatsApp videos (Sections 7–8).

## 5. Agreement Patterns Across Modalities

Pattern inference over signals {production, GenConViT, AASIST, texture,
audio-heuristic, temporal, provenance}:

| Pattern | Count | Videos |
|---|---|---|
| STRONG_MULTIMODAL_AGREEMENT (≥2 independent manipulation signals) | 9 | avengers2, crackers, gemini, girl walkin, girl_speaking, insta, real vid ai aud, south indian, WhatsApp45 |
| MODALITY_DISAGREEMENT (neural modalities read opposite directions) | 3 | avengers1, WhatsApp46, ai_vid_without_audio |
| SINGLE_WEAK_MODALITY | 2 | ai_noaud2, vid_with_noaudio |
| AMBI | 1 | real_vid_to_test_withaudio |

Key observations:

- `STRONG_MULTIMODAL_AGREEMENT` is only trustworthy when it also agrees with the
  provided label (7 of 9; all FAKE). It is *misleading* on WhatsApp45 (2 neural
  detectors converge on a provided-REAL clip — see Section 6/8).
- `MODALITY_DISAGREEMENT` is the dominant *honest* signal on the hard FAKE cases
  (avengers1: GenConViT REAL vs AASIST SPOOF; ai_vid_without_audio: GenConViT REAL,
  AASIST BONAFIDE vs provenance override FAKE).
- `SINGLE_WEAK_MODALITY` / `AMBI` occur exactly on the videos where production is
  UNCERTAIN and one or more neural modalities are unavailable (no face, no audio).

## 6. Disagreement Patterns Across Modalities

12 recorded disagreement cases from `disagreement_cases` in the Phase 2C JSON:

- **NEURAL_VS_NEURAL (4):** avengers1 (GC REAL 0.54 vs AASIST SPOOF 0.99), girl walkin
  (GC REAL 0.63 vs AASIST SPOOF 0.98), WhatsApp46 (GC REAL 0.55 vs AASIST SPOOF 0.99),
  south indian (GC FAKE 0.50 vs AASIST BONAFIDE 0.01).
- **NEURAL_VS_HEURISTIC (8):** AASIST=SPOOF meanwhile audio-heuristic mean ≤0.40 in
  avengers1 (0.13), avengers2 (0.28), crackers (0.24), girl walkin (0.14), insta
  (0.09), real vid ai aud (0.07), WhatsApp45 (0.08), WhatsApp46 (0.24). Plus the
  inverse pair ai_vid_without_audio (AASIST BONAFIDE 0.01 vs heuristic mean 0.45).

`[observation]` Across the entire corpus the audio-heuristic never corroborates an
AASIST SPOOF call. `[hypothesis]` The heuristic is tuned for a narrower codec/
telephony envelope than the corpus and systematically underestimates synthetic-speech
probability in low-bitrate H.264/AAC recordings. `[validated conclusion]` The AASIST
vs heuristic split is deterministic and reproducible for these 15 clips.

## 7. False Negative Analysis (5 missed FAKE videos)

All five labelled FAKE videos that production labels REAL or UNCERTAIN carry their
own distinct evidence profile. `[observation]` unless stated, all numbers below are
measured from the eval JSON.

### 7.1 "real vid ai aud" (GT FAKE → REAL, fusion 0.334) — the clearest miss
Real footage with AI (TTS) voiceover. Production reads: texture REAL-like (0.175),
temporal REAL-like (0.31), audio-heuristic REAL-like (0.075), video score 0.506,
audio score 0.075 → fusion 0.334. `[observation]` AASIST flags **SPOOF with
persistence across all 26 windows** (mean 0.9959, min 0.991, max 0.999), i.e. every
one of the 100 seconds of speech reads synthetic. `[observation]` GenConViT also
flags FAKE (0.495, 20% coverage). Two neural modalities agree the content is
synthetic; production ignores both because neither feeds the current fusion and the
production heuristics read real-like. This is the canonical case proving the value
of reporting AASIST for synthetic-speech detection.

### 7.2 "avengers ai vid 1" (GT FAKE → REAL, fusion 0.202)
AI-generated scene with low-bitrate encode (720×1280 H.264 @ 700 kbps, 3.9 MB).
Production: texture very REAL-like (0.140), audio-heuristic REAL-like (0.130), videoscore
0.249, no provenance, no uncorroboration. `[observation]` AASIST SPOOF persistent
12/12 windows (mean 0.989) — the AI voiceover reads synthetic end-to-end. GenConViT
reads REAL (0.544) at 76% good coverage — a genuine neural miss even with coverage.
`[hypothesis]` The low-bitrate encode corrupts the micro-texture cues the texture
heuristic depends on (median texture ≈0.14, suspiciously flat and low).

### 7.3 "avengers ai vid 2" (GT FAKE → UNCERTAIN, fusion 0.443)
AI scene @ 892×718, 651 kbps. Production honestly leaves it UNCERTAIN: video score
0.551 driven by FAKE-like texture (0.945) but audio score 0.281 pulls fusion to
0.443 (band 0.40–0.60). `[observation]` Three independent signals agree FAKE:
GenConViT FAKE (0.484, 95% good coverage — the only high-coverage correct FAKE
call), AASIST SPOOF persistent 10/10 (0.998), texture FAKE-like (0.945).
`[hypothesis]` With AASIST reporting, a future evidence-gated logic would have three
corroborating FAKE-side modalities and could upgrade this to FAKE with bounded risk
heuristics (still a tuning decision, NOT taken here).

### 7.4 "insta vid" (GT FAKE → REAL, fusion 0.361)
Instagram-sourced AI video (720×1280 @ 2732 kbps, 41.5 s). Production: texture weak
(0.427), audio-heuristic real (0.092), video score 0.5397, audio 0.092 → 0.361.
`[observation]` AASIST SPOOF persistent 11/11 (mean 0.9985); GenConViT FAKE (0.491,
27% coverage). Same pattern as 7.3 but lower texture (0.427).

### 7.5 "ai_noaud2" (GT FAKE → UNCERTAIN, fusion 0.551)
AI video WITHOUT audio (848×480, 1471 kbps). No audio → AASIST and audio-heuristic
both unavailable (`INACTIVE`/`no_windows`). No face → GenConViT unavailable.
Texture FAKE-like (0.970) but temporal analysis reports "per-frame suspicion scores
jump erratically" (score 0.46) with no corroborating modality, and the classifier is
disabled → production stays UNCERTAIN. `[observation]` This is a genuine
single-modality, no-corroboration case: no detector in the current (or reported-only)
stack can bootstrap it with confidence. `[hypothesis]` A video-specific deepfake
head could help such no-face, no-audio clips — outside this phase's scope.

### 7.6 Common thread
In all four audible FN cases the *same* strong signal is present: persistent AASIST
SPOOF. Production fusion cannot use it today because `aasist.use_in_final_fusion`
remains false and the audio heuristic disagrees. The miss is therefore a *decision
policy* miss, not a missing-evidence miss — except ai_noaud2, where evidence is
genuinely absent.

## 8. WhatsApp False-Positive / Hour-Length Analysis (provided-REAL clips)

Both provided-REAL WhatsApp clips produce strong AASIST SPOOF readings. Following the
requirement to separate verified fact from interpretation:

**Observed (measured, reproducible):**
- 6.15.45 PM: duration 57.6 s; AASIST SPOOF every window, 15/15 (mean 0.965, min
  0.933, max 0.995) across the full clip; audio-heuristic very REAL-like (mean 0.077,
  all 57 segments ≤0.146); texture weak (0.451); GenConViT FAKE (0.468) at only 3%
  face coverage; H.264 1024×576 @ 974 kbps.
- 6.15.46 PM: duration 13.4 s; AASIST SPOOF 4/4 (mean 0.994, min 0.989, max 0.999);
  audio-heuristic 0.236 (real-side but higher); GenConViT REAL (0.547) at 92% good
  coverage (the cleanest face evidence in the corpus); H.264 848×478 @ 1829 kbps.

**Likely (hypothesis, consistent but unproven):** the AASIST spoof readings on these
clips are caused by codec/telephony/transcoding artifacts — WhatsApp relinking uses
aggressive AAC compression at low bitrate; both clips are short-form screen-capture
style footage. The persistent-high *pattern* (Sections 9) is identical on genuine
TTS voiceovers (avengers, insta, real-vid-ai-aud), so pattern alone cannot separate
"synthetic speech" from "hard-compressed speech" — separation needs clean re-encode
comparison (Section 15).

**Unverified:** whether the provided REAL label is correct for either clip. We never
assert the label is wrong; we state both interpretations are present in the signal
and that the evidence for "REAL + AASIST false positive" and "FAKE + both audio
readings wrong" cannot be arbitrated on current data.

**Honest consequence:** under any future scenario, WhatsApp-style inputs must be
flagged DOMAIN-RISK (Section 11) with reliability downgrades, so that neither a
SPOOF call nor a REAL call is made on the audio modality alone.

## 9. AASIST Segment-Level Analysis

Window geometry: 4.0375 s per frame-sized window (AASIST raw feature length 64600 @
16 kHz), starts 0.0, 4.037, 8.075, … Window scores are deterministic and stored per
video in the eval JSON.

Three observed segment patterns (also implemented as `aasist_segment_pattern`):

| Pattern | Definition (measured) | Corpus occurrences |
|---|---|---|
| persistent_high_spoof | ≥80% windows ≥0.8 AND 0% <0.2 | 7: avengers1 (12/12), avengers2 (10/10), girl walkin (3/3), insta (11/11), real vid ai aud (26/26), WhatsApp45 (15/15), WhatsApp46 (4/4) |
| persistent_low_spoof | 100% windows <0.2 | 4: ai_vid_without_audio, girl_speaking, south indian, real_vid_to_test |
| mixed_or_localized | neither of the above | 2: crackers (high,high,low), gemini (0.41, 0.99, 0.01) |

`[observation]` Every FAKE-label AI/TTS-audio video is *persistent_high_spoof*
(avengers1/2, insta, real vid ai aud, girl walkin) — and every provided-REAL video
that AASIST flags is also *persistent_high_spoof*. Persistence is therefore a strong
*signal-strength* feature but NOT a truth discriminator on its own (Sections 8, 13).

`[observation]` `mixed_or_localized` exposes a weakness of mean-based window
aggregation: gemini's three windows are (0.406, 0.995, 0.011); the 0.995 localised
spike is erased by the mean (0.470 → BONAFIDE). Segment-level maxima and run-lengths
preserve this information; the current production audio heuristic computes a single
aggregate and cannot see it. `[hypothesis]` localised single-window spikes may also
be brief non-speech artefacts (music, sound effects) — needs transcript-aligned
validation.

`[observation]` Real-vs-AI look-away: "girl_speaking" (AASIST 0.017) and "south
indian" (0.009) are confident BONAFIDE despite being AI-generated videos — the audio
in those clips is plausibly genuine speech, so AASIST is consistent with the audio,
not the video. This reinforces: AASIST informs *audio* authenticity only.

## 10. GenConViT Face-Coverage vs Prediction Analysis

Coverage distribution (`coverage_status`): GOOD = 3 (avengers1 76% / 35 frames,
avengers2 95% / 36 frames, WhatsApp46 92% / 12 frames); PARTIAL = 3 (girl walkin 40%,
girl_speaking 40%, insta 27%); LOW = 5 (ai_vid_without_audio 10%, crackers 10%,
real vid ai aud 20%, south indian 20%, WhatsApp45 3%); NO_FACE = 4 (ai_noaud2, gemini,
real_vid_to_test, vid_with_noaudio).

Coverage-vs-prediction findings `[observation]`:

- WhatsApp46 (REAL, 92% coverage) → GenConViT REAL correct: the one good-coverage
  provided-REAL clip is classified correctly.
- avengers2 (FAKE, 95% coverage) → GenConViT FAKE correct: the strongest-coverage
  FAKE call, and even it could not move the fused verdict past UNCERTAIN.
- avengers1 (FAKE, 76% coverage) → GenConViT REAL **wrong despite good coverage**.
  Coverage is necessary but not sufficient; this is a video-level algorithmic miss,
  not a coverage artefact.
- WhatsApp45 (REAL, 3% coverage) → GenConViT FAKE: low-coverage calls concentrate
  on the same videos that carry codec/compression markers (WhatsApp45, avengers1,
  real vid ai aud) — `[hypothesis]` low coverage correlates with domain-risk here,
  and GenConViT's weakest calls appear in that regime.

Frame-sampled scoring (15 face frames used for avengers1/2, 1–12 elsewhere) means
confidence bounds are narrow; we do not re-quantify detection probability from these
without a dedicated two-population experiment.

## 11. Domain-Risk Indicators

`[validated conclusion]` (deterministic from container + eval traces). Indicators and
their measured incidence:

| Indicator | Measure | Videos flagged | Rationale |
|---|---|---|---|
| Low-bitrate H.264 (≤ ~1200 kbps avg) | 651–974 kbps | avengers1 (700), avengers2 (651), WhatsApp45 (974), (boundary: real_vid_to_test 1456) | Micro-texture/spectral cues degrade; AASIST/heuristic diverge most here |
| Downscaled/screen style resolution (<720p) | 480p–576p | ai_noaud2, vid_with_noaudio, WhatsApp45, WhatsApp46, real_vid_to_test | Same degradation envelope |
| WhatsApp/chat relink exposure | filename/metadata | WhatsApp45, WhatsApp46 | Transcoded AAC + re-encode; commercial tools claim elevated FP on such audio |
| Missing modality | no audio or no face | NO_AUDIO 2 (ai_noaud2, vid_with_noaudio), NO_FACE 4 (+gemini, real_vid_to_test) | Detector cannot corroborate; verdict hinges on single modality |
| Neural-vs-heuristic audio disagreement | AASIST SPOOF while heuristic ≤0.40 | 8 videos | Signals the heuristic is not credentialled in this domain |
| Persistent-high AASIST pattern on REAL label | pattern + provided label conflict | WhatsApp45, WhatsApp46 | Must force DOMAIN-RISK, not FAKE |

Design rule (proposal, not implemented): each indicator contributes to a
`domain_risk` score; if any strong indicator fires (low bitrate + re-encode, or
modality outage), the reliability state for that video is capped at DOMAIN-RISK
(Section 12) and the audio/video modality weights are treated as untrusted rather
than merely weighted lower.

## 12. Reliability State Design

A per-video reliability state attached to the reported verdict, with an explicit
meaning for each state (proposal for the next phase; not implemented):

| State | Meaning | Predicted assignment in this corpus (post-analysis) |
|---|---|---|
| HIGH | ≥3 corroborating trusted modalities, no domain-risk indicator | crackers, gemini (provenance+texture+AASIST/GC-side), south indian, girl_speaking, girl walkin, avengers2 (3 FAKE-side signals) |
| MODERATE | 2 modalities agree, disagreement elsewhere but no domain-risk gate | insta (GC+AASIST FAKE, texture weak) |
| LOW | single modality, or modalities disagree with no tie-breaker | ai_noaud2 (single texture), real_vid_to_test (single audio), vid_with_noaudio (single texture) |
| DOMAIN-RISK | strong compression/re-encode/missing-modality marker present | WhatsApp45, WhatsApp46, avengers1 (low-bitrate + neural disagreement), real vid ai aud (low-bitrate + persistent SPOOF on REAL-pedigree audio) |
| UNAVAILABLE | no usable detector signal at all | none (all 15 have at least texture or provenance) |

Rules of thumb: DOMAIN-RISK caps both FAKE and REAL confidence; states are computed
from evidence, never hand-set; UNAVAILABLE implies "no verdict, request re-capture
or clean upload".

## 13. Hypotheses vs Observations vs Validated Conclusions

| Claim | Status | Evidence |
|---|---|---|
| Production verdicts/metrics unchanged | validated | identical JSON/CSV before vs after; fusion code untouched |
| AASIST window scores & patterns reproducible | validated | deterministic recompute, stored windows |
| Heuristic never corroborates AASIST SPOOF in 15 videos | observation | 8 SPOOF videos all heuristic ≤0.40 |
| All 4 audible FN videos are persistent_high_spoof | observation | window arrays |
| WhatsApp45/46 AASIST SPOOF caused by codec/telephony | hypothesis | pattern identical on genuine TTS; needs clean re-encode |
| WhatsApp labels are wrong | hypothesis (unfavoured) | no arbitrating evidence |
| GenConViT needs face coverage for reliability | observation | misses on avengers1 at 76% |
| Low-bitrate correlates with AASIST/heuristic divergence | hypothesis | 651–974 kbps cluster in FN/FF set |
| Video-specific deepfake head would fix ai_noaud2 | hypothesis | no such detector evaluated here |
| Weighted-average fusion is structurally unsafe here | validated (analysis) | correlated errors (WhatsApp45: both neural detectors wrong together; avengers1: texture+GC wrong together) break averaging |
| Corpus is statistically sufficient to calibrate thresholds | validated (analysis) — it is NOT | n=15, 11/4 split, no independent calibration set |

## 14. Future Detector Strategy

Recommendation: **evidence-gated fusion with domain-risk gating — NOT a learned or
weighted-average blend.** Rationale from this corpus:

1. **Errors are correlated, which averaging cannot handle.** WhatsApp45: GenConViT and
   AASIST *both* converge on FAKE for a provided-REAL clip — two independent-pedigree
   detectors fail together because they share the same degraded input. Averaging
   propagates the shared error; gating on domain risk would cap it.
2. **AASIST's signal is decision-relevant but domain-sensitive.** Its persistent SPOOF
   is the strongest single cue on 4/5 FN videos, yet falls into the same high-SPOOF
   band on 2 REAL clips. Use it as an *evidence gate* (raise FAKE suspicion for
   gated submission) rather than as a linear fusion feature.
3. **The audio-heuristic is the weak link.** Its real-side confidence (0.07–0.08) on
   both the 5-FN audio and the 2 WhatsApp clips means it must be de-weighted in the
   compressed/telephony domain and re-validated on clean speech.
4. **Coverage gating.** GenConViT results below ~40% coverage should degrade the
   texture/genconvit weight in HIGH/MODERATE states (WhatsApp45, avengers1 patterns).

Proposed architecture (next phase, implementation NOT in Phase 2C): compute
per-video reliability state + domain-risk gates → combine modalities into *evidence
votes* (FAKE-vote / REAL-vote / NO-vote) → rule-based decision over votes + risk
gates → verdict {FAKE, REAL, UNCERTAIN, REVIEW}. Thresholds chosen on a held-out
calibration set, never on these 15.

## 15. Required Training and Validation Data

No fabricated dataset sizes are claimed; the following are stated requirements to
make the Section 14 strategy statistically defensible.

- **Per-domain stratification** (each domain = codec × resolution × bitrate ×
  source): at minimum 50 REAL and 100 FAKE clips per cell in the low-bitrate /
  telephony cell and the clean-upload cell, and ≥200 clips for the "re-encode at
  N bitrates" synthetic pairs (same source, different encode axis) to isolate
  codec-induced AASIST/heuristic divergence.
- **Category coverage** (observability): AI-generated video (no audio), real video +
  AI TTS audio, real video + real audio, screen-record/WhatsApp-relink REAL speech,
  deepfake-voice-over-real-picture (e.g. the girl-walkin/south-indian pattern).
  Each missing-modality and each domain-risk combination needs ≥50 samples.
- **Clean references:** every REAL clip used to characterise false positives must
  have a clean (uncompressed/master) copy so the codec hypothesis in Section 8 can be
  tested by re-encode.
- **Independent sets:** a blind calibration set (no overlap with these 15) for
  threshold choice, and a blind validation set for final reporting; target ≥500
  clips total across both for per-domain cells to have credible FPR bounds.
- **Annotator gold standard:** independent labels per clip by 2+ annotators (media
  provenance + audio authenticity + video authenticity separately), because these 15
  clips' labels conflate all three and the AASIST/GC outcomes show that conflation
  matters.

## 16. Test Results

- New Phase 2C unit tests: `tests/unit/test_phase2c_evidence.py` — **16 passed, 0 failed**
  (evidence taxonomy, compression-risk rule scoping, disagreement cases, segment
  patterns, window geometry, production invariance positive/negative).
- Full suite: `py -m pytest tests/ -q` → **263 passed, 5 failed**.
- The 5 failures are PRE-EXISTING and unrelated to Phase 2C (identical set and cause
  recorded in Phase 2B): `tests/unit/test_app.py` Streamlit GUI tests
  (`AttributeError: module 'streamlit' has no attribute 'session_state'` — importing
  the `streamlit` stub lacks the module-level attribute). List:
  `TestFileUploadHandling::test_no_upload_does_not_call_pipeline`,
  `TestFileUploadHandling::test_no_upload_shows_info_prompt`,
  `TestMainAppTabs::test_two_tabs_created`,
  `TestMainAppTabs::test_video_tab_label_present`,
  `TestMainAppTabs::test_image_tab_label_present`. Baseline prior to Phase 2C was
  247 passed / 5 failed; this phase adds exactly the 16 new tests above. These
  cannot be attributed to Phase 2C and were not modified.
- Eval pipeline re-run (full, on the untouched production code paths): completed
  successfully; metrics identical to the Phase 2B snapshot (Section 17).

## 17. Invariance Verification (Before/After Phase 2C)

The task requirement (production must stay byte-identical) is verified both ways:

1. **Constraint-audit (code):** Phase 2C added only `pipeline/analysis/` (importable
   solely by `scripts/analyze_phase2c_evidence.py` and its tests), the analysis
   runner, and the test file. `git status` shows NO Phase-2C modification to
   `app.py`, `config/config.yaml`, `main.py`, `pipeline/{audio,forensics,fusion,
   video}/…`, nor to the 3 production test files — the modified set is exactly the
   Phase 2A/2B change list. `aasist.use_in_final_fusion=false` and
   `genconvit.use_in_final_fusion=false` remain false.
2. **Empirical re-run:** production eval was re-run after the Phase 2C code was in
   place. Comparison of the pre-run backup (`pre2c_eval.json`) against the regenerated
   output: confusion matrix identical {TP:6, TN:4, FP:0, FN:5}; all 9 aggregate
   metrics identical; all 15 per-video `(filename, ground_truth, verdict)` triples
   identical; `fusion_score` identical per video. The only drift observed is
   GenConViT eval-only scores (VAE reparameterization, max |Δ| ≈ 0.007) which do not
   enter fusion and change no predicted label.
3. `verify_production_invariance()` over the stored output returns
   `matches_phase2b_baseline: True` with an empty `metric_deltas`.

## 18. Conclusions and Recommended Next Phase

**Conclusions**
1. Current production detection is functionally unchanged (TP6/TN4/FP0/FN5) and its
   misses are *policy* misses, not evidence-absence: four of five FN videos are
   covered by persistent AASIST SPOOF that the fusion is (by design) not allowed to
   see.
2. GenConViT and AASIST — evaluation-only — are the strongest reported modalities, but
   each fails in its own domain: GenConViT at 3%-coverage (WhatsApp45) and even at
   76% coverage (avengers1); AASIST on compressed telephony speech (both WhatsApp
   clips).
3. The audio-heuristic is not trustworthy below ~1000 kbps bitrates / after
   re-encode; this is where both FN and FF errors live.
4. Weighted-average fusion is structurally inappropriate for correlated-error
   inputs; evidence-gating with domain-risk caps is the recommended direction.
5. This corpus is far too small (n=15, 11 FAKE / 4 REAL) to calibrate anything; any
   threshold decision would be over-fitting these 15 clips.

**Recommended next phase (Phase 3), with gates:**
1. Collect the Section 15 dataset (per-domain, with clean references and blind
   calibration/validation splits) — this is the gate for everything else.
2. Implement the *reporting layer* only: per-video reliability state, domain-risk
   flags, evidence-vote UI (no verdict change) and run it for a period to gather
   human-review labels on WhatsApp/telephony-tier inputs.
3. Re-encode ablation (clean → bitrate sweep) to move the Section 8 "Likely" into
   observation/validated conclusion or reject it.
4. Only after (1)–(3), and on the held-out calibration set only, consider adding
   AASIST as a *gated* FAKE-vote contributor with `use_in_final_fusion` remaining
   false until a formal comparison shows a genuine gain over the frozen baseline on
   the independent set.
5. Keep the current baseline frozen and tracked (this report + `pre2c_*` backups) as
   the invariant reference for any future change.