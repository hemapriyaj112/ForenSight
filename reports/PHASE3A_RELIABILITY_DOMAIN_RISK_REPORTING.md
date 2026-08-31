# ForenSight — Phase 3A: Evidence Reliability & Domain-Risk Reporting Layer

## 1. Executive Summary

Phase 3A adds an **instrumentation / forensic-reporting layer** on top of the
existing ForenSight production pipeline. It makes the system explicitly report
*what* evidence exists, *how strong* that evidence is, whether it is available,
whether the input carries domain-risk indicators, whether modalities agree or
disagree, and whether each detector should currently be trusted — **without**
letting any of this new information modify the production verdict.

The production fusion formula, production thresholds, and all 15 production
verdicts remain byte-identical to the Phase 2C frozen baseline
(`TP=6, TN=4, FP=0, FN=5`; `accuracy=0.6667`, `precision=1.0000`,
`FAKE recall=0.5455`, `F1=0.7059`).

Three concepts are now kept strictly separate:

1. **DETECTOR OUTPUT** — what did the detector say?
2. **EVIDENCE RELIABILITY** — how trustworthy is that output in this media context?
3. **FINAL FORENSIC VERDICT** — what does the complete system conclude?

This report documents the reliability-state design, domain-risk indicators,
evidence-vote representation, cross-modal agreement reporting, AASIST temporal
reporting, GenConViT coverage reporting, the per-video results table, the UI
changes, the test results, the production-invariance verification, limitations,
and the gate to Phase 3B.

## 2. Scope and Constraints

Phase 3A is **reporting-only**. It does **not**:

- add another detector
- train or fine-tune anything
- tune thresholds using the 15-video corpus
- change the production fusion formula or any production thresholds
- allow GenConViT or AASIST into the final fusion
- replace the existing production verdict or create weighted/learned fusion
- automatically upgrade/downgrade REAL/UNCERTAIN/FAKE because of reliability states
- treat AASIST SPOOF as proof of whole-video forgery, or BONAFIDE as proof of authenticity
- treat persistent AASIST SPOOF or compression/WhatsApp suspicion as proven causation

The following configuration remains unchanged:

```yaml
genconvit:
  use_in_final_fusion: false

aasist:
  use_in_final_fusion: false
```

All new code lives in analysis/reporting paths (`pipeline/analysis/`, a Phase 3A
reporting script, and display-only UI additions). No production detector or
fuser code was modified.

## 3. Frozen Production Baseline

The production baseline (unchanged) is:

```text
TP = 6
TN = 4
FP = 0
FN = 5

accuracy  = 0.6667
precision = 1.0000
FAKE recall = 0.5455
F1 = 0.7059
FPR = 0.0000
FNR = 0.4545
```

All 15 `filename / ground_truth / final_verdict / fusion_score` values are
verified identical (see Section 13).

## 4. Reliability State Design

The five evidence reliability states proposed in Phase 2C are implemented as
**evidence reliability states**, not prediction classes:

| State | Meaning |
|---|---|
| `HIGH` | Multiple independent, corroborating, trusted modality readings are available and consistent; no domain-risk gate. |
| `MODERATE` | Some corroboration exists, or a mixed/partial agreement is present, but evidence is not fully consistent or is single-sourced. |
| `LOW` | Only weak/limited evidence is available (e.g. single modality with poor coverage) with no strong tie-breaker. |
| `DOMAIN-RISK` | A strong media-condition marker (low bitrate, re-encode, missing modality, etc.) caps how much any detector output can be trusted. Caps both FAKE- and REAL-oriented confidence. |
| `UNAVAILABLE` | No usable detector evidence at all. |

### Rule provenance

Each rule is explicitly tagged as one of:

- **validated** — grounded in domain knowledge (e.g. missing modality removes that modality's evidence).
- **observed** — repeatedly seen in the 15-video corpus.
- **provisional** — extrapolated from observations; requires independent validation before any threshold is used for decision-making.

The state assignment function `assign_reliability_state` uses a priority
ordering (nothing available → UNAVAILABLE; domain-risk present → DOMAIN-RISK;
multi-source corroboration → HIGH; partial/mixed → MODERATE; single weak →
LOW). These are reporting labels, not prediction classes, and are **not**
tuned on the corpus.

## 5. Domain-Risk Indicators

A structured set of domain-risk flags was implemented in `analyse_domain_risk`.
Each flag carries `flag`, `severity`, `observed_value`, `reason`, and
`evidence_source`. Indicators use measured metadata/detector outputs and are
worded as **domain risk**, never as deepfake evidence.

| Flag | Severity | Basis |
|---|---|---|
| `LOW_BITRATE` | HIGH | ffprobe bitrate ≤ ~1200 kbps (observed) |
| `LOW_RESOLUTION` | MEDIUM | resolution < 720p (observed) |
| `WHATSAPP_OR_CHAT_RELINK` | HIGH | filename containing "whatsapp"/"chat" (observed) |
| `MISSING_AUDIO` | HIGH | no audio stream (validated) |
| `MISSING_FACE` | HIGH | no usable face frames (validated) |
| `LOW_FACE_COVERAGE` | MEDIUM | face coverage < 20% of sampled frames (observed) |
| `NEURAL_AUDIO_HEURISTIC_DISAGREEMENT` | MEDIUM | AASIST=SPOOF while heuristic reads natural (observed) |
| `NEURAL_MODALITY_DISAGREEMENT` | MEDIUM | GenConViT and AASIST map opposite (observed) |
| `PERSISTENT_AASIST_SPOOF_ON_RISKY_MEDIA` | HIGH/MEDIUM | persistent high spoof on faceless and/or low-bitrate media (observed) |

The overall `domain_risk.state` becomes `DOMAIN-RISK` when ≥1 `HIGH`-severity
flag fires; `MODERATE` when only `MEDIUM`/`LOW` flags fire; `HIGH` when no
flags fire.

## 6. Evidence-Vote Representation

Each supported modality gets a reporting-only representation:

```json
{
  "modality": "aasist",
  "evidence_direction": "FAKE_EVIDENCE",
  "evidence_strength": "HIGH",
  "reliability_state": "DOMAIN-RISK"
}
```

`evidence_direction` ∈ {`FAKE_EVIDENCE`, `REAL_EVIDENCE`, `NO_EVIDENCE`}.
This is **not** a verdict. The `evidence_strength` is derived from the
detector's actual score offset from 0.5 and is not an invented confidence.

## 7. Cross-Modal Agreement Reporting

For each video, the cross-modal agreement module reports structured metadata
rather than a single score:

```json
{
  "agreement": {
    "comparisons": {
      "genconvit_vs_aasist": "DISAGREE",
      "aasist_vs_audio_heuristic": "DISAGREE",
      "genconvit_vs_texture": "AGREE"
    },
    "neural_agreement": "NEURAL_DISAGREEMENT",
    "neural_vs_heuristic_disagreement": true,
    "agree_count": 1,
    "disagree_count": 2,
    "summary": "MULTIMODAL_DISAGREEMENT"
  }
}
```

Comparisons are only included when both modalities are actually available.

## 8. AASIST Temporal Reporting

AASIST segment-level evidence (already produced in Phase 2C) is surfaced
verbatim without changing aggregation or adding a decision threshold:

```json
{
  "mean_spoof": 0.996,
  "min_spoof": 0.98,
  "max_spoof": 0.999,
  "window_count": 26,
  "segment_pattern": "persistent_high_spoof",
  "high_spoof_window_count": 26,
  "low_spoof_window_count": 0,
  "high_spoof_fraction": 1.0,
  "timestamps": [0.0, 4.037, ...],
  "detail": "AASIST produced SPOOF-oriented audio evidence across all 26 analyzed windows..."
}
```

The pattern classifier (persistent_high_spoof / persistent_low_spoof /
mixed_or_localized / no_windows) mirrors the existing Phase 2C logic.

## 9. GenConViT Coverage Reporting

GenConViT coverage information from Phase 2A.1 is exposed:

`coverage_status`, `face_coverage`, `frames_used`, `frames_requested`,
`average_face_confidence`, `min_face_confidence`, `max_face_confidence` —
mapped into `GOOD` / `PARTIAL` / `LOW` / `NO_FACE` using the existing coverage
classification (no new threshold introduced, no GenConViT change).

## 10. Per-Video Reliability Results

The full per-video data is in
`reports/phase3a_reliability_evidence.json` and `reports/phase3a_reliability_evidence.csv`.
The 15-video corpus table is in Section 17.

## 11. UI / Reporting Changes

The Streamlit video tab now includes a **"Evidence Reliability & Domain-Risk"**
section that is visually distinct from the **Final production verdict** badge.

Layout (as specified):

```
FINAL VERDICT
FAKE / REAL / UNCERTAIN

Supporting Evidence (existing)
- Texture, Temporal, Provenance

Neural Evaluation Signals  (evaluation-only)
- GenConViT: FAKE / REAL / unavailable
- AASIST: SPOOF / BONAFIDE / unavailable

Evidence Reliability
- Overall: DOMAIN-RISK
- GenConViT: LOW
- AASIST: DOMAIN-RISK

Domain-Risk Indicators
- LOW_BITRATE
- WHATSAPP_OR_CHAT_RELINK
- NEURAL_AUDIO_HEURISTIC_DISAGREEMENT
```

Explicit guard text makes it impossible to confuse `AASIST SPOOF` with
`VIDEO FAKE`. The UI is display-only and does not alter verdict behavior.

## 12. Test Results

New unit tests: `tests/unit/test_phase3a_reliability.py` (35 tests) covering:

- Missing modality (no audio, no face)
- GenConViT coverage (GOOD / PARTIAL / LOW / NO_FACE)
- AASIST patterns (persistent_high_spoof / persistent_low_spoof / mixed_or_localized / no_windows)
- Domain-risk indicators (low bitrate, low resolution, WhatsApp/chat relink, missing modality, neural-vs-heuristic disagreement)
- Reliability states (HIGH / MODERATE / LOW / DOMAIN-RISK / UNAVAILABLE)
- **Evidence-vs-verdict separation** — explicitly asserting that
  `AASIST = SPOOF` does not modify `final_verdict`, and `GenConViT = FAKE`
  does not modify `final_verdict`.

### Full-suite result (`py -m pytest tests/ -q`)

```text
collected 303 items
298 passed, 5 failed
```

- **new tests passed**: 35
- **new tests failed**: 0
- **existing failures**: 5 (all in `test_app.py`, pre-existing — see below)
- **full-suite result**: 298 passed, 5 failed

The 5 failures are all in `tests/unit/test_app.py` and are **pre-existing**:
they fail in `main_app → _inject_theme()` because the test's streamlit stub
does not implement `streamlit.session_state`. They are unrelated to Phase 3A
and were failing before these changes. Phase 3A did not alter them.

## 13. Production Invariance Verification

The evaluation was compared against the Phase 2C frozen baseline:

```text
matches_phase2b_baseline: True
metric_deltas: {}
```

Computed (unchanged):

```text
confusion: {TP: 6, TN: 4, FP: 0, FN: 5}
accuracy: 0.6667
precision_fake: 1.0000
FAKE_RECALL: 0.5455
f1_fake: 0.7059
false_negative_rate: 0.4545
false_positive_rate: 0.0
fake_verdict_rate: 0.4
uncertain_rate: 0.2667
```

All 15 `filename / ground_truth / final_verdict / fusion_score` values remain
identical to the Phase 2C baseline (Section 17 table). No production verdict or
production score changed.

## 14. Limitations

- All rule boundaries here are **reporting** heuristics derived from the
  15-video corpus; they must not be used to tune any decision threshold.
- The `HIGH`, `MODERATE`, `LOW`, `DOMAIN-RISK`, `UNAVAILABLE` states are
  descriptive labels, not calibrated probabilities.
- GenConViT and AASIST remain **evaluation-only**; their reliability labels are
  report-only and do not enter the final verdict.
- Domain-risk detection is partly heuristic (e.g. WhatsApp/chat detection from
  filename patterns) and does not prove causation.
- No scores are calibrated; they must not be presented as true probabilities.

## 15. Conclusions

Phase 3A successfully adds an explicit reliability and domain-risk reporting
layer without changing the production behavior. ForenSight can now say, for
each detector, *what it observed*, *under which media conditions*, *with what
reliability*, and *how that evidence should be interpreted* — while the final
REAL / UNCERTAIN / FAKE verdict remains controlled exclusively by the existing
production fusion.

Calibration status:
```
NOT READY
```

The 15-video corpus continues to be treated as **analysis / evaluation
evidence**, not **threshold calibration data**. No genuine independent data
outside the original 15-video corpus was used in Phase 3A.

## 16. Gate to Phase 3B

Phase 3A is a prerequisite for **Phase 3B: controlled codec/re-encode domain
validation**. The reporting layer is now stable and separates DETECTOR OUTPUT
from EVIDENCE RELIABILITY from FINAL VERDICT. Before deciding whether any
evaluation signal (notably AASIST or GenConViT) may ever influence a verdict
in domain-specific cases, Phase 3B must gather an independent, per-domain
(telephony/WhatsApp-tier) held-out set and formally evaluate the reliability
framework — including whether any state boundaries should be revised. Phase 3B
must **not** build decision gating on the current 15-video corpus.

## 17. 15-Video Corpus Reliability Table

| Video | GT | Verdict | GenConViT | GC Cov. | AASIST | AASIST Pattern | Audio Heur. | DR State | Reliab. | Agreement | Evidence Summary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ai_noaud2.mp4 | FAKE | UNCERTAIN | NO_EVIDENCE | NO_FACE | NO_EVIDENCE | no_windows | n/a | DOMAIN-RISK | DOMAIN-RISK | INSUFFICIENT_DATA | no audio, no face |
| ai_vid_without_audio.mp4 | FAKE | FAKE | REAL_EVIDENCE | LOW | REAL_EVIDENCE | persistent_low_spoof | mean≈0.45 | MODERATE | MODERATE | MIXED_AGREEMENT | GC REAL (0.61), AASIST BONAFIDE (0.01), LOW_FACE_COVERAGE |
| avengers ai vid 1.mp4 | FAKE | REAL | REAL_EVIDENCE | GOOD | FAKE_EVIDENCE | persistent_high_spoof | low | DOMAIN-RISK | DOMAIN-RISK | MIXED_AGREEMENT | LOW_BITRATE, NEUR_HEUR, NEUR_MOD DISAGREE |
| avengers ai vid 2.mp4 | FAKE | UNCERTAIN | FAKE_EVIDENCE | GOOD | FAKE_EVIDENCE | persistent_high_spoof | low | DOMAIN-RISK | DOMAIN-RISK | MIXED_AGREEMENT | LOW_BITRATE, NEUR_HEUR DISAGREE |
| crackers_withaud.mp4 | FAKE | FAKE | FAKE_EVIDENCE | LOW | FAKE_EVIDENCE | mixed_or_localized | low | MODERATE | MODERATE | MIXED_AGREEMENT | LOW_FACE_COVERAGE, NEUR_HEUR DISAGREE |
| gemini ai video.mp4 | FAKE | FAKE | NO_EVIDENCE | NO_FACE | REAL_EVIDENCE | mixed_or_localized | low | DOMAIN-RISK | DOMAIN-RISK | PARTIAL_AGREEMENT | no face |
| girl walkin video.mp4 | FAKE | FAKE | REAL_EVIDENCE | PARTIAL | FAKE_EVIDENCE | persistent_high_spoof | low | MODERATE | MODERATE | MULTIMODAL_DISAGREEMENT | NEUR_MOD DISAGREE, NEUR_HEUR DISAGREE |
| girl_speaking video gemini.mp4 | FAKE | FAKE | REAL_EVIDENCE | PARTIAL | REAL_EVIDENCE | persistent_low_spoof | low | HIGH | MODERATE | MIXED_AGREEMENT | GC REAL (0.73), AASIST BONAFIDE |
| insta vid.mp4 | FAKE | REAL | FAKE_EVIDENCE | PARTIAL | FAKE_EVIDENCE | persistent_high_spoof | low | MODERATE | MODERATE | MIXED_AGREEMENT | NEUR_HEUR DISAGREE |
| real vid ai aud.mp4 | FAKE | REAL | FAKE_EVIDENCE | LOW | FAKE_EVIDENCE | persistent_high_spoof | low | MODERATE | MODERATE | MIXED_AGREEMENT | LOW_FACE_COVERAGE, NEUR_HEUR DISAGREE |
| south indian women speaking.mp4 | FAKE | FAKE | FAKE_EVIDENCE | LOW | REAL_EVIDENCE | persistent_low_spoof | low | MODERATE | MODERATE | MIXED_AGREEMENT | NEUR_MOD DISAGREE |
| WhatsApp …6.15.45 PM.mp4 | REAL | REAL | FAKE_EVIDENCE | LOW | FAKE_EVIDENCE | persistent_high_spoof | low | DOMAIN-RISK | DOMAIN-RISK | MIXED_AGREEMENT | LOW_BITRATE, WHATSAPP_RELINK, LOW_FACE_COV, NEUR_HEUR DISAGREE |
| WhatsApp …6.15.46 PM.mp4 | REAL | REAL | REAL_EVIDENCE | GOOD | FAKE_EVIDENCE | persistent_high_spoof | low | DOMAIN-RISK | DOMAIN-RISK | MIXED_AGREEMENT | WHATSAPP_RELINK, NEUR_HEUR, NEUR_MOD DISAGREE |
| real_vid_to_test_withaudio.mp4 | REAL | UNCERTAIN | NO_EVIDENCE | NO_FACE | REAL_EVIDENCE | persistent_low_spoof | low | DOMAIN-RISK | DOMAIN-RISK | PARTIAL_AGREEMENT | LOW_RESOLUTION, MISSING_FACE |
| vid_with_noaudio.mp4 | REAL | UNCERTAIN | NO_EVIDENCE | NO_FACE | NO_EVIDENCE | no_windows | n/a | DOMAIN-RISK | DOMAIN-RISK | INSUFFICIENT_DATA | LOW_RESOLUTION, MISSING_AUDIO, MISSING_FACE |

Notes:
- `GC Cov.` = GenConViT coverage (GOOD/PARTIAL/LOW/NO_FACE).
- `Audio Heur.` = production audio heuristic aggregate (≈ values shown for context).
- `DR State` = domain-risk state. `Reliab.` = overall evidence reliability.
- `Agreement` = cross-modal agreement summary.
- `Evidence Summary` highlights the dominant domain-risk flags and detector readings.

This table lets a reviewer immediately understand **why** a detector signal is
trustworthy, weak, unavailable, or risky for each of the 15 videos.
