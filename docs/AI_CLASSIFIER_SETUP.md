# Adding a real trained AI-image classifier

This is the piece that closes the gap the rest of ForenSight can't: every
other signal (noise floor, ELA, texture, spectral, metadata) is a
hand-written statistical heuristic, not a model that learned from examples.
A trained classifier is what catches a **fully AI-generated or AI-edited
image with its metadata stripped** — the one case nothing else here can.

## Current status: ENABLED, validated against real photos

`models/ai_classifier/` now contains `prithivMLmods/deepfake-detector-model-v1`
(SigLIP-based) and is **enabled by default**. This replaced an earlier
candidate (`dima806/deepfake_vs_real_image_detection`) that was tested and
rejected — see the comparison below, both results are real, not projected.

**Test 1 — known Gemini-edited photo (hairstyle + blouse swap):**

| | dima806 (rejected) | prithivMLmods (current) |
|---|---|---|
| AI-probability | 0.08% (confidently wrong) | 95% (correct) |
| Fused score, metadata intact | 90% FAKE (metadata alone) | 90% FAKE |
| Fused score, metadata stripped | 13.8% (worse than no classifier at all) | 56.4% UNCERTAIN (correctly suspicious) |

The old model's own README explained the failure in advance: it's trained
on ~3-year-old face-swap data and warns of concept drift against modern
generators. The new model generalizes to this kind of whole-image
generative edit.

**Test 2 — false-positive check on 3 real, unedited photos the user
provided (plant close-up, two book-page photos):** the plant photo was
correctly read as real by both the classifier and the fused score. Both
book photos were **wrongly flagged** (classifier: 82% and unknown-but-high
AI-probability; fused verdict reached FAKE 66% on one). Root cause: flat,
evenly-lit printed paper resembles the "too smooth to be a real camera
sensor" pattern both the classifier and the pre-existing noise-floor
heuristic were tuned to catch — unrelated to AI involvement.

**Mitigation shipped:** `pipeline/forensics/document_detector.py` detects
document/text-page photos (colorfulness + text-edge-density heuristic,
calibrated on those same 4 real photos) and skips the classifier signal
for them. Re-tested after the fix: both book photos moved from false FAKE
to UNCERTAIN — no longer confidently wrong, though not fully clean, because
the noise-floor heuristic's own blind spot on flat paper was deliberately
left untouched (out of scope for this fix; still a known, open limitation).

## Known limitations, stated plainly

- Document-detection is a 4-sample calibration, not a validated classifier.
  A colorful infographic or illustrated page could evade it; a heavily
  desaturated real photo could in theory false-trigger it. Revisit
  thresholds if more real examples surface either failure mode.
- The noise-floor heuristic's false-positive risk on flat/paper-like
  surfaces still exists and was intentionally not addressed here — it
  affects the *heuristic* pipeline, independent of the classifier.
- This classifier's own generalization limits (training data, target
  domain) haven't been independently audited beyond the tests above —
  treat continued monitoring as part of normal use, not a one-time check.

## If you want to try yet another / newer model later

Look for something trained on **recent** generators (2024-2026 diffusion
models), and always test — don't just trust published benchmark numbers,
exactly as this process demonstrated twice now. Get either:
- a single `.onnx` file (lightest — `onnxruntime` is already installed,
  no extra setup), or
- a local Hugging Face folder (`config.json` + weights +
  `preprocessor_config.json`) via `huggingface-cli download <repo_id>
  --local-dir ./folder` on a machine with internet access, then zip and
  upload it here.

I'll load it, run it against real examples (yours or newly provided ones)
before touching any config default, and report the honest result either
way — same process as above.

