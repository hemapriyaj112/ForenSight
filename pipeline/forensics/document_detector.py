"""pipeline/forensics/document_detector.py

Lightweight heuristic to flag when an uploaded image is a photo of a
document/page of text (book, notes, receipt, screenshot of text, etc.)
rather than an ordinary photo of a person/object/scene.

WHY THIS EXISTS
---------------
Real-world testing found that both the noise-floor heuristic and a trained
AI-vs-real image classifier misread genuine document photos as suspicious:
flat, evenly-lit printed paper looks a lot like the "too smooth to be a
real camera sensor" signature both were tuned to catch in photographs of
scenes/people. A real photo of a book page with highlighter annotations
was flagged 82% AI-probability by the classifier and 90%+ suspicious by
the noise-floor check, purely because of what documents naturally look
like, not because of any AI involvement.

This module only gates the trained-classifier signal (per an explicit
decision to leave the noise/ELA/etc. heuristics untouched for now — that's
a separate, still-open limitation, not something this module claims to fix).

METHOD
------
Calibrated (n=4 real photos: 2 documents, 2 ordinary photos — a small
sample, treat this as a first-pass heuristic, not a validated classifier):

- **Colorfulness** (Hasler–Süsstrunk metric): documents scored 11-17,
  ordinary photos scored 42-47 in our test set — a large, clean gap.
  Printed text pages are overwhelmingly white/black/gray with limited
  chromatic range; real-world photos of people/objects/nature have far
  more color variety.
- **Sharp-edge density**: documents have a moderate-to-high density of
  small, high-contrast edges (individual letterforms) against an
  otherwise flat background — used as a secondary corroborating signal,
  not standalone (a grayscale/monochrome photo alone could have low
  colorfulness without being a document).

KNOWN LIMITATIONS (be upfront about these)
-------------------------------------------
- A colorful infographic, glossy magazine page, or illustrated children's
  book could evade this (high colorfulness despite being "text").
- A genuinely monochrome or heavily desaturated photo (fog, night shot,
  black-and-white photography) could trigger a false "is_document" if it
  also happens to have dense small-scale edges — expected to be rare, but
  not tested against such cases.
- This is a 4-sample calibration, not a validated model. Treat as a
  reasonable first-pass gate, revisit thresholds if more real examples
  surface false positives/negatives in either direction.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_COLORFULNESS_CUTOFF = 28.0
_SHARP_EDGE_CUTOFF = 0.03


def looks_like_document(image_rgb: np.ndarray) -> dict[str, Any]:
    img = image_rgb.astype(np.float32)

    maxc = np.max(img / 255.0, axis=-1)
    minc = np.min(img / 255.0, axis=-1)

    gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
    gy, gx = np.gradient(gray)
    gradmag = np.sqrt(gx ** 2 + gy ** 2)
    sharp_edge_fraction = float(np.mean(gradmag > 25))

    rg = img[..., 0] - img[..., 1]
    yb = 0.5 * (img[..., 0] + img[..., 1]) - img[..., 2]
    colorfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

    is_document = colorfulness < _COLORFULNESS_CUTOFF and sharp_edge_fraction > _SHARP_EDGE_CUTOFF

    # Rough confidence: how far past both cutoffs, squashed to 0-1 — purely
    # for display/debugging, not used in any score fusion.
    color_margin = max(0.0, (_COLORFULNESS_CUTOFF - colorfulness) / _COLORFULNESS_CUTOFF)
    edge_margin = max(0.0, min(1.0, (sharp_edge_fraction - _SHARP_EDGE_CUTOFF) / _SHARP_EDGE_CUTOFF))
    confidence = float(np.clip((color_margin + edge_margin) / 2, 0.0, 1.0)) if is_document else 0.0

    return {
        "is_document": is_document,
        "confidence": confidence,
        "colorfulness": round(colorfulness, 2),
        "sharp_edge_fraction": round(sharp_edge_fraction, 4),
    }
