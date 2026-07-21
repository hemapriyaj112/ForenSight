"""pipeline/forensics/ela.py

Error Level Analysis (ELA).

Standard forensic technique: re-save the image at a known JPEG quality and
diff it against the original. Regions that were edited/spliced/inpainted
later than the rest of the image were compressed a different number of
times than their surroundings, so they "light up" differently in the
error map. It's cheap, needs no trained model, and is one of the most
widely used first-pass tools in real image-forensics workflows (e.g.
FotoForensics).

Note this detects *local editing inconsistency*, which is common in
AI-inpainted or spliced images. A fully AI-generated image (nothing
pasted in) can have a uniform ELA response — that's expected and is why
this is only one signal among several, not a standalone verdict.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np


def _ela_block_grid(image_rgb: np.ndarray, quality: int, block: int):
    """Shared helper: returns (ela_gray, block_means_grid, rows, cols)."""
    from PIL import Image

    original = Image.fromarray(image_rgb.astype(np.uint8))
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)

    diff = np.abs(image_rgb.astype(np.float32) - recompressed)
    ela_gray = diff.mean(axis=-1)

    h, w = ela_gray.shape
    rows = h // block
    cols = w // block
    grid = np.zeros((rows, cols), dtype=np.float32)
    for ry in range(rows):
        y = ry * block
        for cx in range(cols):
            x = cx * block
            grid[ry, cx] = float(ela_gray[y : y + block, x : x + block].mean())

    return ela_gray, grid, rows, cols


def error_level_analysis(image_rgb: np.ndarray, quality: int = 90) -> dict[str, Any]:
    """Run ELA on an HxWx3 uint8 RGB array.

    Returns
    -------
    dict with keys: score (0-1), findings (list[str]), heatmap (HxWx3 uint8)
    """
    block = 16
    ela_gray, grid, rows, cols = _ela_block_grid(image_rgb, quality, block)

    findings: list[str] = []
    if rows * cols < 4:
        # Image too small to block-analyse meaningfully
        score = 0.3
    else:
        block_means_arr = grid.flatten()
        median = float(np.median(block_means_arr))
        p95 = float(np.percentile(block_means_arr, 95))
        # Coefficient-of-variation-like ratio: how much do the "hottest"
        # blocks stand out from the typical block? Spliced/edited regions
        # produce a few strongly elevated outlier blocks.
        spread_ratio = (p95 - median) / (median + 1e-6)
        score = float(np.clip(spread_ratio / 6.0, 0.0, 1.0))

        if spread_ratio > 4.0:
            findings.append(
                "Certain regions of the image show a noticeably different "
                "compression history than the rest, consistent with local "
                "editing, splicing, or AI inpainting."
            )
        elif spread_ratio < 1.0:
            findings.append(
                "Compression-error levels are fairly uniform across the image "
                "(no obvious sign of a pasted-in or locally re-edited region)."
            )

    # Normalise heatmap for display
    if ela_gray.max() > 0:
        norm = ela_gray / ela_gray.max()
    else:
        norm = ela_gray
    heatmap = _apply_colormap(norm)

    return {"score": score, "findings": findings, "heatmap": heatmap}


def ela_suspicion_grid(image_rgb: np.ndarray, quality: int = 90, block: int = 16) -> dict[str, Any]:
    """
    Per-block version of ELA, for localizing *where* the compression-error
    inconsistency lives rather than one whole-image score.

    Returns
    -------
    dict with keys: grid (rows x cols float array, 0-1 suspicion),
    block (int), rows (int), cols (int)
    """
    _, grid, rows, cols = _ela_block_grid(image_rgb, quality, block)
    if rows * cols < 4:
        return {"grid": None, "block": block, "rows": rows, "cols": cols}

    median = float(np.median(grid))
    mad = float(np.median(np.abs(grid - median)))
    robust_std = max(mad * 1.4826, 1e-3)

    # Suspicion rises for blocks whose compression-error level stands out
    # (in either direction) from the photo's own typical block — a locally
    # pasted/AI-edited/re-saved patch was compressed a different number of
    # times than its surroundings, so it reads as an outlier either way.
    z = np.abs(grid - median) / robust_std
    suspicion = np.clip((z - 1.0) / 2.5, 0.0, 1.0).astype(np.float32)

    return {"grid": suspicion, "block": block, "rows": rows, "cols": cols}


def _apply_colormap(gray: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4.0 * gray - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * gray - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * gray - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
