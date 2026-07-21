"""pipeline/forensics/noise_residual.py

Sensor-noise-floor analysis.

Every real camera sensor injects photon/read noise into an image, even in
perfectly flat, well-lit areas (a cheek, a wall, a blurred background).
This noise survives moderate JPEG compression and is present at a
measurable level (roughly std 1.5-8 on a 0-255 grayscale scale for
typical phone/camera photos).

AI generators (GANs, diffusion models) do not simulate this stochastic
per-pixel sensor process — flat regions in generated images (skin,
background, fabric) are often synthesized almost perfectly smooth, with
a noise floor far below what any real sensor produces. This is one of
the most reliable and hardest-to-fake forensic signals for exactly the
case that slips past texture/frequency/ELA checks: clean, well-lit,
professionally "photographed"-looking AI portraits.

Caveats (kept conservative / never used alone):
  - Heavy noise-reduction / beautify filters on a real photo can also
    suppress this noise, causing a false positive.
  - Extremely low-resolution or heavily downscaled images naturally
    have less visible per-pixel noise.
These are reflected in a moderate weight rather than an outright verdict.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _compute_blocks(
    image_rgb: np.ndarray, block: int
) -> tuple[list[tuple[int, int, float, float]], int, int]:
    """
    Shared block-decomposition helper used by both the global noise-floor
    score and the per-block localization grid, so the two stay consistent.

    Returns (blocks, rows, cols) where each block tuple is
    (row_index, col_index, gradient_mean, residual_std).
    """
    gray = (0.299 * image_rgb[..., 0]
            + 0.587 * image_rgb[..., 1]
            + 0.114 * image_rgb[..., 2]).astype(np.float32)
    h, w = gray.shape

    denoised = _median_filter5(gray)
    residual = gray - denoised

    gy, gx = np.gradient(gray)
    gradmag = np.sqrt(gx ** 2 + gy ** 2)

    rows = (h - block) // block
    cols = (w - block) // block

    blocks: list[tuple[int, int, float, float]] = []
    for ry in range(rows):
        y = ry * block
        for cx in range(cols):
            x = cx * block
            g = float(gradmag[y : y + block, x : x + block].mean())
            r = float(residual[y : y + block, x : x + block].std())
            blocks.append((ry, cx, g, r))
    return blocks, rows, cols


def analyse_noise_floor(image_rgb: np.ndarray, block: int = 20) -> dict[str, Any]:
    """
    Measure the residual (denoised-subtracted) pixel noise in the flattest
    regions of the image — the regions where any noise present must be
    sensor noise, not real texture/edges.

    Returns
    -------
    dict with keys: score (0-1, higher = more suspicious of AI-generation),
    findings (list[str]), noise_floor (float, the measured std)
    """
    h, w = image_rgb.shape[:2]
    if h < block * 3 or w < block * 3:
        return {"score": 0.3, "findings": [], "noise_floor": None}

    blocks, rows, cols = _compute_blocks(image_rgb, block)
    if len(blocks) < 6:
        return {"score": 0.3, "findings": [], "noise_floor": None}

    by_gradient = sorted(blocks, key=lambda t: t[2])
    n_flat = max(3, len(by_gradient) // 10)  # flattest ~10% of blocks
    flat_residuals = [r for _, _, _, r in by_gradient[:n_flat]]
    noise_floor = float(np.median(flat_residuals))

    # Logistic mapping: noise_floor near 0 -> high suspicion; >= ~2 -> low.
    score = float(1.0 / (1.0 + np.exp((noise_floor - 1.1) / 0.35)))
    score = float(np.clip(score, 0.0, 1.0))

    findings: list[str] = []
    if noise_floor < 0.6:
        findings.append(
            "Flat regions of the image (background/skin/fabric) show almost "
            "no sensor noise — real camera photos, even well-lit ones, "
            "retain measurable pixel-level noise that this image lacks."
        )
    elif noise_floor > 2.0:
        findings.append(
            "Flat regions show a natural level of sensor noise, consistent "
            "with a real camera photo."
        )

    return {"score": score, "findings": findings, "noise_floor": noise_floor}


def noise_suspicion_grid(image_rgb: np.ndarray, block: int = 16) -> dict[str, Any]:
    """
    Per-block version of the noise-floor check, for localizing *where* in
    the image the noise floor looks unnatural — rather than one score for
    the whole photo.

    Only blocks that are locally flat (low gradient — skin, background,
    fabric, sky) are scored; edges/high-detail blocks can't be judged by
    this method (real edges suppress the residual signal too), so they're
    marked as not-assessable rather than guessed at.

    Returns
    -------
    dict with keys:
      grid (rows x cols float array, 0-1 suspicion, NaN where not assessable)
      assessable (rows x cols bool array)
      block (int), rows (int), cols (int)
      reference_floor (float) — the image's own flat-region noise floor,
        used as the "expected" baseline that each block is compared against
    """
    h, w = image_rgb.shape[:2]
    if h < block * 4 or w < block * 4:
        return {"grid": None, "assessable": None, "block": block,
                "rows": 0, "cols": 0, "reference_floor": None}

    blocks, rows, cols = _compute_blocks(image_rgb, block)
    if rows < 3 or cols < 3:
        return {"grid": None, "assessable": None, "block": block,
                "rows": rows, "cols": cols, "reference_floor": None}

    gradients = np.array([g for _, _, g, _ in blocks])
    flat_cutoff = float(np.percentile(gradients, 35))

    flat_residuals = [r for _, _, g, r in blocks if g <= flat_cutoff]
    reference_floor = float(np.median(flat_residuals)) if flat_residuals else 1.1

    # Robust spread (MAD) across the image's own flat blocks. This lets us
    # flag blocks that are statistical *outliers* within this photo — not
    # merely blocks that sit below the median, which half of any real
    # photo's flat blocks would do by construction.
    if flat_residuals:
        mad = float(np.median(np.abs(np.asarray(flat_residuals) - reference_floor)))
    else:
        mad = 0.0
    robust_std = max(mad * 1.4826, 0.05)  # normal-consistent scale, floor to avoid /~0

    grid = np.full((rows, cols), np.nan, dtype=np.float32)
    assessable = np.zeros((rows, cols), dtype=bool)

    for ry, cx, g, r in blocks:
        if g > flat_cutoff:
            continue  # can't reliably judge noise under real texture/edges
        assessable[ry, cx] = True
        # Suspicion rises only for blocks that are markedly smoother than
        # the *typical* flat block in this same photo (a few robust-sigma
        # below the pack), which is what a locally smoothed/inpainted/
        # AI-edited patch looks like next to genuine camera noise.
        z = (reference_floor - r) / robust_std
        grid[ry, cx] = float(np.clip((z - 1.0) / 2.5, 0.0, 1.0))

    return {
        "grid": grid,
        "assessable": assessable,
        "block": block,
        "rows": rows,
        "cols": cols,
        "reference_floor": reference_floor,
    }


def _median_filter5(gray: np.ndarray) -> np.ndarray:
    """Lightweight 5x5 median filter without a scipy dependency requirement."""
    try:
        from scipy.ndimage import median_filter
        return median_filter(gray, size=5)
    except Exception:
        # Fallback: simple box blur if scipy isn't available.
        pad = np.pad(gray, 2, mode="reflect")
        out = np.zeros_like(gray)
        for dy in range(5):
            for dx in range(5):
                out += pad[dy : dy + gray.shape[0], dx : dx + gray.shape[1]]
        return out / 25.0
