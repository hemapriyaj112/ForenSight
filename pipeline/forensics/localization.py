"""pipeline/forensics/localization.py

Turns two of our existing per-image signals (sensor-noise-floor and error-
level-analysis) into a single *spatial* map, so the app can point at *where*
in a photo something looks inconsistent — not just give a whole-image
verdict.

IMPORTANT — what this can and cannot do
----------------------------------------
This localization only makes sense for **partial edits**: a real photo that
had one region (a face swap, a changed hairstyle, an inpainted object)
altered by AI while the rest of the frame is untouched. In that case the
edited region carries a different noise floor / compression history than
its surroundings, and that boundary is exactly what these two signals catch.

For a **fully AI-generated image**, there is no untouched region to compare
against — the whole frame was synthesized together, so it tends to look
uniformly "off" (or uniformly clean) rather than having one anomalous patch.
This module explicitly detects that case (a large fraction of the image
scoring as suspicious) and reports it as "spread across the whole image"
rather than fabricating a specific region to point at.

This is a *display/explanation* layer built on signals that are already
part of the fused score (noise + ela) — it does not add a new vote to the
verdict, so it can't change existing calibration/thresholds.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


_BLOCK = 16
_HOT_THRESHOLD = 0.55          # grid value above this counts as "suspicious"
_GLOBAL_FRACTION_CUTOFF = 0.55  # share of assessable blocks flagged hot ->
                                # treat as "spread across the whole image"
                                # rather than a localized region


def build_suspicious_regions(image_rgb: np.ndarray, block: int = _BLOCK) -> dict[str, Any]:
    """
    Parameters
    ----------
    image_rgb: HxWx3 uint8 array

    Returns
    -------
    dict with keys:
      heatmap (HxWx3 uint8 or None) — combined suspicion map blended on the
        original image, ready to display
      is_localized (bool) — True if the suspicious area is concentrated in
        one part of the image (a plausible local edit); False if it's
        spread broadly (consistent with whole-image generation) or if
        nothing stood out
      suspicious_fraction (float) — share of assessable blocks flagged hot
      regions (list[dict]) — up to 2 largest suspicious clusters, each with
        bbox_px (y0, y1, x0, x1), position_desc (str), area_fraction (float)
      summary (str) — one plain-language sentence for the UI
    """
    from pipeline.forensics import ela as _ela
    from pipeline.forensics import noise_residual as _noise

    h, w = image_rgb.shape[:2]

    noise_out = _noise.noise_suspicion_grid(image_rgb, block=block)
    ela_out = _ela.ela_suspicion_grid(image_rgb, quality=90, block=block)

    noise_grid = noise_out.get("grid")
    ela_grid = ela_out.get("grid")

    if noise_grid is None and ela_grid is None:
        return {
            "heatmap": None, "is_localized": False, "suspicious_fraction": 0.0,
            "regions": [], "summary": "Image too small for region-level analysis.",
        }

    rows = noise_out.get("rows") or ela_out.get("rows")
    cols = noise_out.get("cols") or ela_out.get("cols")

    # Align to the smaller common grid shape (block sizes/rounding can differ
    # by a row/col at the edges).
    if noise_grid is not None and ela_grid is not None:
        r = min(noise_grid.shape[0], ela_grid.shape[0])
        c = min(noise_grid.shape[1], ela_grid.shape[1])
        noise_grid = noise_grid[:r, :c]
        ela_grid = ela_grid[:r, :c]
        assessable = noise_out["assessable"][:r, :c]
        # Where noise couldn't judge a block (real edge/texture), fall back
        # to ELA alone for that block.
        combined = np.where(
            assessable, np.fmax(np.nan_to_num(noise_grid, nan=0.0), ela_grid), ela_grid
        )
    elif noise_grid is not None:
        combined = np.nan_to_num(noise_grid, nan=0.0)
        r, c = combined.shape
    else:
        combined = ela_grid
        r, c = combined.shape

    total_blocks = r * c
    if total_blocks == 0:
        return {
            "heatmap": None, "is_localized": False, "suspicious_fraction": 0.0,
            "regions": [], "summary": "Image too small for region-level analysis.",
        }

    hot_mask = combined >= _HOT_THRESHOLD
    suspicious_fraction = float(hot_mask.sum()) / float(total_blocks)

    regions: list[dict[str, Any]] = []
    is_localized = False
    summary = ""

    if suspicious_fraction < 0.02:
        summary = (
            "No specific region stood out — noise and compression levels "
            "look consistent across the photo."
        )
    elif suspicious_fraction >= _GLOBAL_FRACTION_CUTOFF:
        summary = (
            "Suspicious characteristics are spread broadly across the image "
            "rather than confined to one spot — consistent with the whole "
            "image being AI-generated (or uniformly filtered), not a small "
            "local edit."
        )
    else:
        clusters = _connected_components(hot_mask)
        clusters.sort(key=lambda cl: len(cl), reverse=True)
        for cluster in clusters[:2]:
            ys = [p[0] for p in cluster]
            xs = [p[1] for p in cluster]
            ry0, ry1 = min(ys), max(ys) + 1
            cx0, cx1 = min(xs), max(xs) + 1
            bbox_px = (ry0 * block, min(ry1 * block, h), cx0 * block, min(cx1 * block, w))
            area_fraction = len(cluster) / total_blocks
            if area_fraction < 0.01:
                continue
            position_desc = _describe_position(ry0, ry1, cx0, cx1, r, c)
            regions.append({
                "bbox_px": bbox_px,
                "position_desc": position_desc,
                "area_fraction": round(area_fraction, 3),
            })

        if regions:
            is_localized = True
            top = regions[0]
            pct = round(top["area_fraction"] * 100)
            summary = (
                f"A concentrated suspicious region was found in the "
                f"{top['position_desc']} of the image (~{pct}% of the frame), "
                "where noise/compression characteristics don't match the "
                "rest of the photo — consistent with a local AI edit or "
                "inpainting in that specific area."
            )
        else:
            summary = (
                "A few scattered blocks looked slightly inconsistent, but "
                "nothing formed a clear, concentrated region worth pointing to."
            )

    heatmap = _render_heatmap(image_rgb, combined, block)

    return {
        "heatmap": heatmap,
        "is_localized": is_localized,
        "suspicious_fraction": round(suspicious_fraction, 3),
        "regions": regions,
        "summary": summary,
    }


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connectivity flood fill over a boolean grid; no scipy dependency."""
    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    clusters: list[list[tuple[int, int]]] = []

    for sy in range(rows):
        for sx in range(cols):
            if not mask[sy, sx] or visited[sy, sx]:
                continue
            stack = [(sy, sx)]
            visited[sy, sx] = True
            comp: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            clusters.append(comp)
    return clusters


def _describe_position(ry0: int, ry1: int, cx0: int, cx1: int, rows: int, cols: int) -> str:
    """Map a block-grid bounding box to a plain 3x3-grid position phrase."""
    cy = (ry0 + ry1) / 2.0
    cx = (cx0 + cx1) / 2.0

    v_third = rows / 3.0
    h_third = cols / 3.0

    vertical = "upper" if cy < v_third else ("lower" if cy > 2 * v_third else "middle")
    horizontal = "left" if cx < h_third else ("right" if cx > 2 * h_third else "center")

    if vertical == "middle" and horizontal == "center":
        return "center"
    if vertical == "middle":
        return f"{horizontal} side"
    if horizontal == "center":
        return f"{vertical} area"
    return f"{vertical}-{horizontal}"


def _render_heatmap(image_rgb: np.ndarray, grid: np.ndarray, block: int) -> Optional[bytes]:
    from PIL import Image
    import io as _io

    h, w = image_rgb.shape[:2]
    rows, cols = grid.shape

    # Upsample the coarse block grid to full resolution with a smooth
    # (bilinear-ish) blow-up so the overlay doesn't look like hard tiles.
    small = np.clip(grid, 0.0, 1.0)
    small_img = Image.fromarray((small * 255).astype(np.uint8))
    big = small_img.resize((cols * block, rows * block), Image.BILINEAR)
    big_arr = np.array(big, dtype=np.float32) / 255.0

    # Pad/crop to exact image size (block grid may be a few px smaller).
    canvas = np.zeros((h, w), dtype=np.float32)
    bh, bw = big_arr.shape
    canvas[: min(bh, h), : min(bw, w)] = big_arr[: min(bh, h), : min(bw, w)]

    # Only tint pixels that clear a visibility floor, so calm images stay
    # mostly un-tinted instead of a faint wash over everything.
    alpha = np.clip((canvas - 0.35) / 0.65, 0.0, 1.0) * 0.65

    heat_color = np.zeros((h, w, 3), dtype=np.float32)
    heat_color[..., 0] = 255.0  # red tint for "suspicious"
    heat_color[..., 1] = 60.0
    heat_color[..., 2] = 60.0

    base = image_rgb.astype(np.float32)
    blended = base * (1 - alpha[..., None]) + heat_color * alpha[..., None]
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    buf = _io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return buf.getvalue()
