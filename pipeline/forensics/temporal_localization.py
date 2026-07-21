"""pipeline/forensics/temporal_localization.py

Turns a per-frame fused-score sequence into a *temporal* map, so the app can
point at *when* in a video something looks inconsistent — not just give a
whole-video verdict. This is the direct time-axis analog of
pipeline/forensics/localization.py, which does the same job spatially
within a single frame.

IMPORTANT — what this can and cannot do
----------------------------------------
This only makes sense for **partial edits in time**: a real video where one
stretch (a face-swapped segment, an inserted/replaced shot) was altered
while the rest plays through untouched. In that case the edited segment's
frames carry a different fused score than their neighbours, and that's
exactly what contiguous-run detection below catches.

For a video that is **AI-generated start to finish** (e.g. a fully
synthetic talking-head clip), there is no untouched stretch to compare
against — every frame tends to score similarly (uniformly high or
uniformly clean) rather than having one anomalous segment. This module
explicitly detects that case (a large fraction of frames scoring hot) and
reports it as "spread across the whole video" rather than fabricating a
specific segment to point at — mirroring the spatial module's same caveat
for fully-generated images.

This is a *display/explanation* layer built on the per-frame fused scores
that already feed VideoDetector's overall average — it does not add a new
vote to the verdict, so it can't change existing calibration/thresholds.
"""
from __future__ import annotations

from typing import Any

_HOT_THRESHOLD = 0.55           # frame fused-score above this counts as "suspicious"
_GLOBAL_FRACTION_CUTOFF = 0.55  # share of frames flagged hot -> treat as
                                 # "spread across the whole video" rather
                                 # than a localized segment


def build_suspicious_timeline(
    frame_scores: list[float],
    timestamps: list[float],
) -> dict[str, Any]:
    """
    Parameters
    ----------
    frame_scores: per-frame fused fake-probability, in frame order
    timestamps:   matching per-frame timestamp in seconds

    Returns
    -------
    dict with keys:
      is_localized (bool) — True if suspicion is concentrated in one part
        of the timeline (a plausible local edit); False if it's spread
        broadly (consistent with whole-video generation) or if nothing
        stood out
      suspicious_fraction (float) — share of frames flagged hot
      segments (list[dict]) — up to 2 largest suspicious runs, each with
        start_sec, end_sec, frame_count, area_fraction (float)
      summary (str) — one plain-language sentence for the UI
    """
    n = len(frame_scores)
    if n == 0 or len(timestamps) != n:
        return {
            "is_localized": False, "suspicious_fraction": 0.0,
            "segments": [], "summary": "No per-frame data available for timeline analysis.",
        }

    hot = [s >= _HOT_THRESHOLD for s in frame_scores]
    suspicious_fraction = sum(hot) / n

    segments: list[dict[str, Any]] = []
    is_localized = False
    summary = ""

    if suspicious_fraction < 0.02:
        summary = (
            "No specific moment stood out — per-frame scores look "
            "consistent across the whole clip."
        )
    elif suspicious_fraction >= _GLOBAL_FRACTION_CUTOFF:
        summary = (
            "Suspicious frames are spread broadly across the timeline "
            "rather than confined to one stretch — consistent with the "
            "whole video being AI-generated, not a small local edit."
        )
    else:
        runs = _contiguous_runs(hot)
        runs.sort(key=len, reverse=True)
        for run in runs[:2]:
            i0, i1 = run[0], run[-1]
            area_fraction = len(run) / n
            if area_fraction < 0.01:
                continue
            segments.append({
                "start_sec": round(timestamps[i0], 2),
                "end_sec": round(timestamps[i1], 2),
                "frame_count": len(run),
                "area_fraction": round(area_fraction, 3),
            })

        if segments:
            is_localized = True
            top = segments[0]
            pct = round(top["area_fraction"] * 100)
            summary = (
                f"A concentrated suspicious stretch was found from "
                f"~{top['start_sec']:.1f}s to ~{top['end_sec']:.1f}s "
                f"(~{pct}% of frames), where scores don't match the rest "
                "of the clip — consistent with a local edit or splice in "
                "that time range."
            )
        else:
            summary = (
                "A few scattered frames looked slightly inconsistent, but "
                "nothing formed a clear, concentrated stretch worth pointing to."
            )

    return {
        "is_localized": is_localized,
        "suspicious_fraction": round(suspicious_fraction, 3),
        "segments": segments,
        "summary": summary,
    }


def _contiguous_runs(hot: list[bool]) -> list[list[int]]:
    """Group indices of True values into contiguous runs."""
    runs: list[list[int]] = []
    current: list[int] = []
    for i, is_hot in enumerate(hot):
        if is_hot:
            current.append(i)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs
