"""pipeline/forensics/explain.py

Turns raw sub-scores + per-signal findings into a plain-language summary
that a non-technical person can read and trust, without needing to
understand GradCAM, FFT, or ELA.
"""
from __future__ import annotations

from typing import Any


def summarise(
    verdict: str,
    fused_score: float,
    signals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Parameters
    ----------
    verdict: "REAL" | "FAKE" | "UNCERTAIN"
    fused_score: 0-1 overall probability of AI-generation/manipulation
    signals: {signal_name: {"score": float, "weight": float, "findings": [str, ...]}}

    Returns
    -------
    {
      "headline": str,           # one-line plain summary
      "reasons": list[str],      # ordered, most-important-first
      "confidence_label": str,   # "High" | "Moderate" | "Low"
    }
    """
    pct = round(fused_score * 100)

    if verdict == "FAKE":
        headline = f"This content is likely AI-generated or manipulated ({pct}% confidence)."
    elif verdict == "REAL":
        headline = f"This content appears authentic ({100 - pct}% confidence it's real)."
    else:
        headline = (
            f"This content is ambiguous ({pct}% probability of manipulation) — "
            "the evidence is mixed and a confident call can't be made."
        )

    # Rank signals by their contribution (score * weight), descending, and
    # only surface signals that actually said something meaningful.
    ranked = sorted(
        signals.items(),
        key=lambda kv: kv[1].get("score", 0.0) * kv[1].get("weight", 0.0),
        reverse=True,
    )

    reasons: list[str] = []
    for _name, info in ranked:
        for finding in info.get("findings", []):
            if finding not in reasons:
                reasons.append(finding)

    if not reasons:
        reasons.append(
            "No single signal was strongly conclusive; the verdict reflects "
            "a blend of weak signals."
        )

    spread = abs(fused_score - 0.5)
    if spread > 0.35:
        confidence_label = "High"
    elif spread > 0.15:
        confidence_label = "Moderate"
    else:
        confidence_label = "Low"

    return {"headline": headline, "reasons": reasons, "confidence_label": confidence_label}
