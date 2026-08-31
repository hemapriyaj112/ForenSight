"""
pipeline/video/temporal_analysis.py

Temporal consistency analysis for video deepfake detection.

Real video footage has smooth temporal progression of visual characteristics:
  - Texture scores vary gradually between frames
  - Facial landmark positions follow smooth trajectories
  - Color and brightness change naturally with scene motion

AI-generated video typically exhibits:
  - Jarring frame-to-frame score jumps (the generator produces each frame
    somewhat independently)
  - Unnatural landmark trajectories (faces morphing between frames)
  - Temporal coherence breakdown at the sub-second scale

This module operates on per-frame forensic scores and landmarks (when
available) to measure temporal consistency and produce a 0-1 suspicion score.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def analyse_temporal_consistency(
    frame_scores: list[float],
    timestamps: list[float],
    face_landmarks: list[Any] | None = None,
) -> dict[str, Any]:
    """Analyze temporal consistency across video frames.

    Parameters
    ----------
    frame_scores : list[float]
        Per-frame fused fake-probability scores.
    timestamps : list[float]
        Per-frame timestamps in seconds.
    face_landmarks : list or None
        Optional per-frame MTCNN landmarks (5-point arrays) for
        landmark-trajectory analysis.

    Returns
    -------
    dict with keys:
      score (float 0-1, higher = more temporally inconsistent)
      findings (list[str])
      sub_scores (dict of individual heuristic scores)
    """
    n = len(frame_scores)
    if n < 3:
        return {"score": 0.3, "findings": [], "sub_scores": {}}

    sub_scores: dict[str, float] = {}
    findings: list[str] = []

    score_variance = _score_variance_analysis(frame_scores)
    sub_scores["score_variance"] = score_variance

    if score_variance > 0.5:
        findings.append(
            "Per-frame suspicion scores jump erratically between consecutive "
            "frames, a pattern inconsistent with natural camera footage."
        )

    if face_landmarks is not None and len(face_landmarks) == n:
        landmark_score = _landmark_trajectory_analysis(face_landmarks)
        sub_scores["landmark_instability"] = landmark_score
        if landmark_score > 0.5:
            findings.append(
                "Facial landmark positions shift unnaturally between frames, "
                "suggesting inconsistent face generation across the timeline."
            )

    color_score = _color_consistency_analysis(frame_scores, timestamps)
    sub_scores["color_consistency"] = color_score

    weights = {
        "score_variance": 0.45,
        "landmark_instability": 0.35,
        "color_consistency": 0.20,
    }

    active = {k: v for k, v in sub_scores.items() if k in weights}
    if active:
        total_weight = sum(weights[k] for k in active)
        score = sum(sub_scores[k] * weights[k] for k in active) / total_weight
    else:
        score = 0.3

    score = float(np.clip(score, 0.0, 1.0))

    if not findings and score < 0.35:
        findings.append(
            "Temporal progression of scores looks smooth and consistent "
            "with natural video footage."
        )

    return {"score": score, "findings": findings, "sub_scores": sub_scores}


def _score_variance_analysis(frame_scores: list[float]) -> float:
    """Measure how much per-frame scores jump between consecutive frames.

    Real video has natural temporal variation. AI-generated video can show
    two distinct failure modes:
      1. Jarring frame-to-frame score jumps (each frame generated somewhat
         independently) → high variance
      2. Unnaturally smooth/identical scores (the generator produces a
         consistent output with no natural variation) → very low variance

    Both extremes are suspicious. Natural footage clusters in a mid-range
    of temporal variance.

    Returns 0-1 score (higher = more suspicious).
    """
    if len(frame_scores) < 3:
        return 0.3

    scores = np.array(frame_scores, dtype=np.float32)
    diffs = np.abs(np.diff(scores))

    mean_jump = float(np.mean(diffs))
    max_jump = float(np.max(diffs))

    combined = mean_jump * 0.6 + max_jump * 0.4

    if combined < 0.03:
        return 0.7
    elif combined < 0.08:
        return 0.5
    elif combined < 0.25:
        return 0.3
    elif combined < 0.45:
        return 0.55
    return 0.75


def _landmark_trajectory_analysis(landmarks: list[Any]) -> float:
    """Analyze smoothness of facial landmark trajectories across frames.

    Real faces move smoothly between frames. AI-generated faces can
    morph unnaturally, producing jerky landmark trajectories.

    Parameters
    ----------
    landmarks : list
        Per-frame MTCNN landmark arrays (or None if no face detected).

    Returns
    -------
    float 0-1 (higher = more suspicious).
    """
    valid = []
    for lm in landmarks:
        if lm is not None and hasattr(lm, "__len__") and len(lm) == 5:
            valid.append(np.asarray(lm, dtype=np.float32))

    if len(valid) < 3:
        return 0.3

    valid = np.stack(valid, axis=0)

    diffs = np.diff(valid, axis=0)
    magnitudes = np.linalg.norm(diffs, axis=-1)

    mean_motion = float(np.mean(magnitudes))
    std_motion = float(np.std(magnitudes))

    if mean_motion < 1e-6:
        return 0.2

    cv = std_motion / (mean_motion + 1e-9)

    if cv < 0.3:
        return 0.2
    elif cv < 0.8:
        return 0.4
    elif cv < 2.0:
        return 0.6
    elif cv < 4.0:
        return 0.8
    return 0.9


def _color_consistency_analysis(frame_scores: list[float], timestamps: list[float]) -> float:
    """Proxy for color/brightness consistency using score patterns.

    When combined with actual color histograms in a future enhancement,
    this can directly measure color drift. For now, it uses score patterns
    as a proxy for overall temporal smoothness.

    Returns 0-1 score.
    """
    if len(frame_scores) < 5:
        return 0.3

    scores = np.array(frame_scores, dtype=np.float32)

    window = min(5, len(scores) // 2)
    if window < 2:
        return 0.3

    local_means = np.convolve(scores, np.ones(window) / window, mode="valid")
    if len(local_means) < 2:
        return 0.3

    local_diffs = np.abs(np.diff(local_means))
    mean_local_change = float(np.mean(local_diffs))

    if mean_local_change < 0.02:
        return 0.2
    elif mean_local_change < 0.08:
        return 0.35
    elif mean_local_change < 0.15:
        return 0.55
    elif mean_local_change < 0.25:
        return 0.75
    return 0.9
