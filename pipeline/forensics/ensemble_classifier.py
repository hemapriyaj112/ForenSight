"""
pipeline/forensics/ensemble_classifier.py

Multi-model ensemble classifier for deepfake detection.

Runs multiple free HuggingFace/ONNX deepfake-detection models and averages
their predictions. Diversity across architectures (SigLIP, ViT, EfficientNet)
and training datasets consistently beats any single model on benchmark
evaluations.

Supported backends (same as AIClassifier):
  - HuggingFace transformers (local folder or Hub repo_id)
  - ONNX (single .onnx file)

Configuration in config/config.yaml:

    image_ai_classifier:
      enabled: true
      models:
        - path: "prithivMLmods/deepfake-detector-model-v1"
          backend: "transformers"
          weight: 1.0
        - path: "dima806/deepfake_vs_real_image_detection"
          backend: "transformers"
          weight: 0.8

If only one model is listed, this behaves identically to AIClassifier.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np


class EnsembleClassifier:
    """Runs multiple deepfake-detection models and averages predictions."""

    def __init__(
        self,
        models: list[dict[str, Any]] | None = None,
        default_input_size: int = 224,
    ):
        self.models = []
        self.default_input_size = default_input_size
        self.available = False
        self.load_error: Optional[str] = None
        self._loaded = False

        if not models:
            models = [
                {
                    "path": "prithivMLmods/deepfake-detector-model-v1",
                    "backend": "transformers",
                    "weight": 1.0,
                    "input_size": 224,
                    "ai_class_index": 0,
                }
            ]

        self._model_configs = models

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        self._load_models(self._model_configs)

    def _load_models(self, model_configs: list[dict]) -> None:
        """Load all configured models."""
        errors = []
        for cfg in model_configs:
            try:
                from pipeline.forensics.ai_classifier import AIClassifier
                clf = AIClassifier(
                    model_path=cfg.get("path"),
                    backend=cfg.get("backend", "auto"),
                    input_size=cfg.get("input_size", self.default_input_size),
                    ai_class_index=cfg.get("ai_class_index", 0),
                )
                self.models.append({
                    "classifier": clf,
                    "weight": float(cfg.get("weight", 1.0)),
                    "name": cfg.get("path", "unknown"),
                })
            except Exception as exc:
                errors.append(f"{cfg.get('path', 'unknown')}: {type(exc).__name__}: {exc}")

        if not self.models:
            self.load_error = "; ".join(errors) if errors else "No models configured"
            self.available = False
        else:
            self.available = True
            if errors:
                self.load_error = f"Some models failed to load: {'; '.join(errors)}"
            else:
                self.load_error = None

    def predict(self, image_rgb: np.ndarray) -> dict[str, Any]:
        """Run ensemble prediction on an image.

        Returns dict with:
          available (bool), score (float 0-1 or None),
          findings (list[str]), backend (str), error (str or None),
          individual_scores (list of per-model scores)
        """
        self._ensure_loaded()

        if not self.available:
            return {
                "available": False, "score": None, "findings": [],
                "backend": None, "error": self.load_error,
                "individual_scores": [],
            }

        individual_scores = []
        total_weight = 0.0
        weighted_sum = 0.0
        findings = []
        active_models = []

        for entry in self.models:
            clf = entry["classifier"]
            if not clf.available:
                continue

            try:
                result = clf.predict(image_rgb)
                if result.get("available") and result.get("score") is not None:
                    s = float(result["score"])
                    w = entry["weight"]
                    individual_scores.append({
                        "name": entry["name"],
                        "score": s,
                        "weight": w,
                    })
                    weighted_sum += s * w
                    total_weight += w
                    active_models.append(entry["name"])
                    if result.get("findings"):
                        findings.extend(result["findings"])
            except Exception:
                continue

        if total_weight == 0 or not individual_scores:
            return {
                "available": True, "score": None, "findings": [],
                "backend": "ensemble", "error": "All models failed to predict",
                "individual_scores": individual_scores,
            }

        score = float(np.clip(weighted_sum / total_weight, 0.0, 1.0))

        label = "likely AI-generated" if score >= 0.5 else "likely a real camera photo"
        findings.insert(0,
            f"Ensemble of {len(active_models)} model(s) ({', '.join(active_models)}) "
            f"scored this image as {label} ({score*100:.0f}% AI-probability)."
        )

        return {
            "available": True,
            "score": score,
            "findings": findings,
            "backend": "ensemble",
            "error": None,
            "individual_scores": individual_scores,
        }
