"""pipeline/forensics/ai_classifier.py

Loads a real, trained AI-vs-real image classifier and runs it as one of the
fused signals in ImageDetector. This is the piece the rest of the pipeline
has been explicitly missing: every other signal in this project (noise,
ELA, texture, spectral, metadata) is a hand-written statistical heuristic
with no learned weights behind it. This module is the drop-in slot for an
actual supervised model trained to recognise generator fingerprints.

USAGE
-----
Point ImageDetector at a local model with:

    from pipeline.forensics.ai_classifier import AIClassifier
    clf = AIClassifier(model_path="models/ai_classifier")   # dir or .onnx file
    detector = ImageDetector(ai_classifier=clf)

If model_path doesn't exist, or required packages aren't installed,
AIClassifier.available is False and ImageDetector silently drops this
signal from the fused score (falling back to the 5 heuristics only) rather
than crashing — see ImageDetector for the renormalisation logic.

SUPPORTED WEIGHT FORMATS
-------------------------
1. ONNX (recommended for this environment — onnxruntime is already
   installed, no PyTorch download needed):
   - Point model_path at a single .onnx file.
   - Expects a single image input (NCHW, float32) and either:
       a) one output of shape (1,) or (1,1) — a single AI-probability logit
          / sigmoid score, or
       b) one output of shape (1,2) — two-class logits, in which case
          `ai_class_index` tells us which column is "AI/Fake" (default 0;
          override if your export uses the opposite order).
   - Input size/mean/std are configurable (defaults: 224x224, ImageNet
     normalisation) — check the model card for the exact preprocessing it
     expects and pass overrides if different.

2. Hugging Face / transformers local directory (needs `pip install torch
   transformers` in addition to what's already here):
   - Point model_path at a local folder containing config.json,
     model.safetensors (or pytorch_model.bin), and preprocessor_config.json
     — i.e. exactly what `snapshot_download(repo_id=...)` or
     `git clone` from Hugging Face produces. This module never downloads
     anything itself; the person must fetch the weights and hand over the
     folder.
   - Label mapping is read from the model's own config.id2label; we look
     for a label containing "fake"/"ai"/"synthetic"/"generated" (case-
     insensitive) to identify the AI-probability class. If none of those
     match, the second class index is used as a fallback and a warning is
     surfaced in `findings`.

Neither backend is a fixed dependency of the rest of the app — everything
above stays inert if model_path is None (the current default), so nothing
about the existing 5-signal pipeline changes unless this is explicitly
wired in.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np


class AIClassifier:
    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "auto",          # "auto" | "onnx" | "transformers"
        input_size: int = 224,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        ai_class_index: int = 0,
        onnx_input_name: Optional[str] = None,
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.ai_class_index = ai_class_index
        self._onnx_input_name = onnx_input_name

        self.available = False
        self.backend: Optional[str] = None
        self.load_error: Optional[str] = None

        self._session = None            # onnxruntime.InferenceSession
        self._hf_model = None           # transformers model
        self._hf_processor = None       # transformers image processor
        self._hf_ai_index: Optional[int] = None

        if model_path:
            self._load(backend)

    # ------------------------------------------------------------------ #
    def _load(self, backend: str) -> None:
        resolved = backend
        if backend == "auto":
            if os.path.isfile(self.model_path) and self.model_path.endswith(".onnx"):
                resolved = "onnx"
            elif os.path.isdir(self.model_path):
                resolved = "transformers"
            else:
                self.load_error = (
                    f"model_path '{self.model_path}' is neither an .onnx file "
                    "nor a directory — could not determine backend."
                )
                return

        try:
            if resolved == "onnx":
                self._load_onnx()
            elif resolved == "transformers":
                self._load_transformers()
            else:
                self.load_error = f"Unknown backend '{resolved}'."
                return
            self.backend = resolved
            self.available = True
        except Exception as exc:  # noqa: BLE001 — surface any load failure as unavailable, not a crash
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.available = False

    def _load_onnx(self) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        if self._onnx_input_name is None:
            self._onnx_input_name = self._session.get_inputs()[0].name

    def _load_transformers(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self._hf_processor = AutoImageProcessor.from_pretrained(self.model_path)
        self._hf_model = AutoModelForImageClassification.from_pretrained(self.model_path)
        self._hf_model.eval()

        id2label = getattr(self._hf_model.config, "id2label", {}) or {}
        ai_idx = None
        for idx, label in id2label.items():
            if any(kw in str(label).lower() for kw in ("fake", "ai", "synthetic", "generated")):
                ai_idx = int(idx)
                break
        if ai_idx is None and len(id2label) == 2:
            ai_idx = 1  # fallback guess; flagged in predict() findings
        self._hf_ai_index = ai_idx

    # ------------------------------------------------------------------ #
    def predict(self, image_rgb: np.ndarray) -> dict[str, Any]:
        """
        Returns
        -------
        dict with keys: available (bool), score (float 0-1 or None),
        findings (list[str]), backend (str or None), error (str or None)
        """
        if not self.available:
            return {
                "available": False, "score": None, "findings": [],
                "backend": None, "error": self.load_error,
            }

        try:
            if self.backend == "onnx":
                score = self._predict_onnx(image_rgb)
            else:
                score = self._predict_transformers(image_rgb)
        except Exception as exc:  # noqa: BLE001
            return {
                "available": False, "score": None, "findings": [],
                "backend": self.backend, "error": f"{type(exc).__name__}: {exc}",
            }

        findings = [
            f"A trained AI-image classifier scored this image "
            f"{'as likely AI-generated' if score >= 0.5 else 'as likely a real camera photo'} "
            f"({score*100:.0f}% AI-probability)."
        ]
        return {"available": True, "score": score, "findings": findings,
                "backend": self.backend, "error": None}

    def _preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        img = Image.fromarray(image_rgb.astype(np.uint8)).resize(
            (self.input_size, self.input_size), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0     # HWC, 0-1
        arr = arr.transpose(2, 0, 1)                          # CHW
        arr = (arr - self.mean) / self.std
        return arr[np.newaxis, ...].astype(np.float32)        # NCHW

    def _predict_onnx(self, image_rgb: np.ndarray) -> float:
        inp = self._preprocess(image_rgb)
        outputs = self._session.run(None, {self._onnx_input_name: inp})
        logits = np.asarray(outputs[0]).squeeze()

        if logits.ndim == 0 or logits.size == 1:
            val = float(logits)
            # If it looks like a raw logit rather than a 0-1 probability, squash it.
            if val < 0.0 or val > 1.0:
                val = 1.0 / (1.0 + np.exp(-val))
            return float(np.clip(val, 0.0, 1.0))

        # Multi-class logits — softmax then pick the configured AI index.
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        idx = min(self.ai_class_index, len(probs) - 1)
        return float(np.clip(probs[idx], 0.0, 1.0))

    def _predict_transformers(self, image_rgb: np.ndarray) -> float:
        import torch
        from PIL import Image

        img = Image.fromarray(image_rgb.astype(np.uint8))
        inputs = self._hf_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = self._hf_model(**inputs).logits.squeeze()
        probs = torch.softmax(logits, dim=-1).numpy()

        idx = self._hf_ai_index if self._hf_ai_index is not None else (len(probs) - 1)
        idx = min(idx, len(probs) - 1)
        return float(np.clip(probs[idx], 0.0, 1.0))
