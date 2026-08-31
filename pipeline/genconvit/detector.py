"""pipeline/genconvit/detector.py

GenConViT — a real, free, locally-runnable pretrained video deepfake detector.

This wraps the official GenConViT pretrained weights (MIT license,
https://huggingface.co/Deressa/GenConViT) behind a clean interface so the
video pipeline can score a video with an actual neural network instead of the
Laplacian-variance texture heuristic alone.

DESIGN
------
* The model + weights are loaded LAZILY the first time inference is requested,
  and its status is reported via `genconvit_status` (ACTIVE_MODEL | INACTIVE)
  with an exact reason when inactive. There is NO silent fallback to the
  texture heuristic: if GenConViT cannot load, callers see INACTIVE and a
  null `genconvit_score` — they never receive a texture score disguised as
  GenConViT.
* Frame sampling reuses the project's existing demux frame extraction path
  (the `frames_dir` produced by `utils.demux`), feeding those same frames to
  GenConViT. The minimum preprocessing GenConViT requires (face-crop ->
  224x224 -> ImageNet normalize) is implemented here.
* Output names avoid the word "probability" (not calibrated): we expose
  `genconvit_score` / `genconvit_fake_score` and per-frame raw values.

LABEL MAPPING (verified against the official implementation)
-----------------------------------------------------------
model/pred_func.py of the upstream repo maps the 2-class logits via
`real_or_fake(pred) = {0: 'REAL', 1: 'FAKE'}[pred ^ 1]`, i.e.

    output index 0 -> FAKE
    output index 1 -> REAL

and the aggregated "fake" score is

    mean over frames of sigmoid(logits[:, 0])

with the overall class taken as `argmax(mean sigmoid)`. We reproduce this
exactly so the score means the same thing as the released model's own output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# ImageNet normalization used by the released model (dataset/loader.py).
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_IMG_SIZE = 224

# Default weight directory / filenames (see scripts/download_genconvit_weights.py).
_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "genconvit"
_ED_WEIGHT = "genconvit_ed_inference.pth"
_VAE_WEIGHT = "genconvit_vae_inference.pth"

STATUS_ACTIVE_MODEL = "ACTIVE_MODEL"
STATUS_INACTIVE = "INACTIVE"

# Engineering diagnostics (NOT scientifically validated categories) describing
# how much of a video's timeline GenConViT actually saw a face in. Used to tell
# "the detector was weak" apart from "the detector barely saw any faces".
COVERAGE_NO_FACE = "NO_FACE_DETECTED"
COVERAGE_LOW = "LOW_FACE_COVERAGE"
COVERAGE_PARTIAL = "PARTIAL_FACE_COVERAGE"
COVERAGE_GOOD = "GOOD_FACE_COVERAGE"

# Sanity floor for a face crop before we resize it and hand it to the model:
# a crop smaller than this on its smaller axis is so low-res that upscaling it
# to 224x224 would hand the network a blurry smear. Below this we treat the
# frame as not usable (documented engineering tradeoff, not a quality bar
# change to the face detector itself — MTCNN's own min_face_size still applies).
_MIN_CROP_MINOR_DIM = 24


def _label_for_argmax(argmax: int) -> str:
    """Mirror upstream real_or_fake: index 0 -> FAKE, index 1 -> REAL."""
    return {0: "FAKE", 1: "REAL"}[argmax ^ 1]


def _coverage_status(requested: int, found: int) -> str:
    """Return an engineering coverage label from frames requested vs face frames found.

    Boundaries are engineering heuristics, not validated scientific classes:
      * 0 faces found              -> NO_FACE_DETECTED
      * < 25% of frames had a face -> LOW_FACE_COVERAGE
      * < 60%                      -> PARTIAL_FACE_COVERAGE
      * >= 60%                     -> GOOD_FACE_COVERAGE
    """
    if requested <= 0 or found <= 0:
        return COVERAGE_NO_FACE
    ratio = found / requested
    if ratio < 0.25:
        return COVERAGE_LOW
    if ratio < 0.60:
        return COVERAGE_PARTIAL
    return COVERAGE_GOOD


class GenConViTDetector:
    """Lazy-loading wrapper around the official GenConViT weights.

    Parameters
    ----------
    weights_dir : Path, optional
        Directory containing genconvit_ed_inference.pth and
        genconvit_vae_inference.pth. Defaults to <project>/models/genconvit.
    num_frames : int, optional
        Number of frames sampled evenly across the video (upstream default 15).
    use_vae : bool, optional
        Whether to also load and run the VAE branch. Both ED+VAE form the
        combined "genconvit" model (upstream averages their probabilities).
        If the VAE weight is missing, we fall back to ED-only and report it —
        never to texture.
    fp16 : bool, optional
        Enable half-precision when CUDA is available.
    face_cropper : optional
        An existing FaceCropper instance to reuse for face detection/cropping.
        Falls back to instantiating the project's FaceCropper if None.
    """

    def __init__(
        self,
        weights_dir: Optional[Path | str] = None,
        num_frames: int = 15,
        use_vae: bool = True,
        fp16: bool = True,
        face_cropper=None,
    ):
        self._weights_dir = Path(weights_dir) if weights_dir else _DEFAULT_DIR
        self.num_frames = int(num_frames)
        self.use_vae = bool(use_vae)
        self.fp16 = bool(fp16)
        self._face_cropper = face_cropper

        self.available = False
        self.load_error: Optional[str] = None
        self.model = None
        self.device = "cuda" if self._torch_cuda_available() else "cpu"
        self.branches: list[str] = []
        self._loaded = False

    @staticmethod
    def _torch_cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:  # pragma: no cover - environment dependent
            return False

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    @property
    def status(self) -> str:
        if self._loaded and self.available:
            return STATUS_ACTIVE_MODEL
        return STATUS_INACTIVE

    def ensure_loaded(self) -> bool:
        """Load the weights + model if not already loaded.

        Returns True if the model is ready for inference. On any failure sets
        self.available=False and self.load_error with the exact reason. Never
        raises: callers check `status`/`available` and report INACTIVE.
        """
        if self._loaded:
            return self.available
        self._loaded = True
        try:
            self._do_load()
            self.available = True
        except FileNotFoundError as exc:
            self.load_error = f"missing GenConViT checkpoint: {exc}"
            self.available = False
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.available = False
        return self.available

    def _do_load(self) -> None:
        import torch

        ed_path = self._weights_dir / _ED_WEIGHT
        if not ed_path.exists():
            raise FileNotFoundError(
                f"ED checkpoint not found at {ed_path}. Run "
                "scripts/download_genconvit_weights.py first."
            )

        from pipeline.genconvit.model import GenConViTED, GenConViTVAE

        if self._face_cropper is None:
            from pipeline.forensics.face_crop import FaceCropper

            self._face_cropper = FaceCropper()

        ed_state = torch.load(str(ed_path), map_location="cpu")
        ed_sd = ed_state.get("state_dict", ed_state) if isinstance(ed_state, dict) else ed_state
        model_ed = GenConViTED()
        missing, unexpected = model_ed.load_state_dict(ed_sd, strict=False)
        # Ignore the unused HybridEmbed/swin params that ship in the weight
        # archive but are not on the active inference graph (see model.py note).
        missing = [k for k in missing if not k.startswith(("backbone.patch_embed.", "embedder."))]
        if missing:
            raise RuntimeError(
                f"ED state_dict missing {len(missing)} keys during load: {missing[:5]}"
            )
        model_ed.to(self.device).eval()
        self.branches.append("ed")

        parts: list = [model_ed]

        if self.use_vae:
            vae_path = self._weights_dir / _VAE_WEIGHT
            if vae_path.exists():
                vae_state = torch.load(str(vae_path), map_location="cpu")
                vae_sd = vae_state.get("state_dict", vae_state) if isinstance(vae_state, dict) else vae_state
                model_vae = GenConViTVAE()
                v_missing, _ = model_vae.load_state_dict(vae_sd, strict=False)
                v_missing = [k for k in v_missing if not k.startswith(("backbone.patch_embed.", "embedder."))]
                if v_missing:
                    raise RuntimeError(
                        f"VAE state_dict missing {len(v_missing)} keys during load: {v_missing[:5]}"
                    )
                model_vae.to(self.device).eval()
                parts.append(model_vae)
                self.branches.append("vae")
            else:
                # ED-only fallback is honest: it's still a real GenConViT
                # branch, not a heuristic. Report it clearly.
                self.branches = ["ed"]
                self.load_error = (
                    f"VAE checkpoint not found at {vae_path}; running ED-only."
                )

        if self.fp16 and self.device == "cuda":
            try:
                for p in parts:
                    p.half()
            except Exception:  # noqa: BLE001 - fp16 is best-effort
                pass

        self.model = parts
        self.available = True
        self._loaded = True

    # ------------------------------------------------------------------ #
    # Preprocessing (mirrors upstream preprocess_frame)
    # ------------------------------------------------------------------ #
    def _preprocess(self, face_crops: list[np.ndarray]) -> "torch.Tensor":
        import torch

        tensors = []
        for crop in face_crops:
            img = crop.astype(np.float32) / 255.0
            # HxWx3 -> 3xHxW
            img = np.transpose(img, (2, 0, 1))
            # ImageNet normalize
            for c in range(3):
                img[c] = (img[c] - _MEAN[c]) / _STD[c]
            tensors.append(torch.from_numpy(img))
        x = torch.stack(tensors, dim=0)  # [N,3,H,W]
        return x.to(self.device)

    # ------------------------------------------------------------------ #
    # Face cropping (reuse the project's FaceCropper)
    # ------------------------------------------------------------------ #
    def _collect_face_candidates(
        self, frames: list[np.ndarray]
    ) -> list[dict]:
        """Run the shared FaceCropper over EVERY demuxed frame and return the
        valid face crops as a list of candidate dicts.

        This is the Phase 2A.1 coverage improvement. The previous behaviour
        sampled `num_frames` evenly across the timeline *first* and then kept
        only the sub-subset of those that had a face — so a longer clip whose
        sampled 15 positions happened to miss most of the face-bearing frames
        ended up scoring off 1-3 crops even though valid faces existed in the
        other frames. Here we scan the whole 1-fps timeline (the same frames
        the texture heuristic sees) and collect every frame that passes the
        SAME face-quality rules (MTCNN min_confidence + geometry + bilateral
        texture symmetry — unchanged, we never lower them), then sample evenly
        across those candidates in `infer()`.

        Each candidate dict:
          index       : original frame index (for frame numbering/timestamps)
          crop        : HxWx3 uint8 face crop (already resized to 224x224)
          confidence  : MTCNN detection confidence (or None for Haar backend)
          box        : (x1,y1,x2,y2) original face box (pre-resize), for size metrics
        """
        if self._face_cropper is None or not getattr(
            self._face_cropper, "available", False
        ):
            return []
        candidates: list[dict] = []
        for i, frame in enumerate(frames):
            out = self._face_cropper.crop(frame)
            if not out.get("face_found") or out.get("crop") is None:
                continue
            crop = out["crop"]
            h, w = crop.shape[:2]
            # Guard: refuse to upscale a trivially tiny crop into the model's
            # 224x224 input (would feed it a blurry smear). This is a sanity
            # guard on the crop we resize, not a change to the face detector.
            if min(h, w) < _MIN_CROP_MINOR_DIM:
                continue
            resized = self._resize_to_224(crop)
            candidates.append({
                "index": i,
                "crop": resized,
                "confidence": out.get("confidence"),
                "box": out.get("box"),
            })
        return candidates

    @staticmethod
    def _resize_to_224(crop: np.ndarray) -> np.ndarray:
        import cv2

        h, w = crop.shape[:2]
        if h == 224 and w == 224:
            return crop
        return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def infer(
        self,
        frames: list[np.ndarray],
        timestamps: Optional[list[float]] = None,
    ) -> dict:
        """Run GenConViT over a list of HxWx3 uint8 frames.

        Returns a dict:
        {
          "genconvit_status": "ACTIVE_MODEL" | "INACTIVE",
          "load_error": str | None,
          "genconvit_score": float | None,      # aggregated fake score (0-1)
          "genconvit_fake_score": float | None, # same as genconvit_score
          "genconvit_aggregation_method": str,
          "genconvit_frame_scores": list[float] | None,  # per-frame fake probs
          "frames_used": int,
          "branches": list[str],
          "device": str,
          "note": str | None,
        }

        If the model can't load, `genconvit_score` is None and
        `genconvit_status` is INACTIVE with `load_error` set. No heuristic
        fallback is performed.

        Beyond the score the dict also carries explicit input-coverage metrics
        (`genconvit_frames_requested/sampled`, `genconvit_face_frames_found`,
        `genconvit_face_coverage`, min/avg/max face confidence, per-frame
        indices/timestamps/face-sizes/confidence and a `genconvit_coverage_status`
        label) so a weak GenConViT reading can be distinguished from a video the
        detector barely saw any faces in.
        """
        if not self.ensure_loaded():
            return {
                "genconvit_status": STATUS_INACTIVE,
                "load_error": self.load_error,
                "genconvit_score": None,
                "genconvit_fake_score": None,
                "genconvit_predicted_label": None,
                "genconvit_aggregation_method": None,
                "genconvit_frame_scores": None,
                "genconvit_frames_requested": len(frames),
                "genconvit_frames_sampled": 0,
                "genconvit_face_frames_found": 0,
                "genconvit_face_coverage": 0.0,
                "genconvit_average_face_confidence": None,
                "genconvit_min_face_confidence": None,
                "genconvit_max_face_confidence": None,
                "genconvit_frame_indices": [],
                "genconvit_timestamps": [],
                "genconvit_face_sizes": [],
                "genconvit_face_confidences": [],
                "genconvit_frames_used": 0,
                "genconvit_face_quality": None,
                "genconvit_coverage_status": None,
                "frames_used": 0,
                "branches": self.branches,
                "device": self.device,
                "note": "GenConViT cannot load; no neural video score produced.",
            }

        if not frames:
            return {
                "genconvit_status": STATUS_ACTIVE_MODEL,
                "load_error": None,
                "genconvit_score": None,
                "genconvit_fake_score": None,
                "genconvit_aggregation_method": "mean_sigmoid_index0_+_argmax_index1",
                "genconvit_frame_scores": [],
                "genconvit_frame_indices": [],
                "genconvit_timestamps": [],
                "genconvit_face_sizes": [],
                "genconvit_frames_requested": 0,
                "genconvit_frames_sampled": 0,
                "genconvit_face_frames_found": 0,
                "genconvit_face_coverage": 0.0,
                "genconvit_average_face_confidence": None,
                "genconvit_min_face_confidence": None,
                "genconvit_max_face_confidence": None,
                "genconvit_frames_used": 0,
                "genconvit_face_quality": "NO_FACE_DETECTED",
                "genconvit_coverage_status": COVERAGE_NO_FACE,
                "frames_used": 0,
                "branches": self.branches,
                "device": self.device,
                "note": "No frames provided.",
            }

        if timestamps is None:
            timestamps = [float(i) for i in range(len(frames))]

        import torch

        n_requested = len(frames)

        # Phase 2A.1 coverage: collect face candidates across the WHOLE
        # timeline (not a 15-frame sub-sample), then sample evenly among them.
        candidates = self._collect_face_candidates(frames)
        n_found = len(candidates)

        all_conf = [
            c["confidence"] for c in candidates if c.get("confidence") is not None
        ]
        avg_conf = float(np.mean(all_conf)) if all_conf else None
        min_conf = float(min(all_conf)) if all_conf else None
        max_conf = float(max(all_conf)) if all_conf else None

        coverage_label = _coverage_status(n_requested, n_found)

        if not candidates:
            return {
                "genconvit_status": STATUS_ACTIVE_MODEL,
                "load_error": None,
                "genconvit_score": None,
                "genconvit_fake_score": None,
                "genconvit_predicted_label": None,
                "genconvit_aggregation_method": "mean_sigmoid_index0_+_argmax_index1",
                "genconvit_frame_scores": [],
                "genconvit_frame_indices": [],
                "genconvit_timestamps": [],
                "genconvit_face_sizes": [],
                "genconvit_face_confidences": [],
                "genconvit_frames_requested": n_requested,
                "genconvit_frames_sampled": n_requested,
                "genconvit_face_frames_found": 0,
                "genconvit_face_coverage": 0.0,
                "genconvit_average_face_confidence": avg_conf,
                "genconvit_min_face_confidence": min_conf,
                "genconvit_max_face_confidence": max_conf,
                "genconvit_frames_used": 0,
                "genconvit_face_quality": coverage_label,
                "genconvit_coverage_status": coverage_label,
                "frames_used": 0,
                "branches": self.branches,
                "device": self.device,
                "note": "NO_FACE_DETECTED: no valid face crop found in any "
                        "sampled frame; GenConViT requires a face crop.",
            }

        # Sample up to `num_frames` evenly across the candidate pool (keeps
        # temporal spread AND keeps the inference budget identical to upstream,
        # just from a larger, face-bearing pool instead of a fixed 15).
        count = min(self.num_frames, n_found)
        sel = np.linspace(0, n_found - 1, count, dtype=int)
        chosen = [candidates[int(i)] for i in sel]

        usable = [c["crop"] for c in chosen]
        usable_idxs = [int(c["index"]) for c in chosen]
        usable_conf = [c.get("confidence") for c in chosen]
        usable_boxes = [c.get("box") for c in chosen]
        usable_ts = [
            timestamps[idx] if 0 <= idx < len(timestamps) else float(idx)
            for idx in usable_idxs
        ]
        # Face size = original (pre-resize) box width/height.
        usable_sizes = []
        for b in usable_boxes:
            if b is not None and len(b) == 4:
                w = abs(b[2] - b[0])
                h = abs(b[3] - b[1])
                usable_sizes.append([int(w), int(h)])
            else:
                usable_sizes.append(None)

        x = self._preprocess(usable)

        with torch.no_grad():
            # Shape [branches, N, 2] -> average along branch dim. Keep the
            # batch dimension explicit so a single face crop (batch==1) still
            # yields a [1,2] row instead of unexpectedly collapsing to [2].
            logits = None
            for i, branch in enumerate(self.model):
                with torch.autocast(self.device, enabled=self.fp16 and self.device == "cuda"):
                    out = branch(x)
                out = out[0] if isinstance(out, (tuple, list)) else out
                sl = torch.sigmoid(out).reshape(out.shape[0], -1)
                logits = sl if logits is None else logits + sl
            logits = logits / len(self.model)  # [N, 2], mean across branches

        # Per-frame values, keyed back to original frame indices.
        frame_fake = [float(v) for v in logits[:, 0].tolist()]

        # Aggregated per upstream max_prediction_value: argmax of the MEAN
        # 2-vector -> 0=FAKE, 1=REAL; fake score = mean sigmoid[:,0] (bounded
        # complement of index1 as upstream does).
        mean_vec = logits.mean(dim=0)
        argmax = int(torch.argmax(mean_vec).item())
        agg_fake = float(mean_vec[0].item())
        # Upstream returns max(index0, 1 - index1); equal after sigmoid but
        # keep identical semantics.
        agg_fake = max(agg_fake, abs(1.0 - mean_vec[1].item()))

        # Build per-frame score list aligned to all frames (None where the
        # frame was skipped for lacking a face crop) for transparency.
        per_frame_full: list[Optional[float]] = [None] * n_requested
        for (orig_idx, val) in zip(usable_idxs, frame_fake):
            per_frame_full[orig_idx] = val
        usable_idxs = [int(i) for i in usable_idxs]

        return {
            "genconvit_status": STATUS_ACTIVE_MODEL,
            "load_error": self.load_error,
            "genconvit_score": agg_fake,
            "genconvit_fake_score": agg_fake,
            "genconvit_predicted_label": _label_for_argmax(argmax),
            "genconvit_aggregation_method": "mean_sigmoid_index0_+_argmax_index1",
            "genconvit_frame_scores": [float(v) if v is not None else None for v in per_frame_full],
            "genconvit_frame_indices": usable_idxs,
            "genconvit_timestamps": [float(t) for t in usable_ts],
            "genconvit_face_sizes": usable_sizes,
            "genconvit_face_confidences": [
                float(c) if c is not None else None for c in usable_conf
            ],
            "genconvit_frames_requested": n_requested,
            "genconvit_frames_sampled": n_requested,
            "genconvit_face_frames_found": n_found,
            "genconvit_face_coverage": n_found / n_requested if n_requested else 0.0,
            "genconvit_average_face_confidence": avg_conf,
            "genconvit_min_face_confidence": min_conf,
            "genconvit_max_face_confidence": max_conf,
            "genconvit_frames_used": len(usable),
            "genconvit_face_quality": coverage_label,
            "genconvit_coverage_status": coverage_label,
            "frames_used": len(usable),
            "branches": self.branches,
            "device": self.device,
            "note": None,
        }
