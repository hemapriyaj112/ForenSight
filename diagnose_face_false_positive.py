"""
diagnose_face_false_positive.py

Session 6: the landmark-geometry check added to FaceCropper didn't stop
ai_noaud2.mp4's hand region from still coming back as "face found" --
identical numbers to before the fix (77% face segment / 59% non-face /
96.07% overall). That means the fix landed but isn't actually rejecting
this specific candidate, which usually means one of:

  1. MTCNN's landmark regressor is defaulting to a plausible "average
     face template" layout even on this non-face patch (a known failure
     mode: the regressor was only ever trained on real faces, so on
     out-of-distribution input it tends to output something that still
     LOOKS like a coherent face layout rather than garbage coordinates --
     which would trivially pass generic geometric sanity checks like
     "eyes level, nose between/below them, mouth below the nose").
  2. MTCNN's proposed box is near-square regardless of the underlying
     object (a known MTCNN behavior), so the aspect-ratio check isn't
     doing anything useful here either.

Rather than guessing a third heuristic blind, this script dumps the
REAL numbers for every candidate MTCNN proposes on ai_noaud2.mp4's
frames: confidence, box, aspect ratio, the 5 landmark points, whether
the current geometry check passes, and -- the next hypothesis to test --
the local image contrast/variance at each landmark point vs. a couple of
"plain skin" baseline points on the same crop. A real eye has visibly
higher local edge/contrast energy than a patch of hand skin; if that
holds up on the actual false-positive frame, that's the next check worth
adding.

Usage
-----
    python diagnose_face_false_positive.py /path/to/ai_noaud2.mp4

Run this in YOUR environment (torch/facenet-pytorch installed) since the
sandbox this was written in doesn't have a working torch install. Send
the full printed output back and we'll design the next fix from it
instead of guessing again.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from utils.demux import demux
from pipeline.forensics.face_crop import FaceCropper


def _laplacian(gray: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray.shape
    padded = np.pad(gray, 1, mode="reflect")
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * padded[i : i + h, j : j + w]
    return out


def _local_variance(gray: np.ndarray, cx: float, cy: float, patch: int = 7) -> float | None:
    h, w = gray.shape
    x0, y0 = int(cx - patch // 2), int(cy - patch // 2)
    x1, y1 = x0 + patch, y0 + patch
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    region = gray[y0:y1, x0:x1]
    return float(np.var(_laplacian(region)))


def main(video_path: str) -> None:
    frames_dir, _ = demux(video_path)
    frame_paths = sorted(Path(frames_dir).glob("frame_*.jpg"))
    print(f"Extracted {len(frame_paths)} frames from {video_path}\n")

    cropper = FaceCropper()
    if not cropper.available:
        print(f"FaceCropper unavailable: {cropper.load_error}")
        return

    for fp in frame_paths:
        image_rgb = np.array(Image.open(fp).convert("RGB"))
        gray = (0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2])

        img = Image.fromarray(image_rgb.astype(np.uint8))
        try:
            boxes, probs, landmarks = cropper._mtcnn.detect(img, landmarks=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{fp.name}: detect() raised {type(exc).__name__}: {exc}")
            continue

        if boxes is None or len(boxes) == 0:
            print(f"{fp.name}: no candidates")
            continue

        print(f"--- {fp.name} ---")
        for i, box in enumerate(boxes):
            conf = float(probs[i]) if probs is not None and probs[i] is not None else None
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            aspect = bh / bw if bw > 0 else None
            geo_ok = cropper._has_coherent_face_geometry(
                landmarks[i] if landmarks is not None else None, box
            )
            print(f"  candidate {i}: confidence={conf} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
                  f"aspect(h/w)={aspect} geometry_check_passes={geo_ok}")

            if landmarks is not None:
                pts = landmarks[i]
                names = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
                for name, (px, py) in zip(names, pts):
                    var = _local_variance(gray, px, py)
                    print(f"      {name:>11}: ({px:.1f},{py:.1f})  local_variance={var}")

                # Baseline "plain skin" points: outer-left and outer-right
                # cheek area, same height as the eyes.
                eye_y_avg = (pts[0][1] + pts[1][1]) / 2.0
                left_cheek = (x1 + 0.12 * bw, eye_y_avg)
                right_cheek = (x2 - 0.12 * bw, eye_y_avg)
                for name, (px, py) in [("left_cheek(baseline)", left_cheek),
                                        ("right_cheek(baseline)", right_cheek)]:
                    var = _local_variance(gray, px, py)
                    print(f"      {name:>22}: ({px:.1f},{py:.1f})  local_variance={var}")
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnose_face_false_positive.py /path/to/video.mp4")
        sys.exit(1)
    main(sys.argv[1])
