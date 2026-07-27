"""
pipeline/forensics/face_crop.py
================================
Detects and crops the primary face in a frame before handing it to the
trained AI-image classifier (pipeline/forensics/ai_classifier.py).

WHY THIS EXISTS (session 5 finding)
------------------------------------
The classifier currently wired in (prithivMLmods/deepfake-detector-model-v1,
a SigLIP-based binary "fake"/"real" model tagged deep-fake/detection) was
validated on real VIDEO frames for the first time this session and found
to read BACKWARDS on 2 of 4 test videos -- a confirmed-AI video averaged
0.27 (mostly "real"), and a confirmed-real video averaged 0.78 (mostly
"AI"). That's not just non-discriminative (like noise-floor on video), it's
actively wrong -- worse than a coin flip on half the sample.

`facenet-pytorch` (MTCNN) has been sitting in requirements.txt with a
"face detection & alignment" comment since before this rebuild, but was
never actually wired into the current ImageDetector/VideoDetector pipeline
-- FrameResult.metadata's `face_detected` field is hardcoded True
everywhere rather than computed (see tests/unit/test_video_detector.py's
predecessor, an abandoned MTCNN+EfficientNet design). Deepfake/face-swap
classifiers like this one are almost always trained on cropped, roughly-
centered face images, not full raw camera frames with background --
feeding whole ffmpeg-extracted video frames (arbitrary aspect ratio,
faces small/off-center, lots of non-face background) is a plausible,
concrete explanation for the backwards readings: a genuine input-domain
mismatch, not a preprocessing bug (the classifier's own AutoImageProcessor
already handles resize/normalisation correctly).

This module is intentionally narrow: detect the single largest/most
confident face, crop it with a margin, and return None if no face is
found (callers should skip the classifier for that frame rather than feed
it a whole-frame image it was never trained on -- see the
face_crop_before_classifier flag on ImageDetector.detect).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


class FaceCropper:
    """Lazy-loads MTCNN on first use. Stays inert (available=False) if
    facenet-pytorch/torch aren't installed, mirroring AIClassifier's own
    graceful-unavailability pattern -- callers should treat "no crop
    available" the same way they already treat "no classifier available".
    """

    def __init__(self, margin: float = 0.4, min_face_size: int = 20, min_confidence: float = 0.90):
        self.margin = margin
        self.min_face_size = min_face_size
        # Session 6 finding: ai_noaud2.mp4 (an AI-generated video of a hand,
        # no face anywhere) was getting a "face found" on 1/10 frames.
        # MTCNN.detect() returns every candidate box its cascade proposes,
        # including low-confidence ones -- it deliberately leaves filtering
        # to the caller via the returned `probs`, which this class wasn't
        # doing at all. Skin-tone/blob-like regions (a hand, in this case)
        # can pass the cascade at low confidence. 0.90 is a conservative cut
        # that rejects that class of false positive while still passing the
        # near-1.0 confidence MTCNN reports on genuine frontal/near-frontal
        # faces in this pipeline's other test videos.
        self.min_confidence = min_confidence
        self.available = False
        self.load_error: Optional[str] = None
        self._mtcnn = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from facenet_pytorch import MTCNN
            self._mtcnn = MTCNN(
                keep_all=True, min_face_size=self.min_face_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.available = True
        except Exception as exc:  # noqa: BLE001 — stay inert, don't crash the pipeline
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.available = False

    @staticmethod
    def _has_coherent_face_geometry(points: Optional[np.ndarray], box: np.ndarray) -> bool:
        """
        Sanity-checks MTCNN's 5-point landmark output (left eye, right eye,
        nose, mouth-left, mouth-right, in that fixed order) against what a
        real face's layout looks like, independent of the cascade's own
        confidence score. Rejects candidates with no landmarks at all (the
        regressor failed to converge on anything face-like) or with a
        layout that doesn't look like a face:
          - eyes roughly level with each other (not stacked vertically)
          - nose sits below the eye line and above the mouth line
          - mouth corners sit below the nose
          - overall box isn't a wildly non-face aspect ratio
        Any real face at a normal camera angle satisfies all of these by
        a wide margin; a hand, patch of skin, or other blob that merely
        scores well on "face-shaped enough" usually violates at least one.
        """
        if points is None:
            return False
        try:
            left_eye, right_eye, nose, mouth_l, mouth_r = points
        except (TypeError, ValueError):
            return False

        x1, y1, x2, y2 = box
        box_h = float(y2 - y1)
        box_w = float(x2 - x1)
        if box_h <= 0 or box_w <= 0:
            return False

        # Faces MTCNN detects are roughly upright rectangles, not
        # extremely elongated blobs (a hand/forearm often is).
        aspect = box_h / box_w
        if not (0.7 <= aspect <= 2.3):
            return False

        eye_y_avg = (left_eye[1] + right_eye[1]) / 2.0
        mouth_y_avg = (mouth_l[1] + mouth_r[1]) / 2.0

        # Eyes should be roughly level with each other, not one far above
        # the other (relative to the box height, so it scales with pose).
        if abs(left_eye[1] - right_eye[1]) > 0.35 * box_h:
            return False

        # Vertical ordering: eyes above nose, nose above mouth.
        if not (eye_y_avg < nose[1] < mouth_y_avg):
            return False

        # Nose should sit horizontally between the eyes (with some slack
        # for head turn), not off to one side entirely.
        eye_x_min, eye_x_max = min(left_eye[0], right_eye[0]), max(left_eye[0], right_eye[0])
        eye_span = eye_x_max - eye_x_min
        if eye_span > 0 and not (eye_x_min - 0.6 * eye_span <= nose[0] <= eye_x_max + 0.6 * eye_span):
            return False

        return True

    def crop(self, image_rgb: np.ndarray) -> dict[str, Any]:
        """
        Returns
        -------
        dict with keys:
          face_found (bool), crop (np.ndarray | None) -- the cropped face
          region with margin applied, box (tuple | None) -- (x1,y1,x2,y2)
          in original-image pixel coordinates, confidence (float | None).

        If multiple faces are detected, the largest (by box area) is used
        -- the primary subject is almost always the biggest face in a
        selfie/portrait-style clip, which is what this pipeline's video
        test set has consisted of so far.
        """
        if not self.available:
            return {"face_found": False, "crop": None, "box": None,
                     "confidence": None, "error": self.load_error}

        try:
            from PIL import Image
            img = Image.fromarray(image_rgb.astype(np.uint8))
            boxes, probs, landmarks = self._mtcnn.detect(img, landmarks=True)
        except Exception as exc:  # noqa: BLE001
            return {"face_found": False, "crop": None, "box": None,
                     "confidence": None, "error": f"{type(exc).__name__}: {exc}"}

        if boxes is None or len(boxes) == 0:
            return {"face_found": False, "crop": None, "box": None,
                     "confidence": None, "error": None}

        # Drop any candidate below min_confidence before picking the
        # largest -- MTCNN.detect() intentionally doesn't do this itself.
        if probs is not None:
            keep = [i for i, p in enumerate(probs) if p is not None and p >= self.min_confidence]
        else:
            keep = list(range(len(boxes)))
        if not keep:
            return {"face_found": False, "crop": None, "box": None,
                     "confidence": None, "error": None}

        # Session 6 finding #2: min_confidence alone wasn't enough. On
        # ai_noaud2.mp4 (an AI video of just a hand), one frame's hand
        # region scored >=0.90 confidence from the cascade classifier
        # itself -- a genuinely hard false positive, not something a
        # stricter score cutoff can reliably separate from real faces
        # without also risking rejecting real side-angle/partial faces.
        # Add an independent, geometry-based check on the 5-point
        # landmarks MTCNN also produces (left eye, right eye, nose, mouth
        # corners): a hand or other skin-toned blob can score well on the
        # cascade's "is this face-shaped" pass, but its landmark regressor
        # output won't form a coherent face layout (eyes level with each
        # other, nose below and between them, mouth below the nose). This
        # is a different failure mode than the confidence score and
        # catches false positives the score alone misses.
        keep = [i for i in keep if self._has_coherent_face_geometry(
            landmarks[i] if landmarks is not None else None, boxes[i]
        )]
        if not keep:
            return {"face_found": False, "crop": None, "box": None,
                     "confidence": None, "error": None}

        boxes = [boxes[i] for i in keep]
        probs = [probs[i] for i in keep] if probs is not None else None

        # Largest face by box area (most likely the primary subject).
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        idx = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes[idx]
        confidence = float(probs[idx]) if probs is not None else None

        h, w = image_rgb.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        mx, my = bw * self.margin, bh * self.margin
        cx1 = max(0, int(x1 - mx))
        cy1 = max(0, int(y1 - my))
        cx2 = min(w, int(x2 + mx))
        cy2 = min(h, int(y2 + my))

        if cx2 <= cx1 or cy2 <= cy1:
            return {"face_found": False, "crop": None, "box": None,
                     "confidence": None, "error": "degenerate crop region"}

        crop = image_rgb[cy1:cy2, cx1:cx2]
        return {
            "face_found": True, "crop": crop, "box": (cx1, cy1, cx2, cy2),
            "confidence": confidence, "error": None,
        }
