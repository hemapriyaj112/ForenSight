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

import os
from typing import Any, Optional

import numpy as np


class FaceCropper:
    """Lazy-loads MTCNN on first use. Stays inert (available=False) if
    facenet-pytorch/torch aren't installed, mirroring AIClassifier's own
    graceful-unavailability pattern -- callers should treat "no crop
    available" the same way they already treat "no classifier available".
    """

    def __init__(self, margin: float = 0.4, min_face_size: int = 20, min_confidence: float = 0.80):
        self.margin = margin
        self.min_face_size = min_face_size
        # Session 6 finding: ai_noaud2.mp4 (an AI-generated video of a hand,
        # no face anywhere) was getting a "face found" on 1/10 frames.
        # MTCNN.detect() returns every candidate box its cascade proposes,
        # including low-confidence ones -- it deliberately leaves filtering
        # to the caller via the returned `probs`, which this class wasn't
        # doing at all. Skin-tone/blob-like regions (a hand, in this case)
        # can pass the cascade at low confidence. The cut was originally
        # 0.90, but that rejected genuine faces in AI-generated footage
        # (MTCNN reports noticeably lower confidence on synthetic skin than
        # on photographic faces), silently zeroing out face detection for
        # whole videos -- exactly the content this pipeline must classify.
        # 0.80 still filters casual cascade noise; the geometry and
        # bilateral-texture checks below remain the real FP guards.
        self.min_confidence = min_confidence
        self.available = False
        self.load_error: Optional[str] = None
        # Which detector backend actually loaded: "mtcnn" | "opencv_haar".
        # Surfaced to callers/metadata so a silent downgrade is impossible:
        # if MTCNN can't load we fall back to an OpenCV Haar cascade (weaker
        # -- no landmarks, so the geometry/texture checks are skipped) and
        # say so; only when BOTH are unavailable does detection go dark.
        self.backend: Optional[str] = None
        self._mtcnn = None
        self._haar = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from facenet_pytorch import MTCNN
            self._mtcnn = MTCNN(
                keep_all=True, min_face_size=self.min_face_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.backend = "mtcnn"
            self.available = True
            return
        except Exception as mtcnn_exc:  # noqa: BLE001 — fall through to OpenCV
            mtcnn_err = f"{type(mtcnn_exc).__name__}: {mtcnn_exc}"

        try:
            import cv2
            cascade_dir = getattr(cv2.data, "haarcascades", "")
            cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
            if not cascade_dir or not os.path.exists(cascade_path):
                raise FileNotFoundError(f"haarcascade_frontalface_default.xml not found under {cascade_dir!r}")
            haar = cv2.CascadeClassifier(cascade_path)
            if haar.empty():
                raise RuntimeError(f"failed to load cascade from {cascade_path}")
            self._haar = haar
            self.backend = "opencv_haar"
            self.available = True
            self.load_error = f"facenet_pytorch unavailable ({mtcnn_err}); degraded to OpenCV Haar cascade"
        except Exception as haar_exc:  # noqa: BLE001 — stay inert, don't crash the pipeline
            self.load_error = f"facenet_pytorch: {mtcnn_err}; opencv fallback: {type(haar_exc).__name__}: {haar_exc}"
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

    @staticmethod
    def _local_variance(gray: np.ndarray, cx: float, cy: float, patch: int = 7) -> Optional[float]:
        """Laplacian variance (local edge/contrast energy) in a small patch
        centred on (cx, cy). Returns None if the patch would run off the
        edge of the image."""
        h, w = gray.shape
        half = patch // 2
        x0, y0 = int(cx) - half, int(cy) - half
        x1, y1 = x0 + patch, y0 + patch
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None
        region = gray[y0:y1, x0:x1]
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        padded = np.pad(region, 1, mode="reflect")
        out = np.zeros_like(region, dtype=np.float32)
        for i in range(3):
            for j in range(3):
                out += kernel[i, j] * padded[i : i + region.shape[0], j : j + region.shape[1]]
        return float(np.var(out))

    def _has_bilateral_texture_symmetry(
        self, image_rgb: np.ndarray, points: Optional[np.ndarray],
        max_asymmetry: float = 5.0, min_edge_energy: float = 2.0,
    ) -> bool:
        """
        Session 6 finding #3: the geometry check above (eyes level, nose
        between/below them, etc.) didn't stop ai_noaud2.mp4's hand false
        positive, because MTCNN's landmark regressor was only ever trained
        on real faces -- fed a non-face patch, it tends to output
        coordinates that still form a *plausible-looking* face layout
        rather than garbage, so a purely geometric check can't tell the
        difference.

        Real diagnostic data from that exact false-positive frame showed
        the actual discriminator: a real face's two eyes have local
        contrast/edge energy of the same rough order of magnitude as each
        other (same for the two mouth corners) -- both are the same kind
        of feature. On the hand, one eye-landmark sat on a low-contrast
        patch of skin (variance ~198) while the other happened to land on
        a knuckle/finger-gap edge (variance ~2626, a 13x gap); the two
        mouth-corner points showed an even starker gap (~5.7 vs. ~3498,
        over 600x). A real face doesn't produce that kind of left/right
        asymmetry. This checks for it directly, on top of (not instead
        of) the geometry check.

        Session 7 recalibration: min_edge_energy was originally 8.0, which
        turned out to be a false-negative machine -- AI-GENERATED faces are
        rendered smooth, so BOTH eye patches often sit below 8.0 variance
        while staying symmetric (ratio ~1), and perfectly plausible faces
        were rejected wholesale: 0/46 faces detected across an entire
        multi-face AI video, the classifier never ran once, and the video
        was called REAL from texture alone. The floor's only job is to stop
        two trivially-equal near-zero values (a fully blank patch) sailing
        through the ratio check; 2.0 does that while letting smooth but
        structured faces through. The asymmetry ratio remains the actual FP
        discriminator.
        """
        if points is None:
            return False
        try:
            left_eye, right_eye, _nose, mouth_l, mouth_r = points
        except (TypeError, ValueError):
            return False

        gray = (0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2])
        l_eye_var = self._local_variance(gray, *left_eye)
        r_eye_var = self._local_variance(gray, *right_eye)
        l_mouth_var = self._local_variance(gray, *mouth_l)
        r_mouth_var = self._local_variance(gray, *mouth_r)
        if None in (l_eye_var, r_eye_var, l_mouth_var, r_mouth_var):
            # Landmark patch runs off the frame edge -- not enough data to
            # judge this check either way, don't reject on it alone.
            return True

        # Guards against two trivially-equal near-zero values (a fully
        # flat/blank patch) sailing through the ratio check below.
        if max(l_eye_var, r_eye_var) < min_edge_energy:
            return False

        def _ratio(a: float, b: float) -> float:
            lo = max(min(a, b), 1e-6)
            return max(a, b) / lo

        return (_ratio(l_eye_var, r_eye_var) <= max_asymmetry
                and _ratio(l_mouth_var, r_mouth_var) <= max_asymmetry)

    def _no_face(self, error: Optional[str] = None) -> dict[str, Any]:
        return {"face_found": False, "crop": None, "box": None,
                "confidence": None, "error": error, "landmarks": None,
                "backend": self.backend}

    def crop(self, image_rgb: np.ndarray) -> dict[str, Any]:
        """
        Returns
        -------
        dict with keys:
          face_found (bool), crop (np.ndarray | None) -- the cropped face
          region with margin applied, box (tuple | None) -- (x1,y1,x2,y2)
          in original-image pixel coordinates, confidence (float | None),
          landmarks (array | None), error (str | None), backend (str | None).

        If multiple faces are detected, the largest (by box area) is used
        -- the primary subject is almost always the biggest face in a
        selfie/portrait-style clip, which is what this pipeline's video
        test set has consisted of so far.
        """
        if not self.available:
            return self._no_face(self.load_error)
        if self.backend == "mtcnn":
            return self._crop_mtcnn(image_rgb)
        if self.backend == "opencv_haar":
            return self._crop_haar(image_rgb)
        return self._no_face(f"unknown face-detection backend {self.backend!r}")

    def _crop_mtcnn(self, image_rgb: np.ndarray) -> dict[str, Any]:
        try:
            from PIL import Image
            img = Image.fromarray(image_rgb.astype(np.uint8))
            boxes, probs, landmarks = self._mtcnn.detect(img, landmarks=True)
        except Exception as exc:  # noqa: BLE001
            return self._no_face(f"{type(exc).__name__}: {exc}")

        if boxes is None or len(boxes) == 0:
            return self._no_face()

        # Drop any candidate below min_confidence before picking the
        # largest -- MTCNN.detect() intentionally doesn't do this itself.
        if probs is not None:
            keep = [i for i, p in enumerate(probs) if p is not None and p >= self.min_confidence]
        else:
            keep = list(range(len(boxes)))
        if not keep:
            return self._no_face()

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
            return self._no_face()

        # Session 6 finding #3: real diagnostic data showed the geometry
        # check alone still isn't enough (see _has_bilateral_texture_
        # symmetry's docstring) -- add the content-based check as a
        # second, independent filter.
        keep = [i for i in keep if self._has_bilateral_texture_symmetry(
            image_rgb, landmarks[i] if landmarks is not None else None
        )]
        if not keep:
            return self._no_face()

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
            return self._no_face("degenerate crop region")

        crop = image_rgb[cy1:cy2, cx1:cx2]
        selected_landmarks = landmarks[idx] if landmarks is not None else None
        return {
            "face_found": True, "crop": crop, "box": (cx1, cy1, cx2, cy2),
            "confidence": confidence, "error": None,
            "landmarks": selected_landmarks, "backend": self.backend,
        }

    def _crop_haar(self, image_rgb: np.ndarray) -> dict[str, Any]:
        """
        Degraded fallback path used when facenet-pytorch/torchvision can't
        be imported (the silent total-outage failure mode that produced
        '0 faces detected' across entire videos). The Haar cascade yields
        boxes only -- no confidence scores, no landmarks -- so the
        confidence filter and both landmark-based checks can't run here;
        candidates instead pass an aspect-ratio sanity bound (the same
        bounds the geometry check uses). This backend is weaker than
        MTCNN, but a weak detector that WORKS beats a strong one that is
        silently absent: the caller sees `backend="opencv_haar"` and can
        surface the degradation instead of misreading it as faceless
        footage.
        """
        try:
            import cv2
            gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            raw = self._haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
            )
        except Exception as exc:  # noqa: BLE001
            return self._no_face(f"{type(exc).__name__}: {exc}")

        if raw is None or len(raw) == 0:
            return self._no_face()

        boxes = []
        for fx, fy, fw, fh in raw:
            aspect = float(fh) / max(float(fw), 1e-6)
            if not (0.7 <= aspect <= 2.3):
                continue  # wildly non-face-shaped blob (limb, elongated object)
            boxes.append((float(fx), float(fy), float(fx + fw), float(fy + fh)))
        if not boxes:
            return self._no_face()

        # Largest candidate wins (primary subject), mirroring the MTCNN path.
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        idx = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes[idx]

        h, w = image_rgb.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        mx, my = bw * self.margin, bh * self.margin
        cx1 = max(0, int(x1 - mx))
        cy1 = max(0, int(y1 - my))
        cx2 = min(w, int(x2 + mx))
        cy2 = min(h, int(y2 + my))

        if cx2 <= cx1 or cy2 <= cy1:
            return self._no_face("degenerate crop region")

        return {
            "face_found": True, "crop": image_rgb[cy1:cy2, cx1:cx2],
            "box": (cx1, cy1, cx2, cy2), "confidence": None, "error": None,
            "landmarks": None, "backend": "opencv_haar",
        }
