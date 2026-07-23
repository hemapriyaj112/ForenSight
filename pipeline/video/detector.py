"""pipeline/video/detector.py

Contains:
  _GradCAM          — gradient-weighted class activation mapping helper
  _FrequencyAnalyser — FFT-based artefact detector
  VideoDetector     — per-frame deepfake detection for video
  ImageDetector     — single-image deepfake detection (GradCAM + FFT fusion)
"""
from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from utils.types import FrameResult, ModalResult, ImageResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAKE_THRESHOLD = 0.6
_REAL_THRESHOLD = 0.4
_GRADCAM_WEIGHT = 0.6
_FREQ_WEIGHT    = 0.4

# Image extensions recognised by VideoDetector.run()
_FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assign_verdict(score: float) -> str:
    if score >= _FAKE_THRESHOLD:
        return "FAKE"
    if score <= _REAL_THRESHOLD:
        return "REAL"
    return "UNCERTAIN"


def _png_bytes(array: np.ndarray) -> bytes:
    from PIL import Image
    img = Image.fromarray(array.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _apply_colormap_jet(gray: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4.0 * gray - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * gray - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * gray - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _load_frame(path: Path) -> np.ndarray:
    """Load an image file as HxWx3 uint8 RGB array."""
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _frame_index_to_timestamp(index: int, fps: float = 25.0) -> float:
    return index / fps


# ---------------------------------------------------------------------------
# _GradCAM
# ---------------------------------------------------------------------------

class _GradCAM:
    def __init__(self, model=None, target_layer: str = "layer4"):
        self._model        = model
        self._target_layer = target_layer

    def score_and_overlay(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        if self._model is not None:
            return self._real_forward(image_rgb)
        return self._stub_forward(image_rgb)

    def _stub_forward(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        gray = (0.299 * image_rgb[..., 0]
                + 0.587 * image_rgb[..., 1]
                + 0.114 * image_rgb[..., 2])
        lap     = self._laplacian(gray)
        var_lap = float(np.var(lap))

        # U-shaped scoring, not monotonic. The original version treated
        # "smooth" as unconditionally "real camera photo" — but oversmoothed,
        # denoised skin/backgrounds are exactly what many AI portraits look
        # like. Natural photos cluster in a mid-range of local edge variance;
        # BOTH extremes (unnaturally smooth AND unnaturally noisy/artifacty)
        # are treated as suspicious here.
        log_var = float(np.log10(var_lap + 1e-6))
        center, half_width = -1.4, 0.6
        distance  = max(0.0, abs(log_var - center) - half_width)
        fake_prob = float(np.clip(distance / 1.2, 0.0, 1.0))

        gy = np.gradient(gray, axis=0)
        gx = np.gradient(gray, axis=1)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        return fake_prob, _apply_colormap_jet(magnitude)

    def _real_forward(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray]:  # pragma: no cover
        raise NotImplementedError("Real model inference not wired in this build.")

    @staticmethod
    def _laplacian(gray: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        h, w   = gray.shape
        padded = np.pad(gray, 1, mode="reflect")
        out    = np.zeros((h, w), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                out += kernel[i, j] * padded[i : i + h, j : j + w]
        return out


# ---------------------------------------------------------------------------
# _FrequencyAnalyser
# ---------------------------------------------------------------------------

class _FrequencyAnalyser:
    """
    Spectral-artefact analyser.

    Real photographs follow a well-documented natural-image statistic: the
    radially-averaged power spectrum falls off roughly as a power law
    (~1/f^alpha, alpha typically ~1.5-2.5) with no strong periodic peaks.
    GAN/diffusion upsampling layers (transposed convolutions, pixel-shuffle,
    repeated attention blocks) tend to leave two kinds of tells:
      1. A flatter-than-natural high-frequency falloff (too much energy in
         fine detail relative to a real camera/lens/sensor chain).
      2. Periodic peaks in the azimuthally-averaged spectrum (checkerboard-
         style artefacts), visible as local maxima that spike above their
         neighbours.

    This replaces the old crude "energy outside a central disk" ratio,
    which mostly just measured how detailed/busy an image was and had
    little to do with AI-generation specifically.
    """

    def score_and_heatmap(self, image_rgb: np.ndarray) -> dict[str, object]:
        gray      = (0.299 * image_rgb[..., 0]
                     + 0.587 * image_rgb[..., 1]
                     + 0.114 * image_rgb[..., 2])
        magnitude = self._fft_magnitude(gray)
        log_mag   = np.log1p(magnitude)

        radial_profile, radii = self._radial_average(magnitude)
        findings: list[str] = []

        slope_score = self._slope_score(radial_profile, radii)
        peak_score, has_peak = self._periodic_peak_score(radial_profile)

        if peak_score > 0.5:
            findings.append(
                "Periodic peaks were detected in the frequency spectrum "
                "(a checkerboard-style artefact often left by GAN/diffusion "
                "upsampling layers)."
            )
        if slope_score > 0.6:
            findings.append(
                "The image's high-frequency energy falloff deviates from "
                "the pattern typical of real camera photos."
            )
        if slope_score < 0.3 and peak_score < 0.3:
            findings.append(
                "The frequency spectrum follows a natural falloff pattern "
                "with no periodic artefacts, consistent with a real photo."
            )

        fake_prob = float(np.clip(0.5 * slope_score + 0.5 * peak_score, 0.0, 1.0))

        if log_mag.max() > 0:
            log_mag = log_mag / log_mag.max()
        heatmap = _apply_colormap_jet(log_mag)

        return {
            "score": fake_prob,
            "heatmap": heatmap,
            "findings": findings,
            "slope_score": slope_score,
            "peak_score": peak_score,
        }

    @staticmethod
    def _fft_magnitude(gray: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.fftshift(np.fft.fft2(gray)))

    @staticmethod
    def _radial_average(magnitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.indices((h, w))
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
        max_r = min(cy, cx)
        r = np.clip(r, 0, max_r)
        sums = np.bincount(r.ravel(), weights=magnitude.ravel(), minlength=max_r + 1)
        counts = np.bincount(r.ravel(), minlength=max_r + 1)
        profile = sums / np.maximum(counts, 1)
        return profile, np.arange(max_r + 1)

    @staticmethod
    def _slope_score(profile: np.ndarray, radii: np.ndarray) -> float:
        # Fit log-log slope over the mid-frequency band (skip DC + extreme
        # edge, which are noisy). Natural photos: slope roughly -1.0 to -3.0.
        n = len(profile)
        if n < 20:
            return 0.3
        lo, hi = max(2, int(n * 0.1)), int(n * 0.85)
        r = radii[lo:hi].astype(np.float64)
        p = profile[lo:hi].astype(np.float64)
        valid = p > 0
        if valid.sum() < 10:
            return 0.3
        log_r = np.log(r[valid])
        log_p = np.log(p[valid])
        slope, _ = np.polyfit(log_r, log_p, 1)
        # Natural images cluster around slope in [-3.0, -1.0]. Score rises
        # the further outside that band we land (too flat = too much fine
        # detail energy, common in upsampled/generated images).
        if -3.0 <= slope <= -1.0:
            return 0.15
        distance = min(abs(slope - (-1.0)), abs(slope - (-3.0)))
        return float(np.clip(distance / 2.0, 0.0, 1.0))

    @staticmethod
    def _periodic_peak_score(profile: np.ndarray) -> tuple[float, bool]:
        n = len(profile)
        if n < 20:
            return 0.0, False
        # Look at mid/high frequency band for local maxima that spike well
        # above their local neighbourhood average.
        band = profile[int(n * 0.3):int(n * 0.95)]
        if len(band) < 10 or band.mean() <= 0:
            return 0.0, False
        window = 5
        max_ratio = 0.0
        for i in range(window, len(band) - window):
            local = np.concatenate([band[i - window:i], band[i + 1:i + window + 1]])
            local_mean = local.mean() + 1e-9
            ratio = band[i] / local_mean
            max_ratio = max(max_ratio, ratio)
        # ratio ~1 = no peak; ratio > ~2.5 = clear periodic spike
        score = float(np.clip((max_ratio - 1.5) / 2.0, 0.0, 1.0))
        return score, max_ratio > 2.5


# ---------------------------------------------------------------------------
# VideoDetector
# ---------------------------------------------------------------------------

def _video_provenance(raw_bytes: bytes, filename: Optional[str] = None) -> dict:
    """
    Container-level provenance check for a whole video file — the video
    analog of metadata_forensics.analyse_metadata() for images.

    Reuses the same module because the part that matters most for video
    (the C2PA/Content-Credentials byte-scan and the generator-signature
    text scan) operates on raw bytes regardless of container format, and
    an increasing number of AI video tools (Sora, Runway, etc.) embed C2PA
    manifests or software tags in the MP4 container the same way image
    tools embed them in PNG/JPEG. The one part that doesn't apply is
    EXIF/PNG-chunk parsing (PIL can't open an MP4 as an image) — when that
    fails we swap in a video-appropriate message instead of the image
    module's "could not read image metadata" line, which would be
    misleading here (it's not a failure, it's just not an image).
    """
    from pipeline.forensics import metadata_forensics as _meta

    out = _meta.analyse_metadata(raw_bytes, filename=filename)
    generic_failure = "Could not read image metadata (file may be corrupted or stripped)."
    if generic_failure in out["findings"]:
        out["findings"] = [f for f in out["findings"] if f != generic_failure]
        if not out["generator_tag"] and not out["has_c2pa"]:
            out["findings"].append(
                "No embedded provenance (Content Credentials / generator tag) "
                "found in the video container. This is common and not itself "
                "suspicious — most cameras and editing tools don't embed C2PA "
                "manifests in video files today."
            )
    return out


# ---------------------------------------------------------------------------
# VideoDetector
# ---------------------------------------------------------------------------

class VideoDetector:
    """Per-frame deepfake detector for video streams.

    Brought to parity with ImageDetector (Sprint 9), then corrected
    repeatedly based on real-world testing (Sprint 10, session 5): every
    sampled frame runs through texture analysis (and the trained AI-vs-real
    classifier, if wired in), with document/screen-recording frames gated
    from the classifier the same way document photos are. FOUR of
    ImageDetector's five signals are deliberately EXCLUDED per frame, each
    for a specific, evidence-based reason (image pipeline keeps all five —
    these exclusions are video-specific):

      - metadata/EXIF: an ffmpeg-extracted frame never carries real camera
        EXIF regardless of the source video's authenticity, so this would
        just inject a constant ~0.55 "no info" score into every frame.
      - noise-floor: confirmed via testing to have ~zero discriminative
        power for video specifically — H.264 (and similar codecs) smooth
        flat regions as routine compression behavior, so this scored
        0.88-0.94 on BOTH a known-real phone video and a known-AI video
        (see include_noise's docstring on ImageDetector.detect for the
        measurements). Including it was actively harming real-video
        accuracy (a known-real clip scored 0.476 "Uncertain" with it vs.
        0.174 "Real" without) while adding no real detection benefit.
      - ELA: worse than non-discriminative — confirmed *backwards* for
        video. H.264 allocates bits very unevenly per macroblock based on
        local scene complexity (a real face/hair region gets far less
        quantization than flat background in the SAME real, unedited
        frame); re-JPEG-compressing an already-codec-processed frame then
        measures that pre-existing complexity variance, not editing. Mean
        ELA score was 0.98 (saturated) on a known-real phone video vs.
        0.42 on a known-AI video — the opposite of what the signal means.
        This alone was enough to push a genuinely real video into
        "Uncertain" (see include_ela's docstring on ImageDetector.detect
        for the full measurements).
      - spectral (session 5): sat at its exact floor value (0.075) on
        every real AND AI video frame tested — the FFT periodic-peak check
        is a GAN-era tell that modern diffusion-based generators mostly
        don't leave, and the log-log slope check has a hardcoded floor
        whenever the falloff is anywhere in a broad "natural" range. With
        spectral holding ~57% of the video-only fusion weight, its floor
        value created a hard ceiling (~0.47) on the fused score that a
        maximally-confident texture reading couldn't clear — caught
        directly when a real, un-provenanced AI clip with texture pinned
        at 0.86-1.0 on every frame still landed UNCERTAIN, both with and
        without the trained classifier active (see include_spectral's
        docstring on ImageDetector.detect for the full measurements).

    NOTE: texture ("GradCAM" heuristic) is now the SOLE per-frame heuristic
    for video (before classifier weighting), which raises its stakes
    considerably. It has shown wide, inconsistent swings across different
    real videos in testing (near-0 on one, ~0.83-1.0 on another) that look
    driven by frame sharpness/lighting rather than authenticity — this
    tradeoff (better AI recall, but real "smooth" video is now more exposed
    to false positives) is flagged, not resolved. Needs close monitoring on
    more real footage before being treated as a settled default.

    In their place, two whole-video signals are added:

      - Container-level provenance: a single C2PA/generator-tag scan of the
        raw video file (see _video_provenance above). A confirmed AI-tool
        tag overrides the fused score the same way it does for images
        (floors it at 0.9) rather than being diluted into a per-signal share.
      - Temporal localization: which stretch of the timeline (if any) looks
        edited, built from the per-frame fused-score sequence — the time-
        axis analog of the existing spatial "where" localization, which is
        also computed here (on the single most-suspicious frame) so a
        video gets both a "where" and a "when".

    Two calling conventions (both supported):
      detect(frames)        — list of HxWx3 uint8 arrays  (used by tests)
      run(frames_dir, cfg)  — Path to directory of frame images (used by main.py)
    """

    def __init__(self, cfg=None, model=None, ai_classifier=None, face_cropper=None):
        # cfg is accepted for main.py compatibility but not used in stub mode
        self._cfg = cfg
        self._ai_classifier = ai_classifier  # optional AIClassifier instance, shared with ImageDetector
        # Lazily create a FaceCropper if the caller didn't supply one — mirrors
        # AIClassifier's own lazy-load-then-stay-inert-on-failure pattern (see
        # face_crop_before_classifier's docstring on ImageDetector.detect for
        # why video frames need this before hitting the classifier).
        if face_cropper is None:
            from pipeline.forensics.face_crop import FaceCropper
            face_cropper = FaceCropper()
        self._face_cropper = face_cropper
        self._image_detector = ImageDetector(
            model=model, ai_classifier=ai_classifier, face_cropper=face_cropper,
        )

    # ── Called by main.py ────────────────────────────────────────────────────

    def run(self, frames_dir: str | Path, video_bytes: Optional[bytes] = None,
            video_path: Optional[str] = None) -> ModalResult:
        """Load frames from *frames_dir* and run detect() on them.

        Frame files are sorted alphabetically so ordering is deterministic.
        Returns a ModalResult with fake_prob and frame_results populated.

        video_bytes/video_path (optional): the original video file, for the
        container-level provenance scan. If omitted, provenance is skipped
        (frame-level signals still run) — main.py passes these when available.
        """
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            return ModalResult(
                modality="video", fake_prob=0.5,
                error=f"frames_dir not found: {frames_dir}",
            )

        frame_paths = sorted(
            p for p in frames_dir.iterdir()
            if p.suffix.lower() in _FRAME_EXTS
        )

        if not frame_paths:
            return ModalResult(
                modality="video", fake_prob=0.5,
                error=f"No frame images found in {frames_dir}",
            )

        frames = [_load_frame(p) for p in frame_paths]

        # fps used for real per-frame timestamps (config carries the sampling
        # rate frames were extracted at, e.g. video.frame_rate: 1 in
        # config.yaml — demux.py extracts at that same rate)
        fps = getattr(getattr(self._cfg, "video", None), "frame_rate", 1.0) or 1.0
        timestamps = [_frame_index_to_timestamp(i, fps=fps) for i in range(len(frames))]

        return self.detect(frames, timestamps=timestamps,
                            video_bytes=video_bytes, video_path=video_path)

    # ── Called by tests / main.py ────────────────────────────────────────────

    def detect(
        self,
        frames: list[np.ndarray],
        timestamps: Optional[list[float]] = None,
        video_bytes: Optional[bytes] = None,
        video_path: Optional[str] = None,
    ) -> ModalResult:
        """Analyse a list of HxWx3 uint8 frames → ModalResult.

        Each frame is run through the full ImageDetector fusion (texture,
        spectral, ELA, noise, +classifier if wired in and the frame doesn't
        look like a document/text page) rather than just texture+spectral.
        """
        if timestamps is None:
            timestamps = [float(i) for i in range(len(frames))]

        frame_results: list[FrameResult] = []
        documents_skipped = 0

        for idx, frame in enumerate(frames):
            img_result = self._image_detector.detect(
                frame, image_id=f"frame_{idx:06d}",
                include_metadata=False, include_noise=False, include_ela=False,
                include_spectral=False, face_crop_before_classifier=True,
            )

            frame_results.append(FrameResult(
                frame_index=idx,
                timestamp_sec=timestamps[idx] if idx < len(timestamps) else float(idx),
                fake_prob=img_result.fused_score,
                face_detected=img_result.metadata.get("face_detected", True),
                gradcam_overlay=img_result.gradcam_overlay,
                sub_scores=img_result.sub_scores,
                findings=img_result.findings,
                is_document=img_result.metadata.get("is_document", False),
                classifier_active=img_result.metadata.get("classifier_active", False),
                pre_ceiling_score=img_result.metadata.get("pre_ceiling_score"),
            ))
            if img_result.metadata.get("is_document", False):
                documents_skipped += 1

        overall = float(np.mean([fr.fake_prob for fr in frame_results])) if frame_results else 0.0

        # --- Video-level corroboration gate (aggregate) ---
        # ImageDetector.detect() already applies a PER-FRAME version of this
        # gate (texture alone can't push a single frame's score into FAKE
        # without that frame's own classifier reading corroborating it).
        # That's not sufficient on its own: if even a handful of frames
        # individually have a corroborating classifier reading while most
        # don't, those frames escape their own per-frame cap and pull the
        # AVERAGED video score back above the ceiling — even though the
        # video-wide average classifier reading (the number the UI actually
        # displays, e.g. "Classifier 40.1%") never corroborated at all.
        # Confirmed directly: a real video averaged texture=82.8%,
        # classifier=40.1%, and still fused to ~63% FAKE despite the
        # classifier clearly disagreeing on average, because a minority of
        # frames' individually-high classifier readings weren't capped.
        # Re-apply the same rule here using the same averaged numbers shown
        # in the UI, so the final video-level verdict can't drift out of
        # sync with what the person is actually shown.
        if frame_results:
            texture_vals = [fr.sub_scores.get("texture") for fr in frame_results
                             if fr.sub_scores.get("texture") is not None]
            classifier_vals = [fr.sub_scores.get("classifier") for fr in frame_results
                                if fr.sub_scores.get("classifier") is not None]
            texture_is_sole_heuristic = all(
                set(fr.sub_scores.keys()) <= {"texture", "classifier"} for fr in frame_results
            )
            if texture_is_sole_heuristic and texture_vals:
                avg_classifier = float(np.mean(classifier_vals)) if classifier_vals else None
                from utils.config import CFG as _CFG
                _fake_threshold = float(
                    getattr(getattr(_CFG, "fusion", None), "fake_threshold", 0.60)
                )
                _uncorroborated_ceiling = _fake_threshold - 0.01
                classifier_corroborates = (
                    avg_classifier is not None and avg_classifier >= _fake_threshold
                )
                if classifier_corroborates:
                    # The classifier actively agrees on the frames it could
                    # evaluate — trust texture uniformly across the WHOLE
                    # video rather than leaving `overall` as a mean of
                    # already-capped-per-frame values (which silently
                    # dilutes the score whenever most frames lacked a
                    # detected face/classifier reading and got capped at
                    # the texture-only ceiling individually). Recompute
                    # from the pre-ceiling scores instead. Falls back to
                    # the already-computed (possibly capped) fake_prob for
                    # any frame missing pre_ceiling_score for some reason.
                    pre_ceiling_vals = [
                        fr.pre_ceiling_score if fr.pre_ceiling_score is not None
                        else fr.fake_prob
                        for fr in frame_results
                    ]
                    overall = float(np.mean(pre_ceiling_vals))
                elif overall > _uncorroborated_ceiling:
                    overall = _uncorroborated_ceiling

        # --- Two-segment reporting (session 6, user-requested) ---
        # Split frames into a "face segment" (a face was detected/cropped)
        # and a "non-face segment" (it wasn't), and report each with its
        # own aggregate score/verdict — additional DISPLAY-only detail
        # alongside the single `overall` fused score above, which remains
        # the one number Fuser/main.py uses for the actual verdict.
        face_frames = [fr for fr in frame_results if fr.face_detected]
        non_face_frames = [fr for fr in frame_results if not fr.face_detected]

        def _segment_summary(frames_subset: list[FrameResult]) -> Optional[dict]:
            if not frames_subset:
                return None
            texture_subset = [
                fr.sub_scores.get("texture") for fr in frames_subset
                if fr.sub_scores.get("texture") is not None
            ]
            classifier_subset = [
                fr.sub_scores.get("classifier") for fr in frames_subset
                if fr.sub_scores.get("classifier") is not None
            ]
            # Segment score: mean of each frame's own fused fake_prob (which
            # already carries the per-frame classifier weighting/ceiling
            # logic) restricted to this segment's frames — consistent with
            # how `overall` itself is computed, just over a subset.
            segment_score = float(np.mean([fr.fake_prob for fr in frames_subset]))
            return {
                "frame_count": len(frames_subset),
                "avg_texture": float(np.mean(texture_subset)) if texture_subset else None,
                "avg_classifier": float(np.mean(classifier_subset)) if classifier_subset else None,
                "score": segment_score,
                "verdict": _assign_verdict(segment_score),
            }

        face_segment = _segment_summary(face_frames)
        non_face_segment = _segment_summary(non_face_frames)

        # --- container-level provenance (once per video, not per frame) ---
        provenance: dict = {}
        if video_bytes is not None:
            provenance = _video_provenance(video_bytes, filename=video_path)
            if provenance.get("generator_tag") and provenance.get("score", 0.0) >= 0.9:
                overall = max(overall, 0.9)

        # --- spatial "where": run on the single most-suspicious frame only
        # (running the full block-grid analysis on every frame is wasted
        # cost for frames that already look clean, and localization is only
        # meaningful for the frame(s) that actually look edited) ---
        spatial: dict = {}
        if frames:
            hottest_idx = int(np.argmax([fr.fake_prob for fr in frame_results]))
            from pipeline.forensics import localization as _loc
            spatial = _loc.build_suspicious_regions(frames[hottest_idx])
            spatial["frame_index"] = hottest_idx
            spatial["timestamp_sec"] = frame_results[hottest_idx].timestamp_sec

        # --- temporal "when": which part of the timeline looks edited ---
        from pipeline.forensics import temporal_localization as _temporal
        temporal = _temporal.build_suspicious_timeline(
            [fr.fake_prob for fr in frame_results],
            [fr.timestamp_sec for fr in frame_results],
        )

        metadata = {
            "generator_tag": provenance.get("generator_tag"),
            "has_c2pa": provenance.get("has_c2pa", False),
            "provenance_score": provenance.get("score"),
            "provenance_findings": provenance.get("findings", []),
            "classifier_backend": self._ai_classifier.backend if self._ai_classifier else None,
            "classifier_error": (
                self._ai_classifier.load_error
                if self._ai_classifier is not None and not self._ai_classifier.available
                else None
            ),
            "frames_analysed": len(frame_results),
            "documents_skipped": documents_skipped,
            "face_segment": face_segment,
            "non_face_segment": non_face_segment,
        }

        return ModalResult(
            modality="video",
            fake_prob=float(np.clip(overall, 0.0, 1.0)),
            frame_results=frame_results,
            metadata=metadata,
            spatial_localization=spatial,
            temporal_localization=temporal,
        )


# ---------------------------------------------------------------------------
# ImageDetector
# ---------------------------------------------------------------------------

class ImageDetector:
    """
    Single-image deepfake / AI-generation detector.

    Fuses five independent forensic signals — no one signal decides the
    verdict alone:
      noise     (0.40) — sensor noise-floor absence (strongest signal: real
                          cameras always leave measurable noise in flat
                          regions; clean AI generations usually don't)
      spectral  (0.20) — FFT radial-slope + periodic-peak analysis
      ela       (0.15) — error-level analysis (local editing inconsistency)
      metadata  (0.15) — EXIF / generator-tag / C2PA provenance check
      texture   (0.10) — stub sharpness/edge heuristic (weakest signal,
                          kept only as a minor corroborating input — see note)

    IMPORTANT: `texture` is a hand-written heuristic (Laplacian variance),
    not a trained neural network, and can be actively misleading for smooth
    AI portraits (it reads "smooth" as "real camera", which is backwards
    for oversmoothed generated skin) — that's why it's weighted lowest and
    corroborated by `noise`, which measures the same idea correctly. A real
    trained classifier (local model file or API) can be plugged in later
    via the `model=` constructor arg to `_GradCAM` — see BaseDetector in
    pipeline/forensics for the intended extension point.
    """

    _WEIGHTS = {"texture": 0.15, "spectral": 0.20, "ela": 0.15, "metadata": 0.15, "noise": 0.35}
    # Weight given to a real trained classifier when one is wired in via
    # `ai_classifier=`. Deliberately dominant: a supervised model looking
    # for actual generator fingerprints is categorically stronger evidence
    # than any of the 5 hand-written statistical heuristics above, which is
    # exactly the gap those heuristics can't close on their own (see
    # pipeline/forensics/ai_classifier.py). When no classifier is wired in,
    # this constant is unused and the original 5-signal weights (which sum
    # to 1.0 on their own) apply unchanged.
    _CLASSIFIER_WEIGHT = 0.45

    def __init__(self, model=None, ai_classifier=None, face_cropper=None):
        self._gradcam = _GradCAM(model=model)
        self._freq    = _FrequencyAnalyser()
        self._ai_classifier = ai_classifier  # optional AIClassifier instance
        self._face_cropper  = face_cropper   # optional FaceCropper instance (session 5)

    def detect(
        self,
        image_rgb: np.ndarray,
        image_id: Optional[str] = None,
        image_path: Optional[str] = None,
        raw_bytes: Optional[bytes] = None,
        include_metadata: bool = True,
        include_noise: bool = True,
        include_ela: bool = True,
        include_spectral: bool = True,
        face_crop_before_classifier: bool = False,
    ) -> ImageResult:
        """
        include_metadata: set False when the caller already knows per-image
        EXIF/PNG-chunk metadata is meaningless for this input — e.g.
        VideoDetector calling per sampled video frame, where an
        ffmpeg-extracted JPEG never carries real camera EXIF regardless of
        the source video's authenticity, so the signal would just add a
        constant ~0.55 "no info" score to every frame rather than real
        evidence. Video's actual provenance/C2PA check happens once at the
        container level instead — see pipeline/video/detector._video_provenance.
        When False, the metadata signal is dropped entirely and its weight
        is redistributed proportionally among the remaining signals.

        include_noise: set False when the caller's frames come from a
        compressed VIDEO stream rather than a standalone photo. Confirmed
        via real-world testing (Sprint 10): video codecs (H.264/etc.) apply
        flat-region smoothing as routine, universal compression behavior —
        unlike single-JPEG photos, where a suppressed noise floor is a
        genuine AI-generation signal. Measured noise-floor scores were
        0.88-0.94 on BOTH a known-real phone video and a known-AI-generated
        video — i.e. no discriminative power for video frames as currently
        calibrated, just a constant bias toward "suspicious". Excluding it
        moved a known-real phone video from 0.476 (Uncertain) to 0.174
        (Real) while costing negligible AI-detection sensitivity on the
        known-AI test case (whose detection was carried by the container-
        level provenance check, not the heuristics, either way). The image
        path is untouched — this only affects VideoDetector's per-frame calls.

        include_ela: set False for the same video-vs-photo reason, but with
        an even stronger, *backwards* finding (Sprint 10 testing): ELA
        assumes a real, untouched photo shows fairly uniform compression-
        error across blocks, and a locally edited/spliced region stands out
        as a hot outlier. But H.264 (and similar video codecs) already
        allocate bits very unevenly per macroblock based on local scene
        complexity — a real face/hair/detail region gets far less
        quantization than a flat background in the SAME real, unedited
        frame. Re-compressing an already-codec-processed frame at JPEG
        quality 90 then measures that pre-existing, complexity-driven
        variance, not editing. Result: mean ELA score was 0.98 (saturated)
        on a known-real phone video vs. 0.42 on a known-AI-generated video
        — the OPPOSITE of what the signal is supposed to indicate. This
        alone was enough to push a genuinely real video into "Uncertain".
        Excluding it moved that real video's heuristic-only average from
        0.572 (Uncertain) to 0.398 (crosses the 0.40 REAL threshold), while
        the known-AI video's average was essentially unchanged (0.453 vs
        0.467) — confirming no meaningful detection capability was lost.

        include_spectral: set False for video frames — CONFIRMED (session 5,
        via real un-provenanced AI clip): the FFT periodic-peak check is a
        GAN-era tell (checkerboard artefacts from transposed-conv/pixel-
        shuffle upsampling) that modern diffusion-based video generators
        mostly don't leave, and the log-log slope check has a hardcoded
        floor of 0.15 whenever the falloff sits anywhere in the broad
        natural range [-3.0, -1.0]. In every real AND AI video frame tested
        so far, both sub-checks landed at their floor (peak=0, slope=0.15),
        pinning spectral's score at exactly 0.075 regardless of ground
        truth — zero observed discriminative power, the same bar that
        already got noise/ela excluded above.
        Worse than inert: because spectral still holds real fusion weight
        (0.20 of 0.35, i.e. ~57% of the video-only budget), its 0.075 floor
        creates a hard ceiling. With spectral stuck at 0.075 and texture
        weighted ~43%, the max reachable fused score is
        0.429×1.0 + 0.571×0.075 ≈ 0.47 — BELOW the 0.60 fake_threshold even
        when texture is maximally confident (1.0) on every single frame.
        This was caught directly: a real un-provenanced AI clip with
        texture pinned at 0.86-1.0 across all 10 frames landed at 45.8%
        heuristic-only (59.2% with the trained classifier active) —
        UNCERTAIN both times, purely because of this ceiling, not because
        the evidence was actually ambiguous. Excluding spectral and
        redistributing its weight to texture removes the ceiling; the
        tradeoff (texture is also an imperfect signal — see its own
        docstring above) needs to be re-checked against real footage in
        both directions before this becomes a settled default.

        face_crop_before_classifier: set True for video frames (session 5).
        The wired-in classifier (prithivMLmods/deepfake-detector-model-v1)
        was validated against real video frames for the first time this
        session and found to read BACKWARDS on some real-world clips (a
        confirmed-AI video averaged 0.27 "mostly real"; a confirmed-real
        video averaged 0.78 "mostly AI"). facenet-pytorch (MTCNN) has been
        an unused dependency since before the ImageDetector-fusion rebuild
        — this class is very likely trained on cropped face images (it's
        tagged deep-fake/detection), not whole raw camera frames with
        background, so feeding it whole frames is an input-domain mismatch
        regardless of whether the resize/normalisation is technically
        correct (it is — the transformers backend already uses the model's
        own AutoImageProcessor). When True and a FaceCropper is wired in
        via the face_cropper constructor arg, the classifier only runs on
        the cropped primary face region; if no face is found, the
        classifier is skipped for that frame entirely (findings note why)
        rather than fed a whole-frame image it was never trained on.
        CAVEAT, tested directly: within one video with a genuine mixed
        face/no-face frame split, per-frame classifier scores were nearly
        identical whether or not a face was actually present (~0.245 vs
        ~0.287, both still backwards) — so face-cropping is the technically
        correct thing to do regardless, but is NOT confirmed to fix the
        backwards readings on its own. Needs re-testing on real video with
        this wired in before treating the classifier as trustworthy again.
        """
        from pipeline.forensics import ela as _ela
        from pipeline.forensics import explain as _explain
        from pipeline.forensics import metadata_forensics as _meta
        from pipeline.forensics import noise_residual as _noise

        if image_id is None:
            image_id = str(uuid.uuid4())[:8]

        rgb_f32 = image_rgb.astype(np.float32) / 255.0

        # --- texture (existing heuristic, kept as one signal among several) ---
        texture_score, texture_heatmap = self._gradcam.score_and_overlay(rgb_f32)
        gradcam_overlay_png = _png_bytes(
            self._blend_overlay(image_rgb, texture_heatmap, alpha=0.5)
        )
        texture_findings = []
        if texture_score > 0.6:
            texture_findings.append(
                "Local texture/edge patterns look unnaturally smooth or "
                "synthetic in areas that a real photo would show sensor noise."
            )
        elif texture_score < 0.3:
            texture_findings.append(
                "Texture and edge noise look consistent with a real camera photo."
            )

        # --- spectral (FFT) — skipped for video frames when include_spectral
        # is False; see include_spectral docstring above ---
        freq_out = None
        freq_score = None
        freq_heatmap_png = None
        if include_spectral:
            freq_out = self._freq.score_and_heatmap(rgb_f32)
            freq_score = freq_out["score"]
            freq_heatmap_png = _png_bytes(freq_out["heatmap"])

        # --- error level analysis (skipped for video frames — see
        # include_ela docstring above) ---
        ela_out = None
        ela_heatmap_png = None
        if include_ela:
            ela_out = _ela.error_level_analysis(image_rgb)
            ela_heatmap_png = _png_bytes(ela_out["heatmap"])

        # --- metadata / provenance ---
        meta_out = None
        if include_metadata:
            if raw_bytes is None:
                raw_bytes = _png_bytes(image_rgb)  # best-effort if caller only has array
            meta_out = _meta.analyse_metadata(raw_bytes, filename=image_path)

        # --- sensor noise-floor (strongest signal for clean AI portraits;
        # skipped for video frames — see include_noise docstring above) ---
        noise_out = None
        if include_noise:
            noise_out = _noise.analyse_noise_floor(image_rgb)

        # --- localization: combine noise+ela into a spatial "where" map ---
        from pipeline.forensics import localization as _loc
        loc_out = _loc.build_suspicious_regions(image_rgb)

        # --- trained classifier (optional; the real ML signal) ---
        # Gated off for document/text-page photos: testing found both this
        # classifier and the noise heuristic misread flat printed paper as
        # AI-suspicious. See pipeline/forensics/document_detector.py for
        # why, and its documented limitations.
        from pipeline.forensics import document_detector as _docdet
        doc_out = _docdet.looks_like_document(image_rgb)

        classifier_out = None
        face_crop_out = None
        if (
            self._ai_classifier is not None
            and self._ai_classifier.available
            and not doc_out["is_document"]
        ):
            classifier_input = image_rgb
            if face_crop_before_classifier and self._face_cropper is not None:
                face_crop_out = self._face_cropper.crop(image_rgb)
                if face_crop_out["face_found"]:
                    classifier_input = face_crop_out["crop"]
                else:
                    # No face to crop -- this classifier is very likely
                    # face-trained (see face_crop_before_classifier
                    # docstring above), so feeding it a whole non-face
                    # frame is an out-of-domain input, not a meaningful
                    # reading. Skip rather than guess.
                    classifier_input = None
            if classifier_input is not None:
                classifier_out = self._ai_classifier.predict(classifier_input)

        classifier_active = bool(classifier_out and classifier_out.get("available"))

        # Base weights: start from the full 5-signal set and drop whichever
        # of metadata/noise the caller says isn't meaningful for this input
        # (see include_metadata / include_noise docstrings), renormalizing
        # what's left so the weights still sum to 1.0 rather than silently
        # shrinking the total.
        excluded = set()
        if not include_metadata:
            excluded.add("metadata")
        if not include_noise:
            excluded.add("noise")
        if not include_ela:
            excluded.add("ela")
        if not include_spectral:
            excluded.add("spectral")

        if not excluded:
            base_weights = dict(self._WEIGHTS)
        else:
            remaining_keys = [k for k in self._WEIGHTS if k not in excluded]
            total = sum(self._WEIGHTS[k] for k in remaining_keys)
            base_weights = {k: self._WEIGHTS[k] / total for k in remaining_keys}

        if classifier_active:
            # Trained model present: it takes _CLASSIFIER_WEIGHT of the
            # total, and the heuristics share the remaining budget in the
            # same *proportions* they already had to each other (base
            # weights already sum to 1.0, so scaling all of them by the
            # same factor preserves that).
            remaining = 1.0 - self._CLASSIFIER_WEIGHT
            active_weights = {k: v * remaining for k, v in base_weights.items()}
        else:
            active_weights = dict(base_weights)

        signals = {
            "texture":  {"score": texture_score, "weight": active_weights["texture"],
                         "findings": texture_findings},
        }
        if include_spectral:
            signals["spectral"] = {"score": freq_score, "weight": active_weights["spectral"],
                                    "findings": freq_out["findings"]}
        if include_ela:
            signals["ela"] = {"score": ela_out["score"], "weight": active_weights["ela"],
                               "findings": ela_out["findings"]}
        if include_metadata:
            signals["metadata"] = {"score": meta_out["score"], "weight": active_weights["metadata"],
                                    "findings": meta_out["findings"]}
        if include_noise:
            signals["noise"] = {"score": noise_out["score"], "weight": active_weights["noise"],
                                 "findings": noise_out["findings"]}
        if classifier_active:
            signals["classifier"] = {
                "score": classifier_out["score"],
                "weight": self._CLASSIFIER_WEIGHT,
                "findings": classifier_out["findings"],
            }
        elif (
            self._ai_classifier is not None
            and self._ai_classifier.available
            and doc_out["is_document"]
        ):
            # Classifier exists and would normally run, but this looks like
            # a document/text-page photo — a known blind spot (see
            # document_detector.py), so it was deliberately skipped rather
            # than risk a false "AI-generated" flag on a real photo of a
            # book/receipt/notes page. Attached to whichever of
            # metadata/noise/ela is present — just needs a home so it
            # surfaces in the explanation summary.
            _doc_note_target = (
                "metadata" if include_metadata else
                "noise" if include_noise else
                "ela" if include_ela else
                "spectral" if include_spectral else
                "texture"
            )
            signals[_doc_note_target]["findings"] = signals[_doc_note_target]["findings"] + [
                "This looks like a photo of a document or text page rather "
                "than an ordinary photo, so the trained AI-image classifier "
                "was skipped for this image (it's known to misread flat "
                "printed paper as AI-suspicious)."
            ]

        fused_score = float(np.clip(
            sum(s["score"] * s["weight"] for s in signals.values()), 0.0, 1.0
        ))

        # Captured BEFORE the texture-only ceiling below is applied, so
        # VideoDetector's aggregate gate can later recompute a video-wide
        # mean that trusts texture uniformly across every frame once the
        # classifier has corroborated it on the frames it could see — see
        # "pre_ceiling_score" in VideoDetector.detect()'s aggregate gate.
        # Equal to fused_score whenever the cap doesn't end up firing below.
        pre_ceiling_score = fused_score

        # Session 5: with spectral/noise/ela/metadata all excluded for video,
        # texture is the ONLY per-frame heuristic left. Texture alone has
        # shown wide, inconsistent swings on real footage (see VideoDetector's
        # class docstring) — trusting it enough to single-handedly cross into
        # FAKE produced a false positive on a real video (0.83 fused, purely
        # from texture reading high on smooth/bright frames). Texture pinned
        # at its ceiling on a genuinely un-provenanced AI clip is a real,
        # useful signal that shouldn't be thrown away, though — so texture
        # can push the score up to (but not past) the edge of the UNCERTAIN
        # band on its own, an honest "this looks suspicious but I can't
        # confirm it", and only escalates into FAKE once the trained
        # classifier itself actively corroborates (reads at/above
        # fake_threshold on its own).
        #
        # IMPORTANT (fixed after real-world testing): "classifier active"
        # is NOT the same as "classifier corroborates". The first version
        # of this gate only capped the score when the classifier was
        # unavailable, and assumed that whenever it *was* active its 45%
        # fusion weight would naturally keep texture in check. It doesn't:
        # a real video scored texture=0.828, classifier=0.401 (the
        # classifier correctly leaning REAL, well below fake_threshold)
        # and still fused to 0.636 (0.55*0.828 + 0.45*0.401) — FAKE — purely
        # because texture's 55% share overpowered a disagreeing classifier.
        # The gate must key off whether the classifier's OWN reading
        # corroborates fake, not merely whether it ran.
        texture_is_sole_heuristic = (
            not include_spectral and not include_noise
            and not include_ela and not include_metadata
        )
        if texture_is_sole_heuristic:
            from utils.config import CFG as _CFG
            _fake_threshold = float(
                getattr(getattr(_CFG, "fusion", None), "fake_threshold", 0.60)
            )
            classifier_corroborates = (
                classifier_active and classifier_out is not None
                and classifier_out["score"] >= _fake_threshold
            )
            if not classifier_corroborates:
                _uncorroborated_ceiling = _fake_threshold - 0.01
                if fused_score > _uncorroborated_ceiling:
                    fused_score = _uncorroborated_ceiling

        # A confirmed generator tag (Software/EXIF field, PNG generation
        # metadata, or a C2PA manifest explicitly declaring an AI digital-
        # source-type) is direct provenance evidence, not a statistical
        # guess — it shouldn't get diluted down to "REAL" just because
        # every pixel-level heuristic happened to read clean (which is
        # exactly what a well-blended AI edit looks like). Confirmed
        # provenance overrides the weighted blend rather than merely
        # contributing its 15% share.
        if meta_out and meta_out.get("generator_tag") and meta_out["score"] >= 0.9:
            fused_score = max(fused_score, 0.9)

        verdict = _assign_verdict(fused_score)

        summary = _explain.summarise(verdict, fused_score, signals)

        h, w = image_rgb.shape[:2]
        return ImageResult(
            image_id=image_id,
            verdict=verdict,
            fused_score=fused_score,
            gradcam_score=texture_score,
            freq_score=freq_score,
            gradcam_overlay=gradcam_overlay_png,
            freq_heatmap=freq_heatmap_png,
            findings=summary["reasons"],
            headline=summary["headline"],
            confidence_label=summary["confidence_label"],
            sub_scores={
                "texture": texture_score,
                **({"spectral": freq_score} if freq_out is not None else {}),
                **({"ela": ela_out["score"]} if ela_out is not None else {}),
                **({"noise": noise_out["score"]} if noise_out is not None else {}),
                **({"metadata": meta_out["score"]} if meta_out is not None else {}),
                **({"classifier": classifier_out["score"]} if classifier_active else {}),
            },
            ela_heatmap=ela_heatmap_png,
            localization_heatmap=loc_out.get("heatmap"),
            localization_summary=loc_out.get("summary", ""),
            localization_is_localized=loc_out.get("is_localized", False),
            localization_regions=loc_out.get("regions", []),
            metadata={
                "run_id":          str(uuid.uuid4()),
                "image_path":      image_path or "",
                "width":           w,
                "height":          h,
                "face_detected":   (
                    face_crop_out["face_found"] if face_crop_out is not None else True
                ),
                "face_crop_confidence": (
                    face_crop_out.get("confidence") if face_crop_out is not None else None
                ),
                "classifier_skipped_no_face": bool(
                    face_crop_before_classifier and face_crop_out is not None
                    and not face_crop_out["face_found"]
                ),
                "gradcam_weight":  active_weights["texture"],
                "freq_weight":     active_weights.get("spectral"),
                "ela_weight":      active_weights.get("ela"),
                "metadata_weight": active_weights.get("metadata"),
                "noise_weight":    active_weights.get("noise"),
                "noise_floor":     noise_out.get("noise_floor") if noise_out else None,
                "generator_tag":   meta_out.get("generator_tag") if meta_out else None,
                "has_exif":        meta_out.get("has_exif") if meta_out else None,
                "suspicious_fraction": loc_out.get("suspicious_fraction"),
                "classifier_active": classifier_active,
                "classifier_weight": self._CLASSIFIER_WEIGHT if classifier_active else None,
                "classifier_backend": classifier_out.get("backend") if classifier_out else None,
                "classifier_error": (
                    self._ai_classifier.load_error
                    if self._ai_classifier is not None and not self._ai_classifier.available
                    else None
                ),
                "is_document": doc_out["is_document"],
                "document_colorfulness": doc_out["colorfulness"],
                "pre_ceiling_score": pre_ceiling_score,
            },
        )

    def detect_from_bytes(
        self,
        data: bytes,
        image_id: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> ImageResult:
        from PIL import Image
        pil_img   = Image.open(io.BytesIO(data)).convert("RGB")
        image_rgb = np.array(pil_img, dtype=np.uint8)
        return self.detect(image_rgb, image_id=image_id, image_path=image_path, raw_bytes=data)

    @staticmethod
    def _blend_overlay(
        base_rgb: np.ndarray,
        heatmap_rgb: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        base_f = base_rgb.astype(np.float32)
        heat_f = heatmap_rgb.astype(np.float32)
        if base_f.shape[:2] != heat_f.shape[:2]:
            from PIL import Image
            h, w     = base_f.shape[:2]
            heat_pil = Image.fromarray(heatmap_rgb).resize((w, h), Image.BILINEAR)
            heat_f   = np.array(heat_pil, dtype=np.float32)
        return np.clip(alpha * heat_f + (1 - alpha) * base_f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Module-level convenience (used by app.py / tests)
# ---------------------------------------------------------------------------

def detect_image(image_path: str) -> ImageResult:
    from PIL import Image as _PIL
    pil_img   = _PIL.open(image_path).convert("RGB")
    image_rgb = np.array(pil_img, dtype=np.uint8)
    return ImageDetector().detect(image_rgb, image_path=image_path)