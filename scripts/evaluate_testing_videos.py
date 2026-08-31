"""Evaluates every testing video against the CURRENT ForenSight pipeline.

- Mirrors main.run_pipeline exactly (reusing one classifier instance for status only).
- For the VIDEO pipeline the image classifier is intentionally NOT wired in
  (Task 2: it is validated as reversed/invalid on video frames).
- Ground-truth is inferred ONLY from folder structure (ai/ = FAKE, real/ = REAL).
- Reports: filename, ground_truth, duration, fps, has_audio, video/audio/provenance/
  classifier status + scores, fusion_score, verdict, plus TP/TN/FP/FN, precision,
  recall, F1, FNR, FPR, fake_recall, fake_verdict_rate, uncertain_rate, and a
  per-FAKE "which detector catches which video" breakdown.

Writes reports/testing_videos_evaluation.json, .csv and reports/testing_videos_manifest.csv.

Usage: py scripts/evaluate_testing_videos.py [--root PATH]
"""
import argparse
import csv
import json
import os
import sys
import time
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import main as forensight_main
from utils.config import load_config
from utils.demux import demux
from utils.types import ModalResult
from pipeline.video.detector import VideoDetector
from pipeline.audio.detector import AudioDetector
from pipeline.fusion.fuser import Fuser
from pipeline.forensics.ensemble_classifier import EnsembleClassifier
from pipeline.aasist.detector import AASISTDetector

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

CANDIDATE_ROOTS = [
    os.environ.get("TESTING_VIDEOS_ROOT", ""),
    r"C:\Users\Hemapriya\Documents\ForenSight docs\testing_videos",
    r"C:\Users\Hemapriya\Documents\testing_videos",
    r"C:\Users\Hemapriya\Downloads\testing_videos",
    os.path.join(ROOT, "testing_videos"),
]

CLASSIFIER_INACTIVE_REASON = (
    "Classifier behavior is reversed/invalid for current video preprocessing "
    "and requires independent validation; disabled for video verdicts (Task 2)."
)


def find_videos(root):
    videos, images = [], []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            full = os.path.join(dirpath, f)
            if ext in VIDEO_EXTS:
                videos.append(full)
            elif ext in IMAGE_EXTS:
                images.append(full)
    return sorted(videos), sorted(images)


def infer_label(path):
    parent = os.path.basename(os.path.dirname(path)).lower()
    if parent in ("ai", "fake", "ai_videos", "ai video"):
        return "FAKE"
    if parent in ("real", "real_videos", "reales"):
        return "REAL"
    return "UNKNOWN"


def probe_media(path):
    info = {"duration_seconds": None, "fps": None, "resolution": None,
            "has_audio": False, "codec": None, "filesize_mb": None}
    try:
        info["filesize_mb"] = round(os.path.getsize(path) / (1024 * 1024), 2)
    except Exception:
        pass
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    try:
        out = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=60,
        )
        data = json.loads(out.stdout or "{}")
        fmt = data.get("format", {})
        info["duration_seconds"] = float(fmt["duration"]) if fmt.get("duration") else None
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                w, h = s.get("width"), s.get("height")
                info["resolution"] = f"{w}x{h}" if w and h else None
                info["codec"] = s.get("codec_name")
                fr = s.get("avg_frame_rate")
                if fr and "/" in fr:
                    a, b = fr.split("/")
                    try:
                        info["fps"] = round(float(a) / float(b), 3) if float(b) else None
                    except Exception:
                        pass
            elif s.get("codec_type") == "audio":
                info["has_audio"] = True
    except Exception as e:
        info["probe_error"] = str(e)[:200]
    return info


def load_classifier_for_status(cfg):
    ensemble_cfg = getattr(cfg, "ensemble", None)
    models = []
    if ensemble_cfg and getattr(ensemble_cfg, "models", None):
        for m in ensemble_cfg.models:
            d = dict(m)
            d.setdefault("backend", "transformers")
            d.setdefault("weight", 1.0)
            d.setdefault("input_size", getattr(ensemble_cfg, "input_size", 224))
            models.append(d)
    try:
        clf = EnsembleClassifier(models=models)
        clf._ensure_loaded()
        return clf
    except Exception as e:
        print(f"  [clf] load error: {type(e).__name__}: {str(e)[:160]}", flush=True)
        return None


def build_fuser(cfg):
    fc = getattr(cfg, "fusion", None)
    return Fuser(
        video_weight=getattr(fc, "video_weight", 0.6),
        audio_weight=getattr(fc, "audio_weight", 0.4),
        real_threshold=getattr(fc, "real_threshold", 0.40),
        fake_threshold=getattr(fc, "fake_threshold", 0.60),
        aggregation=getattr(fc, "aggregation", "mean"),
        percentile=getattr(fc, "percentile", 90.0),
        confidence_weighting=getattr(fc, "confidence_weighting", False),
    )


def _run_aasist(aasist_detector, wav_path):
    """Run the Phase 2B AASIST detector on a demuxed wav path.

    Returns a dict with aasist_* keys (identical contract whether the
    detector is ACTIVE or INACTIVE — score None + reason when unavailable).
    NEVER raises; a broken AASIST must not break the production eval loop.
    """
    if aasist_detector is None:
        return {
            "aasist_status": None,
            "aasist_spoof_score": None,
            "aasist_bonafide_score": None,
            "aasist_predicted_label": None,
            "aasist_window_count": 0,
            "aasist_window_scores": [],
            "aasist_note": "AASIST disabled in config.",
            "aasist_load_error": None,
        }
    try:
        return aasist_detector.infer_wav(wav_path)
    except Exception as e:
        return {
            "aasist_status": "INACTIVE",
            "aasist_spoof_score": None,
            "aasist_bonafide_score": None,
            "aasist_predicted_label": None,
            "aasist_window_count": 0,
            "aasist_window_scores": [],
            "aasist_note": f"AASIST inference error: {type(e).__name__}: {str(e)[:200]}",
            "aasist_load_error": None,
        }


def run_video(path, cfg, fuser, aasist_detector=None):
    t0 = time.time()
    rec = {"path": path, "error": None, "frame_scores": [], "audio_segments": []}
    tmp = tempfile.mkdtemp(prefix="forensight_eval_")
    try:
        frames_dir, wav_path = demux(path, output_dir=tmp)
        # TASK 2: video pipeline does NOT use the image classifier.
        vdet = VideoDetector(cfg, ai_classifier=None, fast_mode=False)
        vbytes = open(path, "rb").read()
        vres = vdet.run(frames_dir, video_bytes=vbytes, video_path=str(path))

        adet = AudioDetector(cfg)
        segs = adet.run(wav_path)
        if segs:
            audio_fake = sum(s.fake_prob for s in segs) / len(segs)
            audio_err = None
            audio_status = "ACTIVE"
        else:
            audio_fake = None
            audio_err = "No audio track"
            audio_status = "INACTIVE"
        ares = ModalResult(modality="audio", fake_prob=audio_fake,
                           audio_results=segs, error=audio_err)

        # ---- Phase 2B: AASIST (independent audio anti-spoofing, eval-only).
        # Runs on the SAME demuxed wav as the heuristic; results go into
        # aasist_* keys and NEVER touch ares.fake_prob / the fuser.
        aasist_meta = _run_aasist(aasist_detector, wav_path)
        rec.update(aasist_meta)

        result = fuser.fuse(video_result=vres, audio_result=ares,
                            video_path=path, analysis_id=str(uuid.uuid4()))

        for fr in vres.frame_results:
            rec["frame_scores"].append({
                "frame_index": fr.frame_index,
                "timestamp": fr.timestamp_sec,
                "fake_prob": fr.fake_prob,
                "sub_scores": fr.sub_scores,
                "face_detected": fr.face_detected,
                "classifier_active": getattr(fr, "classifier_active", None),
            })
        for s in segs:
            rec["audio_segments"].append({
                "segment_index": s.segment_index,
                "start": s.start_sec, "end": s.end_sec,
                "fake_prob": s.fake_prob,
            })

        rec["video_score"] = vres.fake_prob
        rec["audio_score"] = ares.fake_prob
        rec["audio_status"] = audio_status
        rec["provenance"] = vres.metadata
        rec["provenance_score"] = vres.metadata.get("provenance_score")
        rec["provenance_override"] = bool(result.metadata.get("provenance_override", False))
        rec["provenance_reason"] = result.metadata.get("provenance_reason")
        # GenConViT (Phase 2A) — real neural video score, stored separately.
        rec["genconvit_status"] = vres.metadata.get("genconvit_status")
        rec["genconvit_score"] = vres.metadata.get("genconvit_score")
        rec["genconvit_predicted_label"] = vres.metadata.get("genconvit_predicted_label")
        rec["genconvit_frames_used"] = vres.metadata.get("genconvit_frames_used")
        rec["genconvit_branches"] = vres.metadata.get("genconvit_branches")
        rec["genconvit_load_error"] = vres.metadata.get("genconvit_load_error")
        rec["genconvit_frame_scores"] = vres.metadata.get("genconvit_frame_scores")
        # Phase 2A.1 input-coverage metrics
        rec["genconvit_frames_requested"] = vres.metadata.get("genconvit_frames_requested")
        rec["genconvit_frames_sampled"] = vres.metadata.get("genconvit_frames_sampled")
        rec["genconvit_face_frames_found"] = vres.metadata.get("genconvit_face_frames_found")
        rec["genconvit_face_coverage"] = vres.metadata.get("genconvit_face_coverage")
        rec["genconvit_average_face_confidence"] = vres.metadata.get("genconvit_average_face_confidence")
        rec["genconvit_min_face_confidence"] = vres.metadata.get("genconvit_min_face_confidence")
        rec["genconvit_max_face_confidence"] = vres.metadata.get("genconvit_max_face_confidence")
        rec["genconvit_face_quality"] = vres.metadata.get("genconvit_face_quality")
        rec["genconvit_coverage_status"] = vres.metadata.get("genconvit_coverage_status")
        rec["genconvit_frame_indices"] = vres.metadata.get("genconvit_frame_indices")
        rec["genconvit_timestamps"] = vres.metadata.get("genconvit_timestamps")
        rec["genconvit_face_sizes"] = vres.metadata.get("genconvit_face_sizes")
        rec["genconvit_face_confidences"] = vres.metadata.get("genconvit_face_confidences")
        rec["fusion_score"] = result.fused_score
        rec["verdict"] = result.verdict.value
        rec["fusion_rationale"] = getattr(result, "rationale", None)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        rec["video_score"] = None
        rec["audio_score"] = None
        rec["fusion_score"] = None
        rec["verdict"] = "ERROR"
        rec["audio_status"] = "INACTIVE"
        rec["provenance_override"] = False
        rec["provenance_score"] = None
        rec["provenance_reason"] = None
        rec["genconvit_status"] = None
        rec["genconvit_score"] = None
        rec["genconvit_predicted_label"] = None
        rec["genconvit_frames_used"] = None
        rec["genconvit_branches"] = None
        rec["genconvit_load_error"] = None
        rec["genconvit_frames_requested"] = None
        rec["genconvit_frames_sampled"] = None
        rec["genconvit_face_frames_found"] = None
        rec["genconvit_face_coverage"] = None
        rec["genconvit_average_face_confidence"] = None
        rec["genconvit_min_face_confidence"] = None
        rec["genconvit_max_face_confidence"] = None
        rec["genconvit_face_quality"] = None
        rec["genconvit_coverage_status"] = None
        rec["genconvit_frame_indices"] = None
        rec["genconvit_timestamps"] = None
        rec["genconvit_face_sizes"] = None
        rec["genconvit_face_confidences"] = None
        rec["aasist_status"] = None
        rec["aasist_spoof_score"] = None
        rec["aasist_bonafide_score"] = None
        rec["aasist_predicted_label"] = None
        rec["aasist_window_count"] = 0
        rec["aasist_window_scores"] = []
        rec["aasist_note"] = None
        rec["aasist_load_error"] = None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rec["processing_seconds"] = round(time.time() - t0, 2)
    return rec


def summarize_subscores(rec):
    out = {"texture_score": None, "frequency_score": None}
    tex, freq = [], []
    for fr in rec.get("frame_scores", []):
        ss = fr.get("sub_scores") or {}
        if isinstance(ss.get("texture"), (int, float)):
            tex.append(ss["texture"])
        if isinstance(ss.get("spectral"), (int, float)):
            freq.append(ss["spectral"])
    if tex:
        out["texture_score"] = round(sum(tex) / len(tex), 4)
    if freq:
        out["frequency_score"] = round(sum(freq) / len(freq), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    root = args.root
    if not root:
        for c in CANDIDATE_ROOTS:
            if c and os.path.isdir(c):
                root = c
                break
    if not root or not os.path.isdir(root):
        print("testing_videos dir not found. Use --root PATH", file=sys.stderr)
        sys.exit(2)
    print(f"[root] {root}", flush=True)

    videos, images = find_videos(root)
    print(f"[videos] {len(videos)} | [images] {len(images)}", flush=True)

    cfg = load_config()

    # ---- TASK 3: honest, structured startup status ----
    print("\n=== DETECTOR STATUS (FAIL-LOUD) ===", flush=True)
    classifier = load_classifier_for_status(cfg)
    statuses = forensight_main.get_detector_status(cfg=cfg, classifier=classifier)
    for s in statuses:
        print(f"[{s['category']}] {s['name']} — {s['detail']}", flush=True)

    fuser = build_fuser(cfg)

    # ---- Phase 2B: AASIST audio anti-spoofing detector (eval-only) ----
    aasist_detector = None
    aasist_cfg = getattr(cfg, "aasist", None)
    if aasist_cfg is not None and getattr(aasist_cfg, "enabled", False):
        w_dir = getattr(aasist_cfg, "weights_dir", "models/aasist")
        if not os.path.isabs(w_dir):
            w_dir = os.path.join(ROOT, w_dir)
        aasist_detector = AASISTDetector(
            weights_dir=w_dir,
            window_samples=getattr(aasist_cfg, "window_samples", 64600) or 64600,
        )
        if not aasist_detector.ensure_loaded():
            print(f"  [aasist] INACTIVE: {aasist_detector.load_error}", flush=True)

    results = []
    for v in videos:
        label = infer_label(v)
        media = probe_media(v)
        print(f"\n[video] {os.path.relpath(v, root)} label={label}", flush=True)
        rec = run_video(v, cfg, fuser, aasist_detector=aasist_detector)
        subs = summarize_subscores(rec)
        prov_score = rec.get("provenance_score")
        prov_override = rec.get("provenance_override", False)
        prov_status = "OVERRIDE" if prov_override else "ACTIVE"
        entry = {
            "filename": os.path.basename(v),
            "path": v,
            "ground_truth": label,
            "duration_seconds": media["duration_seconds"],
            "fps": media["fps"],
            "has_audio": media["has_audio"],
            "video_status": "ACTIVE",
            "video_score": rec.get("video_score"),
            "audio_status": rec.get("audio_status"),
            "audio_score": rec.get("audio_score"),
            "provenance_status": prov_status,
            "provenance_score": prov_score,
            "provenance_override": prov_override,
            "provenance_reason": rec.get("provenance_reason"),
            "classifier_status": "INACTIVE",
            "classifier_score": None,
            "classifier_reason": CLASSIFIER_INACTIVE_REASON,
            "texture_score": subs["texture_score"],
            "frequency_score": subs["frequency_score"],
            "genconvit_status": rec.get("genconvit_status"),
            "genconvit_score": rec.get("genconvit_score"),
            "genconvit_predicted_label": rec.get("genconvit_predicted_label"),
            "genconvit_frames_used": rec.get("genconvit_frames_used"),
            "genconvit_branches": rec.get("genconvit_branches"),
            "genconvit_load_error": rec.get("genconvit_load_error"),
            "genconvit_frames_requested": rec.get("genconvit_frames_requested"),
            "genconvit_frames_sampled": rec.get("genconvit_frames_sampled"),
            "genconvit_face_frames_found": rec.get("genconvit_face_frames_found"),
            "genconvit_face_coverage": rec.get("genconvit_face_coverage"),
            "genconvit_average_face_confidence": rec.get("genconvit_average_face_confidence"),
            "genconvit_min_face_confidence": rec.get("genconvit_min_face_confidence"),
            "genconvit_max_face_confidence": rec.get("genconvit_max_face_confidence"),
            "genconvit_face_quality": rec.get("genconvit_face_quality"),
            "genconvit_coverage_status": rec.get("genconvit_coverage_status"),
            "genconvit_frame_indices": rec.get("genconvit_frame_indices"),
            "genconvit_timestamps": rec.get("genconvit_timestamps"),
            "genconvit_face_sizes": rec.get("genconvit_face_sizes"),
            "genconvit_face_confidences": rec.get("genconvit_face_confidences"),
            "aasist_status": rec.get("aasist_status"),
            "aasist_spoof_score": rec.get("aasist_spoof_score"),
            "aasist_bonafide_score": rec.get("aasist_bonafide_score"),
            "aasist_predicted_label": rec.get("aasist_predicted_label"),
            "aasist_window_count": rec.get("aasist_window_count"),
            "aasist_window_scores": rec.get("aasist_window_scores"),
            "aasist_note": rec.get("aasist_note"),
            "aasist_load_error": rec.get("aasist_load_error"),
            "fusion_score": rec.get("fusion_score"),
            "verdict": rec.get("verdict"),
            "error": rec.get("error"),
            "processing_seconds": rec.get("processing_seconds"),
            "frame_count": len(rec.get("frame_scores", [])),
            "audio_segment_count": len(rec.get("audio_segments", [])),
            "frame_scores": rec.get("frame_scores"),
            "audio_segments": rec.get("audio_segments"),
            "provenance": rec.get("provenance"),
            "genconvit_frame_scores": rec.get("genconvit_frame_scores"),
            "fusion_rationale": rec.get("fusion_rationale"),
        }
        results.append(entry)
        print(f"  verdict={entry['verdict']} fusion={entry['fusion_score']} "
              f"video={entry['video_score']} audio={entry['audio_score']} "
              f"prov={prov_score} override={prov_override} "
              f"tex={subs['texture_score']} gc={entry['genconvit_score']}"
              f"/{entry['genconvit_predicted_label']} "
              f"aasist={entry['aasist_spoof_score']}"
              f"/{entry['aasist_predicted_label']} t={entry['processing_seconds']}s", flush=True)

    image_rows = []
    for im in images:
        media = probe_media(im)
        image_rows.append({
            "filename": os.path.basename(im), "path": im, "type": "image",
            "ground_truth": infer_label(im), "duration_seconds": None,
            "fps": None, "resolution": media["resolution"],
            "has_audio": False, "codec": media["codec"],
            "filesize_mb": media["filesize_mb"],
        })

    metrics, per_fake = compute_metrics(results)
    eval_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": root,
        "detector_status": statuses,
        "metrics": metrics,
        "per_fake_detector_breakdown": per_fake,
        "results": results,
        "images_not_evaluated_as_video": image_rows,
    }
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "testing_videos_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, default=str)
    csv_cols = ["filename", "path", "ground_truth", "duration_seconds", "fps",
                "has_audio", "video_status", "video_score", "audio_status", "audio_score",
                "provenance_status", "provenance_score", "provenance_override",
                "provenance_reason", "classifier_status", "classifier_score",
                "classifier_reason", "texture_score", "frequency_score",
                "genconvit_status", "genconvit_score", "genconvit_predicted_label",
                "genconvit_frames_used", "genconvit_branches", "genconvit_load_error",
                "genconvit_frames_requested", "genconvit_frames_sampled",
                "genconvit_face_frames_found", "genconvit_face_coverage",
                "genconvit_average_face_confidence", "genconvit_min_face_confidence",
                "genconvit_max_face_confidence", "genconvit_face_quality",
                "genconvit_coverage_status",
                "aasist_status", "aasist_spoof_score", "aasist_bonafide_score",
                "aasist_predicted_label", "aasist_window_count", "aasist_note",
                "aasist_load_error",
                "fusion_score", "verdict", "error", "processing_seconds",
                "frame_count", "audio_segment_count"]
    with open(os.path.join(ROOT, "reports", "testing_videos_evaluation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in results:
            w.writerow({c: r.get(c) for c in csv_cols})
    man_cols = ["filename", "path", "type", "ground_truth", "duration_seconds",
                "fps", "resolution", "has_audio", "codec", "filesize_mb"]
    with open(os.path.join(ROOT, "reports", "testing_videos_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=man_cols)
        w.writeheader()
        for r in results:
            w.writerow({c: r.get(c) for c in man_cols})
        for r in image_rows:
            w.writerow({c: r.get(c) for c in man_cols})

    print("\n=== METRICS ===", flush=True)
    print(json.dumps(metrics, indent=2, default=str), flush=True)
    print("\n[done] reports/testing_videos_evaluation.json/.csv and testing_videos_manifest.csv", flush=True)


def compute_metrics(results):
    tp = tn = fp = fn = 0
    rows = [r for r in results if r["ground_truth"] in ("FAKE", "REAL")]
    n = len(rows)
    for r in rows:
        gt, v = r["ground_truth"], r["verdict"]
        if gt == "FAKE":
            if v == "FAKE":
                tp += 1
            else:
                fn += 1
        else:  # REAL
            if v == "FAKE":
                fp += 1
            else:
                tn += 1

    n_fake_total = sum(1 for r in rows if r["ground_truth"] == "FAKE")
    n_real_total = sum(1 for r in rows if r["ground_truth"] == "REAL")
    fake_verdicts = sum(1 for r in rows if r["verdict"] == "FAKE")
    uncertain_verdicts = sum(1 for r in rows if r["verdict"] == "UNCERTAIN")

    accuracy = (tp + tn) / n if n else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall)) else None
    fnr = fn / (tp + fn) if (tp + fn) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    fake_verdict_rate = fake_verdicts / n if n else None
    uncertain_rate = uncertain_verdicts / n if n else None

    # ---- TASK 7: which detector catches which FAKE video ----
    per_fake = []
    fake_pred_fake = fake_pred_unc = fake_pred_real = 0
    for r in rows:
        if r["ground_truth"] != "FAKE":
            continue
        v = r["verdict"]
        if v == "FAKE":
            fake_pred_fake += 1
        elif v == "UNCERTAIN":
            fake_pred_unc += 1
        else:
            fake_pred_real += 1
        prov = r.get("provenance_score")
        per_fake.append({
            "filename": r["filename"],
            "provenance_result": ("FAKE (definitive)" if r.get("provenance_override")
                                  else ("neutral" if prov is None else round(prov, 3))),
            "video_result": r.get("video_score"),
            "audio_result": (r.get("audio_score") if r.get("audio_status") == "ACTIVE" else "INACTIVE"),
            "image_classifier_result": "INACTIVE (disabled for video, Task 2)",
            "genconvit_result": r.get("genconvit_predicted_label"),
            "genconvit_score": r.get("genconvit_score"),
            "aasist_result": r.get("aasist_predicted_label"),
            "aasist_score": r.get("aasist_spoof_score"),
            "final_verdict": v,
        })

    # ---- GenConViT (Phase 2A): independent confusion + disagreement ----
    genconvit_metrics = compute_genconvit_metrics(rows)
    disagreement = compute_disagreements(rows)

    # ---- AASIST (Phase 2B): independent audio anti-spoofing metrics ----
    aasist_metrics = compute_aasist_metrics(rows)
    aasist_disagreement = compute_aasist_disagreements(rows)

    metrics = {
        "n_labeled": n,
        "n_fake": n_fake_total,
        "n_real": n_real_total,
        "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "accuracy": accuracy,
        "precision_fake": precision,
        "recall_fake": recall,
        "FAKE_RECALL": recall,
        "f1_fake": f1,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "fake_verdict_rate": fake_verdict_rate,
        "uncertain_rate": uncertain_rate,
        "fake_predicted_FAKE": fake_pred_fake,
        "fake_predicted_UNCERTAIN": fake_pred_unc,
        "fake_predicted_REAL": fake_pred_real,
        "genconvit": genconvit_metrics,
        "genconvit_disagreement_with_texture": disagreement,
        "aasist": aasist_metrics,
        "aasist_disagreement_with_verdict": aasist_disagreement,
    }
    return metrics, per_fake


def compute_genconvit_metrics(rows):
    """Confusion matrix purely from GenConViT's own predicted label vs truth.

    Only rows where GenConViT was ACTIVE and produced a label are scored
    (genconvit_status == ACTIVE_MODEL and predicted_label is not None).
    GenConViT is reported separately — it is NOT fused into the verdict yet.
    """
    scored = [
        r for r in rows
        if r.get("genconvit_status") == "ACTIVE_MODEL"
        and r.get("genconvit_predicted_label") in ("FAKE", "REAL")
    ]
    tp = tn = fp = fn = 0
    for r in scored:
        gt = r["ground_truth"]
        pred = r["genconvit_predicted_label"]
        if gt == "FAKE":
            if pred == "FAKE":
                tp += 1
            else:
                fn += 1
        else:
            if pred == "FAKE":
                fp += 1
            else:
                tn += 1
    n = len(scored)
    n_fake = sum(1 for r in scored if r["ground_truth"] == "FAKE")
    n_real = sum(1 for r in scored if r["ground_truth"] == "REAL")
    accuracy = (tp + tn) / n if n else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall)) else None
    return {
        "n_scored": n,
        "n_fake": n_fake,
        "n_real": n_real,
        "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "accuracy": accuracy,
        "precision_fake": precision,
        "recall_fake": recall,
        "FAKE_RECALL": recall,
        "f1_fake": f1,
        "false_negative_rate": (fn / (tp + fn)) if (tp + fn) else None,
        "false_positive_rate": (fp / (fp + tn)) if (fp + tn) else None,
        "fake_verdict_rate": (sum(1 for r in scored if r["genconvit_predicted_label"] == "FAKE") / n) if n else None,
        "real_rejection_rate": (sum(1 for r in scored if r["ground_truth"] == "REAL" and r["genconvit_predicted_label"] == "REAL") / n_real) if n_real else None,
        "coverage": compute_coverage_summary(rows),
    }


def compute_coverage_summary(rows):
    """Phase 2A.1 input-coverage diagnostics across every video (scored or not).

    Engineering diagnostics only — how many videos GenConViT saw 0 / 1 /
    multiple usable face frames in, plus aggregate coverage and confidence.
    """
    n_total = len(rows)
    n_found = [r.get("genconvit_face_frames_found") for r in rows]
    n_requested = [r.get("genconvit_frames_requested") for r in rows]
    valid_requested = [x for x in n_requested if isinstance(x, (int, float)) and x > 0]

    n_with_zero = sum(1 for f in n_found if (f or 0) == 0)
    n_with_one = sum(1 for f in n_found if (f or 0) == 1)
    n_with_multi = sum(1 for f in n_found if (f or 0) > 1)
    n_with_face = sum(1 for f in n_found if (f or 0) > 0)

    covers = []
    for r in rows:
        req = r.get("genconvit_frames_requested")
        fnd = r.get("genconvit_face_frames_found")
        if isinstance(req, (int, float)) and req > 0 and isinstance(fnd, (int, float)):
            covers.append(fnd / req)

    confidences = []
    for r in rows:
        avg = r.get("genconvit_average_face_confidence")
        if isinstance(avg, (int, float)):
            confidences.append(avg)

    return {
        "videos_total": n_total,
        "videos_with_any_face": n_with_face,
        "videos_with_zero_faces": n_with_zero,
        "videos_with_one_face": n_with_one,
        "videos_with_multiple_faces": n_with_multi,
        "average_face_coverage": (sum(covers) / len(covers)) if covers else None,
        "average_face_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "status_counts": {
            s: sum(1 for r in rows
                   if (r.get("genconvit_coverage_status") or r.get("genconvit_face_quality")) == s)
            for s in ("NO_FACE_DETECTED", "LOW_FACE_COVERAGE",
                      "PARTIAL_FACE_COVERAGE", "GOOD_FACE_COVERAGE")
        },
    }


def compute_disagreements(rows):
    """Where GenConViT and the production texture/provenance verdict differ."""
    disagree = []
    for r in rows:
        gc = r.get("genconvit_predicted_label")
        if gc not in ("FAKE", "REAL"):
            continue
        text = "FAKE" if (r.get("texture_score") or 1.0) >= 0.6 else (
            "REAL" if (r.get("texture_score") or 0.0) <= 0.4 else "UNCERTAIN"
        )
        disagree.append({
            "filename": r["filename"],
            "ground_truth": r["ground_truth"],
            "genconvit": gc,
            "texture_heuristic": text,
            "texture_score": r.get("texture_score"),
            "verdict": r.get("verdict"),
        })
    return disagree


def _aasist_label_to_video_class(label):
    """Map AASIST's native classes to the video corpus's FAKE/REAL classes.

    AASIST (ASVspoof2019 LA) classifies audio as SPOOF vs BONAFIDE. Mapping
    SPOOF -> FAKE / BONAFIDE -> REAL is only valid under the explicit
    assumption "the video's audio is AI-generated/synthesized if and only if
    the video itself is FAKE" — a simplification that does NOT hold in general
    (a Wav2Lip-forged clip can retain the true audio). The confusion matrix
    below is therefore an *audio anti-spoofing reading of the corpus*, not a
    video-forgery claim.
    """
    return {"SPOOF": "FAKE", "BONAFIDE": "REAL"}.get(label)


def compute_aasist_metrics(rows):
    """Confusion from AASIST's own predicted label (SPOOF/BONAFIDE) vs truth.

    Only rows where AASIST was ACTIVE and produced a label are scored
    (aasist_status == ACTIVE_MODEL and aasist_predicted_label is not None).
    AASIST is reported separately — it is NOT fused into the verdict.
    """
    scored = [
        r for r in rows
        if r.get("aasist_status") == "ACTIVE_MODEL"
        and r.get("aasist_predicted_label") in ("SPOOF", "BONAFIDE")
    ]
    tp = tn = fp = fn = 0
    for r in scored:
        gt = r["ground_truth"]
        pred = _aasist_label_to_video_class(r["aasist_predicted_label"])
        if gt == "FAKE":
            if pred == "FAKE":
                tp += 1
            else:
                fn += 1
        else:
            if pred == "FAKE":
                fp += 1
            else:
                tn += 1
    n = len(scored)
    n_fake = sum(1 for r in scored if r["ground_truth"] == "FAKE")
    n_real = sum(1 for r in scored if r["ground_truth"] == "REAL")
    accuracy = (tp + tn) / n if n else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall)) else None
    statuses = [r.get("aasist_status") for r in rows]
    active = sum(1 for s in statuses if s == "ACTIVE_MODEL")
    return {
        "n_scored": n,
        "n_fake": n_fake,
        "n_real": n_real,
        "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "accuracy": accuracy,
        "precision_fake": precision,
        "recall_fake": recall,
        "FAKE_RECALL": recall,
        "f1_fake": f1,
        "false_negative_rate": (fn / (tp + fn)) if (tp + fn) else None,
        "false_positive_rate": (fp / (fp + tn)) if (fp + tn) else None,
        "fake_verdict_rate": (sum(1 for r in scored if r["aasist_predicted_label"] == "SPOOF") / n) if n else None,
        "real_rejection_rate": (sum(1 for r in scored if r["ground_truth"] == "REAL" and r["aasist_predicted_label"] == "BONAFIDE") / n_real) if n_real else None,
        "audio_anti_spoof_note": "label mapped SPOOF->FAKE / BONAFIDE->REAL; "
                                 "audio spoofing is NOT video forgery.",
        "coverage": {
            "videos_total": len(rows),
            "active_model": active,
            "inactive": len(statuses) - active,
            "no_audio_track": sum(1 for r in rows if "No audio track" in (r.get("aasist_note") or "")),
            "status_counts": {
                s: statuses.count(s) for s in set(statuses) if s
            },
        },
    }


def compute_aasist_disagreements(rows):
    """Where AASIST's label and the production verdict differ (transparency)."""
    disagree = []
    for r in rows:
        aa = r.get("aasist_predicted_label")
        if aa not in ("SPOOF", "BONAFIDE"):
            continue
        disagree.append({
            "filename": r["filename"],
            "ground_truth": r["ground_truth"],
            "aasist": aa,
            "aasist_spoof_score": r.get("aasist_spoof_score"),
            "aasist_window_count": r.get("aasist_window_count"),
            "audio_heuristic_score": r.get("audio_score"),
            "verdict": r.get("verdict"),
        })
    return disagree


if __name__ == "__main__":
    main()
