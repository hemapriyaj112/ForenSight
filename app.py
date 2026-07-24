"""ForenSight — Streamlit dashboard (app.py).

Tabs:
  Video  — upload a video, run full AV pipeline, view results
  Image  — upload an image, run GradCAM + FFT detection, view results

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

import main  # ForenSight CLI orchestrator
from pipeline.forensics import explain as _explain
from pipeline.forensics.ai_classifier import AIClassifier
from pipeline.video.detector import ImageDetector
from utils.config import CFG
from utils.types import ImageResult

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ForenSight",
    page_icon="🔍",
    layout="wide",
)

def _build_detailed_summary(
    headline: str,
    reasons: list[str],
    verdict_str: str,
    extra_lines: list[str] | None = None,
) -> str:
    """Compose a 4-6 sentence, non-technical paragraph explaining WHY the
    verdict came out the way it did: the headline, up to 3 of the strongest
    supporting reasons (already plain-English strings from
    pipeline/forensics/explain.py), any extra context (e.g. a video's face
    vs. background split), and a closing note appropriate to the verdict.
    Meant to sit right under the Verdict/Fused Score, so someone with zero
    technical background can read it and understand the call."""
    lines = [headline]
    lines.extend(reasons[:3])
    if extra_lines:
        lines.extend(extra_lines)
    closing = {
        "FAKE": (
            "Several of our checks agree on this, which is why we're "
            "flagging it as likely AI-generated or edited — if this matters "
            "for a real decision, it's still worth comparing against the "
            "original source if you have one."
        ),
        "REAL": (
            "None of our checks turned up the kind of tell-tale signs — "
            "unnatural texture, mismatched metadata, or classifier "
            "confidence — that AI-generated or edited content usually "
            "leaves behind, so we're treating this as genuine."
        ),
        "UNCERTAIN": (
            "The evidence is mixed: some checks lean toward real and others "
            "toward fake, so we can't confidently call this one way or the "
            "other. Treat the result with caution and check the source if "
            "it matters."
        ),
    }.get(verdict_str, "")
    if closing:
        lines.append(closing)
    return "\n\n".join(lines)


def _render_explanation_panel(headline: str, reasons: list[str], confidence_label: str) -> None:
    """Plain-language 'why' panel — the primary thing a non-technical user reads."""
    conf_color = {"High": "#0f5132", "Moderate": "#664d03", "Low": "#495057"}.get(confidence_label, "#495057")
    conf_bg    = {"High": "#d1e7dd", "Moderate": "#fff3cd", "Low": "#e2e3e5"}.get(confidence_label, "#e2e3e5")
    st.markdown(f"#### {headline}")
    if confidence_label:
        st.markdown(
            f"""<span style="background-color:{conf_bg};color:{conf_color};
            padding:2px 10px;border-radius:8px;font-size:0.85rem;font-weight:600;">
            {confidence_label} confidence</span>""",
            unsafe_allow_html=True,
        )
    if reasons:
        st.markdown("**Why:**")
        for r in reasons:
            st.markdown(f"- {r}")


# ---------------------------------------------------------------------------
# Shared verdict helpers
# ---------------------------------------------------------------------------

_VERDICT_CONFIG: dict[str, dict] = {
    "REAL":      {"color": "green",  "icon": "✅"},
    "FAKE":      {"color": "red",    "icon": "🚨"},
    "UNCERTAIN": {"color": "orange", "icon": "⚠️"},
}


def _verdict_str(verdict) -> str:
    """Accept either a Verdict enum or a plain string."""
    return verdict.value if hasattr(verdict, "value") else str(verdict)


def _verdict_color(verdict) -> str:
    return _VERDICT_CONFIG.get(_verdict_str(verdict), {}).get("color", "gray")


def _verdict_icon(verdict) -> str:
    return _VERDICT_CONFIG.get(_verdict_str(verdict), {}).get("icon", "❓")


def _build_gauge(fused_score: float) -> go.Figure:
    colour = (
        "red"    if fused_score >= 0.6 else
        "orange" if fused_score >= 0.4 else
        "green"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(fused_score, 4),
        number={"valueformat": ".2%"},
        title={"text": "Fused Fake Probability"},
        gauge={
            "axis":  {"range": [0, 1], "tickformat": ".0%"},
            "bar":   {"color": colour},
            "steps": [
                {"range": [0.0, 0.4], "color": "#d4edda"},
                {"range": [0.4, 0.6], "color": "#fff3cd"},
                {"range": [0.6, 1.0], "color": "#f8d7da"},
            ],
            "threshold": {
                "line":      {"color": "black", "width": 3},
                "thickness": 0.75,
                "value":     fused_score,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def _render_verdict_badge(verdict, fused_score: float) -> None:
    color = _verdict_color(verdict)
    icon  = _verdict_icon(verdict)
    label = _verdict_str(verdict)
    st.markdown(
        f"""
        <div style="
            background-color:{color};
            border-radius:12px;
            padding:24px 32px;
            text-align:center;
            color:white;
            font-size:2.2rem;
            font-weight:700;
            letter-spacing:0.05em;
        ">
            {icon}&nbsp;&nbsp;{label}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric(label="Fused score", value=f"{fused_score:.2%}")


# ---------------------------------------------------------------------------
# Video tab — section renderers
# ---------------------------------------------------------------------------

def render_verdict_badge(verdict, fused_score: float) -> None:
    _render_verdict_badge(verdict, fused_score)


def render_gauge(fused_score: float) -> None:
    st.plotly_chart(_build_gauge(fused_score), use_container_width=True)


def render_gradcam_grid(video_result) -> None:
    frames_with_overlay = [
        fr for fr in video_result.frame_results if fr.gradcam_overlay is not None
    ]
    if not frames_with_overlay:
        st.info("No GradCAM overlays available for this video.")
        return
    cols_per_row = 4
    for row_start in range(0, len(frames_with_overlay), cols_per_row):
        row_frames = frames_with_overlay[row_start : row_start + cols_per_row]
        cols = st.columns(len(row_frames))
        for col, fr in zip(cols, row_frames):
            with col:
                st.image(
                    fr.gradcam_overlay,
                    caption=f"t={fr.timestamp_sec:.2f}s  p={fr.fake_prob:.2%}",
                    use_container_width=True,
                )


def _video_visual_finding(video_result) -> str:
    """Build the 'Visual analysis scored X%' explanation sentence from
    whichever per-frame signals actually ran, instead of a hardcoded list.

    Previously this was a fixed string ("texture + frequency-spectrum
    artefacts") that went stale the moment spectral was excluded from video
    fusion (session 5) — it kept describing a signal that was no longer
    part of the score at all. Deriving the wording from fr.sub_scores keys
    means it can't drift out of sync with VideoDetector's actual signal set
    again, whichever heuristics get added/removed from video fusion next.
    """
    frames = video_result.frame_results
    label_by_key = {
        "texture":  "texture",
        "spectral": "frequency-spectrum artefacts",
        "ela":      "error-level analysis",
        "noise":    "noise-floor",
    }
    present_keys = []
    for key in label_by_key:
        if any(key in fr.sub_scores for fr in frames):
            present_keys.append(key)

    if present_keys:
        heuristics_desc = " + ".join(label_by_key[k] for k in present_keys)
        desc = f"Visual analysis ({heuristics_desc} across frames)"
    else:
        desc = "Visual analysis"

    if any(fr.classifier_active for fr in frames):
        desc += " plus a trained AI-image classifier"

    return f"{desc} scored {video_result.fake_prob:.0%} likelihood of manipulation."


def render_video_score_chart(video_result) -> None:
    """Bar chart of per-signal scores, averaged across all analysed frames —
    the video analog of render_image_score_chart. Only signals that were
    actually computed (classifier may be absent) are shown."""
    frames = video_result.frame_results
    if not frames:
        st.info("No per-frame signal data available.")
        return

    keys = ["texture", "spectral", "ela", "noise", "classifier"]
    labels = ["Texture", "Spectral", "ELA", "Noise", "Classifier"]
    avgs, use_labels = [], []
    for key, label in zip(keys, labels):
        vals = [fr.sub_scores[key] for fr in frames if key in fr.sub_scores]
        if vals:
            avgs.append(sum(vals) / len(vals))
            use_labels.append(label)
    use_labels.append("Fused (video)")
    avgs.append(video_result.fake_prob)

    colors = ["red" if v >= 0.6 else "orange" if v >= 0.4 else "green" for v in avgs]
    fig = go.Figure(go.Bar(
        x=use_labels, y=avgs, marker_color=colors,
        text=[f"{v:.1%}" for v in avgs], textposition="outside",
    ))
    fig.update_layout(
        title="Detection score breakdown (averaged across frames)",
        yaxis=dict(title="Fake probability", range=[0, 1.1], tickformat=".0%"),
        height=320, margin=dict(t=40, b=20, l=40, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    n_classifier = sum(1 for fr in frames if fr.classifier_active)
    n_docs = sum(1 for fr in frames if fr.is_document)
    if n_classifier == 0:
        st.caption(
            "ℹ️ The trained AI-image classifier did not run on any frame "
            "(not configured, failed to load, or every frame looked like a "
            "document) — this breakdown reflects the 4 statistical "
            "heuristics only. See the Provenance panel below for details."
        )
    elif n_classifier < len(frames):
        st.caption(
            f"ℹ️ The trained classifier ran on {n_classifier}/{len(frames)} "
            f"frames — skipped on {n_docs} that looked like a document/"
            "text-page/screen-recording frame (see document_detector.py)."
        )


def render_video_provenance(video_result) -> None:
    """Container-level C2PA/generator-tag scan + classifier status — the
    video analog of the image tab's metadata expander, but surfaced more
    prominently since it can single-handedly floor the verdict at 90%."""
    meta = video_result.metadata or {}
    with st.expander("🔎 Provenance & classifier status", expanded=bool(meta.get("generator_tag"))):
        if meta.get("generator_tag"):
            st.warning(
                f"**Provenance match found:** {meta['generator_tag']} "
                f"(container-level C2PA scan, confidence {meta.get('provenance_score', 0):.0%}). "
                "This alone floors the fused score at 90%, regardless of "
                "what the per-frame pixel signals say."
            )
        elif meta.get("has_c2pa"):
            st.info("A Content Credentials (C2PA) manifest was found, but its origin claim wasn't AI-specific.")
        else:
            st.caption("No embedded provenance (C2PA / generator tag) found in the video container.")

        for finding in meta.get("provenance_findings", []):
            st.caption(f"• {finding}")

        st.divider()
        if meta.get("classifier_error"):
            st.caption(f"⚠️ Trained AI classifier unavailable: {meta['classifier_error']}")
        elif meta.get("classifier_backend"):
            st.caption(f"✅ Trained AI classifier active (backend: {meta['classifier_backend']})")
        else:
            st.caption("Trained AI classifier not configured — see config/config.yaml `image_ai_classifier`.")

        st.caption(
            f"Frames analysed: {meta.get('frames_analysed', 0)} · "
            f"Document/text-page frames skipped from classifier: {meta.get('documents_skipped', 0)}"
        )


def render_video_segments(video_result) -> None:
    """Session 6: two-segment reporting. The single fused score above stays
    the one number used for the overall REAL/FAKE/UNCERTAIN verdict — this
    just breaks it down into 'the face' vs. 'the rest of the footage',
    since a video can have a real face composited over a suspicious
    background (or vice versa), and the single number alone can't say
    which part is driving the result."""
    meta = video_result.metadata or {}
    face_seg = meta.get("face_segment")
    non_face_seg = meta.get("non_face_segment")

    if not face_seg and not non_face_seg:
        return

    st.subheader("🧩 Face vs. background breakdown")
    col_face, col_bg = st.columns(2)

    with col_face:
        st.markdown("**Face segment**")
        if face_seg:
            st.write(
                f"{face_seg['verdict']} ({face_seg['score']:.0%}) — "
                f"based on {face_seg['frame_count']} frame(s) where a face was detected."
            )
            details = []
            if face_seg.get("avg_texture") is not None:
                details.append(f"texture avg {face_seg['avg_texture']:.0%}")
            if face_seg.get("avg_classifier") is not None:
                details.append(f"classifier avg {face_seg['avg_classifier']:.0%}")
            if details:
                st.caption(" · ".join(details))
        else:
            st.caption("No face was detected in any sampled frame.")

    with col_bg:
        st.markdown("**Rest of the footage**")
        if non_face_seg:
            st.write(
                f"{non_face_seg['verdict']} ({non_face_seg['score']:.0%}) — "
                f"based on {non_face_seg['frame_count']} frame(s) with no detected face "
                "(texture-only signal; the trained classifier doesn't apply here)."
            )
            if non_face_seg.get("avg_texture") is not None:
                st.caption(f"texture avg {non_face_seg['avg_texture']:.0%}")
        else:
            st.caption("Every sampled frame had a detected face.")


def render_video_localization(video_result) -> None:
    """Spatial 'where' (most-suspicious frame) + temporal 'when' (which
    part of the timeline) — the video analog of render_image_localization."""
    spatial = video_result.spatial_localization or {}
    temporal = video_result.temporal_localization or {}

    st.subheader("📍 Where (most suspicious frame)")
    if spatial.get("heatmap"):
        col_map, col_text = st.columns([3, 2])
        with col_map:
            st.image(
                spatial["heatmap"], use_container_width=True,
                caption=f"Frame {spatial.get('frame_index')} · t={spatial.get('timestamp_sec', 0):.1f}s — "
                        "red = noise/compression inconsistent with the rest of that frame",
            )
        with col_text:
            st.write(spatial.get("summary", ""))
            for reg in spatial.get("regions", []):
                st.caption(f"• {reg['position_desc']} — ~{round(reg['area_fraction']*100)}% of frame")
    else:
        st.info("Region-level analysis wasn't available for this video.")

    st.subheader("🕒 When")
    st.write(temporal.get("summary", "No timeline data available."))
    for seg in temporal.get("segments", []):
        st.caption(f"• {seg['start_sec']:.1f}s–{seg['end_sec']:.1f}s (~{round(seg['area_fraction']*100)}% of frames)")

    st.caption(
        "⚠️ Both only work for **partial edits** — a stretch of an "
        "otherwise-real video that was altered, or one edited region "
        "within a frame. A video that's AI-generated start to finish has "
        "no untouched frame or region to compare against, so it shows as "
        "spread across the whole clip/frame rather than one specific spot."
    )


def render_audio_section(audio_result) -> None:
    # Support both .audio_results (our type) and .segment_results (real type)
    segments = getattr(audio_result, "audio_results", None) \
            or getattr(audio_result, "segment_results", [])
    if not segments:
        if getattr(audio_result, "error", None):
            st.info(f"ℹ️ Audio analysis skipped: {audio_result.error}")
        else:
            st.info("No audio segment data available.")
        return

    seg_labels = [f"{s.start_sec:.1f}–{s.end_sec:.1f}s" for s in segments]
    seg_probs  = [s.fake_prob for s in segments]

    bar_fig = go.Figure(go.Bar(
        x=seg_labels, y=seg_probs,
        marker_color=[
            "red" if p >= 0.6 else "orange" if p >= 0.4 else "green"
            for p in seg_probs
        ],
    ))
    bar_fig.update_layout(
        title="Audio segment fake probability",
        yaxis=dict(title="Fake prob", range=[0, 1], tickformat=".0%"),
        xaxis_title="Segment (seconds)", height=320,
        margin=dict(t=40, b=40, l=40, r=20),
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    mid_times = [(s.start_sec + s.end_sec) / 2 for s in segments]
    wave_fig = go.Figure(go.Scatter(
        x=mid_times, y=seg_probs, mode="lines+markers",
        line=dict(color="steelblue", width=2),
    ))
    wave_fig.update_layout(
        title="Audio waveform / probability envelope",
        yaxis=dict(title="Fake prob", range=[0, 1], tickformat=".0%"),
        xaxis_title="Time (s)", height=260,
        margin=dict(t=40, b=40, l=40, r=20),
    )
    st.plotly_chart(wave_fig, use_container_width=True)


def render_per_frame_chart(video_result) -> None:
    frames = video_result.frame_results
    if not frames:
        st.info("No per-frame data available.")
        return
    fig = go.Figure(go.Scatter(
        x=[fr.timestamp_sec for fr in frames],
        y=[fr.fake_prob      for fr in frames],
        mode="lines", line=dict(color="crimson", width=2),
    ))
    fig.update_layout(
        title="Per-frame fake probability",
        yaxis=dict(title="Fake prob", range=[0, 1], tickformat=".0%"),
        xaxis_title="Time (s)", height=300,
        margin=dict(t=40, b=40, l=40, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_verdict_timeline(metadata: dict) -> None:
    timeline = metadata.get("verdict_timeline", [])
    if not timeline:
        st.info("No verdict timeline data available.")
        return
    times  = [t for t, _ in timeline]
    scores = [s for _, s in timeline]
    fig = go.Figure(go.Scatter(
        x=times, y=scores, mode="lines+markers",
        line=dict(color="darkorange", width=2),
        fill="tozeroy", fillcolor="rgba(255,165,0,0.15)",
    ))
    fig.add_hline(y=0.6, line_dash="dash",  line_color="red",    annotation_text="FAKE threshold")
    fig.add_hline(y=0.4, line_dash="dot",   line_color="orange", annotation_text="UNCERTAIN threshold")
    fig.update_layout(
        title="Verdict timeline",
        yaxis=dict(title="Fused score", range=[0, 1], tickformat=".0%"),
        xaxis_title="Time (s)", height=300,
        margin=dict(t=40, b=40, l=40, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_download_button(result) -> None:
    json_bytes = json.dumps(main._result_to_dict(result), indent=2).encode()
    # AnalysisResult uses analysis_id in the real codebase
    result_id = getattr(result, "analysis_id", getattr(result, "video_id", "unknown"))
    st.download_button(
        label="⬇️ Download full report (JSON)",
        data=json_bytes,
        file_name=f"forensight_{result_id}.json",
        mime="application/json",
    )


def run_pipeline_with_progress(video_path: str):
    progress = st.progress(0, text="Starting analysis…")
    stages = [
        (0.10, "Demuxing video…"),
        (0.40, "Running video detector…"),
        (0.70, "Running audio detector…"),
        (0.90, "Fusing modalities…"),
    ]
    for pct, label in stages[:-1]:
        progress.progress(pct, text=label)
    result = main.run_pipeline(video_path=Path(video_path))
    progress.progress(0.90, text=stages[-1][1])
    progress.progress(1.0,  text="Analysis complete ✓")
    return result


# ---------------------------------------------------------------------------
# Image tab — section renderers
# ---------------------------------------------------------------------------

def render_image_verdict_badge(result: ImageResult) -> None:
    _render_verdict_badge(result.verdict, result.fused_score)


def render_image_score_breakdown(result: ImageResult) -> None:
    sub = result.sub_scores or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Texture",  f"{sub.get('texture', result.gradcam_score):.2%}",
              help="Sharpness/edge heuristic — NOT a trained model, weakest signal alone")
    c2.metric("Spectral", f"{sub.get('spectral', result.freq_score):.2%}",
              help="FFT analysis for GAN/diffusion upsampling artefacts")
    c3.metric("ELA",      f"{sub.get('ela', 0.0):.2%}",
              help="Error-level analysis — detects locally edited/spliced regions")
    c4.metric("Metadata", f"{sub.get('metadata', 0.0):.2%}",
              help="EXIF / AI-generator tag / provenance check")


def render_image_localization(result: ImageResult) -> None:
    st.subheader("📍 Where")
    if not result.localization_heatmap:
        st.info("Region-level analysis wasn't available for this image.")
        return

    col_map, col_text = st.columns([3, 2])
    with col_map:
        st.image(
            result.localization_heatmap,
            use_container_width=True,
            caption="Red = noise/compression inconsistent with the rest of the photo",
        )
    with col_text:
        st.write(result.localization_summary)
        if result.localization_regions:
            for reg in result.localization_regions:
                st.caption(
                    f"• {reg['position_desc']} — ~{round(reg['area_fraction']*100)}% of frame"
                )

        verdict_str = _verdict_str(result.verdict)
        metadata_score = result.sub_scores.get("metadata", 0.0)
        pixel_scores = [
            v for k, v in result.sub_scores.items() if k != "metadata"
        ]
        pixel_signals_clean = bool(pixel_scores) and max(pixel_scores) < 0.4

        if (
            verdict_str in ("FAKE", "UNCERTAIN")
            and not result.localization_is_localized
            and metadata_score >= 0.9
            and pixel_signals_clean
        ):
            st.caption(
                "📎 **This verdict rests almost entirely on the provenance "
                "metadata** (C2PA / generator tag), not on any pixel-level "
                "trace — every noise, compression, and frequency signal on "
                "this photo reads as clean. This typically happens when an "
                "AI editing tool re-renders the **entire** image rather "
                "than pasting a patch into the original file, which erases "
                "any local seam to find. If the metadata were stripped "
                "(e.g. by re-saving or forwarding the file), this specific "
                "edit would currently be very hard to catch — pixel "
                "heuristics alone aren't yet reliable against this class "
                "of full-image AI edit."
            )
        elif (
            verdict_str in ("FAKE", "UNCERTAIN")
            and not result.localization_is_localized
            and result.metadata.get("suspicious_fraction", 0) is not None
            and result.metadata.get("suspicious_fraction", 0) < 0.1
        ):
            st.caption(
                "Since the overall analysis flags this image but no single "
                "region stands out here, that points toward the **whole "
                "image** being AI-generated rather than one small edited "
                "spot."
            )

        st.caption(
            "⚠️ This only works for **partial edits** on an otherwise-real "
            "photo. A fully AI-generated image has no untouched region to "
            "compare against, so it will show as spread across the whole "
            "frame (or nothing localized at all) rather than one specific spot."
        )


def render_image_overlays(result: ImageResult) -> None:
    col_gc, col_fft, col_ela = st.columns(3)
    with col_gc:
        st.subheader("Texture Heatmap")
        if result.gradcam_overlay:
            st.image(result.gradcam_overlay, use_container_width=True,
                     caption="Sharpness/edge activation blended on image")
        else:
            st.info("Texture overlay not available.")
    with col_fft:
        st.subheader("Frequency Heatmap")
        if result.freq_heatmap:
            st.image(result.freq_heatmap, use_container_width=True,
                     caption="FFT magnitude spectrum (log scale)")
        else:
            st.info("Frequency heatmap not available.")
    with col_ela:
        st.subheader("Error Level Analysis")
        if result.ela_heatmap:
            st.image(result.ela_heatmap, use_container_width=True,
                     caption="Bright regions = different compression history")
        else:
            st.info("ELA heatmap not available.")


def render_image_gauge(result: ImageResult) -> None:
    st.plotly_chart(_build_gauge(result.fused_score), use_container_width=True)


def render_image_score_chart(result: ImageResult) -> None:
    sub = result.sub_scores or {}
    labels = ["Texture", "Spectral", "ELA", "Metadata", "Fused"]
    values = [
        sub.get("texture", result.gradcam_score),
        sub.get("spectral", result.freq_score),
        sub.get("ela", 0.0),
        sub.get("metadata", 0.0),
        result.fused_score,
    ]
    colors = [
        "red" if v >= 0.6 else "orange" if v >= 0.4 else "green"
        for v in values
    ]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Detection score breakdown",
        yaxis=dict(title="Fake probability", range=[0, 1.1], tickformat=".0%"),
        height=320, margin=dict(t=40, b=20, l=40, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_image_metadata(result: ImageResult) -> None:
    with st.expander("Image metadata"):
        meta = result.metadata
        st.json({
            "image_id":       result.image_id,
            "width":          meta.get("width"),
            "height":         meta.get("height"),
            "face_detected":  meta.get("face_detected"),
            "gradcam_weight": meta.get("gradcam_weight"),
            "freq_weight":    meta.get("freq_weight"),
            "run_id":         meta.get("run_id"),
        })


def render_image_download_button(result: ImageResult) -> None:
    payload = {
        "image_id":      result.image_id,
        "verdict":       _verdict_str(result.verdict),
        "fused_score":   result.fused_score,
        "gradcam_score": result.gradcam_score,
        "freq_score":    result.freq_score,
        "metadata":      result.metadata,
    }
    st.download_button(
        label="⬇️ Download image report (JSON)",
        data=json.dumps(payload, indent=2).encode(),
        file_name=f"forensight_image_{result.image_id}.json",
        mime="application/json",
    )


@st.cache_resource
def _load_ai_classifier() -> AIClassifier | None:
    """
    Loads the optional trained AI-vs-real image classifier once per server
    process (not once per upload) if config/config.yaml enables it. Returns
    None if disabled/unconfigured — ImageDetector already handles that
    gracefully and falls back to the 5-heuristic pipeline unchanged.
    """
    cfg = getattr(CFG, "image_ai_classifier", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        return None
    model_path = getattr(cfg, "model_path", "") or None
    if not model_path:
        return None
    clf = AIClassifier(
        model_path=model_path,
        backend=getattr(cfg, "backend", "auto"),
        input_size=getattr(cfg, "input_size", 224),
        ai_class_index=getattr(cfg, "ai_class_index", 0),
    )
    if not clf.available:
        st.warning(
            f"⚠️ AI classifier is enabled in config but failed to load: "
            f"{clf.load_error}. Falling back to heuristic-only detection."
        )
    return clf


def run_image_pipeline(uploaded_file) -> ImageResult:
    ai_classifier = _load_ai_classifier()
    detector  = ImageDetector(ai_classifier=ai_classifier)
    image_id  = Path(uploaded_file.name).stem
    raw_bytes = uploaded_file.getvalue()
    return detector.detect_from_bytes(
        raw_bytes,
        image_id=image_id,
        image_path=uploaded_file.name,
    )


# ---------------------------------------------------------------------------
# Tab pages
# ---------------------------------------------------------------------------

def _video_tab() -> None:
    st.header("🎬 Video Deepfake Detection")
    st.caption("Upload a video to analyse it for audio-visual manipulation.")

    uploaded = st.file_uploader(
        "Upload video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="video_uploader",
    )
    if uploaded is None:
        st.info("👆 Upload a video file to get started.")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = str(Path(tmp_dir) / uploaded.name)
        with open(video_path, "wb") as fh:
            fh.write(uploaded.getbuffer())
        try:
            result = run_pipeline_with_progress(video_path)
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            return

    st.divider()
    col_badge, col_gauge = st.columns([1, 2])
    with col_badge:
        st.subheader("Verdict")
        render_verdict_badge(result.verdict, result.fused_score)
    with col_gauge:
        st.subheader("Fused Score")
        render_gauge(result.fused_score)

    # Short, plain-English one-liner right under the verdict/gauge — filled
    # in below once the full explanation is computed, so it doesn't need to
    # duplicate that logic. Anyone should be able to read this and get the
    # gist without scrolling to the detailed Explanation section.
    quick_summary_slot = st.empty()

    st.divider()
    st.subheader("🧾 Explanation")
    audio_available = result.metadata.get("audio_available", True)
    eff_video_weight = result.metadata.get("effective_video_weight", 0.6)
    eff_audio_weight = result.metadata.get("effective_audio_weight", 0.4)

    signals = {
        "video": {
            "score": result.video_result.fake_prob, "weight": eff_video_weight,
            "findings": [_video_visual_finding(result.video_result)],
        },
    }
    if audio_available:
        signals["audio"] = {
            "score": result.audio_result.fake_prob, "weight": eff_audio_weight,
            "findings": [
                "Audio analysis (voice spectral characteristics) scored "
                f"{result.audio_result.fake_prob:.0%} likelihood of manipulation."
            ],
        }
    else:
        # No audio stream in the source video — don't present the neutral
        # 0.5 placeholder as if it were real evidence (it carried zero
        # weight in the actual fused score; see Fuser's audio_available
        # re-weighting). Surfaced as an informational note instead, not a
        # ranked finding.
        st.caption(f"ℹ️ {result.audio_result.error} — verdict is based on video analysis only.")

    summary = _explain.summarise(_verdict_str(result.verdict), result.fused_score, signals)

    verdict_str = _verdict_str(result.verdict)
    _quick_summary_box = {"FAKE": st.error, "REAL": st.success, "UNCERTAIN": st.warning}.get(
        verdict_str, st.info
    )
    _seg_meta = result.video_result.metadata or {}
    _face_seg = _seg_meta.get("face_segment")
    _non_face_seg = _seg_meta.get("non_face_segment")
    _extra_lines: list[str] = []
    if _face_seg and _non_face_seg:
        _extra_lines.append(
            f"Frames where we could see a face read {_face_seg['verdict'].lower()} "
            f"({_face_seg['score']:.0%}), while the rest of the footage — without a "
            f"visible face — read {_non_face_seg['verdict'].lower()} "
            f"({_non_face_seg['score']:.0%})."
        )
    elif _face_seg:
        _extra_lines.append(
            f"Every frame we checked had a visible face, and that reading came out "
            f"{_face_seg['verdict'].lower()} ({_face_seg['score']:.0%})."
        )
    elif _non_face_seg:
        _extra_lines.append(
            f"No face was detected in any sampled frame, so this call is based "
            f"purely on visual texture patterns across the footage "
            f"({_non_face_seg['score']:.0%})."
        )
    with quick_summary_slot.container():
        _quick_summary_box(
            _build_detailed_summary(summary["headline"], summary["reasons"], verdict_str, _extra_lines)
        )

    _render_explanation_panel(summary["headline"], summary["reasons"], summary["confidence_label"])

    st.divider()
    render_video_provenance(result.video_result)

    st.divider()
    render_video_segments(result.video_result)

    with st.expander("🔬 Advanced technical details"):
        st.subheader("Detection Score Breakdown")
        render_video_score_chart(result.video_result)

        st.divider()
        render_video_localization(result.video_result)

        st.divider()
        st.subheader("GradCAM Activation Overlays")
        render_gradcam_grid(result.video_result)

        st.divider()
        st.subheader("Audio Analysis")
        render_audio_section(result.audio_result)

        st.divider()
        st.subheader("Per-Frame Video Score")
        render_per_frame_chart(result.video_result)

        st.divider()
        st.subheader("Verdict Timeline")
        render_verdict_timeline(result.metadata)

    st.divider()
    render_download_button(result)


def _image_tab() -> None:
    st.header("🖼️ Image Deepfake Detection")
    st.caption("Upload an image — GradCAM + FFT analysis will determine if it's real or AI-generated.")

    uploaded = st.file_uploader(
        "Upload image file",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="image_uploader",
    )
    if uploaded is None:
        st.info("👆 Upload an image file to get started.")
        return

    progress = st.progress(0, text="Reading image…")
    try:
        progress.progress(0.3, text="Running GradCAM detector…")
        progress.progress(0.6, text="Running frequency analyser…")
        result = run_image_pipeline(uploaded)
        progress.progress(1.0, text="Analysis complete ✓")
    except Exception as exc:
        st.error(f"Image analysis failed: {exc}")
        return

    st.divider()
    col_img, col_badge = st.columns([1, 1])
    with col_img:
        st.subheader("Uploaded Image")
        st.image(uploaded.getvalue(), use_container_width=True)
    with col_badge:
        st.subheader("Verdict")
        render_image_verdict_badge(result)

        _quick_headline = result.headline or _verdict_str(result.verdict)
        _quick_summary_box = {"FAKE": st.error, "REAL": st.success, "UNCERTAIN": st.warning}.get(
            _verdict_str(result.verdict), st.info
        )
        _quick_summary_box(
            _build_detailed_summary(_quick_headline, result.findings, _verdict_str(result.verdict))
        )

    st.divider()
    st.subheader("🧾 Explanation")
    _render_explanation_panel(
        result.headline or _verdict_str(result.verdict),
        result.findings,
        result.confidence_label,
    )

    st.divider()
    render_image_localization(result)

    with st.expander("🔬 Advanced technical details"):
        render_image_score_breakdown(result)
        st.divider()
        render_image_gauge(result)
        st.divider()
        render_image_overlays(result)
        st.divider()
        render_image_score_chart(result)
        st.divider()
        render_image_metadata(result)

    st.divider()
    render_image_download_button(result)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main_app() -> None:
    st.title("🔍 ForenSight — Deepfake Detection Dashboard")

    tab_video, tab_image = st.tabs(["🎬 Video", "🖼️ Image"])

    with tab_video:
        _video_tab()

    with tab_image:
        _image_tab()


if __name__ == "__main__":
    main_app()