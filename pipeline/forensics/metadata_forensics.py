"""pipeline/forensics/metadata_forensics.py

Metadata / provenance forensics.

Real camera photos almost always carry EXIF data (Make, Model, exposure
settings, GPS, lens info...). AI generators typically strip this entirely,
and several popular tools (Automatic1111/Stable Diffusion, ComfyUI,
Midjourney via Discord export, Adobe Firefly, Bing/DALL-E) leave their own
distinctive fingerprints behind in PNG text chunks, XMP blocks, or the
"Software"/"Make" EXIF fields. C2PA "Content Credentials" manifests
(embedded by an increasing number of AI tools and cameras) are also
checked for, at a shallow byte-signature level.

This is intentionally conservative: EXIF absence alone is only weak
evidence (many legitimate images — screenshots, web-saved photos,
messaging-app re-uploads — also lack it), so it never single-handedly
drives the verdict. A confirmed AI-generator tag, on the other hand, is
very strong evidence and is scored accordingly.
"""
from __future__ import annotations

import io
import re
from typing import Any

_GENERATOR_SIGNATURES = [
    ("stable diffusion", "Stable Diffusion"),
    ("midjourney", "Midjourney"),
    ("dall\u00b7e", "DALL\u00b7E"),
    ("dall-e", "DALL\u00b7E"),
    ("dalle", "DALL\u00b7E"),
    ("adobe firefly", "Adobe Firefly"),
    ("firefly", "Adobe Firefly"),
    ("nightcafe", "NightCafe"),
    ("comfyui", "ComfyUI"),
    ("automatic1111", "Automatic1111 (Stable Diffusion UI)"),
    ("invokeai", "InvokeAI"),
    ("leonardo.ai", "Leonardo.Ai"),
    ("runwayml", "Runway"),
    ("sora", "OpenAI Sora"),
    ("sdxl", "Stable Diffusion XL"),
]

# PNG text chunks written by common generation UIs
_GENERATOR_PNG_KEYS = {"parameters", "prompt", "workflow", "generation_data", "sd-metadata"}

_C2PA_MARKERS = [b"c2pa", b"jumb", b"application/c2pa", b"urn:uuid"]

# C2PA "digitalSourceType" values (IPTC NewsCodes) that indicate AI
# involvement vs. genuine capture. C2PA manifests are typically CBOR-
# encoded, but string values inside CBOR are stored as raw UTF-8, so a
# byte-level substring scan finds them without a full CBOR/COSE parser.
# This is a heuristic text scan, not a validated/signature-checked C2PA
# read — it can't confirm the manifest's signature is authentic, only
# that these strings are present in the file. For an authoritative check,
# a person should still run a real C2PA verifier (e.g. Content Credentials
# Verify at contentcredentials.org/verify).
_C2PA_AI_SOURCE_TYPES = [
    b"trainedalgorithmicmedia",
    b"compositewithtrainedalgorithmicmedia",
    b"algorithmicmedia",
]
_C2PA_AI_ACTION_KEYWORDS = [
    b"c2pa.ai-generative-training",
    b"generativefill",
    b"generative_fill",
    b"c2pa.generated",
]
_C2PA_CAPTURE_SOURCE_TYPES = [
    b"digitalcapture",
    b"negativefilm",
    b"positivefilm",
    b"minorhumanedits",
]

_CAMERA_EXIF_TAGS = {"Make", "Model", "LensModel", "FNumber", "ExposureTime", "FocalLength", "ISOSpeedRatings"}


def analyse_metadata(image_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
    """Inspect raw image bytes for provenance signals.

    Returns
    -------
    dict with keys: score (0-1), findings (list[str]), has_exif (bool),
    generator_tag (str | None), has_camera_tags (bool)
    """
    findings: list[str] = []
    generator_tag: str | None = None
    has_exif = False
    has_camera_tags = False

    # --- C2PA / Content Credentials manifest scan (done first, so its
    # result is known before we decide whether an "EXIF absent" line would
    # be redundant/confusing next to it) ---
    # Search the whole file (capped for very large files) rather than only
    # the first 200KB — the manifest/JUMBF box isn't guaranteed to be near
    # the start, especially in PNG files where it can sit in a trailing
    # chunk.
    scan_bytes = image_bytes if len(image_bytes) <= 8_000_000 else image_bytes[:8_000_000]
    scan_lower = scan_bytes.lower()
    has_c2pa = any(marker in scan_lower for marker in _C2PA_MARKERS)
    ai_hit = capture_hit = None

    if has_c2pa:
        ai_hit = next((k for k in _C2PA_AI_SOURCE_TYPES if k in scan_lower), None) \
            or next((k for k in _C2PA_AI_ACTION_KEYWORDS if k in scan_lower), None)
        capture_hit = next((k for k in _C2PA_CAPTURE_SOURCE_TYPES if k in scan_lower), None)

        if ai_hit:
            generator_tag = "AI tool (per C2PA manifest)"
            findings.append(
                "This image's Content Credentials (C2PA) manifest contains a "
                "digital-source-type or action consistent with AI generation "
                "or AI editing (not a signature-verified read — confirm with "
                "a C2PA verifier for the authoritative claim)."
            )
        elif capture_hit:
            findings.append(
                "This image's Content Credentials (C2PA) manifest contains a "
                "digital-source-type consistent with a genuine camera capture "
                "(not a signature-verified read — confirm with a C2PA verifier "
                "for the authoritative claim)."
            )
        else:
            findings.append(
                "The file appears to contain a Content Credentials (C2PA) "
                "provenance manifest, but its specific origin claim couldn't "
                "be read from a quick scan — check it with a proper C2PA "
                "verifier (e.g. contentcredentials.org/verify) for the "
                "authoritative origin claim."
            )

    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(io.BytesIO(image_bytes))

        # --- PNG text chunks (Stable Diffusion / ComfyUI / etc write here) ---
        png_info = getattr(img, "text", {}) or getattr(img, "info", {}) or {}
        for key, value in png_info.items():
            key_l = str(key).lower()
            if key_l in _GENERATOR_PNG_KEYS or "generation" in key_l:
                tag = _match_generator_signature(str(value)) or "AI image-generation tool"
                generator_tag = tag
                findings.append(
                    f"Metadata contains an AI image-generation field ('{key}') "
                    f"consistent with {tag}."
                )
                break

        # --- EXIF (JPEG/TIFF/some WEBP) ---
        exif = None
        try:
            exif = img.getexif()
        except Exception:
            exif = None

        if exif:
            decoded = {TAGS.get(k, k): v for k, v in exif.items()}
            has_exif = len(decoded) > 0
            has_camera_tags = any(t in decoded and decoded[t] for t in _CAMERA_EXIF_TAGS)

            software = str(decoded.get("Software", "") or "")
            make_model = f"{decoded.get('Make', '')} {decoded.get('Model', '')}"
            sig = _match_generator_signature(software) or _match_generator_signature(make_model)
            if sig:
                generator_tag = sig
                findings.append(f"Image metadata's Software/Make field identifies it as {sig}.")

            if has_camera_tags and not generator_tag:
                findings.append(
                    "Camera metadata (make/model/exposure settings) was found — "
                    "typical of a genuine camera or phone photo."
                )

        if not has_exif and not generator_tag and not has_c2pa:
            fmt = (img.format or "").upper()
            if fmt in ("JPEG", "JPG"):
                findings.append(
                    "No camera metadata (EXIF) was found in this JPEG. Most "
                    "real camera and phone photos retain this data, though it "
                    "can also be stripped by editing apps or messaging platforms."
                )
            else:
                findings.append(
                    f"No embedded metadata found ({fmt or 'this format'} files "
                    "often lack it regardless of origin, so this is weak evidence on its own)."
                )
    except Exception:
        findings.append("Could not read image metadata (file may be corrupted or stripped).")

    # --- score ---
    if generator_tag and ai_hit:
        score = 0.97
    elif generator_tag:
        score = 0.95
    elif capture_hit:
        score = 0.05
    elif has_camera_tags:
        score = 0.05
    elif has_exif:
        score = 0.25
    elif has_c2pa:
        # C2PA present but its claim couldn't be read: mildly more notable
        # than plain metadata absence (real cameras/phones rarely attach
        # C2PA at all today; it's disproportionately common on AI/editing
        # tool output) — but still just a nudge, not a verdict, since C2PA
        # adoption on genuine cameras is real and growing.
        score = 0.62
    else:
        score = 0.55  # mildly suspicious, not conclusive

    return {
        "score": score,
        "findings": findings,
        "has_exif": has_exif,
        "generator_tag": generator_tag,
        "has_camera_tags": has_camera_tags,
        "has_c2pa": has_c2pa,
    }


def _match_generator_signature(text: str) -> str | None:
    text_l = text.lower()
    for needle, label in _GENERATOR_SIGNATURES:
        if needle in text_l:
            return label
    return None
