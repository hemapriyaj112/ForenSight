"""pipeline/forensics — non-ML forensic signal extractors.

These are real, established image-forensics techniques (metadata/provenance
analysis, error-level analysis, spectral artefact analysis) used as
corroborating evidence alongside the texture heuristic in
pipeline/video/detector.py. They do not require any downloaded model
weights, which makes them reliable to run fully offline.

Each module exposes a single analyse_*() function returning a dict with at
least: {"score": float 0-1, "findings": list[str]}. "score" is this
signal's own estimate of "how suspicious of AI-generation/manipulation",
and "findings" are short, plain-language strings explaining *why*, safe to
show directly to an end user.
"""
