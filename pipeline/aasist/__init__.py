"""pipeline/aasist — real pretrained AASIST audio anti-spoofing detector.

AASIST (Jung et al., ICASSP 2022) wrapped behind the same lazy-load contract
used by pipeline/genconvit. Evaluation-only in Phase 2B: the score is exposed
under aasist_* keys and is NOT fused into the production verdict.
"""
from pipeline.aasist.detector import AASISTDetector, STATUS_ACTIVE_MODEL, STATUS_INACTIVE  # noqa: F401

__all__ = ["AASISTDetector", "STATUS_ACTIVE_MODEL", "STATUS_INACTIVE"]