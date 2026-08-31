"""pipeline/genconvit — real pretrained GenConViT video deepfake detector."""
from pipeline.genconvit.detector import GenConViTDetector, STATUS_ACTIVE_MODEL, STATUS_INACTIVE  # noqa: F401

__all__ = ["GenConViTDetector", "STATUS_ACTIVE_MODEL", "STATUS_INACTIVE"]
