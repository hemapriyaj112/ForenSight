# forensight/pipeline/__init__.py
# Exposes the three pipeline stages as a clean public API.

from pipeline.video.detector import VideoDetector       # noqa: F401
from pipeline.audio.detector import AudioDetector       # noqa: F401
from pipeline.fusion.fuser import Fuser                 # noqa: F401
