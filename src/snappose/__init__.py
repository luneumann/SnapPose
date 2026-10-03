"""SnapPose: single-shot 6D pose matching of a known CAD object in one RGB-D frame."""
from .api import PoseMatcher
from .config import Config
from .types import MatchResult

__all__ = ["PoseMatcher", "Config", "MatchResult"]
__version__ = "0.1.0"
