"""ReelForge Core: Configuration, FFmpeg wrappers, Schemas, and Manifest management."""

from .config import Settings, get_settings
from .ffmpeg import probe_video, require_tool, run_ffmpeg, VideoProbe, has_cuda_accel
from .schemas import (
    ShotData,
    SceneData,
    SignalMetrics,
    DialogueData,
    MomentCandidate,
    RightsInfo,
    ManifestSchema,
)
from .manifest import ManifestManager
from .logger import setup_logger, get_logger, write_markdown_report

__all__ = [
    "Settings",
    "get_settings",
    "probe_video",
    "require_tool",
    "run_ffmpeg",
    "VideoProbe",
    "has_cuda_accel",
    "ShotData",
    "SceneData",
    "SignalMetrics",
    "DialogueData",
    "MomentCandidate",
    "RightsInfo",
    "ManifestSchema",
    "ManifestManager",
    "setup_logger",
    "get_logger",
    "write_markdown_report",
]
