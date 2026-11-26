"""Robust FFmpeg and FFprobe subprocess utilities for ReelForge."""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class VideoProbe:
    """Container for probed video stream properties."""

    path: Path
    duration: float  # Precise container/stream duration in seconds (float64)
    width: int
    height: int
    fps: float
    has_audio: bool
    audio_sample_rate: int
    audio_channels: int
    codec_video: str
    codec_audio: str
    bitrate: int


def require_tool(name: str):
    """Verify that an external CLI binary exists on system PATH."""
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable '{name}' was not found on PATH. Please ensure FFmpeg/FFprobe is installed.")


def run_ffmpeg(
    cmd: List[str],
    timeout: Optional[float] = None,
    check: bool = True,
) -> Tuple[int, str, str]:
    """Execute an FFmpeg command safely using communicate() to eliminate stdout/stderr pipe deadlocks.

    Returns:
        (returncode, stdout_str, stderr_str)
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.communicate()
        raise RuntimeError(f"FFmpeg command timed out after {timeout} seconds: {' '.join(cmd)}") from exc

    if check and proc.returncode != 0:
        clean_err = stderr.strip()
        # Keep last 15 lines of error for conciseness
        short_err = "\n".join(clean_err.splitlines()[-15:])
        raise RuntimeError(f"FFmpeg failed with exit code {proc.returncode}:\n{short_err}")

    return proc.returncode, stdout, stderr


def probe_video(input_path: Path) -> VideoProbe:
    """Probe video format, streams, precise duration, and audio specifications."""
    require_tool("ffprobe")
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Video file not found: {input_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(input_path),
    ]
    ret, stdout, stderr = run_ffmpeg(cmd, check=True)
    metadata = json.loads(stdout)

    video_streams = [s for s in metadata.get("streams", []) if s.get("codec_type") == "video"]
    audio_streams = [s for s in metadata.get("streams", []) if s.get("codec_type") == "audio"]

    if not video_streams:
        raise ValueError(f"No video streams found in file: {input_path}")

    v0 = video_streams[0]
    
    # Parse precise duration from format or video stream
    duration = 0.0
    if "duration" in metadata.get("format", {}):
        try:
            duration = float(metadata["format"]["duration"])
        except (ValueError, TypeError):
            pass
    if duration <= 0.0 and "duration" in v0:
        try:
            duration = float(v0["duration"])
        except (ValueError, TypeError):
            pass

    # Fallback: estimate from frame rate and nb_frames if available
    if duration <= 0.0:
        nb_frames = v0.get("nb_frames")
        if nb_frames and "r_frame_rate" in v0:
            try:
                num, den = map(float, v0["r_frame_rate"].split("/"))
                fps_val = num / den if den > 0 else 24.0
                duration = float(nb_frames) / fps_val
            except Exception:
                pass

    if duration <= 0.0:
        raise ValueError(f"Could not determine valid duration for video: {input_path}")

    # Parse resolution
    width = int(v0.get("width", 1920))
    height = int(v0.get("height", 1080))

    # Parse fps
    fps = 24.0
    r_fps = v0.get("r_frame_rate", "24/1")
    if "/" in r_fps:
        try:
            num, den = map(float, r_fps.split("/"))
            fps = num / den if den > 0 else 24.0
        except Exception:
            fps = 24.0

    # Audio details
    has_audio = len(audio_streams) > 0
    audio_sr = int(audio_streams[0].get("sample_rate", 48000)) if has_audio else 0
    audio_ch = int(audio_streams[0].get("channels", 2)) if has_audio else 0
    codec_audio = audio_streams[0].get("codec_name", "none") if has_audio else "none"
    codec_video = v0.get("codec_name", "unknown")

    bitrate = int(metadata.get("format", {}).get("bit_rate", 0))

    return VideoProbe(
        path=input_path,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio=has_audio,
        audio_sample_rate=audio_sr,
        audio_channels=audio_ch,
        codec_video=codec_video,
        codec_audio=codec_audio,
        bitrate=bitrate,
    )


def has_cuda_accel() -> bool:
    """Check if the local FFmpeg build supports NVIDIA CUDA decoding/encoding."""
    try:
        ret, stdout, stderr = run_ffmpeg(["ffmpeg", "-hwaccels"], check=False)
        return "cuda" in stdout.lower()
    except Exception:
        return False
