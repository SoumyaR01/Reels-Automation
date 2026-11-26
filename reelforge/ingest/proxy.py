"""Generate lightweight proxy video for frame-accurate, high-throughput analysis."""

from pathlib import Path
from ..core.ffmpeg import run_ffmpeg, has_cuda_accel, require_tool
from ..core.config import Settings


def generate_video_proxy(
    input_path: Path,
    output_path: Path,
    settings: Settings,
    overwrite: bool = False,
) -> Path:
    """Transcode source video to an efficient low-resolution proxy (e.g. 360p @ 12 fps).

    This eliminates repetitive expensive full-resolution decodes across analysis stages.
    """
    require_tool("ffmpeg")
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.is_file() and not overwrite and output_path.stat().st_size > 1000:
        return output_path

    cuda_available = has_cuda_accel()

    # Build command
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if cuda_available:
        cmd.extend(["-hwaccel", "cuda"])

    filter_graph = f"scale=-2:{settings.proxy_height},fps={settings.proxy_fps}"

    cmd.extend([
        "-i", str(input_path),
        "-vf", filter_graph,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "26",
        "-an",  # Strip audio from video proxy (audio is in dedicated WAV)
        "-movflags", "+faststart",
        str(output_path),
    ])

    run_ffmpeg(cmd, check=True)
    return output_path
