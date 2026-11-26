"""Extract clean, normalized 16kHz mono audio WAV for speech and music analysis."""

from pathlib import Path
from ..core.ffmpeg import run_ffmpeg, require_tool


def extract_normalized_audio(
    input_path: Path,
    output_wav_path: Path,
    sample_rate: int = 16000,
    overwrite: bool = False,
) -> Path:
    """Extract first audio track to a standard 16 kHz 16-bit PCM mono WAV file."""
    require_tool("ffmpeg")
    input_path = Path(input_path).resolve()
    output_wav_path = Path(output_wav_path).resolve()
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    if output_wav_path.is_file() and not overwrite and output_wav_path.stat().st_size > 1000:
        return output_wav_path

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i", str(input_path),
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(output_wav_path),
    ]

    try:
        run_ffmpeg(cmd, check=True)
    except RuntimeError:
        # Fallback if map 0:a:0 failed (e.g. video without explicit map)
        cmd_fallback = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i", str(input_path),
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(output_wav_path),
        ]
        run_ffmpeg(cmd_fallback, check=True)

    return output_wav_path
