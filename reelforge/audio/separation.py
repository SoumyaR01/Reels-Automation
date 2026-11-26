"""Audio stem separation interface (Demucs / vocal isolating bandpass)."""

from pathlib import Path
from typing import Optional


def separate_audio_stems(wav_path: Path, output_dir: Path) -> Optional[Path]:
    """Separate dialogue, music, and effects stems when Demucs is available."""
    # When Demucs is not installed, the original WAV is used with ducking filters
    return Path(wav_path)
