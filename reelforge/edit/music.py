"""Audio ducking and -14 LUFS loudness normalization."""


def build_audio_mix_filter(target_loudness_lufs: float = -14.0) -> str:
    """Build FFmpeg audio filter for EBU R128 (-14 LUFS) broadcast compliance."""
    return f"loudnorm=I={target_loudness_lufs:.1f}:LRA=11.0:TP=-1.5"
