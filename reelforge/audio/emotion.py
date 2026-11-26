"""Speech emotion dimensional estimation (arousal and valence)."""

from typing import Tuple


def estimate_speech_emotion(
    rms_level: float,
    pitch_range: float,
    spectral_centroid: float,
    minor_key_corr: float,
) -> Tuple[float, float, str]:
    """Estimate dimensional emotion (arousal [0, 1] and valence [-1, 1]).

    Returns:
        (arousal, valence, dominant_category)
    """
    # High RMS + high spectral centroid + high pitch range -> high arousal (action/intensity)
    arousal = float(min(1.0, max(0.0, 0.4 * rms_level + 0.3 * (spectral_centroid / 4000.0) + 0.3 * (pitch_range / 200.0))))
    
    # Minor key + low centroid + low RMS -> negative valence (sadness/grief)
    valence = float(min(1.0, max(-1.0, 0.5 - 1.2 * minor_key_corr - 0.4 * (1.0 - rms_level))))

    category = "neutral"
    if valence < -0.2 and arousal < 0.5:
        category = "sad"
    elif arousal > 0.6 and valence > 0.1:
        category = "action"
    elif valence > 0.3 and arousal < 0.5:
        category = "romantic"
    elif arousal > 0.6 and valence < -0.1:
        category = "suspense"

    return arousal, valence, category
