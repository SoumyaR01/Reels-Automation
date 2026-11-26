"""Audio event detection (crying, laughter, silence, music, impact)."""

from typing import List


def detect_audio_events(
    rms_val: float,
    centroid_val: float,
    minor_corr: float,
) -> List[str]:
    """Tag audio events based on acoustic signature thresholds."""
    events = []
    if rms_val < 0.08:
        events.append("silence")
    if minor_corr > 0.55:
        events.append("melancholic_music")
    elif minor_corr < 0.2 and rms_val > 0.3:
        events.append("upbeat_music")
    if centroid_val > 3800.0 and rms_val > 0.4:
        events.append("impact")
    if minor_corr > 0.6 and rms_val < 0.2:
        events.append("crying_or_sighing")

    return events
