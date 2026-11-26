"""Action kinetic energy and intensity estimation."""

import numpy as np


def estimate_action_intensity(motion_magnitudes: np.ndarray) -> float:
    """Calculate kinetic action score based on velocity and acceleration dynamics."""
    if len(motion_magnitudes) == 0:
        return 0.0
    mean_mag = float(np.mean(motion_magnitudes))
    p95_mag = float(np.percentile(motion_magnitudes, 95))
    return float(np.clip(0.6 * mean_mag + 0.4 * p95_mag, 0.0, 1.0))
