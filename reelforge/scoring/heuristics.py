"""Multi-signal heuristic scoring across all 6 supported ReelForge categories."""

from typing import Dict
import numpy as np

from ..core.schemas import SceneData, SignalMetrics
from ..audio.features import AudioAnalysisResult
from ..video.aesthetics import VideoAestheticsResult


def compute_scene_heuristic_scores(
    scene: SceneData,
    audio_res: AudioAnalysisResult,
    video_res: VideoAestheticsResult,
    shot_durations: list,
) -> Dict[str, float]:
    """Compute raw multi-signal scores across the 6 categories for a given scene."""
    start_t, end_t = scene.start, scene.end

    # Audio slice
    a_idx = np.where((audio_res.time_points >= start_t) & (audio_res.time_points <= end_t))[0]
    if len(a_idx) > 0:
        mean_rms = float(np.mean(audio_res.rms[a_idx]))
        mean_minor = float(np.mean(audio_res.minor_correlation[a_idx]))
        mean_sc = float(np.mean(audio_res.spectral_centroid[a_idx]))
    else:
        mean_rms, mean_minor, mean_sc = 0.2, 0.0, 2000.0

    # Video slice
    v_idx = np.where((video_res.time_points >= start_t) & (video_res.time_points <= end_t))[0]
    if len(v_idx) > 0:
        mean_comp = float(np.mean(video_res.composition[v_idx]))
        mean_color = float(np.mean(video_res.color_grading[v_idx]))
        mean_stab = float(np.mean(video_res.motion_stability[v_idx]))
        mean_slow = float(np.mean(video_res.slow_motion[v_idx]))
        mean_bright = float(np.mean(video_res.low_brightness[v_idx]))
    else:
        mean_comp, mean_color, mean_stab, mean_slow, mean_bright = 0.5, 0.5, 0.5, 0.0, 0.5

    # Shot pacing variance
    if len(shot_durations) > 1:
        shot_cv = float(np.std(shot_durations) / (np.mean(shot_durations) + 1e-5))
        var_score = float(np.clip(shot_cv / 1.0, 0.0, 1.0))
    else:
        var_score = 0.65  # Continuous long take baseline

    low_rms_score = float(1.0 - np.clip(mean_rms / 0.35, 0.0, 1.0))
    comp_norm = float(np.clip(mean_comp / 0.35, 0.0, 1.0))

    # 1. Sadness score
    audio_sad = float(0.45 * low_rms_score + 0.40 * mean_minor + 0.15 * (1.0 - np.clip(mean_sc / 3500.0, 0.0, 1.0)))
    sad_score = float(0.40 * audio_sad + 0.30 * mean_slow + 0.30 * mean_bright)

    # 2. Cinematic score
    cinematic_score = float(0.25 * var_score + 0.25 * comp_norm + 0.25 * mean_color + 0.25 * mean_stab)

    # 3. Powerful dialogue score
    dialogue_score = float(0.50 * mean_comp + 0.30 * mean_stab + 0.20 * (1.0 - mean_slow))

    # 4. Romantic score
    romantic_score = float(0.40 * mean_color + 0.30 * mean_stab + 0.30 * (1.0 - mean_bright * 0.5))

    # 5. Action score
    action_score = float(0.40 * (1.0 - mean_stab) + 0.35 * (1.0 - low_rms_score) + 0.25 * var_score)

    # 6. Suspense score
    suspense_score = float(0.40 * mean_bright + 0.35 * mean_minor + 0.25 * low_rms_score)

    return {
        "sad": sad_score,
        "cinematic": cinematic_score,
        "powerful_dialogue": dialogue_score,
        "romantic": romantic_score,
        "action": action_score,
        "suspense": suspense_score,
    }
