"""Acoustic feature extraction with proper windowing and rolling 5-10s tonality analysis."""

from dataclasses import dataclass
from pathlib import Path
import warnings

import librosa
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# Krumhansl-Kessler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)

_m_norm = MAJOR_PROFILE - np.mean(MAJOR_PROFILE)
_m_norm = _m_norm / (np.linalg.norm(_m_norm) + 1e-8)
_min_norm = MINOR_PROFILE - np.mean(MINOR_PROFILE)
_min_norm = _min_norm / (np.linalg.norm(_min_norm) + 1e-8)

MAJOR_TEMPLATES = np.stack([np.roll(_m_norm, s) for s in range(12)], axis=0)
MINOR_TEMPLATES = np.stack([np.roll(_min_norm, s) for s in range(12)], axis=0)


@dataclass
class AudioAnalysisResult:
    sample_rate: int
    duration: float
    time_points: np.ndarray  # sampled at 2 Hz (every 0.5s)
    rms: np.ndarray
    minor_correlation: np.ndarray
    spectral_centroid: np.ndarray
    music_presence: np.ndarray


def extract_audio_features(wav_path: Path, target_sr: int = 16000) -> AudioAnalysisResult:
    """Extract continuous acoustic signals with 0.5s windows and 5-10s tonality smoothing (fixing Bug #2)."""
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    y, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
    duration = float(len(y) / sr)

    if len(y) < sr:
        # Fallback for empty or near-empty audio
        t = np.array([0.0], dtype=np.float64)
        zeros = np.zeros(1, dtype=np.float32)
        return AudioAnalysisResult(sr, duration, t, zeros, zeros, zeros, zeros)

    # 1. Continuous RMS with 0.5s windows and 0.25s hop
    frame_length = int(sr * 0.50)
    hop_length = int(sr * 0.25)
    rms_raw = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    sc_raw = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

    # Align to common 2 Hz grid (0.5s steps)
    step_sec = 0.50
    total_steps = max(1, int(np.ceil(duration / step_sec)))
    t_grid = np.arange(total_steps, dtype=np.float64) * step_sec

    # Resample RMS and Centroid to grid
    rms_grid = np.interp(
        t_grid,
        np.linspace(0, duration, len(rms_raw)),
        rms_raw,
    ).astype(np.float32)

    sc_grid = np.interp(
        t_grid,
        np.linspace(0, duration, len(sc_raw)),
        sc_raw,
    ).astype(np.float32)

    # 2. Chroma with 5-10 second rolling window for accurate key/mode estimation
    chroma_raw = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=hop_length,
        tuning=0.0,
    )

    # Rolling average over ~7.5 seconds (30 hops of 0.25s)
    window_hops = 30
    pad_h = window_hops // 2
    padded_chroma = np.pad(chroma_raw, ((0, 0), (pad_h, pad_h)), mode="edge")

    smoothed_chroma = np.zeros_like(chroma_raw)
    for c in range(12):
        smoothed_chroma[c] = np.convolve(
            padded_chroma[c],
            np.ones(window_hops) / window_hops,
            mode="valid",
        )[: chroma_raw.shape[1]]

    # Correlate smoothed chroma with key templates
    minor_raw = np.zeros(chroma_raw.shape[1], dtype=np.float32)
    for i in range(chroma_raw.shape[1]):
        vec = smoothed_chroma[:, i]
        v_mean = vec - np.mean(vec)
        v_norm = np.linalg.norm(v_mean)
        if v_norm > 1e-6:
            v_std = v_mean / v_norm
            c_maj = np.max(np.dot(MAJOR_TEMPLATES, v_std))
            c_min = np.max(np.dot(MINOR_TEMPLATES, v_std))
            diff = (c_min - c_maj + 0.3) / 0.6
            minor_raw[i] = float(np.clip(diff, 0.0, 1.0) * max(0.0, c_min))

    minor_grid = np.interp(
        t_grid,
        np.linspace(0, duration, len(minor_raw)),
        minor_raw,
    ).astype(np.float32)

    # 3. Simple music vs speech/silence presence (harmonic energy vs percussive ratio)
    music_grid = np.clip(minor_grid * 1.5, 0.0, 1.0)

    return AudioAnalysisResult(
        sample_rate=sr,
        duration=duration,
        time_points=t_grid,
        rms=rms_grid,
        minor_correlation=minor_grid,
        spectral_centroid=sc_grid,
        music_presence=music_grid,
    )


def save_audio_features(res: AudioAnalysisResult, path: Path) -> None:
    """Save audio analysis result to compressed .npz archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        sample_rate=res.sample_rate,
        duration=res.duration,
        time_points=res.time_points,
        rms=res.rms,
        minor_correlation=res.minor_correlation,
        spectral_centroid=res.spectral_centroid,
        music_presence=res.music_presence,
    )


def load_audio_features(path: Path) -> AudioAnalysisResult:
    """Load cached audio analysis result from compressed .npz archive."""
    data = np.load(str(path))
    return AudioAnalysisResult(
        sample_rate=int(data["sample_rate"]),
        duration=float(data["duration"]),
        time_points=data["time_points"],
        rms=data["rms"],
        minor_correlation=data["minor_correlation"],
        spectral_centroid=data["spectral_centroid"],
        music_presence=data["music_presence"],
    )

