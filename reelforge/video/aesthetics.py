"""Visual aesthetics analysis: rule-of-thirds, teal/orange grading, brightness, and burst stability."""

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass
class VideoAestheticsResult:
    time_points: np.ndarray  # sampled at analysis fps
    composition: np.ndarray
    color_grading: np.ndarray
    motion_stability: np.ndarray
    slow_motion: np.ndarray
    low_brightness: np.ndarray


def create_rule_of_thirds_mask(width: int, height: int) -> np.ndarray:
    """Precompute a 2D weighting mask prioritizing rule-of-thirds lines and power-points."""
    x1, x2 = width / 3.0, 2.0 * width / 3.0
    y1, y2 = height / 3.0, 2.0 * height / 3.0

    sigma_x = width / 18.0
    sigma_y = height / 18.0

    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)

    dist_x = np.minimum(np.abs(xs - x1), np.abs(xs - x2))
    dist_y = np.minimum(np.abs(ys - y1), np.abs(ys - y2))

    wx = np.exp(-(dist_x**2) / (2.0 * sigma_x**2))
    wy = np.exp(-(dist_y**2) / (2.0 * sigma_y**2))

    grid_x, grid_y = np.meshgrid(wx, wy)
    lines_weight = 0.5 * (grid_x + grid_y)
    points_weight = grid_x * grid_y

    mask = 0.5 * lines_weight + 0.5 * points_weight
    return mask.astype(np.float32)


def extract_video_aesthetics(proxy_path: Path, fps: float = 12.0, sample_stride: int = 4) -> VideoAestheticsResult:
    """Stream proxy video to extract dense aesthetics signals with burst-level motion stability (Fixing Bug #4).
    
    sample_stride: evaluate every Nth frame (default: 4, giving 3 fps sample density for fast, accurate coverage).
    """
    proxy_path = Path(proxy_path)
    cap = cv2.VideoCapture(str(proxy_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open proxy video for aesthetics: {proxy_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    thirds_mask = create_rule_of_thirds_mask(width, height)

    times = []
    comp_list = []
    color_list = []
    stab_list = []
    slow_list = []
    bright_list = []

    prev_gray = None
    prev_disp = np.array([0.0, 0.0], dtype=np.float32)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_stride != 0:
            frame_idx += 1
            continue

        t = float(frame_idx / fps)
        times.append(t)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # 1. Rule-of-thirds composition
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        edge_sum = np.sum(edges)
        if edge_sum > 1e-4:
            comp_score = float(np.sum(edges * thirds_mask) / (edge_sum + 1e-4))
        else:
            comp_score = 0.0
        comp_list.append(comp_score)

        # 2. Color grading: Teal/Orange and desaturated tones
        chromatic = s_chan > 35
        chromatic_count = np.count_nonzero(chromatic)
        if chromatic_count > 100:
            orange_mask = chromatic & (h_chan >= 5) & (h_chan <= 25)
            teal_mask = chromatic & (h_chan >= 85) & (h_chan <= 115)
            to_fraction = (np.count_nonzero(orange_mask) + np.count_nonzero(teal_mask)) / chromatic_count

            highlights = v_chan > 128
            shadows = v_chan <= 128
            warm_hl = np.count_nonzero(orange_mask & highlights) / (np.count_nonzero(chromatic & highlights) + 1e-5)
            cool_sh = np.count_nonzero(teal_mask & shadows) / (np.count_nonzero(chromatic & shadows) + 1e-5)
            to_score = 0.65 * to_fraction + 0.35 * (0.5 * warm_hl + 0.5 * cool_sh)
        else:
            to_score = 0.2

        mean_sat = np.mean(s_chan)
        desat_score = np.exp(-((mean_sat - 65.0) ** 2) / (2.0 * (35.0**2)))
        color_score = float(0.6 * to_score + 0.4 * desat_score)
        color_list.append(color_score)

        # 3. Brightness & Deep Shadow
        mean_v = np.mean(v_chan)
        shadow_ratio = np.count_nonzero(v_chan < 45) / float(v_chan.size)
        darkness = 1.0 - np.clip(mean_v / 220.0, 0.0, 1.0)
        low_bright = float(0.65 * darkness + 0.35 * shadow_ratio)
        bright_list.append(low_bright)

        # 4. Motion Stability & Slow Motion (sampled at >=12 fps for real physical camera motion)
        if prev_gray is not None:
            pix_diff = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
            if pix_diff < 0.35:  # within shot
                disp, response = cv2.phaseCorrelate(prev_gray.astype(np.float32), gray.astype(np.float32))
                if response > 0.15:
                    disp_vec = np.clip(np.array(disp, dtype=np.float32), -25.0, 25.0)
                else:
                    disp_vec = np.array([0.0, 0.0], dtype=np.float32)

                mag = float(np.linalg.norm(disp_vec))
                jerk = float(np.linalg.norm(disp_vec - prev_disp))
                prev_disp = disp_vec
                stab_score = float(np.clip(1.0 - (jerk / 5.0), 0.0, 1.0))
                slow_score = float(np.clip(1.0 - (mag / 3.0), 0.0, 1.0))
            else:
                stab_score = 0.5
                slow_score = 0.0
                prev_disp = np.array([0.0, 0.0], dtype=np.float32)
        else:
            stab_score = 0.5
            slow_score = 0.0

        stab_list.append(stab_score)
        slow_list.append(slow_score)

        prev_gray = gray
        frame_idx += 1

    cap.release()

    return VideoAestheticsResult(
        time_points=np.asarray(times, dtype=np.float64),
        composition=np.asarray(comp_list, dtype=np.float32),
        color_grading=np.asarray(color_list, dtype=np.float32),
        motion_stability=np.asarray(stab_list, dtype=np.float32),
        slow_motion=np.asarray(slow_list, dtype=np.float32),
        low_brightness=np.asarray(bright_list, dtype=np.float32),
    )


def save_video_aesthetics(res: VideoAestheticsResult, path: Path) -> None:
    """Save video aesthetics result to compressed .npz archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        time_points=res.time_points,
        composition=res.composition,
        color_grading=res.color_grading,
        motion_stability=res.motion_stability,
        slow_motion=res.slow_motion,
        low_brightness=res.low_brightness,
    )


def load_video_aesthetics(path: Path) -> VideoAestheticsResult:
    """Load cached video aesthetics result from compressed .npz archive."""
    data = np.load(str(path))
    return VideoAestheticsResult(
        time_points=data["time_points"],
        composition=data["composition"],
        color_grading=data["color_grading"],
        motion_stability=data["motion_stability"],
        slow_motion=data["slow_motion"],
        low_brightness=data["low_brightness"],
    )

