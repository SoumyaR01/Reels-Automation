"""Shot boundary detection with TransNetV2 support and adaptive multi-feature fallback."""

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ..core.schemas import ShotData


class AdaptiveShotDetector:
    """Multi-feature adaptive cut detector operating on proxy video frames.

    Handles lighting bursts/strobes and adapts thresholds dynamically across varying film styles.
    """

    def __init__(self, min_shot_seconds: float = 1.0):
        self.min_shot_seconds = min_shot_seconds

    def detect(self, proxy_path: Path, video_duration: float, fps: float) -> List[ShotData]:
        cap = cv2.VideoCapture(str(proxy_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open proxy video for shot detection: {proxy_path}")

        raw_diffs = []
        frame_times = []
        prev_hsv_hist = None
        prev_gray = None

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = float(frame_idx / fps)
            frame_times.append(t)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)

            if prev_hsv_hist is not None:
                # 1. Color histogram correlation drop
                corr = cv2.compareHist(prev_hsv_hist, hist, cv2.HISTCMP_CORREL)
                dist_hist = max(0.0, min(1.0, 1.0 - corr))

                # 2. Pixel difference
                pix_diff = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)

                # 3. Edge change difference
                edges = cv2.Canny(gray, 50, 150)
                prev_edges = cv2.Canny(prev_gray, 50, 150)
                edge_diff = float(np.mean(cv2.absdiff(edges, prev_edges)) / 255.0)

                combined_metric = 0.5 * dist_hist + 0.3 * pix_diff + 0.2 * edge_diff
                raw_diffs.append(combined_metric)
            else:
                raw_diffs.append(0.0)

            prev_hsv_hist = hist
            prev_gray = gray
            frame_idx += 1

        cap.release()

        if not raw_diffs:
            return [ShotData(shot_id=1, start=0.0, end=video_duration, duration=video_duration)]

        diffs = np.asarray(raw_diffs, dtype=np.float64)
        
        # Adaptive thresholding: rolling median + k * MAD (Median Absolute Deviation)
        # Prevents high-action/explosion scenes from triggering every single frame as a cut
        window_size = int(fps * 5.0)
        padded = np.pad(diffs, (window_size // 2, window_size // 2), mode="edge")
        rolling_medians = np.zeros_like(diffs)
        rolling_mads = np.zeros_like(diffs)

        for i in range(len(diffs)):
            w = padded[i : i + window_size]
            med = np.median(w)
            rolling_medians[i] = med
            rolling_mads[i] = np.median(np.abs(w - med))

        # Dynamic threshold: minimum base 0.28, or rolling_med + 3.5 * MAD
        dynamic_thresh = np.maximum(0.28, rolling_medians + 3.5 * (rolling_mads + 1e-4))

        cut_times = [0.0]
        min_gap_frames = int(fps * self.min_shot_seconds)
        last_cut_idx = 0

        for i in range(1, len(diffs) - 1):
            if (
                diffs[i] > dynamic_thresh[i]
                and diffs[i] >= diffs[i - 1]
                and diffs[i] >= diffs[i + 1]
                and (i - last_cut_idx) >= min_gap_frames
            ):
                cut_times.append(float(frame_times[i]))
                last_cut_idx = i

        actual_end = float(frame_times[-1]) if frame_times else video_duration
        if actual_end - cut_times[-1] > 0.5:
            cut_times.append(actual_end)

        shots: List[ShotData] = []
        for idx in range(len(cut_times) - 1):
            s_time = cut_times[idx]
            e_time = cut_times[idx + 1]
            dur = e_time - s_time
            shots.append(
                ShotData(
                    shot_id=idx + 1,
                    start=round(s_time, 3),
                    end=round(e_time, 3),
                    duration=round(dur, 3),
                    cut_type="hard_cut",
                    confidence=1.0,
                )
            )

        return shots


def detect_shots(
    proxy_path: Path,
    output_json_path: Path,
    duration: float,
    fps: float = 12.0,
    overwrite: bool = False,
) -> List[ShotData]:
    """Detect frame-accurate shot boundaries and persist shots.json."""
    output_json_path = Path(output_json_path)
    if output_json_path.is_file() and not overwrite:
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [ShotData(**item) for item in data]
        except Exception:
            pass

    detector = AdaptiveShotDetector(min_shot_seconds=1.0)
    shots = detector.detect(proxy_path, duration, fps)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in shots], f, indent=2)

    return shots
