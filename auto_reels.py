"""Auto Reels: Best Sad & Cinematic 30-Second Compilation Reels.

Extracts the best sad and cinematic scenes from a movie and compiles them into
continuous 30-second vertical 9:16 reels:
1. Detects complete scene boundaries using hard cuts from start to end of movie.
2. Scores each scene for:
   - Sadness: audio low RMS, minor-key spectral features via librosa, low brightness, slow motion.
   - Cinematic Quality: rule-of-thirds composition, teal/orange & desaturated color grading, motion stability.
3. Collects all scenes passing the sadness threshold and cinematic threshold.
4. Compiles qualifying sad scenes into a single continuous 30-second reel (from start to end).
5. Compiles qualifying cinematic scenes into a separate single 30-second reel.
6. Concatenates with FFmpeg, trims to exactly 30 seconds, and exports vertical 9:16 reels:
   - sad_reel.mp4
   - cinematic_reel.mp4
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import cv2
import librosa
import numpy as np

# Suppress minor audio tuning warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

SAMPLE_WIDTH = 320
SAMPLE_HEIGHT = 180

# Krumhansl-Kessler cognitive key profiles for tonality and key detection
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float32,
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float32,
)

# Standardize profiles to zero-mean, unit-norm for fast correlation
_major_norm = MAJOR_PROFILE - np.mean(MAJOR_PROFILE)
_major_norm = _major_norm / (np.linalg.norm(_major_norm) + 1e-8)
_minor_norm = MINOR_PROFILE - np.mean(MINOR_PROFILE)
_minor_norm = _minor_norm / (np.linalg.norm(_minor_norm) + 1e-8)

MAJOR_TEMPLATES = np.stack([np.roll(_major_norm, shift) for shift in range(12)], axis=0)
MINOR_TEMPLATES = np.stack([np.roll(_minor_norm, shift) for shift in range(12)], axis=0)


def require_tool(name: str):
    """Verify that an external CLI executable exists on PATH."""
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable '{name}' was not found on PATH.")


def probe_video(input_path: Path):
    """Probe video format, duration, dimensions, and audio stream existence."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    video_streams = [
        s for s in metadata.get("streams", []) if s.get("codec_type") == "video"
    ]
    if not video_streams:
        raise ValueError(f"No video stream found in {input_path}")

    duration = float(metadata.get("format", {}).get("duration", 0))
    if duration <= 0:
        duration = float(video_streams[0].get("duration", 0))
    if duration <= 0:
        raise ValueError(f"Could not determine the duration of {input_path}")

    has_audio = any(
        s.get("codec_type") == "audio" for s in metadata.get("streams", [])
    )
    width = int(video_streams[0].get("width", 1920))
    height = int(video_streams[0].get("height", 1080))

    return duration, has_audio, width, height


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


def extract_audio_features(input_path: Path, duration: float, sample_rate: int = 16000):
    """Extract audio via FFmpeg pipe and compute per-second RMS, minor-key correlation, and spectral centroid."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise RuntimeError(f"Could not start ffmpeg for audio: {exc}") from exc

    raw_audio, _ = proc.communicate()
    sec_count = int(np.ceil(duration))

    if proc.returncode != 0 or len(raw_audio) < 4:
        return {
            "rms": np.zeros(sec_count, dtype=np.float32),
            "minor_score": np.zeros(sec_count, dtype=np.float32),
            "spectral_centroid": np.zeros(sec_count, dtype=np.float32),
        }

    y = np.frombuffer(raw_audio, dtype=np.float32)

    hop_length = sample_rate
    n_fft = 2048

    rms_feat = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    sc_feat = librosa.feature.spectral_centroid(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    chroma_feat = librosa.feature.chroma_stft(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, tuning=0.0)

    # Pad or truncate to match sec_count
    if len(rms_feat) < sec_count:
        pad_len = sec_count - len(rms_feat)
        rms_feat = np.pad(rms_feat, (0, pad_len), mode="edge")
        sc_feat = np.pad(sc_feat, (0, pad_len), mode="edge")
        chroma_feat = np.pad(chroma_feat, ((0, 0), (0, pad_len)), mode="edge")
    else:
        rms_feat = rms_feat[:sec_count]
        sc_feat = sc_feat[:sec_count]
        chroma_feat = chroma_feat[:, :sec_count]

    # Compute minor vs major key correlation for each second
    minor_scores = np.zeros(sec_count, dtype=np.float32)
    for t in range(sec_count):
        c = chroma_feat[:, t]
        c_mean = c - np.mean(c)
        c_norm = np.linalg.norm(c_mean)
        if c_norm > 1e-6:
            c_std = c_mean / c_norm
            corr_major = np.max(np.dot(MAJOR_TEMPLATES, c_std))
            corr_minor = np.max(np.dot(MINOR_TEMPLATES, c_std))
            diff = (corr_minor - corr_major + 0.3) / 0.6
            minor_scores[t] = float(np.clip(diff, 0.0, 1.0) * max(0.0, corr_minor))
        else:
            minor_scores[t] = 0.0

    return {
        "rms": rms_feat.astype(np.float32),
        "minor_score": minor_scores,
        "spectral_centroid": sc_feat.astype(np.float32),
    }


def analyze_video_stream(input_path: Path, duration: float, sample_fps: float = 2.0):
    """Stream downscaled video via FFmpeg pipe, detect hard cuts, and extract visual features."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-vf",
        f"fps={sample_fps},scale={SAMPLE_WIDTH}:{SAMPLE_HEIGHT}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise RuntimeError(f"Could not start ffmpeg for video streaming: {exc}") from exc

    frame_bytes = SAMPLE_WIDTH * SAMPLE_HEIGHT * 3
    thirds_mask = create_rule_of_thirds_mask(SAMPLE_WIDTH, SAMPLE_HEIGHT)

    timestamps = []
    frame_diffs = []
    composition_scores = []
    color_grading_scores = []
    motion_magnitudes = []
    motion_jerks = []
    low_brightness_scores = []
    frozen_frame_flags = []

    prev_gray = None
    prev_hsv_hist = None
    prev_disp = np.array([0.0, 0.0], dtype=np.float32)

    frame_idx = 0
    log_interval = max(1, int(sample_fps * 15))

    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((SAMPLE_HEIGHT, SAMPLE_WIDTH, 3))
        timestamp = frame_idx / sample_fps
        timestamps.append(timestamp)

        if frame_idx % log_interval == 0 and duration > 0:
            pct = min(100.0, (timestamp / duration) * 100.0)
            print(f"\r  Analyzing frames: {timestamp:6.1f}s / {duration:.1f}s ({pct:5.1f}%)...", end="", flush=True)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # 1. Composition: Rule-of-Thirds edges
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        edge_sum = np.sum(edges)
        if edge_sum > 1e-4:
            comp_score = float(np.sum(edges * thirds_mask) / (edge_sum + 1e-4))
        else:
            comp_score = 0.0
        composition_scores.append(comp_score)

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
        color_grading_scores.append(color_score)

        # 3. Brightness & Deep Shadow for Sadness
        mean_v = np.mean(v_chan)
        shadow_ratio = np.count_nonzero(v_chan < 45) / float(v_chan.size)
        darkness = 1.0 - np.clip(mean_v / 220.0, 0.0, 1.0)
        low_bright = float(0.65 * darkness + 0.35 * shadow_ratio)
        low_brightness_scores.append(low_bright)

        # 4. Motion, Stability & Hard Cut Detection
        hsv_hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hsv_hist, hsv_hist, alpha=1.0, norm_type=cv2.NORM_L1)

        if prev_gray is not None:
            hist_corr = cv2.compareHist(prev_hsv_hist, hsv_hist, cv2.HISTCMP_CORREL)
            hist_dist = max(0.0, min(1.0, 1.0 - hist_corr))
            pix_diff = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
            cut_metric = 0.6 * hist_dist + 0.4 * pix_diff
            frame_diffs.append(cut_metric)

            # Ignore inter-shot cut transitions for motion & stability metrics
            if cut_metric < 0.35:
                disp, response = cv2.phaseCorrelate(prev_gray.astype(np.float32), gray.astype(np.float32))
                if response > 0.15:
                    disp_vec = np.clip(np.array(disp, dtype=np.float32), -25.0, 25.0)
                else:
                    disp_vec = np.array([0.0, 0.0], dtype=np.float32)

                mag = float(np.linalg.norm(disp_vec))
                jerk = float(np.linalg.norm(disp_vec - prev_disp))
                prev_disp = disp_vec
                is_frozen = 1.0 if (pix_diff < 0.035 and mag < 2.0) else 0.0
            else:
                mag = 0.0
                jerk = 0.0
                prev_disp = np.array([0.0, 0.0], dtype=np.float32)
                is_frozen = 0.0

            motion_magnitudes.append(mag)
            motion_jerks.append(jerk)
            frozen_frame_flags.append(is_frozen)
        else:
            frame_diffs.append(0.0)
            motion_magnitudes.append(0.0)
            motion_jerks.append(0.0)
            frozen_frame_flags.append(0.0)

        prev_gray = gray
        prev_hsv_hist = hsv_hist
        frame_idx += 1

    proc.stdout.close()
    proc.wait()

    timestamps = np.asarray(timestamps, dtype=np.float32)
    frame_diffs = np.asarray(frame_diffs, dtype=np.float32)

    total_time = timestamps[-1] if len(timestamps) else duration
    print(f"\r  Analyzing frames: {total_time:.1f}s / {duration:.1f}s (100.0%)... Done!     ")

    # Detect hard cuts from start to end
    cuts = [0.0]
    min_shot_frames = int(sample_fps * 1.5)  # minimum 1.5 seconds per shot
    last_cut_idx = 0

    cut_threshold = 0.35
    for i in range(1, len(frame_diffs) - 1):
        if (
            frame_diffs[i] > cut_threshold
            and frame_diffs[i] >= frame_diffs[i - 1]
            and frame_diffs[i] >= frame_diffs[i + 1]
            and (i - last_cut_idx) >= min_shot_frames
        ):
            cuts.append(float(timestamps[i]))
            last_cut_idx = i

    video_duration = float(timestamps[-1]) if len(timestamps) else duration
    if video_duration - cuts[-1] > 2.0:
        cuts.append(video_duration)

    return {
        "sample_fps": sample_fps,
        "duration": video_duration,
        "timestamps": timestamps,
        "cuts": np.asarray(cuts, dtype=np.float32),
        "composition": np.asarray(composition_scores, dtype=np.float32),
        "color_grading": np.asarray(color_grading_scores, dtype=np.float32),
        "motion_magnitudes": np.asarray(motion_magnitudes, dtype=np.float32),
        "motion_jerks": np.asarray(motion_jerks, dtype=np.float32),
        "low_brightness": np.asarray(low_brightness_scores, dtype=np.float32),
        "frozen_flags": np.asarray(frozen_frame_flags, dtype=np.float32),
    }


def find_scene_boundaries(cuts: np.ndarray, min_duration: float = 2.0, max_duration: float = 10.0):
    """Detect complete scene boundaries using hard cuts from start to end."""
    scenes = []
    num_cuts = len(cuts)

    # 1. Single-shot scenes between consecutive cuts
    for i in range(num_cuts - 1):
        start_t = cuts[i]
        end_t = cuts[i + 1]
        duration = end_t - start_t
        if min_duration <= duration <= max_duration:
            scenes.append((float(start_t), float(end_t)))

    # 2. Multi-shot complete scenes between hard cuts (up to 3 consecutive shots within max_duration)
    for i in range(num_cuts - 2):
        start_t = cuts[i]
        for j in range(i + 2, min(num_cuts, i + 4)):
            end_t = cuts[j]
            duration = end_t - start_t
            if min_duration <= duration <= max_duration:
                scenes.append((float(start_t), float(end_t)))

    return scenes


def score_scene(scene, video_feats, audio_feats):
    """Score a scene for sadness and cinematic quality."""
    start_t, end_t = scene
    duration = end_t - start_t
    sample_fps = video_feats["sample_fps"]

    start_idx = max(0, int(np.floor(start_t * sample_fps)))
    end_idx = min(len(video_feats["timestamps"]), int(np.ceil(end_t * sample_fps)))

    if end_idx <= start_idx:
        return 0.0, 0.0, {}

    # --- 1. CINEMATIC QUALITY SCORING ---
    # 1a. Rule-of-thirds visual composition
    comp_slice = video_feats["composition"][start_idx:end_idx]
    comp_mean = float(np.mean(comp_slice)) if len(comp_slice) else 0.0
    comp_score = float(np.clip(comp_mean / 0.35, 0.0, 1.0))

    # 1b. Color grading (teal/orange & desaturation)
    color_slice = video_feats["color_grading"][start_idx:end_idx]
    color_score = float(np.mean(color_slice)) if len(color_slice) else 0.0

    # 1c. Motion stability (smoothness / low jerk)
    jerk_slice = video_feats["motion_jerks"][start_idx:end_idx]
    mean_jerk = float(np.mean(jerk_slice)) if len(jerk_slice) else 0.0
    stability_score = float(np.clip(1.0 - (mean_jerk / 6.0), 0.0, 1.0))

    cinematic_score = float(
        0.35 * comp_score
        + 0.35 * color_score
        + 0.30 * stability_score
    )

    # --- 2. SADNESS SCORING ---
    # 2a. Audio analysis: low RMS, minor-key spectral features, low centroid
    a_start = max(0, int(np.floor(start_t)))
    a_end = min(len(audio_feats["rms"]), int(np.ceil(end_t)))
    if a_end > a_start:
        rms_slice = audio_feats["rms"][a_start:a_end]
        minor_slice = audio_feats["minor_score"][a_start:a_end]
        sc_slice = audio_feats["spectral_centroid"][a_start:a_end]

        mean_rms = float(np.mean(rms_slice))
        ref_rms = float(np.percentile(audio_feats["rms"], 90)) if len(audio_feats["rms"]) else 0.1
        low_rms_score = float(1.0 - np.clip(mean_rms / (ref_rms + 1e-5), 0.0, 1.0))

        minor_key_score = float(np.mean(minor_slice))
        mean_sc = float(np.mean(sc_slice))
        low_sc_score = float(1.0 - np.clip(mean_sc / 3500.0, 0.0, 1.0))

        audio_sad_score = float(
            0.45 * low_rms_score + 0.40 * minor_key_score + 0.15 * low_sc_score
        )
    else:
        audio_sad_score = 0.5
        low_rms_score = 0.5
        minor_key_score = 0.0

    # 2b. Slow motion & frozen frames
    frozen_slice = video_feats["frozen_flags"][start_idx:end_idx]
    motion_slice = video_feats["motion_magnitudes"][start_idx:end_idx]
    frozen_ratio = float(np.mean(frozen_slice)) if len(frozen_slice) else 0.0
    mean_motion = float(np.mean(motion_slice)) if len(motion_slice) else 0.0
    slow_score = float(np.clip(1.0 - (mean_motion / 5.0), 0.0, 1.0))
    slow_frozen_score = float(0.5 * frozen_ratio + 0.5 * slow_score)

    # 2c. Low brightness & deep shadows
    bright_slice = video_feats["low_brightness"][start_idx:end_idx]
    low_bright_score = float(np.mean(bright_slice)) if len(bright_slice) else 0.0

    sadness_score = float(
        0.40 * audio_sad_score
        + 0.30 * slow_frozen_score
        + 0.30 * low_bright_score
    )

    details = {
        "start": start_t,
        "end": end_t,
        "duration": duration,
        "cinematic": {
            "total": cinematic_score,
            "composition": comp_score,
            "color_grading": color_score,
            "stability": stability_score,
        },
        "sadness": {
            "total": sadness_score,
            "audio": audio_sad_score,
            "low_rms": low_rms_score,
            "minor_key": minor_key_score,
            "slow_frozen": slow_frozen_score,
            "low_brightness": low_bright_score,
        },
    }

    return cinematic_score, sadness_score, details


def temporal_overlap(seg_a, seg_b):
    """Check if two time intervals overlap significantly."""
    start_a, end_a = seg_a[0], seg_a[1]
    start_b, end_b = seg_b[0], seg_b[1]
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    return intersection > 0.5


def collect_qualifying_scenes(
    scored_scenes, score_key, threshold, target_duration=30.0, max_overlap_iou=0.15
):
    """Collect qualifying scenes passing threshold, select the best non-overlapping clips,

    and order them chronologically from start to end to reach at least target_duration.
    """
    # 1. Filter all scenes passing the threshold
    qualifying = [s for s in scored_scenes if s[score_key] >= threshold]

    # If qualifying scenes don't have enough duration, supplement with top-ranked available scenes
    qualifying_duration = sum(s["duration"] for s in qualifying)
    if qualifying_duration < target_duration:
        pool = sorted(scored_scenes, key=lambda s: s[score_key], reverse=True)
    else:
        pool = sorted(qualifying, key=lambda s: s[score_key], reverse=True)

    # 2. Greedily select top non-overlapping scenes until cumulative duration >= target_duration
    selected = []
    accumulated_duration = 0.0

    for item in pool:
        seg = (item["start"], item["end"])
        if not any(temporal_overlap(seg, (sel["start"], sel["end"])) for sel in selected):
            selected.append(item)
            accumulated_duration += item["duration"]
            if accumulated_duration >= target_duration:
                break

    # 3. Sort chosen scenes chronologically from start to end
    selected.sort(key=lambda s: s["start"])
    return selected


def export_reel(
    input_path: Path,
    selected_scenes: list,
    output_path: Path,
    target_duration: float = 30.0,
    width: int = 1080,
    height: int = 1920,
    crf: int = 20,
    preset: str = "medium",
):
    """Compile scenes into intermediate 9:16 clips, concatenate them, and trim to exactly target_duration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output_path.parent) as temp_dir:
        temp_dir_path = Path(temp_dir)
        clip_paths = []

        # 1. Export each individual scene formatted as vertical 9:16
        for idx, sc in enumerate(selected_scenes, start=1):
            clip_file = temp_dir_path / f"part_{idx:03d}.mp4"
            filter_graph = (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},setsar=1,fps=30"
            )
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{sc['start']:.3f}",
                "-i",
                str(input_path),
                "-t",
                f"{sc['duration']:.3f}",
                "-vf",
                filter_graph,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                str(crf),
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                str(clip_file),
            ]
            subprocess.run(cmd, check=True)
            clip_paths.append(clip_file)

        # 2. Write concat playlist
        concat_file = temp_dir_path / "concat_list.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for p in clip_paths:
                f.write(f"file '{p.resolve().as_posix()}'\n")

        # 3. Concatenate and trim final output to exactly target_duration seconds
        concat_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-t",
            f"{target_duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(concat_cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract the best sad clips and cinematic clips and compile them into 30-second reels."
    )
    parser.add_argument("input", type=Path, help="Input movie file")
    parser.add_argument("--output-dir", type=Path, default=Path("reels"), help="Output directory")
    parser.add_argument(
        "--reel-duration",
        type=float,
        default=30.0,
        help="Target length of each compiled reel in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--min-clip-duration",
        type=float,
        default=2.5,
        help="Minimum scene duration in seconds (default: 2.5)",
    )
    parser.add_argument(
        "--max-clip-duration",
        type=float,
        default=9.0,
        help="Maximum scene duration in seconds (default: 9.0)",
    )
    parser.add_argument(
        "--min-sadness",
        type=float,
        default=0.40,
        help="Minimum threshold score for sad scenes (default: 0.40)",
    )
    parser.add_argument(
        "--min-cinematic",
        type=float,
        default=0.40,
        help="Minimum threshold score for cinematic scenes (default: 0.40)",
    )
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Video analysis sampling rate (default: 2.0)")
    parser.add_argument("--width", type=int, default=1080, help="Output reel width (default: 1080)")
    parser.add_argument("--height", type=int, default=1920, help="Output reel height (default: 1920)")
    parser.add_argument("--crf", type=int, default=20, help="H.264 quality factor (default: 20)")
    parser.add_argument("--preset", default="medium", help="FFmpeg encoding preset (default: medium)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input video file not found: {args.input}")

    require_tool("ffmpeg")
    require_tool("ffprobe")

    duration, has_audio, orig_w, orig_h = probe_video(args.input)
    print(f"Movie: {args.input.name}")
    print(f"Duration: {duration:.1f}s (~{duration/60:.1f} mins) | Resolution: {orig_w}x{orig_h} | Audio: {has_audio}")

    # Pass 1: Audio Analysis via librosa
    if has_audio:
        print("\n[1/3] Extracting and analyzing audio features via librosa (RMS, minor-key, spectral)...")
        audio_feats = extract_audio_features(args.input, duration)
    else:
        print("\n[1/3] No audio track detected; skipping audio analysis.")
        audio_feats = {
            "rms": np.zeros(int(np.ceil(duration)), dtype=np.float32),
            "minor_score": np.zeros(int(np.ceil(duration)), dtype=np.float32),
            "spectral_centroid": np.zeros(int(np.ceil(duration)), dtype=np.float32),
        }

    # Pass 2: Video Analysis and Hard Cut Detection
    print(f"\n[2/3] Analyzing visual features & detecting hard cuts at {args.sample_fps} fps...")
    video_feats = analyze_video_stream(args.input, duration, sample_fps=args.sample_fps)
    cuts = video_feats["cuts"]
    print(f"Detected {len(cuts)} hard cuts from start to end.")

    # Pass 3: Candidate Scene Formation and Scoring
    print(f"\n[3/3] Detecting complete scenes ({args.min_clip_duration}s - {args.max_clip_duration}s) and scoring...")
    candidate_scenes = find_scene_boundaries(cuts, args.min_clip_duration, args.max_clip_duration)
    print(f"Found {len(candidate_scenes)} candidate complete scenes.")

    if not candidate_scenes:
        raise RuntimeError(
            f"No complete scenes found between {args.min_clip_duration}s and {args.max_clip_duration}s between cuts."
        )

    scored_scenes = []
    for scene in candidate_scenes:
        cinematic_score, sadness_score, details = score_scene(scene, video_feats, audio_feats)
        scored_scenes.append(
            {
                "start": details["start"],
                "end": details["end"],
                "duration": details["duration"],
                "cinematic": cinematic_score,
                "sadness": sadness_score,
                "details": details,
            }
        )

    # Collect qualifying scenes and compile into 30-second sequences (from start to end)
    sad_selected = collect_qualifying_scenes(
        scored_scenes, "sadness", args.min_sadness, target_duration=args.reel_duration
    )
    cinematic_selected = collect_qualifying_scenes(
        scored_scenes, "cinematic", args.min_cinematic, target_duration=args.reel_duration
    )

    print("\n" + "=" * 65)
    print("SELECTED SCENES FOR REEL COMPILATION (CHRONOLOGICAL ORDER)")
    print("=" * 65)
    print(f"\n[Sad Reel Compilation] ({len(sad_selected)} scenes, total duration: {sum(s['duration'] for s in sad_selected):.1f}s):")
    for idx, sc in enumerate(sad_selected, start=1):
        d = sc["details"]["sadness"]
        print(
            f"  Clip #{idx:02d} [{sc['start']:.1f}s - {sc['end']:.1f}s] ({sc['duration']:.1f}s) -> "
            f"Sadness: {sc['sadness']:.3f} | Audio: {d['audio']:.2f} (Minor: {d['minor_key']:.2f}, LowRMS: {d['low_rms']:.2f}), "
            f"Slow/Frozen: {d['slow_frozen']:.2f}, LowBright: {d['low_brightness']:.2f}"
        )

    print(f"\n[Cinematic Reel Compilation] ({len(cinematic_selected)} scenes, total duration: {sum(s['duration'] for s in cinematic_selected):.1f}s):")
    for idx, sc in enumerate(cinematic_selected, start=1):
        d = sc["details"]["cinematic"]
        print(
            f"  Clip #{idx:02d} [{sc['start']:.1f}s - {sc['end']:.1f}s] ({sc['duration']:.1f}s) -> "
            f"Cinematic: {sc['cinematic']:.3f} | Comp: {d['composition']:.2f}, "
            f"Color: {d['color_grading']:.2f}, Stab: {d['stability']:.2f}"
        )
    print("=" * 65)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sad_reel_path = args.output_dir / "sad_reel.mp4"
    cinematic_reel_path = args.output_dir / "cinematic_reel.mp4"

    # Export sad reel
    print(f"\nCompiling and exporting sad reel (trimmed to exactly {args.reel_duration:.1f}s) -> {sad_reel_path.name}...")
    export_reel(
        args.input,
        sad_selected,
        sad_reel_path,
        target_duration=args.reel_duration,
        width=args.width,
        height=args.height,
        crf=args.crf,
        preset=args.preset,
    )
    print(f"  Successfully exported: {sad_reel_path.resolve()}")

    # Export cinematic reel
    print(f"\nCompiling and exporting cinematic reel (trimmed to exactly {args.reel_duration:.1f}s) -> {cinematic_reel_path.name}...")
    export_reel(
        args.input,
        cinematic_selected,
        cinematic_reel_path,
        target_duration=args.reel_duration,
        width=args.width,
        height=args.height,
        crf=args.crf,
        preset=args.preset,
    )
    print(f"  Successfully exported: {cinematic_reel_path.resolve()}")

    print("\nProcessing complete! Both 30-second reels successfully generated.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
