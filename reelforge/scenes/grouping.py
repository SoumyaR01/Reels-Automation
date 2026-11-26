"""Hierarchical shot clustering into semantic narrative scenes."""

import json
from pathlib import Path
from typing import List

import cv2
import numpy as np

from ..core.schemas import SceneData, ShotData


def extract_shot_fingerprint(cap: cv2.VideoCapture, shot: ShotData, fps: float) -> np.ndarray:
    """Extract a robust visual color/spatial distribution vector for a shot's middle frames."""
    mid_time = (shot.start + shot.end) / 2.0
    frame_idx = int(mid_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return np.zeros(64, dtype=np.float32)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L2)
    return hist.flatten().astype(np.float32)


def group_shots_into_scenes(
    shots: List[ShotData],
    proxy_path: Path,
    output_json_path: Path,
    fps: float = 12.0,
    similarity_threshold: float = 0.65,
    max_scene_seconds: float = 300.0,
    overwrite: bool = False,
) -> List[SceneData]:
    """Group consecutive shots into narrative scenes based on environment/palette continuity.

    Shots within the same physical environment (e.g. shot-reverse-shot dialogue) are
    coalesced into complete multi-shot narrative scenes.
    """
    output_json_path = Path(output_json_path)
    if output_json_path.is_file() and not overwrite:
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [SceneData(**item) for item in data]
        except Exception:
            pass

    if not shots:
        return []

    cap = cv2.VideoCapture(str(proxy_path))
    fingerprints = [extract_shot_fingerprint(cap, shot, fps) for shot in shots]
    cap.release()

    scenes: List[SceneData] = []
    current_shots = [shots[0]]
    current_start = shots[0].start
    current_sims = []

    for i in range(1, len(shots)):
        prev_shot = shots[i - 1]
        curr_shot = shots[i]

        fp_prev = fingerprints[i - 1]
        fp_curr = fingerprints[i]

        # Cosine similarity between shot visual environments
        sim = float(np.dot(fp_prev, fp_curr))
        current_sims.append(sim)

        # Environment change or exceeded max scene length
        is_scene_boundary = (sim < similarity_threshold) or (
            (curr_shot.end - current_start) > max_scene_seconds
        )

        if is_scene_boundary:
            # Seal current scene
            avg_sim = float(np.mean(current_sims)) if current_sims else 1.0
            scenes.append(
                SceneData(
                    scene_id=f"scene_{len(scenes) + 1:04d}",
                    shot_ids=[s.shot_id for s in current_shots],
                    start=round(current_start, 3),
                    end=round(prev_shot.end, 3),
                    duration=round(prev_shot.end - current_start, 3),
                    visual_similarity=round(avg_sim, 3),
                    dialogue_gap=0.0,
                )
            )
            # Reset
            current_shots = [curr_shot]
            current_start = curr_shot.start
            current_sims = []
        else:
            current_shots.append(curr_shot)

    # Final trailing scene
    if current_shots:
        avg_sim = float(np.mean(current_sims)) if current_sims else 1.0
        scenes.append(
            SceneData(
                scene_id=f"scene_{len(scenes) + 1:04d}",
                shot_ids=[s.shot_id for s in current_shots],
                start=round(current_start, 3),
                end=round(current_shots[-1].end, 3),
                duration=round(current_shots[-1].end - current_start, 3),
                visual_similarity=round(avg_sim, 3),
                dialogue_gap=0.0,
            )
        )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump([sc.model_dump() for sc in scenes], f, indent=2)

    return scenes
