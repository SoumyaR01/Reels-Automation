"""Scene database manager with persistent indexing, filtering, and artifact derivation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from .schema import SceneRecord, DialogueLine, AudioFeatures
from ..core.schemas import ShotData, SceneData
from ..audio.features import AudioAnalysisResult
from ..video.aesthetics import VideoAestheticsResult
from ..audio.asr import TranscriptSegment
from ..scoring.fusion import normalize_movie_scores


class SceneDatabase:
    """Manages scene records and metadata index for hybrid RAG retrieval."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else None
        self.scenes: Dict[str, SceneRecord] = {}
        self.movie_metadata: Dict[str, Any] = {}
        if self.db_path and self.db_path.is_file():
            self.load()

    def load(self, path: Optional[Path] = None):
        """Load scene records from JSON database file."""
        target_path = Path(path) if path else self.db_path
        if not target_path or not target_path.is_file():
            return
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.movie_metadata = data.get("movie_metadata", {})
        self.scenes = {
            s["scene_id"]: SceneRecord(**s)
            for s in data.get("scenes", [])
        }

    def save(self, path: Optional[Path] = None):
        """Save scene records to JSON database file."""
        target_path = Path(path) if path else self.db_path
        if not target_path:
            raise ValueError("No database path specified to save.")
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "movie_metadata": self.movie_metadata,
            "scene_count": len(self.scenes),
            "scenes": [s.model_dump() for s in self.scenes.values()],
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def add_scene(self, scene: SceneRecord):
        """Insert or update a scene record."""
        self.scenes[scene.scene_id] = scene

    def get_scene(self, scene_id: str) -> Optional[SceneRecord]:
        """Retrieve scene by its scene_id."""
        return self.scenes.get(scene_id)

    def list_scenes(self) -> List[SceneRecord]:
        """Return all scenes sorted chronologically."""
        return sorted(self.scenes.values(), key=lambda s: s.start_time_seconds)

    def filter_scenes(
        self,
        category: Optional[str] = None,
        min_score: float = 0.5,
        min_duration: float = 3.0,
        max_duration: float = 60.0,
    ) -> List[SceneRecord]:
        """Filter scenes by category score threshold and duration bounds."""
        results = []
        for s in self.scenes.values():
            if not (min_duration <= s.duration_seconds <= max_duration):
                continue
            if category:
                score = s.category_scores.get(category, 0.0)
                if score < min_score:
                    continue
            results.append(s)
        return sorted(results, key=lambda s: s.start_time_seconds)

    def build_from_artifacts(
        self,
        movie_id: str,
        movie_title: str,
        shots: List[ShotData],
        scenes: List[SceneData],
        audio_res: AudioAnalysisResult,
        video_res: VideoAestheticsResult,
        transcripts: List[TranscriptSegment],
        db_path: Optional[Path] = None,
    ) -> "SceneDatabase":
        """Build rich Section 5 SceneRecords from video, audio, and shot clustering artifacts."""
        self.movie_metadata = {
            "movie_id": movie_id,
            "movie_title": movie_title,
            "shots_count": len(shots),
            "scenes_count": len(scenes),
            "duration_seconds": scenes[-1].end if scenes else (shots[-1].end if shots else 0.0),
        }
        self.scenes.clear()

        # Pre-compute heuristic scores for normalization
        raw_scores_list = []
        scene_candidates = []

        for sc in scenes:
            start_t, end_t = sc.start, sc.end
            dur = max(0.1, end_t - start_t)

            # 1. Match dialogue lines
            scene_dialogue = []
            for t in transcripts:
                if (t.start >= start_t and t.start <= end_t) or (t.end >= start_t and t.end <= end_t):
                    scene_dialogue.append(
                        DialogueLine(
                            start_time_seconds=round(max(start_t, t.start), 3),
                            end_time_seconds=round(min(end_t, t.end), 3),
                            speaker="speaker_01",
                            text=t.text.strip(),
                            language="en",
                        )
                    )

            # 2. Audio features slice
            a_idx = np.where((audio_res.time_points >= start_t) & (audio_res.time_points <= end_t))[0]
            if len(a_idx) > 0:
                mean_rms = float(np.mean(audio_res.rms[a_idx]))
                mean_minor = float(np.mean(audio_res.minor_correlation[a_idx]))
                mean_sc = float(np.mean(audio_res.spectral_centroid[a_idx]))
                silence_ratio = float(np.count_nonzero(audio_res.rms[a_idx] < 0.05) / len(a_idx))
                music_pres = float(np.mean(audio_res.music_presence[a_idx]))
            else:
                mean_rms, mean_minor, mean_sc, silence_ratio, music_pres = 0.15, 0.1, 2000.0, 0.2, 0.3

            # 3. Video features slice
            v_idx = np.where((video_res.time_points >= start_t) & (video_res.time_points <= end_t))[0]
            if len(v_idx) > 0:
                mean_comp = float(np.mean(video_res.composition[v_idx]))
                mean_color = float(np.mean(video_res.color_grading[v_idx]))
                mean_stab = float(np.mean(video_res.motion_stability[v_idx]))
                mean_slow = float(np.mean(video_res.slow_motion[v_idx]))
                mean_bright = float(np.mean(video_res.low_brightness[v_idx]))
            else:
                mean_comp, mean_color, mean_stab, mean_slow, mean_bright = 0.5, 0.5, 0.5, 0.0, 0.5

            # Shot durations in this scene
            sc_shot_durs = [s.duration for s in shots if s.start >= start_t and s.end <= end_t]
            shot_cv = float(np.std(sc_shot_durs) / (np.mean(sc_shot_durs) + 1e-5)) if len(sc_shot_durs) > 1 else 0.65
            var_score = float(np.clip(shot_cv / 1.0, 0.0, 1.0))

            # Speech rate
            dialogue_word_count = sum(len(d.text.split()) for d in scene_dialogue)
            wpm = int((dialogue_word_count / max(1.0, dur)) * 60) if dialogue_word_count > 0 else 0

            # Raw multi-signal scores across all 6 categories
            low_rms_score = float(1.0 - np.clip(mean_rms / 0.35, 0.0, 1.0))
            comp_norm = float(np.clip(mean_comp / 0.35, 0.0, 1.0))

            # Category 1: Sad
            audio_sad = float(0.45 * low_rms_score + 0.40 * mean_minor + 0.15 * (1.0 - np.clip(mean_sc / 3500.0, 0.0, 1.0)))
            raw_sad = float(0.40 * audio_sad + 0.30 * mean_slow + 0.30 * mean_bright)

            # Category 2: Action
            raw_action = float(0.40 * (1.0 - mean_stab) + 0.35 * (1.0 - low_rms_score) + 0.25 * var_score)

            # Category 3: Powerful Dialogue
            dialogue_pres_score = 0.8 if len(scene_dialogue) > 0 else 0.2
            raw_dialogue = float(0.40 * dialogue_pres_score + 0.30 * comp_norm + 0.30 * mean_stab)

            # Category 4: Romantic
            raw_romantic = float(0.40 * mean_color + 0.30 * mean_stab + 0.30 * (1.0 - mean_bright * 0.5))

            # Category 5: Motivational
            raw_motivational = float(0.35 * comp_norm + 0.35 * (1.0 - mean_slow) + 0.30 * mean_stab)

            # Category 6: Suspense
            raw_suspense = float(0.40 * mean_bright + 0.35 * mean_minor + 0.25 * low_rms_score)

            scores_dict = {
                "sad": raw_sad,
                "action": raw_action,
                "powerful_dialogue": raw_dialogue,
                "romantic": raw_romantic,
                "motivational": raw_motivational,
                "suspense": raw_suspense,
            }
            raw_scores_list.append(scores_dict)

            # Infer lighting & visual mood
            lighting = "low_key" if mean_bright > 0.6 else ("high_key" if mean_bright < 0.3 else "cinematic_contrast")
            vis_moods = []
            if mean_color > 0.6:
                vis_moods.append("vibrant")
            if mean_bright > 0.5:
                vis_moods.append("somber")
            if mean_stab > 0.7:
                vis_moods.append("steady")
            if not vis_moods:
                vis_moods.append("cinematic")

            # Camera motion & shot types
            cam_motion = ["tracking"] if mean_stab > 0.6 else (["handheld"] if mean_stab < 0.4 else ["static"])
            s_types = ["close_up"] if comp_norm > 0.7 else (["wide"] if comp_norm < 0.3 else ["medium"])

            scene_candidates.append({
                "scene": sc,
                "dur": dur,
                "dialogue": scene_dialogue,
                "mean_rms": mean_rms,
                "mean_minor": mean_minor,
                "silence_ratio": silence_ratio,
                "music_pres": music_pres,
                "wpm": wpm,
                "lighting": lighting,
                "vis_moods": vis_moods,
                "cam_motion": cam_motion,
                "s_types": s_types,
                "mean_comp": mean_comp,
            })

        # Normalize scores per movie using percentile ranking
        categories = ["sad", "action", "powerful_dialogue", "romantic", "motivational", "suspense"]
        norm_scores = normalize_movie_scores(raw_scores_list, categories)

        for cand, scores in zip(scene_candidates, norm_scores):
            sc = cand["scene"]
            dur = cand["dur"]
            dialogue = cand["dialogue"]

            # Derive dominant emotions and themes
            dominant_cat = max(scores.items(), key=lambda x: x[1])[0]
            emotions = [dominant_cat.replace("_", " ")]
            themes = [dominant_cat.replace("_", " ")]
            if scores["motivational"] > 0.7:
                themes.extend(["resilience", "perseverance"])
                emotions.append("determination")
            if scores["sad"] > 0.7:
                themes.extend(["loss", "reflection"])
                emotions.append("grief")
            if scores["action"] > 0.7:
                themes.extend(["conflict", "survival"])
                emotions.append("intensity")
            if scores["romantic"] > 0.7:
                themes.extend(["connection", "love"])
                emotions.append("affection")
            if scores["suspense"] > 0.7:
                themes.extend(["mystery", "tension"])
                emotions.append("fear")

            # Actions derived from shot pace and motion
            actions = ["speaking"] if dialogue else (["running", "struggling"] if scores["action"] > 0.7 else ["reflecting", "observing"])

            # Brief summary
            summary = (
                f"Scene {sc.scene_id} at {sc.start:.1f}s-{sc.end:.1f}s ({dur:.1f}s). "
                f"Mood: {dominant_cat.replace('_', ' ').title()}. "
                f"{'Spoken dialogue present. ' if dialogue else 'Visual atmosphere. '}"
            )
            vis_desc = f"{cand['lighting'].replace('_', ' ').title()} visual lighting with {', '.join(cand['cam_motion'])} camera motion."
            audio_desc = f"{'Dynamic audio track' if cand['mean_rms'] > 0.2 else 'Quiet ambient audio'} with {cand['silence_ratio']*100:.0f}% pause ratio."

            record = SceneRecord(
                movie_id=movie_id,
                scene_id=sc.scene_id,
                start_time_seconds=round(sc.start, 3),
                end_time_seconds=round(sc.end, 3),
                duration_seconds=round(dur, 3),
                scene_summary=summary,
                visual_description=vis_desc,
                characters=["character_01"],
                actions=actions,
                emotions=list(set(emotions)),
                themes=list(set(themes)),
                dialogue=dialogue,
                audio_description=audio_desc,
                shot_types=cand["s_types"],
                camera_motion=cand["cam_motion"],
                lighting_mood=cand["lighting"],
                visual_mood=cand["vis_moods"],
                reel_tags=[dominant_cat] + cand["vis_moods"],
                category_scores=scores,
                audio_features=AudioFeatures(
                    speech_rate_wpm=cand["wpm"],
                    pitch_variance="high" if cand["mean_minor"] > 0.4 else "medium",
                    energy_rms=round(cand["mean_rms"], 3),
                    music_intensity="high" if cand["music_pres"] > 0.6 else "low",
                    silence_ratio=round(cand["silence_ratio"], 3),
                ),
                context_requirements="low" if cand["mean_comp"] > 0.5 or dialogue else "medium",
                reel_suitability={
                    "15_seconds": round(float(np.clip(1.0 - abs(dur - 15.0) / 30.0, 0.4, 0.98)), 2),
                    "30_seconds": round(float(np.clip(1.0 - abs(dur - 30.0) / 45.0, 0.5, 0.99)), 2),
                },
                content_warnings=[],
            )
            self.scenes[sc.scene_id] = record

        if db_path or self.db_path:
            self.save(db_path or self.db_path)

        return self
