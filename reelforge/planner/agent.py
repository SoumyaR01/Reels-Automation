"""Reel Planning Agent: Turns natural language prompts into structured Section 7 Reel Plans."""

import uuid
from typing import Dict, List, Optional
from .schema import ReelPlan, ReelRequest, SelectedClip, TimedCaption, HindiShayari
from ..rag.retrieval import HybridRetriever, RetrievalCandidate
from ..rag.db import SceneDatabase
from ..edit.shayari import ShayariEngine


class ReelPlanningAgent:
    """Orchestrates RAG retrieval, 5-beat narrative timeline planning, and grounding verification."""

    def __init__(self, db: SceneDatabase, retriever: Optional[HybridRetriever] = None):
        self.db = db
        self.retriever = retriever or HybridRetriever(db)

    def plan_reel(
        self,
        prompt: str,
        target_duration: float = 30.0,
        include_shayari: bool = False,
        aspect_ratio: str = "9:16",
    ) -> ReelPlan:
        """Create a grounded, structured Section 7 Reel Plan from natural language user request."""
        # 1. Parse prompt
        params = self.retriever.parse_query(prompt)
        target_dur = target_duration or params.target_duration
        category = params.target_category or "motivational"

        # 2. Retrieve scene candidates using hybrid RAG
        candidates = self.retriever.retrieve(prompt, top_k=8)
        if not candidates:
            # Fallback to general scenes in DB
            scenes = self.db.list_scenes()
            candidates = [
                RetrievalCandidate(
                    scene_id=s.scene_id,
                    start_time_seconds=s.start_time_seconds,
                    end_time_seconds=s.end_time_seconds,
                    duration_seconds=s.duration_seconds,
                    relevance_score=0.7,
                    category=category,
                    category_score=s.category_scores.get(category, 0.5),
                    hook_score=0.8,
                    context_independence=0.8,
                    reason="Default scene fallback",
                    scene_summary=s.scene_summary,
                    dialogue_excerpt="",
                    themes=s.themes,
                    emotions=s.emotions,
                )
                for s in scenes[:4]
            ]

        # 3. Assemble 5-beat narrative timeline
        # Target beats: Hook (0-3s), Setback (3-9s), Turning Point (9-19s), Payoff (19-26s), Closing (26-30s)
        beat_durations = [3.0, 6.0, 10.0, 7.0, 4.0]
        beat_purposes = [
            ("Hook", "Strong opening hook to capture viewer attention immediately"),
            ("Setback", "Establish context, struggle, or emotional premise"),
            ("Turning Point", "Character realization, emotional shift, or decision to act"),
            ("Payoff", "Climactic action, emotional release, or resolution"),
            ("Closing", "Poignant closing visual resolution beat"),
        ]

        if target_dur < 25.0:
            # For 15s reels: Hook (3s), Core (8s), Closing (4s)
            beat_durations = [3.0, 8.0, 4.0]
            beat_purposes = [
                ("Hook", "Immediate visual or dialogue hook"),
                ("Core", "Central dramatic moment or action"),
                ("Closing", "Final visual beat and takeaway quote"),
            ]

        selected_clips: List[SelectedClip] = []
        timed_captions: List[TimedCaption] = []
        current_reel_time = 0.0

        num_beats = len(beat_durations)
        # Select distinct scenes for each beat where possible
        used_scene_ids = set()

        for beat_idx in range(num_beats):
            desired_len = beat_durations[beat_idx]
            purpose_name, purpose_desc = beat_purposes[beat_idx]

            # Find best candidate for this beat that hasn't been overused
            cand = None
            for c in candidates:
                if c.scene_id not in used_scene_ids:
                    cand = c
                    used_scene_ids.add(c.scene_id)
                    break
            if cand is None:
                cand = candidates[beat_idx % len(candidates)]

            # Grounding enforcement: Ensure source range is strictly inside scene boundaries
            scene_record = self.db.get_scene(cand.scene_id)
            if scene_record is None:
                continue

            scene_start = scene_record.start_time_seconds
            scene_end = scene_record.end_time_seconds
            scene_dur = scene_record.duration_seconds

            clip_len = min(desired_len, scene_dur)
            source_start = scene_start
            source_end = min(scene_end, source_start + clip_len)

            clip_reel_start = current_reel_time
            clip_reel_end = round(current_reel_time + (source_end - source_start), 3)

            selected_clips.append(
                SelectedClip(
                    movie_id=scene_record.movie_id,
                    scene_id=scene_record.scene_id,
                    source_start_seconds=round(source_start, 3),
                    source_end_seconds=round(source_end, 3),
                    reel_start_seconds=round(clip_reel_start, 3),
                    reel_end_seconds=round(clip_reel_end, 3),
                    reason=f"{purpose_name}: {purpose_desc}",
                )
            )

            # Map actual dialogue lines from this clip onto reel timeline
            for d in scene_record.dialogue:
                if d.start_time_seconds >= source_start and d.start_time_seconds <= source_end:
                    diag_rel_start = d.start_time_seconds - source_start
                    diag_rel_end = min(source_end - source_start, d.end_time_seconds - source_start)
                    timed_captions.append(
                        TimedCaption(
                            start_seconds=round(clip_reel_start + diag_rel_start, 2),
                            end_seconds=round(clip_reel_start + diag_rel_end, 2),
                            text=d.text,
                            speaker=d.speaker,
                        )
                    )

            current_reel_time = clip_reel_end

        # 4. Generate Hindi Shayari for the closing beat
        shayari_obj = None
        if include_shayari:
            shayari_start = max(0.0, current_reel_time - 8.0)
            shayari_obj = ShayariEngine.generate_shayari(
                category=category,
                start_time=shayari_start,
                end_time=current_reel_time,
            )

        # 5. Formulate music and transition direction
        music_map = {
            "sad": "Melancholic piano and soft strings, slow crescendo into reflective silence",
            "motivational": "Inspirational cinematic build, rising brass and percussion into triumphant payoff",
            "romantic": "Acoustic warmth, delicate melodic strings, gentle heartfelt conclusion",
            "action": "Driving rhythmic sub-bass, fast cinematic percussion, abrupt hard impact",
            "powerful_dialogue": "Subtle low ambient drone ducking under dialogue, rising harmonic resonance",
            "suspense": "Low-frequency atmospheric tension, discordant string swells, sudden silence",
        }
        music_dir = music_map.get(category, "Balanced cinematic orchestral underscore with dynamic ducking")

        # Create unique plan ID
        plan_id = f"plan_{category}_{uuid.uuid4().hex[:8]}"
        title = f"{category.replace('_', ' ').title()} Reel: {prompt[:40]}"

        return ReelPlan(
            plan_id=plan_id,
            title=title,
            request=ReelRequest(
                theme=category,
                emotion=params.emotion or category,
                target_duration_seconds=round(current_reel_time, 1),
                aspect_ratio=aspect_ratio,
                dialogue_preference="optional" if params.requires_dialogue is None else ("required" if params.requires_dialogue else "none"),
            ),
            selected_clips=selected_clips,
            captions=timed_captions,
            shayari=shayari_obj,
            effects=["vignette", "smart_crop_9_16"],
            music_direction=music_dir,
            transition_notes=[
                "Smooth audio crossfade (0.2s) between clips to eliminate audio cut pops",
                "Straight hard visual cuts aligned with shot boundaries",
            ],
            approved_by_user=False,
        )
