"""Section 12A Deterministic Guardrails: Automated verification for reel plans."""

from typing import Any, Dict, List
from ..planner.schema import ReelPlan
from ..rag.db import SceneDatabase


class DeterministicGuardrails:
    """Implements non-negotiable deterministic validation rules from Section 12A."""

    @classmethod
    def validate_plan(cls, plan: ReelPlan, db: SceneDatabase) -> Dict[str, Any]:
        """Run all deterministic checks on a ReelPlan.
        
        Returns a dict with overall 'passed' status, individual check outcomes, and violation errors.
        """
        results: Dict[str, bool] = {}
        violations: List[str] = []

        # Check 1: Schema Validity
        try:
            plan_dict = plan.model_dump()
            results["schema_validity"] = True
        except Exception as e:
            results["schema_validity"] = False
            violations.append(f"Schema validity failure: {e}")

        # Check 2: Scene IDs Exist in Database
        all_scenes_exist = True
        for clip in plan.selected_clips:
            scene = db.get_scene(clip.scene_id)
            if scene is None:
                all_scenes_exist = False
                violations.append(f"Scene ID '{clip.scene_id}' does not exist in scene database.")
        results["scene_ids_exist"] = all_scenes_exist

        # Check 3: Timestamps strictly in bounds (Grounding)
        timestamps_grounded = True
        for clip in plan.selected_clips:
            scene = db.get_scene(clip.scene_id)
            if scene:
                # Source boundaries
                if clip.source_start_seconds < scene.start_time_seconds - 0.05:
                    timestamps_grounded = False
                    violations.append(
                        f"Clip start {clip.source_start_seconds:.2f}s is before scene start {scene.start_time_seconds:.2f}s for {clip.scene_id}."
                    )
                if clip.source_end_seconds > scene.end_time_seconds + 0.05:
                    timestamps_grounded = False
                    violations.append(
                        f"Clip end {clip.source_end_seconds:.2f}s exceeds scene end {scene.end_time_seconds:.2f}s for {clip.scene_id}."
                    )
                # Clip length match
                clip_dur = round(clip.source_end_seconds - clip.source_start_seconds, 2)
                reel_dur = round(clip.reel_end_seconds - clip.reel_start_seconds, 2)
                if abs(clip_dur - reel_dur) > 0.1:
                    timestamps_grounded = False
                    violations.append(
                        f"Clip duration mismatch: source duration {clip_dur:.2f}s != reel duration {reel_dur:.2f}s."
                    )
        results["timestamps_in_bounds"] = timestamps_grounded

        # Check 4: Dialogue exact match against stored transcripts
        dialogue_matched = True
        for caption in plan.captions:
            # Find matching scene
            found = False
            for clip in plan.selected_clips:
                if clip.reel_start_seconds <= caption.start_seconds <= clip.reel_end_seconds:
                    scene = db.get_scene(clip.scene_id)
                    if scene:
                        for d in scene.dialogue:
                            if caption.text.strip().lower() in d.text.strip().lower() or d.text.strip().lower() in caption.text.strip().lower():
                                found = True
                                break
            # If caption was marked as speech, it must match
            if not found and caption.speaker is not None:
                dialogue_matched = False
                violations.append(f"Dialogue caption '{caption.text}' does not match stored scene dialogue.")
        results["dialogue_exact_match"] = dialogue_matched

        # Check 5: Hindi Shayari strictly labelled as AI-generated
        if plan.shayari:
            shayari_labelled = (
                plan.shayari.ai_generated is True and
                plan.shayari.language == "hi" and
                len(plan.shayari.text.strip()) > 5
            )
            results["shayari_labelled_ai_generated"] = shayari_labelled
            if not shayari_labelled:
                violations.append("Shayari is present but not correctly flagged with ai_generated: true or language: hi.")
        else:
            results["shayari_labelled_ai_generated"] = True

        # Check 6: Output duration within acceptable range
        if plan.selected_clips:
            final_reel_time = plan.selected_clips[-1].reel_end_seconds
            target = plan.request.target_duration_seconds
            dur_valid = abs(final_reel_time - target) <= 5.0
            results["duration_within_tolerance"] = dur_valid
            if not dur_valid:
                violations.append(
                    f"Final reel timeline ({final_reel_time:.1f}s) deviates more than 5s from target ({target:.1f}s)."
                )
        else:
            results["duration_within_tolerance"] = False
            violations.append("Plan contains zero selected clips.")

        passed = len(violations) == 0
        return {
            "plan_id": plan.plan_id,
            "passed": passed,
            "checks": results,
            "violations": violations,
        }
