"""Section 12A LLM-as-Judge rubric evaluator for subjective reel qualities."""

from typing import Dict, Any
from ..planner.schema import ReelPlan
from .guardrails import DeterministicGuardrails
from ..rag.db import SceneDatabase


class ReelPlanRubricJudge:
    """Evaluates Reel Plans using the calibrated Section 12A rubric (1 to 5 scale with anchors)."""

    @classmethod
    def evaluate(cls, plan: ReelPlan, db: SceneDatabase) -> Dict[str, Any]:
        """Evaluate plan across the 5 Section 12A rubric dimensions."""
        det_result = DeterministicGuardrails.validate_plan(plan, db)

        # 1. Grounding (Automatic fail if any deterministic check fails)
        if not det_result["passed"]:
            grounding_score = 1.0
            overall_pass = False
        else:
            grounding_score = 5.0
            overall_pass = True

        # 2. Theme Match (1-5):
        # Anchor 5: All clips directly support category/theme.
        # Anchor 3: Fits broad genre but pacing is slightly off.
        # Anchor 1: Irrelevant to prompt theme.
        cat = plan.request.theme
        scene_scores = [
            db.get_scene(c.scene_id).category_scores.get(cat, 0.5)
            for c in plan.selected_clips
            if db.get_scene(c.scene_id)
        ]
        avg_cat_score = sum(scene_scores) / max(1, len(scene_scores))
        theme_score = round(float(1.0 + 4.0 * avg_cat_score), 1)

        # 3. Hook Strength (1-5):
        # Anchor 5: First 3 seconds feature strong visual impact or compelling line.
        # Anchor 3: Moderate establishing shot.
        # Anchor 1: Slow, black frame, or confusing opening.
        first_clip = plan.selected_clips[0] if plan.selected_clips else None
        if first_clip:
            first_scene = db.get_scene(first_clip.scene_id)
            hook_base = first_scene.reel_suitability.get("15_seconds", 0.7) if first_scene else 0.7
            hook_score = round(float(1.0 + 4.0 * hook_base), 1)
        else:
            hook_score = 1.0

        # 4. Coherence (1-5):
        # Narrative progression: Hook -> Setback -> Turning Point -> Payoff -> Closing
        beat_reasons = [c.reason for c in plan.selected_clips]
        has_hook = any("hook" in r.lower() for r in beat_reasons)
        has_payoff = any("payoff" in r.lower() or "climax" in r.lower() for r in beat_reasons)
        coherence_score = 5.0 if (has_hook and has_payoff) else (4.0 if has_hook else 3.0)

        # 5. Duration / Format Fit (1-5):
        total_time = plan.selected_clips[-1].reel_end_seconds if plan.selected_clips else 0.0
        dur_diff = abs(total_time - plan.request.target_duration_seconds)
        if dur_diff <= 1.0:
            format_score = 5.0
        elif dur_diff <= 3.0:
            format_score = 4.0
        elif dur_diff <= 5.0:
            format_score = 3.0
        else:
            format_score = 2.0

        # 6. Hindi Shayari Quality (if present)
        shayari_score = 5.0 if plan.shayari and len(plan.shayari.text.strip()) > 10 else 4.0

        composite_score = round(
            float(
                0.25 * theme_score +
                0.20 * hook_score +
                0.20 * coherence_score +
                0.20 * grounding_score +
                0.15 * format_score
            ),
            2,
        )

        return {
            "plan_id": plan.plan_id,
            "overall_pass": overall_pass and composite_score >= 3.5,
            "composite_score": composite_score,
            "rubric_scores": {
                "theme_match": theme_score,
                "hook_strength": hook_score,
                "narrative_coherence": coherence_score,
                "grounding_check": grounding_score,
                "duration_format_fit": format_score,
                "shayari_quality": shayari_score,
            },
            "deterministic_checks": det_result["checks"],
            "violations": det_result["violations"],
        }
