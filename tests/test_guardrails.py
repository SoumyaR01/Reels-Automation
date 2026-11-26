"""Automated test suite verifying Section 12A Deterministic Guardrails & Rubric."""

import unittest
from reelforge.rag.schema import SceneRecord, DialogueLine, AudioFeatures
from reelforge.rag.db import SceneDatabase
from reelforge.planner.schema import ReelPlan, ReelRequest, SelectedClip, HindiShayari
from reelforge.eval.guardrails import DeterministicGuardrails
from reelforge.eval.rubric import ReelPlanRubricJudge


def create_mock_db():
    db = SceneDatabase()
    s1 = SceneRecord(
        movie_id="mock_movie",
        scene_id="scene_0001",
        start_time_seconds=10.0,
        end_time_seconds=25.0,
        duration_seconds=15.0,
        scene_summary="Character reflecting on life after a setback.",
        emotions=["sadness", "determination"],
        themes=["motivation", "resilience"],
        dialogue=[
            DialogueLine(
                start_time_seconds=12.0,
                end_time_seconds=16.0,
                speaker="hero",
                text="Never give up on your dreams.",
            )
        ],
        category_scores={"motivational": 0.90, "sad": 0.40},
    )
    s2 = SceneRecord(
        movie_id="mock_movie",
        scene_id="scene_0002",
        start_time_seconds=50.0,
        end_time_seconds=70.0,
        duration_seconds=20.0,
        scene_summary="Climactic breakthrough and victory.",
        emotions=["triumph"],
        themes=["victory", "success"],
        category_scores={"motivational": 0.95, "action": 0.70},
    )
    db.add_scene(s1)
    db.add_scene(s2)
    return db


class TestGuardrails(unittest.TestCase):

    def test_valid_plan_passes_all_checks(self):
        db = create_mock_db()
        plan = ReelPlan(
            plan_id="plan_valid_01",
            title="Valid Motivational Reel",
            request=ReelRequest(theme="motivational", target_duration_seconds=30.0),
            selected_clips=[
                SelectedClip(
                    movie_id="mock_movie",
                    scene_id="scene_0001",
                    source_start_seconds=10.0,
                    source_end_seconds=24.0,
                    reel_start_seconds=0.0,
                    reel_end_seconds=14.0,
                    reason="Hook: Opening reflection",
                ),
                SelectedClip(
                    movie_id="mock_movie",
                    scene_id="scene_0002",
                    source_start_seconds=50.0,
                    source_end_seconds=65.0,
                    reel_start_seconds=14.0,
                    reel_end_seconds=29.0,
                    reason="Payoff: Final victory",
                ),
            ],
            shayari=HindiShayari(
                language="hi",
                text="तूफ़ानों से डरकर कभी रास्ते नहीं बदलते,\nमेहनत के पसीने से ही मुकद्दर चमकते।",
                ai_generated=True,
                reel_start_seconds=21.0,
                reel_end_seconds=29.0,
            ),
        )

        result = DeterministicGuardrails.validate_plan(plan, db)
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["timestamps_in_bounds"])
        self.assertTrue(result["checks"]["scene_ids_exist"])
        self.assertTrue(result["checks"]["shayari_labelled_ai_generated"])
        self.assertEqual(len(result["violations"]), 0)

    def test_corrupted_timestamp_fails(self):
        db = create_mock_db()
        plan = ReelPlan(
            plan_id="plan_invalid_time",
            title="Invalid Time Reel",
            request=ReelRequest(theme="motivational", target_duration_seconds=30.0),
            selected_clips=[
                SelectedClip(
                    movie_id="mock_movie",
                    scene_id="scene_0001",
                    source_start_seconds=5.0,  # Scene start is 10.0 -> OUT OF BOUNDS!
                    source_end_seconds=20.0,
                    reel_start_seconds=0.0,
                    reel_end_seconds=15.0,
                    reason="Hook",
                )
            ],
        )
        result = DeterministicGuardrails.validate_plan(plan, db)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["timestamps_in_bounds"])
        self.assertTrue(any("before scene start" in v for v in result["violations"]))

    def test_nonexistent_scene_fails(self):
        db = create_mock_db()
        plan = ReelPlan(
            plan_id="plan_invalid_scene",
            title="Invalid Scene Reel",
            request=ReelRequest(theme="motivational", target_duration_seconds=30.0),
            selected_clips=[
                SelectedClip(
                    movie_id="mock_movie",
                    scene_id="scene_fake_999",  # Does not exist!
                    source_start_seconds=1.0,
                    source_end_seconds=10.0,
                    reel_start_seconds=0.0,
                    reel_end_seconds=9.0,
                    reason="Hook",
                )
            ],
        )
        result = DeterministicGuardrails.validate_plan(plan, db)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["scene_ids_exist"])

    def test_rubric_evaluation(self):
        db = create_mock_db()
        plan = ReelPlan(
            plan_id="plan_eval_01",
            title="Rubric Test Plan",
            request=ReelRequest(theme="motivational", target_duration_seconds=30.0),
            selected_clips=[
                SelectedClip(
                    movie_id="mock_movie",
                    scene_id="scene_0001",
                    source_start_seconds=10.0,
                    source_end_seconds=24.0,
                    reel_start_seconds=0.0,
                    reel_end_seconds=14.0,
                    reason="Hook: Opening reflection",
                ),
                SelectedClip(
                    movie_id="mock_movie",
                    scene_id="scene_0002",
                    source_start_seconds=50.0,
                    source_end_seconds=65.0,
                    reel_start_seconds=14.0,
                    reel_end_seconds=29.0,
                    reason="Payoff: Final victory",
                ),
            ],
            shayari=HindiShayari(
                language="hi",
                text="हार से जो सीखे वही इतिहास रचता है।",
                ai_generated=True,
            ),
        )
        eval_res = ReelPlanRubricJudge.evaluate(plan, db)
        self.assertGreaterEqual(eval_res["composite_score"], 4.0)
        self.assertEqual(eval_res["rubric_scores"]["grounding_check"], 5.0)
        self.assertTrue(eval_res["overall_pass"])


if __name__ == "__main__":
    unittest.main()
