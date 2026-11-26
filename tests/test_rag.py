"""Automated test suite verifying Hybrid RAG Retrieval & Question Library."""

import unittest
from reelforge.rag.schema import SceneRecord, DialogueLine
from reelforge.rag.db import SceneDatabase
from reelforge.rag.retrieval import HybridRetriever


def create_test_db():
    db = SceneDatabase()
    s_mot = SceneRecord(
        movie_id="movie_test",
        scene_id="scene_001",
        start_time_seconds=10.0,
        end_time_seconds=40.0,
        duration_seconds=30.0,
        scene_summary="A boxer stands up after a knockdown, refusing to quit.",
        emotions=["determination", "courage"],
        themes=["motivation", "perseverance", "overcoming failure"],
        dialogue=[
            DialogueLine(
                start_time_seconds=15.0,
                end_time_seconds=20.0,
                speaker="coach",
                text="Get up on your feet! The fight is not over yet!",
            )
        ],
        category_scores={"motivational": 0.95, "sad": 0.20, "action": 0.85},
    )
    s_sad = SceneRecord(
        movie_id="movie_test",
        scene_id="scene_002",
        start_time_seconds=100.0,
        end_time_seconds=130.0,
        duration_seconds=30.0,
        scene_summary="Character weeps at the grave of his beloved friend.",
        emotions=["sadness", "grief"],
        themes=["heartbreak", "loss"],
        dialogue=[],
        lighting_mood="low_key",
        category_scores={"sad": 0.98, "motivational": 0.10, "action": 0.05},
    )
    db.add_scene(s_mot)
    db.add_scene(s_sad)
    return db


class TestRAG(unittest.TestCase):

    def test_query_parser(self):
        retriever = HybridRetriever(create_test_db())
        p1 = retriever.parse_query("Create a 30-second motivational reel about overcoming failure")
        self.assertEqual(p1.target_category, "motivational")
        self.assertEqual(p1.target_duration, 30.0)
        self.assertTrue(any("resilience" in v for v in p1.query_variants))

        p2 = retriever.parse_query("Find the saddest scenes in this movie in 15 seconds")
        self.assertEqual(p2.target_category, "sad")
        self.assertEqual(p2.target_duration, 15.0)

    def test_hybrid_retrieval_ranking(self):
        db = create_test_db()
        retriever = HybridRetriever(db)

        # Search for motivational
        results_mot = retriever.retrieve("overcoming failure and perseverance", top_k=2)
        self.assertGreater(len(results_mot), 0)
        self.assertEqual(results_mot[0].scene_id, "scene_001")
        self.assertEqual(results_mot[0].category, "motivational")

        # Search for sad
        results_sad = retriever.retrieve("Find the saddest scenes about grief and crying", top_k=2)
        self.assertGreater(len(results_sad), 0)
        self.assertEqual(results_sad[0].scene_id, "scene_002")
        self.assertEqual(results_sad[0].category, "sad")

    def test_question_library(self):
        db = create_test_db()
        retriever = HybridRetriever(db)
        ans = retriever.answer_question("Find the most emotional scene in this movie")
        self.assertIn("question", ans)
        self.assertGreater(len(ans["top_scene_matches"]), 0)
        self.assertIn("answer_summary", ans)


if __name__ == "__main__":
    unittest.main()
