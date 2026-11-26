"""Hybrid RAG retrieval engine implementing Section 6 ranking and Section 4 Question Library."""

import re
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from .schema import SceneRecord
from .db import SceneDatabase
from .embeddings import EmbeddingService


class RetrievalCandidate(BaseModel):
    """A retrieved scene candidate with hybrid ranking signals, confidence, and editorial rationale."""

    scene_id: str
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    relevance_score: float = Field(ge=0.0, le=1.0)
    category: str
    category_score: float
    hook_score: float
    context_independence: float
    reason: str
    scene_summary: str
    dialogue_excerpt: str = ""
    themes: List[str] = Field(default_factory=list)
    emotions: List[str] = Field(default_factory=list)


class QueryParameters(BaseModel):
    """Parsed user prompt intent and constraints."""

    raw_query: str
    target_category: Optional[str] = None
    target_duration: float = 30.0
    theme: str = ""
    emotion: str = ""
    requires_dialogue: Optional[bool] = None
    visual_only: bool = False
    query_variants: List[str] = Field(default_factory=list)


class HybridRetriever:
    """Performs hybrid semantic vector search, keyword matching, and Section 6 multi-signal ranking."""

    def __init__(self, db: SceneDatabase, embedding_service: Optional[EmbeddingService] = None):
        self.db = db
        self.embedding_service = embedding_service or EmbeddingService()
        self._initialize_index()

    def _initialize_index(self):
        """Fit embedding service on all scene searchable texts."""
        scenes = self.db.list_scenes()
        if scenes:
            corpus = [s.get_searchable_text() for s in scenes]
            self.embedding_service.fit(corpus)
            for s in scenes:
                vec = self.embedding_service.encode(s.get_searchable_text())
                s.metadata_embedding = vec.tolist()

    def parse_query(self, query: str) -> QueryParameters:
        """Parse natural language query into intent, target category, emotion, and duration."""
        q_lower = query.lower()
        
        # 1. Target duration detection
        dur = 30.0
        dur_match = re.search(r"(\d+)\s*[- ]*(?:seconds?|secs?|s)\b", q_lower)
        if dur_match:
            dur = float(dur_match.group(1))

        # 2. Category / emotion detection
        cat = None
        if any(w in q_lower for w in ["sad", "crying", "heartbreak", "grief", "pain", "loss"]):
            cat = "sad"
        elif any(w in q_lower for w in ["action", "fight", "chase", "combat", "intense", "battle"]):
            cat = "action"
        elif any(w in q_lower for w in ["dialogue", "quote", "speech", "saying", "words", "monologue"]):
            cat = "powerful_dialogue"
        elif any(w in q_lower for w in ["romantic", "love", "romance", "couple", "kiss", "date"]):
            cat = "romantic"
        elif any(w in q_lower for w in ["motivat", "inspire", "overcom", "failure", "success", "courage", "persevere"]):
            cat = "motivational"
        elif any(w in q_lower for w in ["suspense", "thriller", "mystery", "tension", "scary", "fear"]):
            cat = "suspense"

        # 3. Visual-only vs dialogue preference
        visual_only = any(w in q_lower for w in ["no dialogue", "without dialogue", "visual only", "only visuals", "silent"])
        req_dialogue = True if ("dialogue" in q_lower and not visual_only) else (False if visual_only else None)

        # 4. Generate query expansion variants
        variants = [query]
        if "overcoming failure" in q_lower or "motivat" in q_lower:
            variants.extend(["resilience after setback", "perseverance and determination", "character rising up"])
        elif "sad" in q_lower:
            variants.extend(["heartbreak and loneliness", "grief and sorrow", "quiet somber reflection"])
        elif "action" in q_lower:
            variants.extend(["intense fight sequence", "danger and survival", "fast paced conflict"])

        return QueryParameters(
            raw_query=query,
            target_category=cat,
            target_duration=dur,
            theme=cat or "cinematic",
            emotion=cat or "cinematic",
            requires_dialogue=req_dialogue,
            visual_only=visual_only,
            query_variants=variants,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.35,
    ) -> List[RetrievalCandidate]:
        """Execute hybrid search using the Section 6 weighting formula.

        Ranking formula:
        - 30% prompt/theme relevance (semantic similarity)
        - 20% emotional match (category score)
        - 15% visual match (lighting & stability)
        - 15% hook strength (hook suitability)
        - 10% story clarity / context independence
        - 10% duration fit
        """
        params = self.parse_query(query)
        q_vec = self.embedding_service.encode(" ".join(params.query_variants))

        scenes = self.db.list_scenes()
        candidates: List[RetrievalCandidate] = []

        for s in scenes:
            # Check dialogue constraint
            has_dialogue = len(s.dialogue) > 0
            if params.visual_only and has_dialogue:
                continue
            if params.requires_dialogue and not has_dialogue:
                continue

            # 1. Semantic relevance (30%)
            s_vec = np.array(s.metadata_embedding) if s.metadata_embedding else self.embedding_service.encode(s.get_searchable_text())
            sim = max(0.0, self.embedding_service.cosine_similarity(q_vec, s_vec))
            semantic_score = float(sim)

            # 2. Emotional match (20%)
            cat = params.target_category or "motivational"
            emotion_score = float(s.category_scores.get(cat, 0.5))

            # 3. Visual match (15%)
            visual_score = 0.8 if s.lighting_mood in ["low_key", "cinematic_contrast"] else 0.6
            if "steady" in s.visual_mood or "tracking" in s.camera_motion:
                visual_score = min(1.0, visual_score + 0.15)

            # 4. Hook strength (15%)
            hook_score = 0.85 if (s.reel_suitability.get("15_seconds", 0.5) > 0.7 or has_dialogue) else 0.60

            # 5. Context independence (10%)
            context_score = 0.90 if s.context_requirements == "low" else 0.60

            # 6. Duration fit (10%)
            dur_diff = abs(s.duration_seconds - params.target_duration)
            dur_fit = float(np.clip(1.0 - (dur_diff / max(30.0, params.target_duration)), 0.0, 1.0))

            # Composite hybrid score
            final_score = float(
                0.30 * semantic_score +
                0.20 * emotion_score +
                0.15 * visual_score +
                0.15 * hook_score +
                0.10 * context_score +
                0.10 * dur_fit
            )

            if final_score >= min_score:
                dialogue_text = " ".join(f'"{d.text}"' for d in s.dialogue)
                reason = (
                    f"Matches '{cat}' theme (score {emotion_score:.2f}) with semantic relevance {semantic_score:.2f}. "
                    f"Lighting: {s.lighting_mood}, duration {s.duration_seconds:.1f}s."
                )
                candidates.append(
                    RetrievalCandidate(
                        scene_id=s.scene_id,
                        start_time_seconds=s.start_time_seconds,
                        end_time_seconds=s.end_time_seconds,
                        duration_seconds=s.duration_seconds,
                        relevance_score=round(final_score, 3),
                        category=cat,
                        category_score=round(emotion_score, 3),
                        hook_score=round(hook_score, 3),
                        context_independence=round(context_score, 3),
                        reason=reason,
                        scene_summary=s.scene_summary,
                        dialogue_excerpt=dialogue_text[:180],
                        themes=s.themes,
                        emotions=s.emotions,
                    )
                )

        # Sort descending by hybrid relevance score
        candidates.sort(key=lambda c: c.relevance_score, reverse=True)
        return candidates[:top_k]

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Handle structured questions from Section 4 Movie RAG Question Library."""
        candidates = self.retrieve(question, top_k=5)
        meta = self.db.movie_metadata

        return {
            "question": question,
            "movie_id": meta.get("movie_id", ""),
            "movie_title": meta.get("movie_title", "Movie"),
            "total_scenes_analyzed": len(self.db.scenes),
            "top_scene_matches": [c.model_dump() for c in candidates],
            "answer_summary": (
                f"Found {len(candidates)} grounded scenes answering: '{question}'. "
                f"Top scene is {candidates[0].scene_id} ({candidates[0].start_time_seconds:.1f}s - {candidates[0].end_time_seconds:.1f}s) "
                f"with confidence {candidates[0].relevance_score:.2f}."
                if candidates else "No matching scenes found for query."
            ),
        }
