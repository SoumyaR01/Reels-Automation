"""Section 5 Scene Data Schema for Movie RAG Reel Generation System."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DialogueLine(BaseModel):
    """A single transcribed spoken utterance with exact timing."""

    start_time_seconds: float
    end_time_seconds: float
    speaker: str = "speaker_01"
    text: str
    language: str = "en"


class AudioFeatures(BaseModel):
    """Acoustic and prosodic properties extracted for the scene."""

    speech_rate_wpm: int = 120
    pitch_variance: str = "medium"  # low, medium, high
    energy_rms: float = 0.20
    music_intensity: str = "medium"  # low, medium, high
    silence_ratio: float = 0.15


class SceneRecord(BaseModel):
    """Core scene metadata record conforming strictly to Section 5 of Project Specification."""

    movie_id: str
    scene_id: str
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    scene_summary: str = ""
    visual_description: str = ""
    characters: List[str] = Field(default_factory=lambda: ["character_01"])
    actions: List[str] = Field(default_factory=list)
    emotions: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    dialogue: List[DialogueLine] = Field(default_factory=list)
    audio_description: str = ""
    shot_types: List[str] = Field(default_factory=lambda: ["medium"])
    camera_motion: List[str] = Field(default_factory=lambda: ["static"])
    lighting_mood: str = "neutral"
    visual_mood: List[str] = Field(default_factory=lambda: ["cinematic"])
    reel_tags: List[str] = Field(default_factory=list)
    
    # 6 Core Categories from Project Specification (Section 1A, 4.5A, 5)
    category_scores: Dict[str, float] = Field(
        default_factory=lambda: {
            "sad": 0.0,
            "action": 0.0,
            "powerful_dialogue": 0.0,
            "romantic": 0.0,
            "motivational": 0.0,
            "suspense": 0.0,
        }
    )
    
    audio_features: AudioFeatures = Field(default_factory=AudioFeatures)
    context_requirements: str = "low"  # low, medium, high
    reel_suitability: Dict[str, float] = Field(
        default_factory=lambda: {
            "15_seconds": 0.80,
            "30_seconds": 0.85,
        }
    )
    content_warnings: List[str] = Field(default_factory=list)

    # Embeddings references (stored natively or as vector lists)
    transcript_embedding: Optional[List[float]] = None
    visual_embedding: Optional[List[float]] = None
    metadata_embedding: Optional[List[float]] = None

    def get_searchable_text(self) -> str:
        """Combine metadata, dialogue, themes, and descriptions into a dense semantic string."""
        dialogue_text = " ".join(d.text for d in self.dialogue)
        components = [
            self.scene_summary,
            self.visual_description,
            " ".join(self.themes),
            " ".join(self.emotions),
            " ".join(self.actions),
            " ".join(self.reel_tags),
            dialogue_text,
            f"category: {' '.join(k for k, v in self.category_scores.items() if v >= 0.6)}",
            f"lighting: {self.lighting_mood}",
            f"audio: {self.audio_description}",
        ]
        return " ".join(c for c in components if c).strip()
