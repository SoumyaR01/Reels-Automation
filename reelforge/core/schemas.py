"""Pydantic data schemas representing the ReelForge core data contract."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ShotData(BaseModel):
    """Single continuous shot between two hard cuts or transitions."""

    shot_id: int
    start: float  # in seconds (float64 precision)
    end: float
    duration: float
    cut_type: str = "hard_cut"  # hard_cut, dissolve, fade
    confidence: float = 1.0


class SceneData(BaseModel):
    """A semantic narrative scene grouping one or more consecutive shots."""

    scene_id: str
    shot_ids: List[int]
    start: float
    end: float
    duration: float
    visual_similarity: float = 0.0
    dialogue_gap: float = 0.0


class SignalMetrics(BaseModel):
    """Multimodal signals extracted across visual, acoustic, and emotional dimensions."""

    arousal: float = Field(default=0.5, ge=0.0, le=1.0)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    speech_rate: float = Field(default=2.0, description="Words/syllables per second")
    f0_range: float = Field(default=100.0, description="Pitch dynamic range in Hz")
    pause_ratio: float = Field(default=0.2, description="Proportion of silence within dialogue")
    
    # Visual signals
    visual_composition: float = Field(default=0.5, ge=0.0, le=1.0, description="Rule-of-thirds edge score")
    color_grading: float = Field(default=0.5, ge=0.0, le=1.0, description="Teal/orange and film desaturation score")
    motion_stability: float = Field(default=0.5, ge=0.0, le=1.0, description="Camera tracking stability / low jerk")
    slow_motion: float = Field(default=0.0, ge=0.0, le=1.0, description="Still / frozen frame presence")
    low_brightness: float = Field(default=0.0, ge=0.0, le=1.0, description="Low illumination and shadow depth")
    
    # Audio signals
    low_rms: float = Field(default=0.5, ge=0.0, le=1.0, description="Subdued audio energy score")
    minor_key_correlation: float = Field(default=0.0, ge=0.0, le=1.0, description="Krumhansl-Kessler minor profile alignment")
    low_spectral_centroid: float = Field(default=0.5, ge=0.0, le=1.0, description="Warm / dark acoustic spectrum")
    
    # Semantic tags
    face_emotion: str = "neutral"
    music_mood: str = "neutral"
    audio_events: List[str] = Field(default_factory=list)


class DialogueData(BaseModel):
    """Speech transcript and narrative semantic ratings."""

    text: str = ""
    standalone_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Self-contained comprehensibility without movie context")
    quotability: float = Field(default=0.5, ge=0.0, le=1.0, description="Punchiness, impact, and rhetorical quality")
    contains_profanity: bool = False


class RightsInfo(BaseModel):
    """Source material rights registry and publishing gate information."""

    source_id: str
    source_title: str
    source_status: str = Field(
        default="unknown",
        description="One of: 'licensed', 'public_domain', 'own', 'fair_dealing_commentary', 'unknown'"
    )
    rights_holder: Optional[str] = None
    notes: Optional[str] = None
    verified: bool = False
    verified_at: Optional[str] = None


class MomentCandidate(BaseModel):
    """A candidate 25-35s compilation window conforming to the core ReelForge data contract."""

    movie_id: str
    moment_id: str
    scene_id: str
    start: float
    end: float
    duration: float
    
    # Normalized scores across genres
    labels: Dict[str, float] = Field(
        default_factory=lambda: {
            "sad": 0.0,
            "cinematic": 0.0,
            "powerful_dialogue": 0.0,
            "romantic": 0.0,
            "action": 0.0,
            "suspense": 0.0,
        }
    )
    
    signals: SignalMetrics = Field(default_factory=SignalMetrics)
    dialogue: DialogueData = Field(default_factory=DialogueData)
    
    # Editorial quality metrics
    hook_score: float = Field(default=0.5, description="Strength of visual or audio hook in first 3 seconds")
    spoiler_risk: float = Field(default=0.1, description="Probability of revealing climactic twists or endings")
    standalone_score: float = Field(default=0.7, description="Self-contained engagement score")
    final_score: float = Field(default=0.5, description="Blended overall editorial rating")
    
    selected_category: str = "cinematic"
    rights: RightsInfo = Field(default_factory=lambda: RightsInfo(source_id="m_default", source_title="Movie"))


class ManifestSchema(BaseModel):
    """Comprehensive job execution manifest recording state, caches, and artifacts for resumability."""

    movie_id: str
    movie_path: str
    movie_title: str
    duration: float
    created_at: str
    updated_at: str
    
    # Pipeline stage progression
    stages: Dict[str, str] = Field(
        default_factory=lambda: {
            "ingest": "pending",
            "shots": "pending",
            "scenes": "pending",
            "audio": "pending",
            "video": "pending",
            "scoring": "pending",
            "selection": "pending",
            "render": "pending",
        }
    )
    
    # Persisted artifact paths
    artifacts: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
