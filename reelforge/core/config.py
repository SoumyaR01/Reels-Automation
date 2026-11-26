"""Configuration and settings management for ReelForge."""

from pathlib import Path
from typing import List
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Global system settings for ReelForge pipeline execution."""

    work_dir: Path = Field(default=Path("workdir"), description="Root working directory for cached proxies and artifacts")
    output_dir: Path = Field(default=Path("reels"), description="Output folder for compiled 9:16 reels")
    
    # Ingest / Proxy settings
    proxy_height: int = Field(default=360, description="Height for downscaled analysis proxy video (e.g. 360p)")
    proxy_fps: float = Field(default=12.0, description="Analysis proxy frame rate for accurate motion and cut sampling")
    audio_sample_rate: int = Field(default=16000, description="Standard sample rate for audio analysis (16 kHz mono)")

    # Moment & Window constraints
    target_duration: float = Field(default=30.0, description="Exact target duration in seconds for exported reels")
    min_moment_duration: float = Field(default=25.0, description="Minimum candidate moment length in seconds")
    max_moment_duration: float = Field(default=35.0, description="Maximum candidate moment length in seconds")
    min_shot_duration: float = Field(default=1.0, description="Minimum shot duration in seconds")
    
    # Reel visual specifications
    reel_width: int = Field(default=1080, description="Vertical reel output width")
    reel_height: int = Field(default=1920, description="Vertical reel output height")
    crf: int = Field(default=20, description="H.264 quality CRF factor (lower is higher quality)")
    preset: str = Field(default="medium", description="FFmpeg encoding preset")
    
    # Audio & Music specifications
    target_loudness_lufs: float = Field(default=-14.0, description="Integrated loudness target for short-form video (LUFS)")
    ducking_attenuation_db: float = Field(default=-16.0, description="Attenuation dB for background music ducking under dialogue")
    
    # Rights & Compliance
    enforce_rights_gate: bool = Field(default=True, description="Hard-block final reel export if rights status is 'unknown'")
    ai_disclosure_label: bool = Field(default=True, description="Include AI-curated/edited disclosure tag in reel metadata")
    
    # Categories
    categories: List[str] = Field(
        default=["sad", "cinematic", "powerful_dialogue", "romantic", "action", "suspense"],
        description="Supported emotion and style categories"
    )

    class Config:
        arbitrary_types_allowed = True


_settings_instance = None


def get_settings() -> Settings:
    """Singleton getter for active ReelForge settings."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
