"""Idempotent manifest management for ReelForge pipelines."""

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import ManifestSchema


def compute_movie_id(file_path: Path) -> str:
    """Compute a deterministic, clean movie_id based on file stem and size."""
    file_path = Path(file_path)
    stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in file_path.stem)
    stem = stem[:32].strip("_")
    try:
        size = file_path.stat().st_size
        h = hashlib.sha256(f"{file_path.name}_{size}".encode("utf-8")).hexdigest()[:8]
    except Exception:
        h = hashlib.sha256(file_path.name.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{h}".lower()


class ManifestManager:
    """Manages the job lifecycle and stage caching in workdir/<movie_id>/manifest.json."""

    def __init__(self, movie_path: Path, work_dir: Path):
        self.movie_path = Path(movie_path).resolve()
        self.movie_id = compute_movie_id(self.movie_path)
        self.job_dir = Path(work_dir) / self.movie_id
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.job_dir / "manifest.json"
        self.manifest: ManifestSchema = self._load_or_init()

    def _now_iso(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _load_or_init(self) -> ManifestSchema:
        if self.manifest_file.is_file():
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ManifestSchema(**data)
            except Exception:
                pass

        now = self._now_iso()
        manifest = ManifestSchema(
            movie_id=self.movie_id,
            movie_path=str(self.movie_path),
            movie_title=self.movie_path.stem,
            duration=0.0,
            created_at=now,
            updated_at=now,
        )
        self._write_to_disk(manifest)
        return manifest

    def _write_to_disk(self, manifest: ManifestSchema):
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)

    def save(self):
        """Save current manifest state to disk."""
        self.manifest.updated_at = self._now_iso()
        self._write_to_disk(self.manifest)

    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a pipeline stage has already completed successfully."""
        return self.manifest.stages.get(stage_name) == "completed"

    def mark_stage_started(self, stage_name: str):
        """Mark a pipeline stage as running."""
        self.manifest.stages[stage_name] = "running"
        self.save()

    def mark_stage_completed(self, stage_name: str, artifacts: Optional[Dict[str, str]] = None):
        """Mark a pipeline stage as completed and persist produced artifact paths."""
        self.manifest.stages[stage_name] = "completed"
        if artifacts:
            for k, v in artifacts.items():
                self.manifest.artifacts[k] = str(Path(v).resolve())
        self.save()

    def mark_stage_failed(self, stage_name: str, error_message: str):
        """Mark a pipeline stage as failed with error context."""
        self.manifest.stages[stage_name] = f"failed: {error_message}"
        self.save()

    def get_artifact(self, key: str) -> Optional[Path]:
        """Retrieve path for an artifact if it exists on disk."""
        val = self.manifest.artifacts.get(key)
        if val:
            p = Path(val)
            if p.exists():
                return p
        return None

    def set_duration(self, duration: float):
        """Set probed duration in manifest."""
        self.manifest.duration = float(duration)
        self.save()
