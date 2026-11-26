"""Upload management and async job tracking for video processing pipeline."""

import json
import shutil
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStage(BaseModel):
    """Individual processing stage within a job."""
    name: str
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    message: str = ""


class ProcessingJob(BaseModel):
    """Represents a video processing job from upload through scene indexing."""
    job_id: str
    filename: str
    original_filename: str
    file_size: int = 0
    movie_id: str = ""
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    stages: List[JobStage] = Field(default_factory=list)
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class UploadManager:
    """Manages video file uploads and processing job lifecycle."""

    def __init__(self, upload_dir: Path, work_dir: Path):
        self.upload_dir = Path(upload_dir)
        self.work_dir = Path(work_dir)
        self.jobs_file = self.work_dir / "jobs.json"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _load_jobs(self) -> Dict[str, ProcessingJob]:
        if self.jobs_file.is_file():
            try:
                with open(self.jobs_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {k: ProcessingJob(**v) for k, v in data.items()}
            except Exception:
                return {}
        return {}

    def _save_jobs(self, jobs: Dict[str, ProcessingJob]):
        with open(self.jobs_file, "w", encoding="utf-8") as f:
            json.dump({k: v.model_dump() for k, v in jobs.items()}, f, indent=2)

    def generate_movie_id(self, filename: str) -> str:
        """Generate a stable movie_id from filename."""
        stem = Path(filename).stem.lower()
        # Clean up common patterns
        clean = stem.replace('.', '_').replace('-', '_').replace(' ', '_')
        # Remove duplicate underscores
        while '__' in clean:
            clean = clean.replace('__', '_')
        # Create a short hash for uniqueness
        h = hashlib.md5(filename.encode()).hexdigest()[:8]
        # Keep it reasonably short
        if len(clean) > 40:
            clean = clean[:40]
        return f"{clean}_{h}"

    def register_upload(self, filename: str, file_size: int) -> ProcessingJob:
        """Register a new upload and create a processing job."""
        movie_id = self.generate_movie_id(filename)
        job_id = f"job_{movie_id}"

        job = ProcessingJob(
            job_id=job_id,
            filename=filename,
            original_filename=filename,
            file_size=file_size,
            movie_id=movie_id,
            status=JobStatus.QUEUED,
            stages=[
                JobStage(name="upload", status="completed", progress=100.0, message="File uploaded"),
                JobStage(name="ingest", status="pending", message="Waiting to start"),
                JobStage(name="shots", status="pending", message="Shot detection"),
                JobStage(name="scenes", status="pending", message="Scene grouping"),
                JobStage(name="audio", status="pending", message="Audio analysis"),
                JobStage(name="video", status="pending", message="Visual scoring"),
                JobStage(name="scoring", status="pending", message="RAG indexing"),
                JobStage(name="ready", status="pending", message="Ready for reels"),
            ],
        )

        jobs = self._load_jobs()
        jobs[job_id] = job
        self._save_jobs(jobs)
        return job

    def update_job(self, job_id: str, **kwargs) -> Optional[ProcessingJob]:
        """Update a job's status or fields."""
        jobs = self._load_jobs()
        if job_id not in jobs:
            return None
        job = jobs[job_id]
        for k, v in kwargs.items():
            if hasattr(job, k):
                setattr(job, k, v)
        job.updated_at = time.time()
        jobs[job_id] = job
        self._save_jobs(jobs)
        return job

    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        jobs = self._load_jobs()
        return jobs.get(job_id)

    def list_jobs(self) -> List[ProcessingJob]:
        jobs = self._load_jobs()
        return sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get_indexed_videos(self) -> List[Dict[str, Any]]:
        """List all videos that have been fully indexed with scene databases."""
        indexed = []

        # Check workdir for scene databases
        if self.work_dir.is_dir():
            for d in sorted(self.work_dir.iterdir()):
                if d.is_dir() and (d / "scene_db.json").is_file():
                    info = {
                        "movie_id": d.name,
                        "has_scene_db": True,
                        "scene_count": 0,
                        "title": d.name.replace("_", " ").title(),
                    }
                    try:
                        with open(d / "scene_db.json", "r", encoding="utf-8") as f:
                            scenes = json.load(f)
                            info["scene_count"] = len(scenes)
                    except Exception:
                        pass

                    # Try to load manifest for more info
                    manifest_file = d / "manifest.json"
                    if manifest_file.is_file():
                        try:
                            with open(manifest_file, "r", encoding="utf-8") as f:
                                manifest = json.load(f)
                                info["title"] = manifest.get("movie_title", info["title"])
                                info["duration"] = manifest.get("duration", 0)
                                info["filename"] = manifest.get("original_filename", "")
                        except Exception:
                            pass

                    indexed.append(info)

        return indexed

    def cleanup_job(self, job_id: str):
        """Remove a job and its uploaded file."""
        jobs = self._load_jobs()
        if job_id in jobs:
            job = jobs[job_id]
            # Remove uploaded file
            upload_path = self.upload_dir / job.filename
            if upload_path.is_file():
                upload_path.unlink(missing_ok=True)
            del jobs[job_id]
            self._save_jobs(jobs)
