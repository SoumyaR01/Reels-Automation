"""FastAPI service: AI-Powered Reels Generator with upload, scene detection, planning, and caption generation."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

from ..core.config import get_settings
from ..rights.registry import RightsRegistry
from ..rag.db import SceneDatabase
from ..rag.retrieval import HybridRetriever
from ..planner.agent import ReelPlanningAgent
from ..planner.schema import ReelPlan
from ..planner.auto_mode import AutoReelBatchGenerator
from ..eval.guardrails import DeterministicGuardrails
from ..eval.rubric import ReelPlanRubricJudge
from ..edit.render import render_reel_plan
from ..captions.generator import CaptionEngine
from .jobs import UploadManager


# ─── Request Models ───────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    query: str
    movie_id: str = ""
    top_k: int = 5


class PlanGenerateRequest(BaseModel):
    prompt: str
    movie_id: str = ""
    duration: float = 30.0
    style: str = ""
    include_shayari: bool = False


class AutoGenerateRequest(BaseModel):
    movie_id: str = ""
    styles: List[str] = []
    reels_per_style: int = 1
    duration: float = 30.0


class CaptionRequest(BaseModel):
    style: str = "cinematic"
    title: str = ""
    movie_title: str = ""
    duration: float = 30.0
    dialogue_excerpt: str = ""
    beat_count: int = 5
    music_direction: str = ""


class RightsUpdateRequest(BaseModel):
    status: str
    holder: Optional[str] = None
    notes: Optional[str] = None


# ─── App Factory ──────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="ReelForge AI Studio", version="2.0.0")
    rights_reg = RightsRegistry(settings.work_dir / "rights_registry.json")
    upload_mgr = UploadManager(
        upload_dir=Path("uploads"),
        work_dir=settings.work_dir,
    )

    # ─── Helpers ──────────────────────────────────────────────

    def _get_default_movie_id() -> str:
        """Find the default or active indexed movie ID in workdir."""
        if settings.work_dir.is_dir():
            for d in settings.work_dir.iterdir():
                if d.is_dir() and (d / "scene_db.json").is_file():
                    return d.name
            for d in settings.work_dir.iterdir():
                if d.is_dir() and (d / "manifest.json").is_file():
                    return d.name
        return ""

    def _resolve_movie_id(movie_id: str = "") -> str:
        """Resolve a movie ID, falling back to default."""
        return movie_id.strip() or _get_default_movie_id()

    def _get_db(movie_id: str) -> SceneDatabase:
        mid = _resolve_movie_id(movie_id)
        if not mid:
            raise HTTPException(status_code=404, detail="No indexed video found. Please upload and process a video first.")
        db_path = settings.work_dir / mid / "scene_db.json"
        if not db_path.is_file():
            raise HTTPException(status_code=404, detail=f"Scene database not found for '{mid}'")
        return SceneDatabase(db_path)

    def _get_movie_info(movie_id: str) -> Dict[str, Any]:
        """Get comprehensive movie info for the UI."""
        mid = _resolve_movie_id(movie_id)
        info = {
            "movie_id": mid,
            "title": mid,
            "duration": 0.0,
            "scene_count": 0,
            "filename": "",
        }
        if not mid:
            return info

        # Manifest
        manifest_file = settings.work_dir / mid / "manifest.json"
        if manifest_file.is_file():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    info["title"] = data.get("movie_title", mid)
                    info["duration"] = data.get("duration", 0.0)
                    info["filename"] = data.get("original_filename", "")
            except Exception:
                pass

        # Scene count
        db_file = settings.work_dir / mid / "scene_db.json"
        if db_file.is_file():
            try:
                with open(db_file, "r", encoding="utf-8") as f:
                    info["scene_count"] = len(json.load(f))
            except Exception:
                pass

        return info

    # ─── Health ───────────────────────────────────────────────

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "ReelForge AI Studio", "version": "2.0.0"}

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    # ─── Video Upload ─────────────────────────────────────────

    @app.post("/upload")
    async def upload_video(file: UploadFile = File(...)):
        """Upload a video file for processing."""
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Validate file type
        allowed_ext = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv', '.flv', '.m4v'}
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_ext:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(allowed_ext))}"
            )

        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / file.filename

        file_size = 0
        with open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                f.write(chunk)
                file_size += len(chunk)

        # Register job
        job = upload_mgr.register_upload(file.filename, file_size)

        return {
            "status": "uploaded",
            "job_id": job.job_id,
            "movie_id": job.movie_id,
            "filename": file.filename,
            "file_size": file_size,
            "message": f"Video uploaded successfully. Use /process/{job.job_id} to start scene detection.",
        }

    @app.post("/process/{job_id}")
    def process_video(job_id: str):
        """Start processing an uploaded video through the scene detection pipeline."""
        job = upload_mgr.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        upload_path = Path("uploads") / job.filename
        if not upload_path.is_file():
            raise HTTPException(status_code=404, detail="Uploaded video file not found")

        # Update job status
        upload_mgr.update_job(job_id, status="processing", progress=10.0)

        try:
            from ..workers.runner import PipelineRunner

            runner = PipelineRunner(upload_path, settings=settings)
            runner.run_all(allow_unverified_rights=True)

            upload_mgr.update_job(
                job_id,
                status="completed",
                progress=100.0,
                result={"movie_id": job.movie_id, "scene_db": str(settings.work_dir / job.movie_id / "scene_db.json")},
            )

            return {
                "status": "completed",
                "job_id": job_id,
                "movie_id": job.movie_id,
                "message": "Video processed successfully. Scene database is ready for reel generation.",
            }
        except Exception as e:
            upload_mgr.update_job(job_id, status="failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    @app.get("/jobs")
    def list_jobs():
        """List all upload/processing jobs."""
        jobs = upload_mgr.list_jobs()
        return {"jobs": [j.model_dump() for j in jobs]}

    @app.get("/jobs/{job_id}")
    def get_job_status(job_id: str):
        job = upload_mgr.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return job.model_dump()

    # ─── Indexed Videos ───────────────────────────────────────

    @app.get("/videos/indexed")
    def list_indexed_videos():
        """List all videos that have been indexed and are ready for reel generation."""
        indexed = upload_mgr.get_indexed_videos()
        return {"videos": indexed, "count": len(indexed)}

    # ─── Scenes ───────────────────────────────────────────────

    @app.get("/scenes/{movie_id}")
    def list_scenes(movie_id: str, category: Optional[str] = None, min_score: float = 0.5):
        db = _get_db(movie_id)
        if category:
            scenes = db.filter_scenes(category=category, min_score=min_score)
        else:
            scenes = db.list_scenes()
        return {"movie_id": movie_id, "scene_count": len(scenes), "scenes": [s.model_dump() for s in scenes]}

    # ─── RAG Query ────────────────────────────────────────────

    @app.post("/rag/query")
    def rag_query(req: RAGQueryRequest):
        mid = _resolve_movie_id(req.movie_id)
        db = _get_db(mid)
        retriever = HybridRetriever(db)
        return retriever.answer_question(req.query)

    # ─── Style Presets ────────────────────────────────────────

    @app.get("/styles")
    def get_styles():
        """Get available reel style presets."""
        return {"styles": CaptionEngine.get_style_presets()}

    # ─── Reel Planning ────────────────────────────────────────

    @app.post("/planner/generate")
    def generate_plan(req: PlanGenerateRequest):
        mid = _resolve_movie_id(req.movie_id)
        db = _get_db(mid)
        agent = ReelPlanningAgent(db)

        # Map style to internal category if provided
        if req.style:
            category = CaptionEngine.map_style_to_category(req.style)
            prompt = req.prompt or f"Create a {req.duration:.0f}-second {req.style} reel"
        else:
            prompt = req.prompt
            category = None

        plan = agent.plan_reel(
            prompt=prompt,
            target_duration=req.duration,
            include_shayari=req.include_shayari,
        )

        # Save to review queue
        queue_dir = Path("reels/review_queue")
        queue_dir.mkdir(parents=True, exist_ok=True)
        plan_path = queue_dir / f"{plan.plan_id}.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            f.write(plan.model_dump_json(indent=2))

        # Auto-generate caption
        style_name = CaptionEngine.map_category_to_style(plan.request.theme)
        movie_info = _get_movie_info(mid)
        caption = CaptionEngine.generate_caption(
            style=style_name,
            title=plan.title,
            movie_title=movie_info.get("title", ""),
            duration=plan.request.target_duration_seconds,
            dialogue_excerpt=plan.captions[0].text if plan.captions else "",
            beat_count=len(plan.selected_clips),
            music_direction=plan.music_direction,
        )

        return {
            "status": "queued_for_review",
            "plan_id": plan.plan_id,
            "title": plan.title,
            "plan": plan.model_dump(),
            "caption": caption.model_dump(),
        }

    @app.post("/auto/generate")
    def auto_generate(req: AutoGenerateRequest):
        """Auto-generate reel plans across selected styles."""
        mid = _resolve_movie_id(req.movie_id)
        db = _get_db(mid)
        generator = AutoReelBatchGenerator(db, output_queue_dir=Path("reels/review_queue"))

        results = generator.generate_all_categories(reels_per_category=req.reels_per_style)
        total = sum(len(plans) for plans in results.values())

        return {
            "status": "generated",
            "total_plans": total,
            "categories": {cat: [p.model_dump() for p in plans] for cat, plans in results.items()},
        }

    # ─── Queue ────────────────────────────────────────────────

    @app.get("/plans/queue")
    def list_queue():
        queue_dir = Path("reels/review_queue")
        if not queue_dir.is_dir():
            return {"queue_count": 0, "plans": []}
        plans = []
        for p in sorted(list(queue_dir.glob("plan_*.json"))):
            with open(p, "r", encoding="utf-8") as f:
                plans.append(json.load(f))
        return {"queue_count": len(plans), "plans": plans}

    @app.get("/plans/{plan_id}")
    def get_plan(plan_id: str):
        queue_file = Path("reels/review_queue") / f"{plan_id}.json"
        if not queue_file.is_file():
            queue_file = Path("reels") / f"{plan_id}.json"
        if not queue_file.is_file():
            raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found")
        with open(queue_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @app.post("/plans/{plan_id}/approve")
    def approve_and_render(plan_id: str, movie_id: Optional[str] = None):
        target_mid = _resolve_movie_id(movie_id or "")
        queue_file = Path("reels/review_queue") / f"{plan_id}.json"
        if not queue_file.is_file():
            queue_file = Path("reels") / f"{plan_id}.json"
        if not queue_file.is_file():
            raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found")

        with open(queue_file, "r", encoding="utf-8") as f:
            plan = ReelPlan(**json.load(f))
        plan.approved_by_user = True

        # Find the video file
        movie_file = None
        for ext in ['.mkv', '.mp4', '.avi', '.mov', '.webm']:
            candidates = list(Path(".").glob(f"*{ext}"))
            if candidates:
                movie_file = candidates[0]
                break
        # Also check uploads directory
        if not movie_file:
            for ext in ['.mkv', '.mp4', '.avi', '.mov', '.webm']:
                candidates = list(Path("uploads").glob(f"*{ext}")) if Path("uploads").is_dir() else []
                if candidates:
                    movie_file = candidates[0]
                    break

        if not movie_file:
            raise HTTPException(status_code=404, detail="Source video file not found")

        out_file = Path("reels") / f"{plan.plan_id}.mp4"
        render_reel_plan(
            input_path=movie_file,
            plan=plan,
            output_path=out_file,
            work_dir=settings.work_dir / "render_tmp",
        )

        return {
            "status": "rendered",
            "plan_id": plan_id,
            "output_file": str(out_file),
            "video_url": f"/videos/reel/{out_file.name}",
        }

    # ─── Evaluation ───────────────────────────────────────────

    @app.get("/eval/{plan_id}")
    def evaluate_plan(plan_id: str, movie_id: Optional[str] = None):
        target_mid = _resolve_movie_id(movie_id or "")
        queue_file = Path("reels/review_queue") / f"{plan_id}.json"
        if not queue_file.is_file():
            queue_file = Path("reels") / f"{plan_id}.json"
        if not queue_file.is_file():
            raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found")
        db = _get_db(target_mid)
        with open(queue_file, "r", encoding="utf-8") as f:
            plan = ReelPlan(**json.load(f))
        return ReelPlanRubricJudge.evaluate(plan, db)

    # ─── Caption Generation ───────────────────────────────────

    @app.post("/captions/generate")
    def generate_caption(req: CaptionRequest):
        """Generate optimized social media captions and hashtags."""
        caption = CaptionEngine.generate_caption(
            style=req.style,
            title=req.title,
            movie_title=req.movie_title,
            duration=req.duration,
            dialogue_excerpt=req.dialogue_excerpt,
            beat_count=req.beat_count,
            music_direction=req.music_direction,
        )
        return caption.model_dump()

    # ─── Video Serving ────────────────────────────────────────

    @app.get("/videos/reel/{filename}")
    def serve_reel_video(filename: str):
        file_path = settings.output_dir / filename
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="Video file not found")
        return FileResponse(file_path, media_type="video/mp4")

    # ─── Rights ───────────────────────────────────────────────

    @app.post("/rights/{source_id}")
    def update_rights(source_id: str, req: RightsUpdateRequest):
        info = rights_reg.set_rights(source_id=source_id, status=req.status, rights_holder=req.holder, notes=req.notes)
        return {"status": "updated", "rights": info.model_dump()}

    # ─── Review UI ────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    @app.get("/review", response_class=HTMLResponse)
    @app.get("/review/", response_class=HTMLResponse)
    def default_review_ui():
        movie_id = _get_default_movie_id()
        return review_ui(movie_id) if movie_id else review_ui_empty()

    @app.get("/review/{movie_id}", response_class=HTMLResponse)
    def review_ui(movie_id: str):
        info = _get_movie_info(movie_id)
        rights = rights_reg.get_rights(movie_id, info["title"])

        # Rendered reels
        reels_dir = settings.output_dir
        rendered_reels = []
        if reels_dir.is_dir():
            for p in sorted(list(reels_dir.glob("*.mp4"))):
                if p.stat().st_size > 1000:
                    rendered_reels.append(p.name)

        # Pending queue
        queue_dir = reels_dir / "review_queue"
        pending_plans = []
        if queue_dir.is_dir():
            for p in sorted(list(queue_dir.glob("plan_*.json"))):
                with open(p, "r", encoding="utf-8") as f:
                    pending_plans.append(json.load(f))

        # Indexed videos
        indexed = upload_mgr.get_indexed_videos()

        # Pipeline stages
        stages = {
            "ingest": "completed", "shots": "completed", "scenes": "completed",
            "audio": "completed", "video": "completed", "scoring": "completed",
            "selection": "completed", "render": "completed",
        }
        manifest_file = settings.work_dir / movie_id / "manifest.json"
        if manifest_file.is_file():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    stages = data.get("stages", stages)
            except Exception:
                pass

        # Style presets
        style_presets = CaptionEngine.get_style_presets()

        template_path = Path(__file__).parent / "templates" / "review.html"
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        html = (
            template.replace("__TITLE__", info["title"])
            .replace("__MOVIE_ID__", movie_id)
            .replace("__DURATION_MINS__", f"{info['duration'] / 60.0:.1f}")
            .replace("__DURATION_SECS__", f"{info['duration']:.1f}")
            .replace("__SCENE_COUNT__", str(info["scene_count"]))
            .replace("__RIGHTS_STATUS__", rights.source_status.upper())
            .replace("__QUEUE_COUNT__", str(len(pending_plans)))
            .replace("__RENDERED_COUNT__", str(len(rendered_reels)))
            .replace("__RENDERED_REELS_JSON__", json.dumps(rendered_reels))
            .replace("__PENDING_PLANS_JSON__", json.dumps(pending_plans))
            .replace("__STAGES_JSON__", json.dumps(stages))
            .replace("__INDEXED_VIDEOS_JSON__", json.dumps(indexed))
            .replace("__STYLE_PRESETS_JSON__", json.dumps(style_presets))
        )

        return html

    def review_ui_empty():
        """Render the UI when no video is indexed yet."""
        template_path = Path(__file__).parent / "templates" / "review.html"
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        html = (
            template.replace("__TITLE__", "No Video Loaded")
            .replace("__MOVIE_ID__", "")
            .replace("__DURATION_MINS__", "0.0")
            .replace("__DURATION_SECS__", "0.0")
            .replace("__SCENE_COUNT__", "0")
            .replace("__RIGHTS_STATUS__", "NONE")
            .replace("__QUEUE_COUNT__", "0")
            .replace("__RENDERED_COUNT__", "0")
            .replace("__RENDERED_REELS_JSON__", "[]")
            .replace("__PENDING_PLANS_JSON__", "[]")
            .replace("__STAGES_JSON__", json.dumps({}))
            .replace("__INDEXED_VIDEOS_JSON__", json.dumps([]))
            .replace("__STYLE_PRESETS_JSON__", json.dumps(CaptionEngine.get_style_presets()))
        )

        return html

    return app
