"""Pipeline stage orchestrator with idempotent caching and resume support."""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from ..core.config import Settings, get_settings
from ..core.ffmpeg import probe_video
from ..core.logger import setup_logger, write_markdown_report
from ..core.manifest import ManifestManager
from ..core.schemas import MomentCandidate, ShotData, SceneData
from ..ingest.proxy import generate_video_proxy
from ..ingest.audio_extract import extract_normalized_audio
from ..shots.transnet import detect_shots
from ..scenes.grouping import group_shots_into_scenes
from ..audio.features import extract_audio_features
from ..audio.asr import transcribe_audio
from ..video.aesthetics import extract_video_aesthetics
from ..video.faces import detect_faces
from ..scoring.heuristics import compute_scene_heuristic_scores
from ..scoring.fusion import normalize_movie_scores
from ..scoring.llm_judge import evaluate_moment_narrative
from ..selection.windows import generate_candidate_windows
from ..selection.diversity import select_diverse_moments
from ..edit.render import render_final_reel
from ..rights.registry import RightsRegistry, assert_export_allowed, record_reel_audit_log


class PipelineRunner:
    """Orchestrates end-to-end execution across stages 0 through 10 with idempotent caching."""

    def __init__(self, movie_path: Path, settings: Optional[Settings] = None):
        self.movie_path = Path(movie_path).resolve()
        self.settings = settings or get_settings()
        self.manifest = ManifestManager(self.movie_path, self.settings.work_dir)
        self.rights_reg = RightsRegistry(self.settings.work_dir / "rights_registry.json")

        # Configure dual-pipe logging to console and persistent log files
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.settings.output_dir / "reelforge.log"
        self.work_log_file = self.manifest.job_dir / "reelforge.log"
        self.logger = setup_logger([self.log_file, self.work_log_file])

    def log(self, message: str):
        """Log message to both terminal and persistent log file."""
        self.logger.info(message)

    def run_all(
        self,
        category_quotas: Optional[Dict[str, int]] = None,
        allow_unverified_rights: bool = False,
    ) -> List[Path]:
        """Execute full end-to-end pipeline and return paths to generated reels."""
        quotas = category_quotas or {"sad": 1, "cinematic": 1}
        self.log(f"=== ReelForge Engine: Processing '{self.movie_path.name}' ===")

        # Stage 0: Ingest
        probe, proxy_path, wav_path = self.stage_ingest()

        # Stage 1: Shots
        shots = self.stage_shots(proxy_path, probe.duration)

        # Stage 2: Scenes
        scenes = self.stage_scenes(shots, proxy_path)

        # Stage 3: Audio AI & ASR
        audio_res, transcripts = self.stage_audio(wav_path)

        # Stage 4: Visual AI & Aesthetics
        video_res, faces = self.stage_video(proxy_path)

        # Stage 5 & 6: Moment Extraction & Scoring
        scored_moments = self.stage_scoring(shots, scenes, audio_res, video_res, transcripts)

        # Stage 7: Moment Selection (MMR diversity + category quotas)
        selected_moments = self.stage_selection(scored_moments, quotas)

        # Stage 8, 9 & 10: Rights Check, Single-Pass Render & Audit Log
        exported_reels = self.stage_render(selected_moments, allow_unverified=allow_unverified_rights)

        # Generate Comprehensive Analysis Report File
        report_path = write_markdown_report(
            output_dir=self.settings.output_dir,
            movie_title=self.manifest.manifest.movie_title,
            movie_path=str(self.movie_path),
            duration=probe.duration,
            shots_count=len(shots),
            scenes_count=len(scenes),
            selected_moments=selected_moments,
            exported_reels=exported_reels,
            stages=self.manifest.manifest.stages,
        )
        try:
            shutil.copy2(report_path, self.manifest.job_dir / "ANALYSIS_REPORT.md")
        except Exception:
            pass

        self.log(f"\n[Report Generated] Saved report: {report_path.resolve()}")
        self.log(f"[Execution Log] Saved log:    {self.log_file.resolve()}")
        self.log(f"\n=== ReelForge Finished: {len(exported_reels)} reels exported successfully! ===")
        return exported_reels

    def stage_ingest(self):
        self.log("\n[Stage 0: Ingest] Probing video and generating proxy / audio cache...")
        probe = probe_video(self.movie_path)
        self.manifest.set_duration(probe.duration)

        proxy_path = self.manifest.job_dir / "proxy_360p.mp4"
        wav_path = self.manifest.job_dir / "audio_16k.wav"

        if not self.manifest.is_stage_completed("ingest") or not proxy_path.exists() or not wav_path.exists():
            self.manifest.mark_stage_started("ingest")
            generate_video_proxy(self.movie_path, proxy_path, self.settings)
            extract_normalized_audio(self.movie_path, wav_path, self.settings.audio_sample_rate)
            self.manifest.mark_stage_completed("ingest", {
                "proxy_video": str(proxy_path),
                "audio_wav": str(wav_path),
            })
        else:
            self.log("  (Reusing cached proxy and audio from manifest)")

        return probe, proxy_path, wav_path

    def stage_shots(self, proxy_path: Path, duration: float) -> List[ShotData]:
        self.log("\n[Stage 1: Shots] Frame-accurate adaptive cut detection...")
        shots_json = self.manifest.job_dir / "shots.json"

        if not self.manifest.is_stage_completed("shots") or not shots_json.exists():
            self.manifest.mark_stage_started("shots")
            shots = detect_shots(proxy_path, shots_json, duration, fps=self.settings.proxy_fps)
            self.manifest.mark_stage_completed("shots", {"shots_json": str(shots_json)})
        else:
            self.log("  (Reusing cached shots.json)")
            with open(shots_json, "r", encoding="utf-8") as f:
                shots = [ShotData(**item) for item in json.load(f)]

        self.log(f"  Detected {len(shots)} shots across film.")
        return shots

    def stage_scenes(self, shots: List[ShotData], proxy_path: Path) -> List[SceneData]:
        self.log("\n[Stage 2: Scenes] Semantic environment shot clustering...")
        scenes_json = self.manifest.job_dir / "scenes.json"

        if not self.manifest.is_stage_completed("scenes") or not scenes_json.exists():
            self.manifest.mark_stage_started("scenes")
            scenes = group_shots_into_scenes(shots, proxy_path, scenes_json, fps=self.settings.proxy_fps)
            self.manifest.mark_stage_completed("scenes", {"scenes_json": str(scenes_json)})
        else:
            self.log("  (Reusing cached scenes.json)")
            with open(scenes_json, "r", encoding="utf-8") as f:
                scenes = [SceneData(**item) for item in json.load(f)]

        self.log(f"  Formed {len(scenes)} cohesive narrative scenes.")
        return scenes

    def stage_audio(self, wav_path: Path):
        self.log("\n[Stage 3: Audio AI] Continuous acoustic signal & rolling tonality extraction...")
        audio_res = extract_audio_features(wav_path, self.settings.audio_sample_rate)
        transcript_json = self.manifest.job_dir / "transcript.json"
        transcripts = transcribe_audio(wav_path, transcript_json)
        self.manifest.mark_stage_completed("audio")
        return audio_res, transcripts

    def stage_video(self, proxy_path: Path):
        self.log("\n[Stage 4: Visual AI] Dense visual aesthetics and face tracking...")
        video_res = extract_video_aesthetics(proxy_path, fps=self.settings.proxy_fps)
        faces = detect_faces(proxy_path, sample_interval_sec=1.0)
        self.manifest.mark_stage_completed("video")
        return video_res, faces

    def stage_scoring(self, shots, scenes, audio_res, video_res, transcripts) -> List[MomentCandidate]:
        self.log("\n[Stage 5: Scoring] Candidate 25-35s moment generation and percentile fusion...")
        moments = generate_candidate_windows(
            shots,
            self.manifest.movie_id,
            min_duration=self.settings.min_moment_duration,
            max_duration=self.settings.max_moment_duration,
        )
        self.log(f"  Formed {len(moments)} candidate moment windows.")

        if not moments:
            # Fallback for short clips
            total_dur = shots[-1].end if shots else 30.0
            moments = [
                MomentCandidate(
                    movie_id=self.manifest.movie_id,
                    moment_id=f"m_{self.manifest.movie_id}_0001",
                    scene_id="scene_0001",
                    start=0.0,
                    end=min(30.0, total_dur),
                    duration=min(30.0, total_dur),
                )
            ]

        # Compute multi-signal scores
        raw_scores = []
        for m in moments:
            dummy_scene = SceneData(scene_id=m.scene_id, shot_ids=[], start=m.start, end=m.end, duration=m.duration)
            s_durations = [s.duration for s in shots if s.start >= m.start and s.end <= m.end]
            scores = compute_scene_heuristic_scores(dummy_scene, audio_res, video_res, s_durations)
            raw_scores.append(scores)

        # Normalize per movie using percentile ranking (Fixing Bug #8)
        norm_scores = normalize_movie_scores(raw_scores, self.settings.categories)
        for m, s in zip(moments, norm_scores):
            m.labels = s

        self.manifest.mark_stage_completed("scoring")
        return moments

    def stage_selection(self, moments: List[MomentCandidate], quotas: Dict[str, int]) -> List[MomentCandidate]:
        self.log("\n[Stage 7: Moment Selection] Maximal Marginal Relevance (MMR) & category quotas...")
        selected = select_diverse_moments(moments, quotas, max_overlap_iou=0.20)
        
        # Attach rights
        rights_info = self.rights_reg.get_rights(self.manifest.movie_id, self.manifest.manifest.movie_title)
        for m in selected:
            m.rights = rights_info

        self.log(f"  Selected {len(selected)} diverse moments for production export:")
        for idx, m in enumerate(selected, 1):
            self.log(f"    #{idx:02d} [{m.start:6.1f}s - {m.end:6.1f}s] ({m.duration:4.1f}s) | Category: {m.selected_category:12s} | Score: {m.final_score:.3f}")

        self.manifest.mark_stage_completed("selection")
        return selected

    def stage_render(self, moments: List[MomentCandidate], allow_unverified: bool = False) -> List[Path]:
        self.log("\n[Stage 9: Render & Export] Single-pass 1080x1920 vertical reel compilation...")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        rendered_paths = []

        for idx, m in enumerate(moments, 1):
            if self.settings.enforce_rights_gate:
                assert_export_allowed(m.rights, allow_unverified=allow_unverified)

            out_filename = f"{m.selected_category}_reel_{idx:02d}.mp4"
            if len([x for x in moments if x.selected_category == m.selected_category]) == 1:
                out_filename = f"{m.selected_category}_reel.mp4"

            out_path = self.settings.output_dir / out_filename
            self.log(f"  Rendering {out_path.name} ({m.start:.1f}s - {m.end:.1f}s, exact {self.settings.target_duration:.1f}s)...")

            render_final_reel(
                self.movie_path,
                m,
                out_path,
                width=self.settings.reel_width,
                height=self.settings.reel_height,
                target_duration=self.settings.target_duration,
                crf=self.settings.crf,
                preset=self.settings.preset,
                reframe_mode="smart_crop",
                ai_disclosure=self.settings.ai_disclosure_label,
            )

            record_reel_audit_log(out_path, m)
            rendered_paths.append(out_path)

        self.manifest.mark_stage_completed("render")
        return rendered_paths
