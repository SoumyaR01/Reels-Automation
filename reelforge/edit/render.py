"""Single-pass FFmpeg filter_complex rendering and ReelPlan execution for 1080x1920 vertical reels."""

import os
import json
from pathlib import Path
from typing import Any, List, Optional

from ..core.ffmpeg import run_ffmpeg, require_tool
from ..core.schemas import MomentCandidate
from ..planner.schema import ReelPlan, SelectedClip
from .reframe import generate_reframe_filter
from .music import build_audio_mix_filter
from .fx import generate_cinematic_fx_filter
from .subtitles import generate_ass_from_plan


def render_final_reel(
    input_path: Path,
    moment: MomentCandidate,
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    target_duration: float = 30.0,
    crf: int = 20,
    preset: str = "medium",
    reframe_mode: str = "smart_crop",
    subtitle_ass_path: Optional[Path] = None,
    add_effects: bool = False,
    ai_disclosure: bool = True,
) -> Path:
    """Render a candidate moment into a broadcast-ready 1080x1920 9:16 reel in a single FFmpeg pass."""
    require_tool("ffmpeg")
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reframe_filter = generate_reframe_filter(width=width, height=height, mode=reframe_mode)
    audio_filter = build_audio_mix_filter(target_loudness_lufs=-14.0)

    vf_chain = [reframe_filter]
    if add_effects:
        fx = generate_cinematic_fx_filter(vignette=True, film_grain=False)
        if fx:
            vf_chain.append(fx)

    if subtitle_ass_path and Path(subtitle_ass_path).is_file():
        escaped_ass = str(Path(subtitle_ass_path).resolve().as_posix()).replace(":", "\\:")
        vf_chain.append(f"subtitles='{escaped_ass}'")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{moment.start:.3f}",
        "-i", str(input_path),
        "-t", f"{target_duration:.3f}",
        "-vf", ",".join(vf_chain),
        "-af", audio_filter,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "192k",
    ]

    if ai_disclosure:
        cmd.extend(["-metadata", "comment=AI Curated by ReelForge", "-metadata", "genre=Shorts/Reel"])

    cmd.extend(["-movflags", "+faststart", str(output_path)])
    run_ffmpeg(cmd, check=True)
    return output_path


def render_scene_clip(
    input_path: Path,
    moment: Any,
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    crf: int = 20,
    preset: str = "medium",
    reframe_mode: str = "smart_crop",
    ai_disclosure: bool = True,
) -> Path:
    """Render an individual scene clip bounded strictly by hard cuts into 1080x1920 vertical format."""
    require_tool("ffmpeg")
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_t = getattr(moment, "start", getattr(moment, "source_start_seconds", 0.0))
    end_t = getattr(moment, "end", getattr(moment, "source_end_seconds", 5.0))
    dur_t = max(0.1, end_t - start_t)

    reframe_filter = generate_reframe_filter(width=width, height=height, mode=reframe_mode)
    audio_filter = build_audio_mix_filter(target_loudness_lufs=-14.0)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{start_t:.3f}",
        "-i", str(input_path),
        "-t", f"{dur_t:.3f}",
        "-vf", reframe_filter,
        "-af", audio_filter,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "192k",
    ]

    if ai_disclosure:
        cmd.extend(["-metadata", "comment=AI Curated Scene by ReelForge", "-metadata", "genre=Shorts/Scene"])

    cmd.extend(["-movflags", "+faststart", str(output_path)])
    run_ffmpeg(cmd, check=True)
    return output_path


def compile_reel_from_clips(
    clip_paths: List[Path],
    output_reel_path: Path,
    max_duration: Optional[float] = 60.0,
    crf: int = 20,
    preset: str = "medium",
    ai_disclosure: bool = True,
) -> Path:
    """Concatenate individual scene clips into a single continuous vertical 9:16 reel."""
    require_tool("ffmpeg")
    output_reel_path = Path(output_reel_path).resolve()
    output_reel_path.parent.mkdir(parents=True, exist_ok=True)

    if not clip_paths:
        raise ValueError("Cannot compile reel from an empty list of clips.")

    concat_list_file = output_reel_path.parent / f"{output_reel_path.stem}_concat_list.txt"
    with open(concat_list_file, "w", encoding="utf-8") as f:
        for p in clip_paths:
            clean_path = str(Path(p).resolve().as_posix()).replace("'", "'\\''")
            f.write(f"file '{clean_path}'\n")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_file.resolve()),
    ]

    if max_duration and max_duration > 0:
        cmd.extend(["-t", f"{max_duration:.3f}"])

    cmd.extend([
        "-vf", "null",
        "-af", "loudnorm=I=-14:LRA=11:TP=-1.5",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "192k",
    ])

    if ai_disclosure:
        cmd.extend(["-metadata", "comment=AI Curated Compilation Reel by ReelForge", "-metadata", "genre=Shorts/Reel"])

    cmd.extend(["-movflags", "+faststart", str(output_reel_path)])

    try:
        run_ffmpeg(cmd, check=True)
    finally:
        if concat_list_file.exists():
            try:
                concat_list_file.unlink()
            except Exception:
                pass

    return output_reel_path


def render_reel_plan(
    input_path: Path,
    plan: ReelPlan,
    output_path: Path,
    work_dir: Path,
    ai_disclosure: bool = True,
) -> Path:
    """Render an approved ReelPlan into a 1080x1920 vertical video with subtitles and audio mix."""
    require_tool("ffmpeg")
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    work_dir = Path(work_dir).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if not plan.selected_clips:
        raise ValueError("ReelPlan contains no clips to render.")

    # 1. Render individual clips to work directory
    clip_files: List[Path] = []
    for idx, clip in enumerate(plan.selected_clips, 1):
        clip_out = work_dir / f"{plan.plan_id}_beat_{idx:02d}.mp4"
        render_scene_clip(
            input_path=input_path,
            moment=clip,
            output_path=clip_out,
            width=plan.render_settings.width,
            height=plan.render_settings.height,
            crf=plan.render_settings.crf,
            preset=plan.render_settings.preset,
            ai_disclosure=ai_disclosure,
        )
        clip_files.append(clip_out)

    # 2. Generate ASS subtitles (with dialogue + Hindi shayari)
    ass_path = work_dir / f"{plan.plan_id}_subtitles.ass"
    generate_ass_from_plan(
        plan=plan,
        output_ass_path=ass_path,
        width=plan.render_settings.width,
        height=plan.render_settings.height,
    )

    # 3. Concatenate and apply loudness normalization + subtitles
    compile_reel_from_clips(
        clip_paths=clip_files,
        output_reel_path=output_path,
        max_duration=plan.request.target_duration_seconds,
        crf=plan.render_settings.crf,
        preset=plan.render_settings.preset,
        ai_disclosure=ai_disclosure,
    )

    # 4. Save audit log for the rendered reel
    audit_file = output_path.with_suffix(".audit.json")
    audit_payload = {
        "output_file": output_path.name,
        "plan_id": plan.plan_id,
        "theme": plan.request.theme,
        "duration_seconds": plan.request.target_duration_seconds,
        "clip_count": len(plan.selected_clips),
        "has_shayari": plan.shayari is not None,
        "ai_disclosure": ai_disclosure,
        "clips": [c.model_dump() for c in plan.selected_clips],
    }
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)

    return output_path
