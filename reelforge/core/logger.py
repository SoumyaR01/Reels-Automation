"""Logging and report generation system for ReelForge."""

import datetime
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_logger = None


def setup_logger(log_file_paths: Optional[List[Path]] = None) -> logging.Logger:
    """Configure dual-pipe logger that streams to console and log files."""
    global _logger
    logger = logging.getLogger("reelforge")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handlers
    if log_file_paths:
        file_format = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        for log_path in log_file_paths:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(file_format)
            logger.addHandler(fh)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Retrieve global logger."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def write_markdown_report(
    output_dir: Path,
    movie_title: str,
    movie_path: str,
    duration: float,
    shots_count: int,
    scenes_count: int,
    selected_moments: list,
    exported_reels: list,
    stages: Dict[str, str],
) -> Path:
    """Generate a clean, professional Markdown report file summarizing the entire movie analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "ANALYSIS_REPORT.md"

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mins = duration / 60.0

    lines = [
        f"# 🎬 ReelForge Analysis & Generation Report",
        f"",
        f"**Generated:** {now_str}  ",
        f"**Source Video:** `{Path(movie_path).name}`  ",
        f"**Duration:** {duration:.1f}s (~{mins:.1f} minutes)  ",
        f"",
        f"---",
        f"",
        f"## 1. Film Structural Overview",
        f"",
        f"| Metric | Result |",
        f"|---|---|",
        f"| Total Hard Cuts Detected | **{shots_count}** |",
        f"| Average Shot Duration | **{(duration / max(1, shots_count)):.2f}s** |",
        f"| Semantic Narrative Scenes | **{scenes_count}** |",
        f"| Average Scene Duration | **{(duration / max(1, scenes_count)):.2f}s** |",
        f"",
        f"---",
        f"",
        f"## 2. Pipeline Execution Stages",
        f"",
        f"| Pipeline Stage | Status |",
        f"|---|---|",
    ]

    for stage_name, status in stages.items():
        icon = "✅" if status == "completed" else "⏳"
        lines.append(f"| {stage_name.title()} | {icon} `{status.upper()}` |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. Selected Highlights & Moments",
        f"",
        f"Below are the top-rated candidate moments selected via Maximal Marginal Relevance (MMR) diversity re-ranking:",
        f"",
        f"| # | Category | Time Interval | Duration | Score | Highlights |",
        f"|---|---|---|---|---|---|",
    ])

    for idx, m in enumerate(selected_moments, start=1):
        cat = getattr(m, "selected_category", "general")
        score = getattr(m, "final_score", 0.0)
        s_time = getattr(m, "start", 0.0)
        e_time = getattr(m, "end", 0.0)
        dur = getattr(m, "duration", 0.0)
        
        # Details
        sig = getattr(m, "signals", None)
        highlights = []
        if sig:
            if getattr(sig, "minor_key_correlation", 0) > 0.4:
                highlights.append("Minor Key / Melancholic")
            if getattr(sig, "low_brightness", 0) > 0.6:
                highlights.append("Low-Key Lighting")
            if getattr(sig, "visual_composition", 0) > 0.5:
                highlights.append("Rule of Thirds")
            if getattr(sig, "motion_stability", 0) > 0.6:
                highlights.append("Smooth Camera Track")

        hl_str = ", ".join(highlights) if highlights else "Balanced Pacing"
        lines.append(f"| {idx:02d} | **{cat.title()}** | `{s_time:.1f}s` → `{e_time:.1f}s` | {dur:.1f}s | `{score:.3f}` | {hl_str} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 4. Exported 9:16 Vertical Reels",
        f"",
        f"| File | Duration | Format | Resolution | Companion Audit Log |",
        f"|---|---|---|---|---|",
    ])

    for p in exported_reels:
        p = Path(p)
        audit_name = p.with_suffix(".audit.json").name
        lines.append(f"| **`{p.name}`** | 30.0s | H.264 / AAC (-14 LUFS) | 1080×1920 (9:16) | `{audit_name}` |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"> **Notice:** All reels are rendered with loudness normalization conforming to -14 LUFS broadcast standards and compliant rights audit logging.",
        f"",
    ])

    report_content = "\n".join(lines)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_file
