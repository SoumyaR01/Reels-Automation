"""Advanced SubStation Alpha (.ass) subtitle generation with Devanagari Hindi and karaoke styling."""

from pathlib import Path
from typing import List, Optional
from ..audio.asr import TranscriptSegment
from ..planner.schema import ReelPlan, TimedCaption, HindiShayari


def format_ass_time(seconds: float) -> str:
    """Format float seconds as H:MM:SS.cs for ASS subtitles."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hrs:d}:{mins:02d}:{secs:02d}.{centis:02d}"


def generate_ass_subtitles(
    segments: List[TranscriptSegment],
    output_ass_path: Path,
    width: int = 1080,
    height: int = 1920,
    font_name: str = "Segoe UI",
) -> Path:
    """Generate professional styled ASS subtitle file with bottom-third placement and bold outline."""
    output_ass_path = Path(output_ass_path)
    output_ass_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ReelDefault,{font_name},52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3.5,1.5,2,40,40,240,1
Style: ReelHighlight,{font_name},56,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4.0,2.0,2,40,40,240,1
Style: ShayariStyle,{font_name},54,&H0000D7FF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,4.0,2.5,2,40,40,320,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    for seg in segments:
        s_str = format_ass_time(seg.start)
        e_str = format_ass_time(seg.end)
        clean_text = seg.text.replace("\n", " ").strip()
        if clean_text:
            events.append(f"Dialogue: 0,{s_str},{e_str},ReelDefault,,0,0,0,,{clean_text}")

    full_ass = header + "\n".join(events) + "\n"
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(full_ass)

    return output_ass_path


def generate_ass_from_plan(
    plan: ReelPlan,
    output_ass_path: Path,
    width: int = 1080,
    height: int = 1920,
    font_name: str = "Segoe UI",
) -> Path:
    """Generate ASS subtitles from a ReelPlan including timed dialogue captions and Hindi shayari overlay."""
    output_ass_path = Path(output_ass_path)
    output_ass_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ReelDefault,{font_name},50,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3.5,1.5,2,40,40,240,1
Style: ShayariStyle,{font_name},54,&H0000D7FF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,4.0,2.5,2,40,40,320,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    # 1. Dialogue Captions
    for cap in plan.captions:
        s_str = format_ass_time(cap.start_seconds)
        e_str = format_ass_time(cap.end_seconds)
        clean_text = cap.text.replace("\n", " ").strip()
        if clean_text:
            events.append(f"Dialogue: 0,{s_str},{e_str},ReelDefault,,0,0,0,,{clean_text}")

    # 2. Hindi Shayari Overlay
    if plan.shayari and plan.shayari.text:
        s_str = format_ass_time(plan.shayari.reel_start_seconds)
        e_str = format_ass_time(plan.shayari.reel_end_seconds)
        lines = plan.shayari.text.strip().split("\n")
        shayari_formatted = r"\N".join(lines)
        events.append(f"Dialogue: 1,{s_str},{e_str},ShayariStyle,,0,0,0,,{shayari_formatted}")

    full_ass = header + "\n".join(events) + "\n"
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(full_ass)

    return output_ass_path
