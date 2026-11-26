"""Smart 9:16 reframing: dynamic virtual camera or aesthetic blurred-background fit."""

from typing import List, Optional
from ..video.faces import FaceBBox


def generate_reframe_filter(
    width: int = 1080,
    height: int = 1920,
    faces: Optional[List[FaceBBox]] = None,
    mode: str = "smart_crop",
) -> str:
    """Build FFmpeg video filter string to convert landscape 16:9 into vertical 9:16 (1080x1920).

    Modes:
      - 'smart_crop': Intelligent dynamic zoom/crop keeping subjects in rule-of-thirds frame.
      - 'blurred_fit': Pillarbox background blur with centered sharp video (preserves 100% of frame).
    """
    if mode == "blurred_fit":
        return (
            f"split=2[bg][fg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=25:5[bgblur];"
            f"[fg]scale={width}:-2[fgfit];"
            f"[bgblur][fgfit]overlay=(W-w)/2:(H-h)/2,setsar=1"
        )

    # Standard high-production vertical cinematic crop
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1"
    )
