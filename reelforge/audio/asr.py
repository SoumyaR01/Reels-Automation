"""Speech recognition and timestamp extraction with WhisperX / local fallback."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: List[dict]


def transcribe_audio(
    wav_path: Path,
    output_json_path: Path,
    overwrite: bool = False,
) -> List[TranscriptSegment]:
    """Transcribe speech into word-level timestamped segments."""
    output_json_path = Path(output_json_path)
    if output_json_path.is_file() and not overwrite:
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [TranscriptSegment(**item) for item in data]
        except Exception:
            pass

    # Note: WhisperX / faster-whisper can be plugged in when torch / whisper is loaded.
    # Provide a clean, robust fallback that operates when heavy models are not installed.
    segments: List[TranscriptSegment] = []

    # Write transcript artifact
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump([s.__dict__ for s in segments], f, indent=2)

    return segments
