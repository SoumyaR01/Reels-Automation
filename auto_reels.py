"""ReelForge: Production AI Movie Analysis & 9:16 Vertical Reel Studio.

CLI Wrapper and entrypoint running the full ReelForge engine:
- Ingest & 360p Proxy/WAV caching
- TransNetV2 / Adaptive frame-accurate shot detection
- Semantic environment scene grouping
- Continuous acoustic & rolling tonality extraction (0.5s windows, 5-10s key analysis)
- Dense aesthetics (rule-of-thirds, teal/orange, film desaturation, motion stability)
- Multimodal percentile fusion scoring across categories (sad, cinematic, etc.)
- 25-35s candidate moment windows snapped to cuts
- MMR diversity re-ranking & category quotas
- Rights compliance gate & audit logging
- Single-pass FFmpeg vertical 1080x1920 reel rendering trimmed to exact duration
"""

import argparse
import sys
from pathlib import Path

from reelforge.core.config import get_settings
from reelforge.workers.runner import PipelineRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="ReelForge: Production AI Movie Analysis & 9:16 Vertical Reel Studio."
    )
    parser.add_argument("input", type=Path, help="Path to input movie file")
    parser.add_argument("--output-dir", type=Path, default=Path("reels"), help="Output directory for reels")
    parser.add_argument("--work-dir", type=Path, default=Path("workdir"), help="Work cache directory for proxies & manifests")
    parser.add_argument("--reel-duration", type=float, default=30.0, help="Target duration of each compiled reel in seconds")
    parser.add_argument("--sad-quota", type=int, default=1, help="Number of sad reels to compile (default: 1)")
    parser.add_argument("--cinematic-quota", type=int, default=1, help="Number of cinematic reels to compile (default: 1)")
    parser.add_argument("--crf", type=int, default=20, help="H.264 quality factor")
    parser.add_argument("--preset", default="medium", help="FFmpeg encoding preset")
    parser.add_argument(
        "--allow-unverified-rights",
        action="store_true",
        default=True,  # Default True for local developer workflows; can be set False for production publishing gate
        help="Permit rendering without verified rights clearance in registry",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.is_file():
        print(f"Error: Video file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    settings = get_settings()
    settings.output_dir = args.output_dir
    settings.work_dir = args.work_dir
    settings.target_duration = args.reel_duration
    settings.crf = args.crf
    settings.preset = args.preset

    quotas = {
        "sad": args.sad_quota,
        "cinematic": args.cinematic_quota,
    }

    runner = PipelineRunner(args.input, settings=settings)
    runner.run_all(
        category_quotas=quotas,
        allow_unverified_rights=args.allow_unverified_rights,
    )


if __name__ == "__main__":
    main()
