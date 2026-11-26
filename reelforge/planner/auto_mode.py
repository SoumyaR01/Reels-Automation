"""Auto-Reel Batch Mode (Section 2A & Phase 5A): Generates diverse 30s reel plans across all 6 categories."""

from pathlib import Path
from typing import Dict, List, Optional
import json

from .schema import ReelPlan
from .agent import ReelPlanningAgent
from ..rag.db import SceneDatabase


class AutoReelBatchGenerator:
    """Generates distinct, diverse 30-second vertical reel plans across all six core categories."""

    CATEGORIES = [
        "sad",
        "action",
        "powerful_dialogue",
        "romantic",
        "motivational",
        "suspense",
    ]

    CATEGORY_PROMPTS = {
        "sad": "Create a poignant 30-second emotional reel capturing heartbreak, loss, and quiet sorrow.",
        "action": "Create an intense 30-second high-octane action reel with fast cuts, high energy, and thrilling combat.",
        "powerful_dialogue": "Create a 30-second reel featuring the most impactful dialogue, memorable lines, and dramatic delivery.",
        "romantic": "Create a 30-second heartwarming romantic reel with tender moments and gentle music.",
        "motivational": "Create an inspirational 30-second motivational reel about overcoming failure, perseverance, and triumph.",
        "suspense": "Create a gripping 30-second suspenseful thriller reel with dark atmosphere, rising mystery, and sudden tension.",
    }

    def __init__(self, db: SceneDatabase, output_queue_dir: Optional[Path] = None):
        self.db = db
        self.agent = ReelPlanningAgent(db)
        self.queue_dir = Path(output_queue_dir) if output_queue_dir else Path("reels/review_queue")

    def generate_all_categories(
        self,
        reels_per_category: int = 2,
        target_duration: float = 30.0,
    ) -> Dict[str, List[ReelPlan]]:
        """Generate distinct 30s reel plans across all six categories and store in review queue."""
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        results: Dict[str, List[ReelPlan]] = {}

        for cat in self.CATEGORIES:
            cat_plans: List[ReelPlan] = []
            prompt = self.CATEGORY_PROMPTS.get(cat, f"Create a 30-second {cat} reel.")

            # Filter top candidate scenes for this category
            top_scenes = self.db.filter_scenes(category=cat, min_score=0.5)
            if not top_scenes:
                top_scenes = self.db.list_scenes()

            # Generate multiple distinct plans per category with staggered scene selections
            for plan_num in range(1, reels_per_category + 1):
                # Stagger prompt slightly to yield diverse scenes
                custom_prompt = f"{prompt} (Variant {plan_num})"
                plan = self.agent.plan_reel(
                    prompt=custom_prompt,
                    target_duration=target_duration,
                    include_shayari=False,
                )
                plan.plan_id = f"plan_{cat}_{plan_num:02d}"
                plan.title = f"{cat.replace('_', ' ').title()} Reel #{plan_num:02d}"

                # Save plan to review queue directory
                plan_file = self.queue_dir / f"{plan.plan_id}.json"
                with open(plan_file, "w", encoding="utf-8") as f:
                    f.write(plan.model_dump_json(indent=2))

                cat_plans.append(plan)

            results[cat] = cat_plans

        return results

    def list_review_queue(self) -> List[Path]:
        """List all pending reel plans awaiting review in the review queue."""
        if not self.queue_dir.is_dir():
            return []
        return sorted(list(self.queue_dir.glob("plan_*.json")))
