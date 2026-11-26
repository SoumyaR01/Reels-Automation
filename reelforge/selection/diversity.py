"""Diversity re-ranking using Maximal Marginal Relevance (MMR) and category quotas."""

from typing import Dict, List
from ..core.schemas import MomentCandidate


def temporal_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Compute temporal Intersection-over-Union (IoU)."""
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return float(inter / (union + 1e-8))


def select_diverse_moments(
    candidates: List[MomentCandidate],
    category_quotas: Dict[str, int],
    max_overlap_iou: float = 0.20,
) -> List[MomentCandidate]:
    """Select high-scoring, non-redundant moments across designated category quotas."""
    selected_all: List[MomentCandidate] = []

    for cat, quota in category_quotas.items():
        # Filter and rank candidates by this category's score
        cat_candidates = [c for c in candidates if cat in c.labels]
        cat_candidates.sort(key=lambda c: c.labels.get(cat, 0.0), reverse=True)

        selected_for_cat: List[MomentCandidate] = []
        for cand in cat_candidates:
            # Check overlap against all already selected moments across all categories
            overlaps = any(
                temporal_overlap(cand.start, cand.end, sel.start, sel.end) > max_overlap_iou
                for sel in selected_all
            )
            if not overlaps:
                cand_copy = cand.model_copy()
                cand_copy.selected_category = cat
                cand_copy.final_score = cand.labels.get(cat, 0.5)
                selected_for_cat.append(cand_copy)
                selected_all.append(cand_copy)
                if len(selected_for_cat) >= quota:
                    break

    # Sort final moments chronologically for review
    selected_all.sort(key=lambda c: c.start)
    return selected_all
