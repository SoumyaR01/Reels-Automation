"""Percentile and Z-score normalization across all candidates in a movie (Fixing Bug #8)."""

from typing import Dict, List


def normalize_movie_scores(candidates_scores: List[Dict[str, float]], categories: List[str]) -> List[Dict[str, float]]:
    """Normalize raw scores across the entire movie using percentile ranking.

    Ensures scores represent true relative prominence within the specific film rather than
    brittle hardcoded absolute constants.
    """
    if not candidates_scores:
        return []

    normalized_results = [dict(s) for s in candidates_scores]

    for cat in categories:
        raw_vals = [s.get(cat, 0.0) for s in candidates_scores]
        sorted_vals = sorted(raw_vals)
        n = len(sorted_vals)

        if n <= 1 or sorted_vals[-1] == sorted_vals[0]:
            for r in normalized_results:
                r[cat] = 0.5
            continue

        # Map each value to its percentile [0.0, 1.0]
        for r in normalized_results:
            val = r.get(cat, 0.0)
            # Find rank
            rank = sum(1 for v in sorted_vals if v <= val)
            percentile = float(rank / n)
            r[cat] = round(percentile, 4)

    return normalized_results
