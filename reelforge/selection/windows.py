"""Generate candidate 25-35s moment windows snapped strictly to shot cut boundaries."""

from typing import List
from ..core.schemas import MomentCandidate, ShotData, SignalMetrics, DialogueData, RightsInfo


def generate_candidate_windows(
    shots: List[ShotData],
    movie_id: str,
    min_duration: float = 25.0,
    max_duration: float = 35.0,
) -> List[MomentCandidate]:
    """Extract candidate moment windows that start at a hard cut and end at a hard cut (no mid-shot chops)."""
    candidates: List[MomentCandidate] = []
    num_shots = len(shots)

    for i in range(num_shots):
        start_t = shots[i].start
        for j in range(i, num_shots):
            end_t = shots[j].end
            dur = end_t - start_t
            if min_duration <= dur <= max_duration:
                m_id = f"m_{movie_id}_{len(candidates)+1:04d}"
                candidates.append(
                    MomentCandidate(
                        movie_id=movie_id,
                        moment_id=m_id,
                        scene_id=f"scene_s{shots[i].shot_id}_s{shots[j].shot_id}",
                        start=round(start_t, 3),
                        end=round(end_t, 3),
                        duration=round(dur, 3),
                        signals=SignalMetrics(),
                        dialogue=DialogueData(),
                        rights=RightsInfo(source_id=movie_id, source_title=movie_id),
                    )
                )
            elif dur > max_duration:
                break

    return candidates
