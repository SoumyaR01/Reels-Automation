"""Rights verification gate and publishing compliance audit registry (Section 6)."""

import datetime
import json
from pathlib import Path
from typing import Optional

from ..core.schemas import MomentCandidate, RightsInfo


class RightsRegistry:
    """Manages legal source clearance status and audit compliance."""

    def __init__(self, registry_file: Optional[Path] = None):
        self.registry_file = Path(registry_file) if registry_file else Path("workdir/rights_registry.json")
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict:
        if self.registry_file.is_file():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save(self):
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def get_rights(self, source_id: str, title: str = "") -> RightsInfo:
        if source_id in self.data:
            return RightsInfo(**self.data[source_id])
        return RightsInfo(source_id=source_id, source_title=title or source_id, source_status="unknown")

    def set_rights(
        self,
        source_id: str,
        status: str,
        title: str = "",
        rights_holder: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> RightsInfo:
        valid_statuses = ("licensed", "public_domain", "own", "fair_dealing_commentary", "unknown")
        if status not in valid_statuses:
            raise ValueError(f"Invalid rights status '{status}'. Must be one of {valid_statuses}")

        info = RightsInfo(
            source_id=source_id,
            source_title=title or source_id,
            source_status=status,
            rights_holder=rights_holder,
            notes=notes,
            verified=(status != "unknown"),
            verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat() if status != "unknown" else None,
        )
        self.data[source_id] = info.model_dump()
        self._save()
        return info


def assert_export_allowed(rights: RightsInfo, allow_unverified: bool = False):
    """Publishing Gate: Hard-blocks export if rights status is 'unknown' (Plan Section 6)."""
    if rights.source_status == "unknown" and not allow_unverified:
        raise PermissionError(
            f"Export blocked by ReelForge Rights Gate for source '{rights.source_title}'.\n"
            f"Status is currently 'unknown'. Set rights clearance status via:\n"
            f"  reelforge rights --source-id {rights.source_id} --status licensed|public_domain|own\n"
            f"Or explicitly override using the --allow-unverified-rights flag."
        )


def record_reel_audit_log(output_reel_path: Path, moment: MomentCandidate):
    """Write an immutable audit log JSON alongside the exported reel documenting rights, timestamps, and model usage."""
    audit_path = Path(output_reel_path).with_suffix(".audit.json")
    record = {
        "output_file": output_reel_path.name,
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_movie_id": moment.movie_id,
        "moment_id": moment.moment_id,
        "start_time": moment.start,
        "end_time": moment.end,
        "duration": moment.duration,
        "category": moment.selected_category,
        "final_score": moment.final_score,
        "rights_status": moment.rights.source_status,
        "rights_holder": moment.rights.rights_holder,
        "ai_disclosure_tag": True,
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
