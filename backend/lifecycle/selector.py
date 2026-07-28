"""LifecycleSelector — Stage D "Candidate Selection" of the reference notebook.

Rules (unchanged):
  1. Keep only Accepted candidates.
  2. None accepted -> None.
  3. Exactly one accepted -> select it.
  4. Multiple accepted -> longest duration wins.
"""
from __future__ import annotations

from typing import Any, Optional

from backend.lifecycle.farm_builder import LocationRecord


class LifecycleSelector:
    def select(self, record: LocationRecord) -> Optional[dict[str, Any]]:
        valid = [c for c in record.candidate_cycles if c["status"] == "Accepted"]

        if len(valid) == 0:
            record.accepted_cycle = None
            return None

        if len(valid) == 1:
            record.accepted_cycle = valid[0]
            return valid[0]

        best = max(valid, key=lambda c: c["duration"])
        record.accepted_cycle = best
        return best
