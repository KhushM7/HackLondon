from __future__ import annotations

from math import sqrt

from sqlalchemy.orm import Session

from ..models import ConjunctionEvent


class AvoidanceSimulator:
    def __init__(self, db: Session):
        self.db = db

    def simulate(self, event: ConjunctionEvent, delta_v_mps: float) -> tuple[float, list[dict]]:
        # Simplified non-authoritative displacement approximation.
        drift_seconds = 900.0
        offset_km = (delta_v_mps * drift_seconds) / 1000.0

        shifted = []
        for sample in event.pre_trajectory:
            vx, vy, vz = sample["velocity_kms"]
            norm = sqrt(vx * vx + vy * vy + vz * vz) or 1.0
            ux, uy, uz = vx / norm, vy / norm, vz / norm
            px, py, pz = sample["position_km"]
            shifted.append(
                {
                    "t": sample["t"],
                    "position_km": [px + ux * offset_km, py + uy * offset_km, pz + uz * offset_km],
                    "velocity_kms": sample["velocity_kms"],
                }
            )

        new_min = min(
            self._distance(a["position_km"], b["position_km"])
            for a, b in zip(shifted, event.intruder_trajectory)
        )

        event.post_trajectory = shifted
        event.post_miss_distance_km = new_min
        self.db.commit()
        self.db.refresh(event)
        return new_min, shifted

    @staticmethod
    def _distance(a: list[float], b: list[float]) -> float:
        return sqrt(sum((ax - bx) ** 2 for ax, bx in zip(a, b)))
