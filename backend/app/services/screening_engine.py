from __future__ import annotations

from datetime import datetime, timedelta
from math import sqrt
from time import sleep

from sqlalchemy import and_, delete, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from ..models import ConjunctionEvent, TLERecord
from .propagate_engine import PropagateEngine


class ScreeningEngine:
    def __init__(self, db: Session, propagate_engine: PropagateEngine):
        self.db = db
        self.propagate_engine = propagate_engine

    def find_conjunctions(self, defended_norad_id: int, days: int = 3) -> list[ConjunctionEvent]:
        defended = self.db.execute(
            select(TLERecord).where(TLERecord.norad_id == defended_norad_id)
        ).scalar_one_or_none()
        if defended is None:
            return []

        candidates = self.db.execute(
            select(TLERecord).where(
                and_(
                    TLERecord.norad_id != defended_norad_id,
                    TLERecord.inclination_deg.between(
                        defended.inclination_deg - 5.0, defended.inclination_deg + 5.0
                    ),
                    TLERecord.mean_motion.between(
                        defended.mean_motion - 0.6, defended.mean_motion + 0.6
                    ),
                )
            )
        ).scalars().all()

        start = datetime.utcnow()
        pre = self.propagate_engine.propagate(defended.line1, defended.line2, start=start)

        created: list[ConjunctionEvent] = []
        cutoff = start + timedelta(days=days)
        for candidate in candidates:
            intr = self.propagate_engine.propagate(candidate.line1, candidate.line2, start=start)
            event = self._closest_approach(defended_norad_id, candidate.norad_id, pre, intr, cutoff)
            if event:
                created.append(event)

        self._replace_events_for_asset(defended_norad_id, created)
        for item in created:
            self.db.refresh(item)
        return created

    def _replace_events_for_asset(self, defended_norad_id: int, events: list[ConjunctionEvent]) -> None:
        attempts = 3
        for idx in range(attempts):
            try:
                self.db.execute(
                    delete(ConjunctionEvent).where(
                        ConjunctionEvent.defended_norad_id == defended_norad_id
                    )
                )
                for event in events:
                    self.db.add(event)
                self.db.commit()
                return
            except OperationalError as exc:
                self.db.rollback()
                if "database is locked" not in str(exc).lower() or idx == attempts - 1:
                    raise
                sleep(0.5 * (idx + 1))

    def _closest_approach(
        self,
        defended_norad_id: int,
        intruder_norad_id: int,
        defended_samples: list[dict],
        intruder_samples: list[dict],
        cutoff: datetime,
    ) -> ConjunctionEvent | None:
        sample_count = min(len(defended_samples), len(intruder_samples))
        min_idx = -1
        min_distance = 1e12

        for i in range(sample_count):
            t = datetime.fromisoformat(defended_samples[i]["t"].replace("Z", ""))
            if t > cutoff:
                break
            d = self._distance(defended_samples[i]["position_km"], intruder_samples[i]["position_km"])
            if d < min_distance:
                min_distance = d
                min_idx = i

        if min_idx < 0 or min_distance > 10.0:
            return None

        rv = self._distance(defended_samples[min_idx]["velocity_kms"], intruder_samples[min_idx]["velocity_kms"])
        risk_tier = self._risk_tier(min_distance)

        lo = max(0, min_idx - 45)
        hi = min(sample_count, min_idx + 45)

        return ConjunctionEvent(
            defended_norad_id=defended_norad_id,
            intruder_norad_id=intruder_norad_id,
            tca_utc=datetime.fromisoformat(defended_samples[min_idx]["t"].replace("Z", "")),
            miss_distance_km=min_distance,
            relative_velocity_kms=rv,
            risk_tier=risk_tier,
            pre_trajectory=defended_samples[lo:hi],
            intruder_trajectory=intruder_samples[lo:hi],
        )

    @staticmethod
    def _distance(a: list[float], b: list[float]) -> float:
        return sqrt(sum((ax - bx) ** 2 for ax, bx in zip(a, b)))

    @staticmethod
    def _risk_tier(miss_distance_km: float) -> str:
        if miss_distance_km < 1.0:
            return "High"
        if miss_distance_km < 5.0:
            return "Medium"
        return "Low"
