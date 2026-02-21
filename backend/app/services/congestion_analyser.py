from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import ConjunctionEvent, TLERecord
from .ingest_service import IngestService


class CongestionAnalyser:
    def __init__(self, db: Session):
        self.db = db

    def compute(self) -> list[dict]:
        bands = defaultdict(lambda: {"object_count": 0, "conjunction_count": 0})

        objects = self.db.execute(select(TLERecord)).scalars().all()
        for obj in objects:
            alt = IngestService._approx_altitude_from_mean_motion(obj.mean_motion * (2.0 * 3.141592653589793) / 1440.0)
            band_floor = int(alt // 10) * 10
            label = f"{band_floor}-{band_floor + 10}"
            bands[label]["object_count"] += 1

        window_start = datetime.utcnow() - timedelta(days=7)
        events = self.db.execute(
            select(ConjunctionEvent).where(ConjunctionEvent.created_at >= window_start)
        ).scalars().all()

        defended_lookup = {obj.norad_id: obj for obj in objects}
        for event in events:
            defended = defended_lookup.get(event.defended_norad_id)
            if not defended:
                continue
            alt = IngestService._approx_altitude_from_mean_motion(defended.mean_motion * (2.0 * 3.141592653589793) / 1440.0)
            band_floor = int(alt // 10) * 10
            label = f"{band_floor}-{band_floor + 10}"
            bands[label]["conjunction_count"] += 1

        response = []
        for label in sorted(bands.keys(), key=lambda x: int(x.split("-")[0])):
            obj_count = bands[label]["object_count"]
            conj_count = bands[label]["conjunction_count"]
            rate = float(conj_count) / obj_count if obj_count else 0.0
            response.append(
                {
                    "altitude_band_km": label,
                    "object_count": obj_count,
                    "conjunction_rate": rate,
                }
            )
        return response
