from __future__ import annotations

from datetime import datetime, timedelta

from sgp4.api import Satrec, jday


class PropagateEngine:
    def __init__(self, horizon_days: int, resolution_seconds: int):
        self.horizon_days = horizon_days
        self.resolution_seconds = resolution_seconds

    def propagate(self, line1: str, line2: str, start: datetime | None = None) -> list[dict]:
        sat = Satrec.twoline2rv(line1, line2)
        start_time = start or datetime.utcnow()
        end = start_time + timedelta(days=self.horizon_days)

        samples: list[dict] = []
        current = start_time
        while current <= end:
            jd, fr = jday(
                current.year,
                current.month,
                current.day,
                current.hour,
                current.minute,
                current.second + current.microsecond / 1_000_000.0,
            )
            err, position, velocity = sat.sgp4(jd, fr)
            if err == 0:
                samples.append(
                    {
                        "t": current.isoformat() + "Z",
                        "position_km": [float(v) for v in position],
                        "velocity_kms": [float(v) for v in velocity],
                    }
                )
            current += timedelta(seconds=self.resolution_seconds)
        return samples
