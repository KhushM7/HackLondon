from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import httpx
from sgp4.api import Satrec
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import TLERecord

logger = logging.getLogger(__name__)


class TLESourceError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class IngestService:
    SQLITE_SAFE_MAX_BIND_VARS = 900
    HTTP_TIMEOUT_SECONDS = 30
    HTTP_MAX_RETRIES = 3
    RETRYABLE_HTTP_STATUSES = {403, 429, 500, 502, 503, 504}

    def __init__(self, db: Session):
        self.db = db

    async def ingest_latest_tles(self) -> int:
        text = await self._download_tle_text()
        records = self._parse_tle_text(text)
        self._upsert_records(records)
        self.db.commit()
        return len(records)

    async def _download_tle_text(self) -> str:
        last_error: TLESourceError | None = None
        for source_url in self._bulk_tle_source_urls():
            try:
                text = await self._fetch_text_with_retries(source_url)
                if text.strip():
                    logger.info("Fetched TLE catalog from %s", source_url)
                    return text
            except TLESourceError as exc:
                last_error = exc
                logger.warning("TLE fetch failed for %s: %s", source_url, exc)
                continue

        if last_error is not None:
            raise last_error
        raise TLESourceError("TLE source returned an empty payload")

    def _parse_tle_text(self, text: str) -> list[dict]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        parsed = []
        idx = 0
        while idx + 2 < len(lines):
            name, line1, line2 = lines[idx], lines[idx + 1], lines[idx + 2]
            idx += 3
            if not line1.startswith("1 ") or not line2.startswith("2 "):
                continue
            try:
                sat = Satrec.twoline2rv(line1, line2)
                altitude_km = self._approx_altitude_from_mean_motion(sat.no_kozai)
                if altitude_km > settings.leo_max_altitude_km:
                    continue
                parsed.append(
                    {
                        "norad_id": int(line1[2:7]),
                        "name": name,
                        "line1": line1,
                        "line2": line2,
                        "inclination_deg": float(line2[8:16]),
                        "mean_motion": float(line2[52:63]),
                    }
                )
            except Exception:
                continue
        return parsed

    def _upsert_records(self, records: list[dict]) -> None:
        if not records:
            return

        now = datetime.utcnow()
        payload = [{**record, "updated_at": now} for record in records]

        bind = self.db.get_bind()
        if bind is not None and bind.dialect.name == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            single_row_stmt = sqlite_insert(TLERecord).values(payload[0])
            bind_vars_per_row = max(1, len(single_row_stmt.compile(dialect=bind.dialect).params))
            batch_size = max(1, self.SQLITE_SAFE_MAX_BIND_VARS // bind_vars_per_row)

            for start in range(0, len(payload), batch_size):
                stmt = sqlite_insert(TLERecord).values(payload[start : start + batch_size])
                stmt = stmt.on_conflict_do_update(
                    index_elements=[TLERecord.norad_id],
                    set_={
                        "name": stmt.excluded.name,
                        "line1": stmt.excluded.line1,
                        "line2": stmt.excluded.line2,
                        "inclination_deg": stmt.excluded.inclination_deg,
                        "mean_motion": stmt.excluded.mean_motion,
                        "updated_at": stmt.excluded.updated_at,
                    },
                )
                self.db.execute(stmt)
            return

        existing = {
            row.norad_id: row
            for row in self.db.execute(select(TLERecord)).scalars().all()
        }
        for record in payload:
            entity = existing.get(record["norad_id"])
            if entity:
                entity.name = record["name"]
                entity.line1 = record["line1"]
                entity.line2 = record["line2"]
                entity.inclination_deg = record["inclination_deg"]
                entity.mean_motion = record["mean_motion"]
                entity.updated_at = now
            else:
                self.db.add(TLERecord(**record))

    async def ingest_satellite_by_norad_id(self, norad_id: int) -> TLERecord:
        text = ""
        last_error: TLESourceError | None = None
        for url in self._single_satellite_source_urls(norad_id):
            try:
                text = (await self._fetch_text_with_retries(url)).strip()
                if text:
                    logger.info("Fetched TLE for NORAD %s from %s", norad_id, url)
                    break
            except TLESourceError as exc:
                last_error = exc
                logger.warning("TLE fetch failed for NORAD %s from %s: %s", norad_id, url, exc)
                continue

        if not text and last_error is not None:
            raise last_error

        if not text or "No GP data found" in text:
            raise ValueError(f"No TLE data found for NORAD ID {norad_id} on CelesTrak")

        parsed = self._parse_single_tle(text)
        if parsed is None:
            raise ValueError(f"Could not parse TLE for NORAD ID {norad_id}")
        if parsed["altitude_km"] > settings.leo_max_altitude_km:
            raise ValueError(
                f"'{parsed['name']}' is not LEO (altitude ≈ {parsed['altitude_km']:.0f} km; limit {settings.leo_max_altitude_km:.0f} km)"
            )

        self._upsert_records([parsed])
        self.db.commit()
        record = self.db.execute(select(TLERecord).where(TLERecord.norad_id == norad_id)).scalar_one_or_none()
        if record is None:
            raise ValueError(f"Failed to store satellite {norad_id}")
        return record

    def _parse_single_tle(self, text: str) -> dict | None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 3:
            return None
        name, line1, line2 = lines[0], lines[1], lines[2]
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            return None
        try:
            sat = Satrec.twoline2rv(line1, line2)
            return {
                "norad_id": int(line1[2:7]),
                "name": name,
                "line1": line1,
                "line2": line2,
                "inclination_deg": float(line2[8:16]),
                "mean_motion": float(line2[52:63]),
                "altitude_km": self._approx_altitude_from_mean_motion(sat.no_kozai),
            }
        except Exception:
            return None

    @staticmethod
    def _approx_altitude_from_mean_motion(no_kozai: float) -> float:
        # Derives semi-major axis from mean motion in radians/minute.
        mu = 398600.4418
        n = no_kozai / 60.0
        a = (mu / (n * n)) ** (1.0 / 3.0)
        return a - 6378.137

    @staticmethod
    def _http_headers() -> dict[str, str]:
        return {
            "User-Agent": "OrbitGuard/1.0 (+https://github.com/)",
            "Accept": "text/plain, */*;q=0.9",
            "Accept-Language": "en-US,en;q=0.8",
            "Referer": "https://celestrak.org/",
        }

    @staticmethod
    def _dedupe_urls(urls: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for url in urls:
            key = url.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    def _bulk_tle_source_urls(self) -> list[str]:
        primary = settings.tle_source_url
        primary_lower = primary.lower()
        celestrak_fallbacks = [
            "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=3le",
            "https://celestrak.org/NORAD/elements/active.txt",
            "https://www.celestrak.com/NORAD/elements/active.txt",
        ]
        if "celestrak.org" in primary_lower or "celestrak.com" in primary_lower:
            return self._dedupe_urls([primary, *celestrak_fallbacks])
        return [primary]

    def _single_satellite_source_urls(self, norad_id: int) -> list[str]:
        return self._dedupe_urls(
            [
                f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle",
                f"https://www.celestrak.com/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle",
            ]
        )

    async def _fetch_text_with_retries(self, url: str) -> str:
        retry_delay_seconds = 0.5
        async with httpx.AsyncClient(
            timeout=self.HTTP_TIMEOUT_SECONDS,
            headers=self._http_headers(),
            follow_redirects=True,
        ) as client:
            for attempt in range(1, self.HTTP_MAX_RETRIES + 1):
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    return response.text
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    if status in self.RETRYABLE_HTTP_STATUSES and attempt < self.HTTP_MAX_RETRIES:
                        await asyncio.sleep(retry_delay_seconds)
                        retry_delay_seconds *= 2
                        continue
                    raise TLESourceError(f"TLE source responded with HTTP {status}", status_code=status) from exc
                except httpx.RequestError as exc:
                    if attempt < self.HTTP_MAX_RETRIES:
                        await asyncio.sleep(retry_delay_seconds)
                        retry_delay_seconds *= 2
                        continue
                    raise TLESourceError(f"TLE source request failed: {exc}") from exc
