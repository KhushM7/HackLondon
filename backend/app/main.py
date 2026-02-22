import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .api import router
from .config import settings
from .db import Base, SessionLocal, engine
from .services.ingest_service import IngestService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


async def ingestion_loop() -> None:
    while True:
        db = SessionLocal()
        try:
            count = await IngestService(db).ingest_latest_tles()
            logger.info("Ingested %s TLE records", count)
        except Exception as exc:
            logger.exception("Ingestion failed: %s", exc)
            db.rollback()
        finally:
            db.close()
        await asyncio.sleep(settings.ingestion_interval_hours * 3600)


@app.on_event("startup")
async def startup_event() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Improves candidate filtering speed for screening queries.
        db.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_tle_screening_inc_mean "
                "ON tle_records (inclination_deg, mean_motion)"
            )
        )
        db.commit()
    finally:
        db.close()
    asyncio.create_task(ingestion_loop())


@app.get("/")
def health() -> dict:
    return {
        "name": settings.app_name,
        "status": "ok",
        "disclaimer": "Public TLE only. Screening-grade outputs. No formal collision probability.",
    }
