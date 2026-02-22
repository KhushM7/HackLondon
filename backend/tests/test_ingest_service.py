import pytest
pytest.skip("SQLAlchemy-specific test disabled after MongoDB migration", allow_module_level=True)

from datetime import datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.models import TLERecord
from app.services.ingest_service import IngestService


def _make_record(norad_id: int) -> dict:
    return {
        "norad_id": norad_id,
        "name": f"TEST-{norad_id}",
        "line1": f"1 {norad_id:05d}U 24001A   26052.50000000  .00010000  00000-0  15000-3 0  9991",
        "line2": f"2 {norad_id:05d}  51.6400 210.5000 0005000  75.0000 285.0000 15.50000000000017",
        "inclination_deg": 51.64,
        "mean_motion": 15.5,
    }


def test_sqlite_upsert_batches_for_large_payload():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()
    try:
        service = IngestService(session)
        service.SQLITE_SAFE_MAX_BIND_VARS = 30

        records = [_make_record(10000 + i) for i in range(25)]
        service._upsert_records(records)
        session.commit()

        stored = session.execute(select(TLERecord)).scalars().all()
        assert len(stored) == 25
        assert all(rec.updated_at <= datetime.utcnow() for rec in stored)
    finally:
        session.close()
