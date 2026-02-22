"""Tests for custom satellite ingestion pipeline."""
import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db import Base, get_session
from app.main import app
from app.models import TLERecord, ConjunctionEvent

# Valid ISS-like TLE for testing
VALID_TLE_LINE1 = "1 90000U 24001A   26052.50000000  .00010000  00000-0  15000-3 0  9991"
VALID_TLE_LINE2 = "2 90000  51.6400 210.5000 0005000  75.0000 285.0000 15.50000000000017"

# Second valid TLE for duplicate name tests
VALID_TLE2_LINE1 = "1 90001U 24002A   26052.50000000  .00010000  00000-0  15000-3 0  9984"
VALID_TLE2_LINE2 = "2 90001  51.6400 210.5000 0005000  75.0000 285.0000 15.50000000000010"


def _compute_checksum(line: str) -> int:
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


# Pre-compute valid TLEs with correct checksums
def _make_valid_tle():
    l1 = "1 90000U 24001A   26052.50000000  .00010000  00000-0  15000-3 0  999"
    l2 = "2 90000  51.6400 210.5000 0005000  75.0000 285.0000 15.5000000000001"
    l1 += str(_compute_checksum(l1))
    l2 += str(_compute_checksum(l2))
    return l1, l2


def _make_valid_tle2():
    l1 = "1 90001U 24002A   26052.50000000  .00010000  00000-0  15000-3 0  998"
    l2 = "2 90001  51.6400 210.5000 0005000  75.0000 285.0000 15.5000000000001"
    l1 += str(_compute_checksum(l1))
    l2 += str(_compute_checksum(l2))
    return l1, l2


VALID_L1, VALID_L2 = _make_valid_tle()
VALID2_L1, VALID2_L2 = _make_valid_tle2()


@pytest.fixture()
def db_session():
    from sqlalchemy.pool import StaticPool
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = TestSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db_session):
    def _override():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_session] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestAddCustomSatellite:
    """POST /catalog/custom-satellite"""

    def test_valid_custom_tle_returns_201(self, client, db_session):
        resp = client.post(
            "/catalog/custom-satellite",
            json={"name": "TestSat-Valid", "line1": VALID_L1, "line2": VALID_L2},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["satellite"]["name"] == "TestSat-Valid"
        assert data["satellite"]["norad_id"] >= 90000
        assert data["satellite"]["is_synthetic"] is True
        assert data["satellite"]["source"] == "custom"
        assert "conjunctions_found" in data
        assert isinstance(data["events"], list)

        # Verify stored in DB
        rec = db_session.execute(
            select(TLERecord).where(TLERecord.name == "TestSat-Valid")
        ).scalar_one_or_none()
        assert rec is not None
        assert rec.is_synthetic is True

    def test_malformed_tle_line1_returns_422(self, client):
        resp = client.post(
            "/catalog/custom-satellite",
            json={
                "name": "BadSat",
                "line1": "INVALID LINE",
                "line2": VALID_L2,
            },
        )
        assert resp.status_code == 422

    def test_checksum_auto_corrected_returns_201(self, client):
        """Checksums are auto-corrected for hand-crafted TLEs, so wrong checksums should still succeed."""
        bad_l1 = VALID_L1[:68] + str((int(VALID_L1[68]) + 1) % 10)
        resp = client.post(
            "/catalog/custom-satellite",
            json={"name": "ChecksumFixed", "line1": bad_l1, "line2": VALID_L2},
        )
        assert resp.status_code == 201
        assert resp.json()["satellite"]["name"] == "ChecksumFixed"

    def test_duplicate_name_returns_409(self, client):
        # First insert should succeed
        resp1 = client.post(
            "/catalog/custom-satellite",
            json={"name": "DuplicateSat", "line1": VALID_L1, "line2": VALID_L2},
        )
        assert resp1.status_code == 201

        # Second insert with same name should fail
        resp2 = client.post(
            "/catalog/custom-satellite",
            json={"name": "DuplicateSat", "line1": VALID2_L1, "line2": VALID2_L2},
        )
        assert resp2.status_code == 409
        assert "already exists" in resp2.json()["detail"]


class TestDeleteCustomSatellite:
    """DELETE /catalog/custom-satellite/{norad_id}"""

    def test_delete_synthetic_satellite_returns_200(self, client, db_session):
        # Create satellite first
        resp = client.post(
            "/catalog/custom-satellite",
            json={"name": "DeleteMe", "line1": VALID_L1, "line2": VALID_L2},
        )
        assert resp.status_code == 201
        norad_id = resp.json()["satellite"]["norad_id"]

        # Delete it
        resp2 = client.delete(f"/catalog/custom-satellite/{norad_id}")
        assert resp2.status_code == 200

        # Verify gone from DB
        rec = db_session.execute(
            select(TLERecord).where(TLERecord.norad_id == norad_id)
        ).scalar_one_or_none()
        assert rec is None

    def test_delete_real_satellite_returns_403(self, client, db_session):
        # Insert a real satellite
        now = datetime.utcnow()
        real = TLERecord(
            norad_id=25544,
            name="ISS",
            line1=VALID_L1,
            line2=VALID_L2,
            inclination_deg=51.64,
            mean_motion=15.5,
            source="celestrak",
            is_synthetic=False,
            updated_at=now,
        )
        db_session.add(real)
        db_session.commit()

        resp = client.delete("/catalog/custom-satellite/25544")
        assert resp.status_code == 403
        assert "real catalog" in resp.json()["detail"].lower()


class TestListCustomSatellites:
    """GET /catalog/custom-satellites"""

    def test_list_returns_only_synthetic(self, client, db_session):
        # Add a real satellite
        now = datetime.utcnow()
        real = TLERecord(
            norad_id=25544, name="ISS-Real", line1=VALID_L1, line2=VALID_L2,
            inclination_deg=51.64, mean_motion=15.5,
            source="celestrak", is_synthetic=False, updated_at=now,
        )
        db_session.add(real)
        db_session.commit()

        # Add a custom satellite
        resp = client.post(
            "/catalog/custom-satellite",
            json={"name": "SyntheticOnly", "line1": VALID_L1, "line2": VALID_L2},
        )
        assert resp.status_code == 201

        # List should only show synthetic
        resp2 = client.get("/catalog/custom-satellites")
        assert resp2.status_code == 200
        items = resp2.json()
        assert len(items) == 1
        assert items[0]["name"] == "SyntheticOnly"
        assert items[0]["is_synthetic"] is True
        assert "conjunction_count" in items[0]


class TestScreeningOnCreation:
    """Verify conjunction screening runs on custom satellite creation."""

    def test_screening_triggered_on_creation(self, client, db_session):
        # Add a real satellite nearby in orbital elements
        now = datetime.utcnow()
        real = TLERecord(
            norad_id=25544, name="ISS-Screening", line1=VALID_L1, line2=VALID_L2,
            inclination_deg=51.64, mean_motion=15.5,
            source="celestrak", is_synthetic=False, updated_at=now,
        )
        db_session.add(real)
        db_session.commit()

        # Add custom satellite — screening should run (may or may not find conjunctions)
        resp = client.post(
            "/catalog/custom-satellite",
            json={"name": "ScreenTest", "line1": VALID2_L1, "line2": VALID2_L2},
        )
        assert resp.status_code == 201
        data = resp.json()
        # Screening was triggered - conjunctions_found should be an integer
        assert isinstance(data["conjunctions_found"], int)
        assert isinstance(data["events"], list)
