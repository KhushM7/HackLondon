import pytest
pytest.skip("SQLite migration test disabled after MongoDB migration", allow_module_level=True)

from sqlalchemy import create_engine, inspect, text

from app.schema_migrations import apply_startup_sqlite_migrations


def test_sqlite_startup_migration_adds_missing_conjunction_columns():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE conjunction_events (
                    id INTEGER PRIMARY KEY,
                    defended_norad_id INTEGER,
                    intruder_norad_id INTEGER,
                    tca_utc DATETIME,
                    miss_distance_km FLOAT,
                    relative_velocity_kms FLOAT,
                    risk_tier VARCHAR(16),
                    pre_trajectory JSON,
                    intruder_trajectory JSON,
                    post_trajectory JSON,
                    post_miss_distance_km FLOAT,
                    created_at DATETIME
                )
                """
            )
        )

    apply_startup_sqlite_migrations(engine)

    cols = {c["name"] for c in inspect(engine).get_columns("conjunction_events")}
    assert "maneuver_confidence" in cols
    assert "maneuver_optimizer_version" in cols
