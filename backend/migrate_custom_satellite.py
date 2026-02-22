"""Add custom satellite columns to tle_records table."""
import sqlite3
import sys

DB_PATH = "orbitguard.db"

NEW_COLUMNS = [
    ("source", "VARCHAR(32) DEFAULT 'celestrak'"),
    ("is_synthetic", "BOOLEAN DEFAULT 0"),
    ("eccentricity", "FLOAT"),
    ("raan_deg", "FLOAT"),
    ("arg_perigee_deg", "FLOAT"),
    ("bstar", "FLOAT"),
    ("epoch", "DATETIME"),
    ("mass_kg", "FLOAT"),
    ("drag_area_m2", "FLOAT"),
    ("radar_cross_section_m2", "FLOAT"),
    ("created_at", "DATETIME"),
]


def migrate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("PRAGMA table_info(tle_records)")
    existing = {row[1] for row in cursor.fetchall()}
    added = 0
    for col, col_type in NEW_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE tle_records ADD COLUMN {col} {col_type}")
            print(f"  Added column: {col}")
            added += 1
    conn.commit()
    conn.close()
    print(f"Migration complete. {added} columns added.")


if __name__ == "__main__":
    migrate()
