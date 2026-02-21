from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class TLERecord(Base):
    __tablename__ = "tle_records"
    __table_args__ = (UniqueConstraint("norad_id", name="uq_tle_norad"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    norad_id: Mapped[int] = mapped_column(Integer, index=True)
    name: Mapped[str] = mapped_column(String(256))
    line1: Mapped[str] = mapped_column(String(80))
    line2: Mapped[str] = mapped_column(String(80))
    inclination_deg: Mapped[float] = mapped_column(Float)
    mean_motion: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ConjunctionEvent(Base):
    __tablename__ = "conjunction_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    defended_norad_id: Mapped[int] = mapped_column(Integer, index=True)
    intruder_norad_id: Mapped[int] = mapped_column(Integer, index=True)
    tca_utc: Mapped[datetime] = mapped_column(DateTime, index=True)
    miss_distance_km: Mapped[float] = mapped_column(Float)
    relative_velocity_kms: Mapped[float] = mapped_column(Float)
    risk_tier: Mapped[str] = mapped_column(String(16), index=True)
    pre_trajectory: Mapped[list] = mapped_column(JSON)
    intruder_trajectory: Mapped[list] = mapped_column(JSON)
    post_trajectory: Mapped[list | None] = mapped_column(JSON, nullable=True)
    post_miss_distance_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
