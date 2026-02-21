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

    # --- High-fidelity fields (Phase 7) ---

    # Trajectory quality
    propagation_method: Mapped[str | None] = mapped_column(String(16), nullable=True, default="sgp4")
    covariance_source: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # TCA (high-precision)
    tca_miss_vector_km: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tca_relative_speed_kms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # B-plane
    b_dot_t_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    b_dot_r_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    b_magnitude_km: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Probability of Collision
    pc_foster: Mapped[float | None] = mapped_column(Float, nullable=True)
    pc_chan: Mapped[float | None] = mapped_column(Float, nullable=True)
    pc_monte_carlo: Mapped[float | None] = mapped_column(Float, nullable=True)
    pc_monte_carlo_ci_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    pc_monte_carlo_ci_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    pc_method_used: Mapped[str | None] = mapped_column(String(16), nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String(8), nullable=True)

    # Covariances (stored as JSON-flattened arrays)
    covariance_primary_6x6: Mapped[list | None] = mapped_column(JSON, nullable=True)
    covariance_secondary_6x6: Mapped[list | None] = mapped_column(JSON, nullable=True)
    hard_body_radius_km: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.020)

    # Post-maneuver
    post_maneuver_delta_v_mps: Mapped[float | None] = mapped_column(Float, nullable=True)
    post_maneuver_direction: Mapped[str | None] = mapped_column(String(16), nullable=True)
    post_maneuver_burn_epoch: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    post_maneuver_pc: Mapped[float | None] = mapped_column(Float, nullable=True)
    post_maneuver_miss_distance_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    post_maneuver_fuel_cost_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    post_maneuver_trajectory: Mapped[list | None] = mapped_column(JSON, nullable=True)
