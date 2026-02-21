"""Avoidance simulator — now backed by the high-fidelity maneuver planner.

Maintains the original API interface while using physically correct
RTN frame burns and proper Pc computation.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
from sqlalchemy.orm import Session

from ..models import ConjunctionEvent, TLERecord
from .bplane_analysis import compute_bplane
from .covariance_handler import estimate_covariance_from_tle, default_covariance, project_covariance_to_bplane
from .maneuver_planner import (
    simulate_maneuver,
    optimize_maneuver,
    compute_trade_study,
    tsiolkovsky_fuel_cost,
    BURN_DIRECTIONS,
)
from .orbital_constants import DEFAULT_HBR_KM, classify_risk
from .pc_calculator import compute_pc_foster
from .tca_finder import find_tca_sgp4, TCAResult

logger = logging.getLogger(__name__)


class AvoidanceSimulator:
    def __init__(self, db: Session):
        self.db = db

    def simulate(
        self,
        event: ConjunctionEvent,
        delta_v_mps: float,
        direction: str = "prograde",
    ) -> tuple[float, list[dict]]:
        """Simulate an avoidance maneuver for a conjunction event.

        Parameters:
            event: The conjunction event to mitigate.
            delta_v_mps: Delta-v magnitude [m/s].
            direction: RTN direction (prograde, retrograde, etc.).

        Returns:
            (updated_miss_distance_km, post_trajectory)
        """
        from sqlalchemy import select

        # Get TLE records for both objects
        defended_tle = self.db.execute(
            select(TLERecord).where(TLERecord.norad_id == event.defended_norad_id)
        ).scalar_one_or_none()
        intruder_tle = self.db.execute(
            select(TLERecord).where(TLERecord.norad_id == event.intruder_norad_id)
        ).scalar_one_or_none()

        if defended_tle is None or intruder_tle is None:
            # Fallback: can't do proper simulation without TLEs
            logger.warning("TLE records not found; cannot simulate maneuver")
            return event.miss_distance_km, event.pre_trajectory

        # Build TCA from stored event data
        tca = self._tca_from_event(event)
        if tca is None:
            logger.warning("Could not reconstruct TCA from event data")
            return event.miss_distance_km, event.pre_trajectory

        # Get covariance
        from sgp4.api import Satrec
        sat_d = Satrec.twoline2rv(defended_tle.line1, defended_tle.line2)
        sat_i = Satrec.twoline2rv(intruder_tle.line1, intruder_tle.line2)
        epoch_age_d = max(0.0, (datetime.utcnow() - defended_tle.updated_at).total_seconds() / 86400.0)
        epoch_age_i = max(0.0, (datetime.utcnow() - intruder_tle.updated_at).total_seconds() / 86400.0)
        cov_d = estimate_covariance_from_tle(sat_d.bstar, epoch_age_d)
        cov_i = estimate_covariance_from_tle(sat_i.bstar, epoch_age_i)

        try:
            result = simulate_maneuver(
                defended_tle.line1, defended_tle.line2,
                intruder_tle.line1, intruder_tle.line2,
                tca, delta_v_mps, direction,
                cov_primary=cov_d, cov_secondary=cov_i,
            )
        except Exception as e:
            logger.error("Maneuver simulation failed: %s", e)
            return event.miss_distance_km, event.pre_trajectory

        # Update event in database
        event.post_trajectory = result.post_trajectory
        event.post_miss_distance_km = result.post_miss_distance_km
        event.post_maneuver_delta_v_mps = result.delta_v_mps
        event.post_maneuver_direction = result.direction
        event.post_maneuver_burn_epoch = result.burn_epoch
        event.post_maneuver_pc = result.post_pc
        event.post_maneuver_miss_distance_km = result.post_miss_distance_km
        event.post_maneuver_fuel_cost_kg = result.fuel_cost_kg
        event.post_maneuver_trajectory = result.post_trajectory
        self.db.commit()
        self.db.refresh(event)

        return result.post_miss_distance_km, result.post_trajectory

    def run_optimize(
        self,
        event: ConjunctionEvent,
        target_pc: float = 1e-4,
        direction: str = "prograde",
        max_dv_mps: float = 10.0,
        burn_lead_time_hours: float = 24.0,
    ) -> dict:
        """Find minimum delta-v to achieve target Pc."""
        from sqlalchemy import select

        defended_tle = self.db.execute(
            select(TLERecord).where(TLERecord.norad_id == event.defended_norad_id)
        ).scalar_one_or_none()
        intruder_tle = self.db.execute(
            select(TLERecord).where(TLERecord.norad_id == event.intruder_norad_id)
        ).scalar_one_or_none()

        if defended_tle is None or intruder_tle is None:
            raise ValueError("TLE records not found for this event")

        tca = self._tca_from_event(event)
        if tca is None:
            raise ValueError("Could not reconstruct TCA from event data")

        result = optimize_maneuver(
            defended_tle.line1, defended_tle.line2,
            intruder_tle.line1, intruder_tle.line2,
            tca, target_pc, direction, max_dv_mps, burn_lead_time_hours,
        )

        # Persist
        event.post_maneuver_delta_v_mps = result.delta_v_mps
        event.post_maneuver_direction = result.direction
        event.post_maneuver_burn_epoch = result.burn_epoch
        event.post_maneuver_pc = result.post_pc
        event.post_maneuver_miss_distance_km = result.post_miss_distance_km
        event.post_maneuver_fuel_cost_kg = result.fuel_cost_kg
        event.post_maneuver_trajectory = result.post_trajectory
        event.post_trajectory = result.post_trajectory
        event.post_miss_distance_km = result.post_miss_distance_km
        self.db.commit()

        return {
            "event_id": event.id,
            "optimal_delta_v_mps": result.delta_v_mps,
            "direction": result.direction,
            "burn_epoch": result.burn_epoch,
            "pre_pc": result.pre_pc,
            "post_pc": result.post_pc,
            "post_miss_distance_km": result.post_miss_distance_km,
            "fuel_cost_kg": result.fuel_cost_kg,
        }

    def run_trade_study(self, event: ConjunctionEvent) -> list[dict]:
        """Compute trade study across all RTN directions."""
        from sqlalchemy import select

        defended_tle = self.db.execute(
            select(TLERecord).where(TLERecord.norad_id == event.defended_norad_id)
        ).scalar_one_or_none()
        intruder_tle = self.db.execute(
            select(TLERecord).where(TLERecord.norad_id == event.intruder_norad_id)
        ).scalar_one_or_none()

        if defended_tle is None or intruder_tle is None:
            raise ValueError("TLE records not found for this event")

        tca = self._tca_from_event(event)
        if tca is None:
            raise ValueError("Could not reconstruct TCA from event data")

        return compute_trade_study(
            defended_tle.line1, defended_tle.line2,
            intruder_tle.line1, intruder_tle.line2,
            tca,
        )

    def _tca_from_event(self, event: ConjunctionEvent) -> TCAResult | None:
        """Reconstruct TCA from stored event data."""
        if not event.pre_trajectory or not event.intruder_trajectory:
            return None

        # Find closest approach sample
        min_dist = float('inf')
        min_idx = 0
        for i in range(min(len(event.pre_trajectory), len(event.intruder_trajectory))):
            d_pos = np.array(event.pre_trajectory[i]["position_km"])
            i_pos = np.array(event.intruder_trajectory[i]["position_km"])
            dist = float(np.linalg.norm(d_pos - i_pos))
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        d_sample = event.pre_trajectory[min_idx]
        i_sample = event.intruder_trajectory[min_idx]

        r1 = np.array(d_sample["position_km"])
        v1 = np.array(d_sample["velocity_kms"])
        r2 = np.array(i_sample["position_km"])
        v2 = np.array(i_sample["velocity_kms"])

        miss_vec = r1 - r2
        rel_vel = v1 - v2

        return TCAResult(
            tca_epoch=event.tca_utc,
            miss_distance_km=float(np.linalg.norm(miss_vec)),
            miss_vector_km=miss_vec,
            relative_speed_kms=float(np.linalg.norm(rel_vel)),
            r_primary_km=r1,
            v_primary_kms=v1,
            r_secondary_km=r2,
            v_secondary_kms=v2,
            tca_offset_seconds=0.0,
        )
