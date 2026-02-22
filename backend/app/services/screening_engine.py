from __future__ import annotations

import logging
from datetime import datetime, timedelta
from time import perf_counter
from time import sleep

import numpy as np
from sqlalchemy import and_, delete, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, load_only

from ..models import ConjunctionEvent, TLERecord
from .bplane_analysis import compute_bplane
from .covariance_handler import estimate_covariance_from_tle, default_covariance, project_covariance_to_bplane
from .orbital_constants import classify_risk, DEFAULT_HBR_KM
from .pc_calculator import compute_pc_foster, compute_pc_chan, compute_pc_monte_carlo
from .propagate_engine import PropagateEngine
from .tca_finder import find_tca_sgp4, find_tca_from_states

logger = logging.getLogger(__name__)


class ScreeningEngine:
    def __init__(self, db: Session, propagate_engine: PropagateEngine):
        self.db = db
        self.propagate_engine = propagate_engine

    def find_conjunctions(self, defended_norad_id: int, days: int = 3) -> list[ConjunctionEvent]:
        started_at = perf_counter()
        logger.info(
            "Screening start: defended_norad_id=%s days=%s",
            defended_norad_id,
            days,
        )
        defended = self.db.execute(
            select(TLERecord)
            .options(
                load_only(
                    TLERecord.norad_id,
                    TLERecord.line1,
                    TLERecord.line2,
                    TLERecord.inclination_deg,
                    TLERecord.mean_motion,
                    TLERecord.updated_at,
                    TLERecord.bstar,
                )
            )
            .where(TLERecord.norad_id == defended_norad_id)
        ).scalar_one_or_none()
        if defended is None:
            logger.warning("Screening aborted: defended asset %s not found", defended_norad_id)
            return []

        query_started = perf_counter()
        candidates = self.db.execute(
            select(TLERecord)
            .options(
                load_only(
                    TLERecord.norad_id,
                    TLERecord.line1,
                    TLERecord.line2,
                    TLERecord.updated_at,
                    TLERecord.bstar,
                )
            )
            .where(
                and_(
                    TLERecord.norad_id != defended_norad_id,
                    TLERecord.inclination_deg.between(
                        defended.inclination_deg - 5.0, defended.inclination_deg + 5.0
                    ),
                    TLERecord.mean_motion.between(
                        defended.mean_motion - 0.6, defended.mean_motion + 0.6
                    ),
                )
            )
        ).scalars().all()
        logger.info(
            "Screening candidates loaded: defended_norad_id=%s candidates=%s query_ms=%.1f",
            defended_norad_id,
            len(candidates),
            (perf_counter() - query_started) * 1000.0,
        )

        propagate_started = perf_counter()
        start = datetime.utcnow()
        pre = self.propagate_engine.propagate(defended.line1, defended.line2, start=start)
        logger.info(
            "Screening defended propagation complete: defended_norad_id=%s samples=%s elapsed_ms=%.1f",
            defended_norad_id,
            len(pre),
            (perf_counter() - propagate_started) * 1000.0,
        )

        created: list[ConjunctionEvent] = []
        cutoff = start + timedelta(days=days)
        progress_stride = max(1, len(candidates) // 10)
        for idx, candidate in enumerate(candidates, start=1):
            intr = self.propagate_engine.propagate(candidate.line1, candidate.line2, start=start)
            event = self._closest_approach(
                defended_norad_id, candidate.norad_id,
                defended, candidate,
                pre, intr, cutoff,
            )
            if event:
                created.append(event)
            if idx == 1 or idx == len(candidates) or idx % progress_stride == 0:
                logger.info(
                    "Screening progress: defended_norad_id=%s processed=%s/%s hits=%s elapsed_s=%.1f",
                    defended_norad_id,
                    idx,
                    len(candidates),
                    len(created),
                    perf_counter() - started_at,
                )

        write_started = perf_counter()
        self._replace_events_for_asset(defended_norad_id, created)
        stored_events = self.db.execute(
            select(ConjunctionEvent).where(ConjunctionEvent.defended_norad_id == defended_norad_id)
        ).scalars().all()
        logger.info(
            "Screening complete: defended_norad_id=%s events=%s total_s=%.1f write_ms=%.1f",
            defended_norad_id,
            len(stored_events),
            perf_counter() - started_at,
            (perf_counter() - write_started) * 1000.0,
        )
        return stored_events

    def _replace_events_for_asset(self, defended_norad_id: int, events: list[ConjunctionEvent]) -> None:
        attempts = 3
        for idx in range(attempts):
            try:
                self.db.execute(
                    delete(ConjunctionEvent).where(
                        ConjunctionEvent.defended_norad_id == defended_norad_id
                    )
                )
                for event in events:
                    self.db.add(event)
                self.db.commit()
                return
            except OperationalError as exc:
                self.db.rollback()
                if "database is locked" not in str(exc).lower() or idx == attempts - 1:
                    raise
                sleep(0.5 * (idx + 1))

    def _closest_approach(
        self,
        defended_norad_id: int,
        intruder_norad_id: int,
        defended_tle: TLERecord,
        intruder_tle: TLERecord,
        defended_samples: list[dict],
        intruder_samples: list[dict],
        cutoff: datetime,
    ) -> ConjunctionEvent | None:
        # Phase 1: coarse scan using SGP4 samples to find close approaches
        sample_count = min(len(defended_samples), len(intruder_samples))
        if sample_count == 0:
            return None

        d_pos = np.array([s["position_km"] for s in defended_samples[:sample_count]])
        i_pos = np.array([s["position_km"] for s in intruder_samples[:sample_count]])
        d_vel = np.array([s["velocity_kms"] for s in defended_samples[:sample_count]])
        i_vel = np.array([s["velocity_kms"] for s in intruder_samples[:sample_count]])

        # Filter to cutoff
        cutoff_idx = sample_count
        for i in range(sample_count):
            t = datetime.fromisoformat(defended_samples[i]["t"].replace("Z", ""))
            if t > cutoff:
                cutoff_idx = i
                break

        if cutoff_idx == 0:
            return None

        d_pos = d_pos[:cutoff_idx]
        i_pos = i_pos[:cutoff_idx]
        d_vel = d_vel[:cutoff_idx]
        i_vel = i_vel[:cutoff_idx]

        # GPU-accelerated distance computation when available
        from .compute_backend import get_xp, asnumpy
        xp = get_xp()
        d_pos_xp = xp.asarray(d_pos)
        i_pos_xp = xp.asarray(i_pos)
        distances_xp = xp.linalg.norm(d_pos_xp - i_pos_xp, axis=1)
        min_idx = int(xp.argmin(distances_xp))
        min_distance = float(distances_xp[min_idx])
        distances = asnumpy(distances_xp)

        if min_distance > 10.0:
            return None

        # Phase 2: refine TCA using root-finding on state vectors
        start_epoch = datetime.fromisoformat(defended_samples[0]["t"].replace("Z", ""))
        times = np.array([
            (datetime.fromisoformat(defended_samples[i]["t"].replace("Z", "")) - start_epoch).total_seconds()
            for i in range(cutoff_idx)
        ])

        tca = find_tca_from_states(times, d_pos, d_vel, i_pos, i_vel, start_epoch)
        if tca is None:
            return None

        miss_distance = tca.miss_distance_km
        rel_speed = tca.relative_speed_kms
        risk_tier = self._risk_tier(miss_distance)

        # Phase 3: B-plane analysis
        bplane = compute_bplane(tca)

        # Phase 4: Covariance estimation from TLE quality
        from sgp4.api import Satrec
        sat_d = Satrec.twoline2rv(defended_tle.line1, defended_tle.line2)
        sat_i = Satrec.twoline2rv(intruder_tle.line1, intruder_tle.line2)

        epoch_age_d = max(0.0, (datetime.utcnow() - defended_tle.updated_at).total_seconds() / 86400.0)
        epoch_age_i = max(0.0, (datetime.utcnow() - intruder_tle.updated_at).total_seconds() / 86400.0)

        cov_d = estimate_covariance_from_tle(sat_d.bstar, epoch_age_d)
        cov_i = estimate_covariance_from_tle(sat_i.bstar, epoch_age_i)

        # Phase 5: Probability of Collision
        cov_bp = project_covariance_to_bplane(cov_d.cov_3x3_pos, cov_i.cov_3x3_pos, bplane)
        miss_bp = np.array([bplane.b_dot_t_km, bplane.b_dot_r_km])

        pc_foster_result = compute_pc_foster(miss_bp, cov_bp, DEFAULT_HBR_KM)
        pc_chan_result = compute_pc_chan(miss_distance, cov_bp, DEFAULT_HBR_KM)
        pc_mc_result = compute_pc_monte_carlo(miss_bp, cov_bp, DEFAULT_HBR_KM, n_samples=5000, seed=42)

        risk_level = classify_risk(pc_foster_result.pc)

        # Trajectory window around TCA for visualization
        lo = max(0, min_idx - 45)
        hi = min(cutoff_idx, min_idx + 45)

        return ConjunctionEvent(
            defended_norad_id=defended_norad_id,
            intruder_norad_id=intruder_norad_id,
            tca_utc=tca.tca_epoch,
            miss_distance_km=miss_distance,
            relative_velocity_kms=rel_speed,
            risk_tier=risk_tier,
            pre_trajectory=defended_samples[lo:hi],
            intruder_trajectory=intruder_samples[lo:hi],
            # High-fidelity fields
            propagation_method="sgp4",
            covariance_source=cov_d.source,
            tca_miss_vector_km=tca.miss_vector_km.tolist(),
            tca_relative_speed_kms=rel_speed,
            b_dot_t_km=bplane.b_dot_t_km,
            b_dot_r_km=bplane.b_dot_r_km,
            b_magnitude_km=bplane.b_magnitude_km,
            pc_foster=pc_foster_result.pc,
            pc_chan=pc_chan_result.pc,
            pc_monte_carlo=pc_mc_result.pc,
            pc_monte_carlo_ci_low=pc_mc_result.ci_low,
            pc_monte_carlo_ci_high=pc_mc_result.ci_high,
            pc_method_used="foster",
            risk_level=risk_level.value,
            hard_body_radius_km=DEFAULT_HBR_KM,
            covariance_primary_6x6=cov_d.cov_6x6.flatten().tolist(),
            covariance_secondary_6x6=cov_i.cov_6x6.flatten().tolist(),
        )

    @staticmethod
    def _risk_tier(miss_distance_km: float) -> str:
        if miss_distance_km < 1.0:
            return "High"
        if miss_distance_km < 5.0:
            return "Medium"
        return "Low"
