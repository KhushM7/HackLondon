"""Avoidance simulator — now backed by the high-fidelity maneuver planner.

Maintains the original API interface while using physically correct
RTN frame burns and proper Pc computation.

Includes two-stage optimization engine:
  Stage A (coarse): grid search over RTN directions × delta-v ladder.
  Stage B (refine): local continuous optimization around top candidates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import perf_counter
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import AvoidancePlan, ConjunctionEvent, TLERecord
from .bplane_analysis import compute_bplane
from .compute_backend import get_xp, asnumpy
from .covariance_handler import estimate_covariance_from_tle, default_covariance, project_covariance_to_bplane
from .maneuver_planner import (
    simulate_maneuver,
    optimize_maneuver,
    compute_trade_study,
    tsiolkovsky_fuel_cost,
    apply_impulsive_burn_sgp4,
    BURN_DIRECTIONS,
)
from .orbital_constants import DEFAULT_HBR_KM, classify_risk
from .pc_calculator import compute_pc_foster
from .propagate_engine import PropagateEngine
from .screening_engine import ScreeningEngine
from .tca_finder import find_tca_sgp4, find_tca_from_states, _sgp4_state_at, TCAResult

logger = logging.getLogger(__name__)


class OptimizationCancelled(Exception):
    """Raised when an in-flight optimization is cancelled by the user."""


# Coarse grid delta-v ladder (m/s)
_DV_LADDER = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
_FAST_OPT_STEP_SECONDS = 300.0
_FAST_OPT_POST_PAD_HOURS = 6.0
_FINAL_PATH_STEP_SECONDS = 45.0
_FINAL_PATH_POST_PAD_HOURS = 24.0
_SAFETY_MISS_FLOOR_KM = 5.0
_HARD_COLLISION_MISS_KM = 1.0
_HARD_COLLISION_PC = 1e-3
_ALLOWED_PC_BASE = 1e-4


def _eci_to_lla(pos_km: list[float]) -> list[float]:
    """Convert ECI position to approximate [lat, lon, alt_km] for map display."""
    from .orbital_constants import R_EARTH_KM
    x, y, z = pos_km
    r = np.sqrt(x * x + y * y + z * z)
    alt = r - R_EARTH_KM
    lat = np.degrees(np.arcsin(z / r)) if r > 0 else 0.0
    lon = np.degrees(np.arctan2(y, x))
    return [round(float(lat), 4), round(float(lon), 4), round(float(alt), 2)]


@dataclass
class _CandidateScore:
    direction: str
    dv_mps: float
    burn_lead_hours: float
    miss_distance_km: float
    pc: float
    score: float
    rtn_vector: list[float]


@dataclass
class _EventContext:
    event_id: int
    intruder_norad_id: int
    intruder_tle: TLERecord
    tca: TCAResult
    cov_secondary: object
    pre_pc: float


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

    # ------------------------------------------------------------------
    # Two-stage optimization engine
    # ------------------------------------------------------------------

    def run_full_optimization(
        self,
        norad_id: int,
        plan_id: int | None = None,
        max_delta_v_mps: float = 5.0,
        burn_window_hours: float = 48.0,
        weight_miss_distance: float = 1.0,
        weight_delta_v: float = 0.3,
        top_n_events: int = 3,
    ) -> AvoidancePlan:
        """Run two-stage avoidance optimization for an asset's top-risk conjunctions.

        Stage A: coarse RTN grid search across 6 directions × delta-v ladder.
        Stage B: local refinement around top candidates using scipy.optimize.

        Returns a persisted AvoidancePlan with the best maneuver recommendation.
        """
        started_at = perf_counter()
        logger.info(
            "Optimization start: norad_id=%s max_dv=%.2f window_h=%.1f",
            norad_id, max_delta_v_mps, burn_window_hours,
        )

        # Reuse caller-created plan when available, otherwise create one.
        plan: AvoidancePlan | None = None
        if plan_id is not None:
            plan = self.db.execute(
                select(AvoidancePlan).where(AvoidancePlan.id == plan_id)
            ).scalar_one_or_none()
        if plan is None:
            plan = AvoidancePlan(
                asset_norad_id=norad_id,
                status="running",
                progress_stage="initializing",
                progress_done=0,
                progress_total=None,
                progress_message="Loading conjunction context",
                heartbeat_at=datetime.utcnow(),
                optimizer_version="v1",
            )
            self.db.add(plan)
        else:
            if plan.status == "cancelled":
                logger.info("Optimization not started: plan_id=%s is already cancelled", plan.id)
                return plan
            plan.status = "running"
            plan.error_message = None
            plan.completed_at = None
            plan.progress_stage = "initializing"
            plan.progress_done = 0
            plan.progress_total = None
            plan.progress_message = "Loading conjunction context"
            plan.heartbeat_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(plan)

        try:
            result = self._optimize_inner(
                norad_id, plan, max_delta_v_mps, burn_window_hours,
                weight_miss_distance, weight_delta_v, top_n_events, started_at,
            )
            plan.status = "completed"
            plan.completed_at = datetime.utcnow()
            plan.optimization_elapsed_s = perf_counter() - started_at
            plan.progress_stage = "completed"
            plan.progress_done = 1
            plan.progress_total = 1
            plan.progress_message = "Optimization completed"
            plan.heartbeat_at = datetime.utcnow()
            logger.info(
                "Optimization complete: norad_id=%s plan_id=%s elapsed_s=%.2f candidates=%s",
                norad_id, plan.id, plan.optimization_elapsed_s,
                plan.candidates_evaluated,
            )
        except OptimizationCancelled as exc:
            plan.status = "cancelled"
            plan.error_message = str(exc)[:500]
            plan.completed_at = datetime.utcnow()
            plan.optimization_elapsed_s = perf_counter() - started_at
            plan.progress_stage = "cancelled"
            plan.progress_done = 1
            plan.progress_total = 1
            plan.progress_message = "Optimization cancelled"
            plan.heartbeat_at = datetime.utcnow()
            logger.info(
                "Optimization cancelled: norad_id=%s plan_id=%s elapsed_s=%.2f",
                norad_id, plan.id, plan.optimization_elapsed_s,
            )
        except Exception as exc:
            plan.status = "failed"
            plan.error_message = str(exc)[:500]
            plan.completed_at = datetime.utcnow()
            plan.optimization_elapsed_s = perf_counter() - started_at
            plan.progress_stage = "failed"
            plan.progress_done = 1
            plan.progress_total = 1
            plan.progress_message = "Optimization failed"
            plan.heartbeat_at = datetime.utcnow()
            logger.error(
                "Optimization failed: norad_id=%s plan_id=%s error=%s elapsed_s=%.2f",
                norad_id, plan.id, exc, plan.optimization_elapsed_s,
            )

        self.db.commit()
        self.db.refresh(plan)
        return plan

    def _optimize_inner(
        self,
        norad_id: int,
        plan: AvoidancePlan,
        max_delta_v_mps: float,
        burn_window_hours: float,
        weight_miss: float,
        weight_dv: float,
        top_n_events: int,
        started_at: float,
    ) -> None:
        """Core optimization logic (runs inside try/except wrapper)."""
        last_progress_commit = perf_counter()

        def _update_progress(
            stage: str,
            message: str | None = None,
            done: int | None = None,
            total: int | None = None,
            force: bool = False,
        ) -> None:
            nonlocal last_progress_commit
            if plan.id is not None:
                current_status = self.db.execute(
                    select(AvoidancePlan.status).where(AvoidancePlan.id == plan.id)
                ).scalar_one_or_none()
                if current_status == "cancelled":
                    raise OptimizationCancelled(f"Plan {plan.id} cancelled by user")
            if stage:
                plan.progress_stage = stage
            if message is not None:
                plan.progress_message = message
            if done is not None:
                plan.progress_done = done
            if total is not None:
                plan.progress_total = total
            now_perf = perf_counter()
            if force or (now_perf - last_progress_commit) >= 1.0:
                plan.optimization_elapsed_s = now_perf - started_at
                plan.heartbeat_at = datetime.utcnow()
                self.db.commit()
                last_progress_commit = now_perf

        def _load_events() -> list[ConjunctionEvent]:
            # A small grace window avoids missing near-now events due to clock skew.
            tca_floor = datetime.utcnow() - timedelta(minutes=5)
            return self.db.execute(
                select(ConjunctionEvent)
                .where(
                    ConjunctionEvent.defended_norad_id == norad_id,
                    ConjunctionEvent.tca_utc >= tca_floor,
                )
                .order_by(ConjunctionEvent.miss_distance_km.asc())
                .limit(top_n_events)
            ).scalars().all()

        # Find top-risk conjunction events for this asset.
        _update_progress("loading_events", "Searching for upcoming conjunction events", done=0, total=1, force=True)
        events = _load_events()

        if not events:
            screening_days = max(
                1,
                min(
                    settings.propagation_horizon_days,
                    int(np.ceil(max(burn_window_hours, 24.0) / 24.0)),
                ),
            )
            logger.info(
                "Optimization pre-screen: no upcoming cached events for %s, running screening (days=%s)",
                norad_id,
                screening_days,
            )
            _update_progress(
                "screening",
                f"No cached events found. Running screening for {screening_days} day(s)",
                done=0,
                total=1,
                force=True,
            )
            ScreeningEngine(
                self.db,
                PropagateEngine(screening_days, settings.propagation_resolution_seconds),
            ).find_conjunctions(norad_id, days=screening_days)
            events = _load_events()

        if not events:
            raise ValueError(
                f"No upcoming conjunction events for asset {norad_id} after screening refresh"
            )

        now = datetime.utcnow()

        # Use the highest-risk event as the primary target
        target_event = events[0]
        plan.event_id = target_event.id
        plan.pre_miss_distance_km = target_event.miss_distance_km
        pre_pc_ref = target_event.pc_foster

        # Load TLEs
        defended_tle = self.db.execute(
            select(TLERecord).where(TLERecord.norad_id == target_event.defended_norad_id)
        ).scalar_one_or_none()
        intruder_tle = self.db.execute(
            select(TLERecord).where(TLERecord.norad_id == target_event.intruder_norad_id)
        ).scalar_one_or_none()

        if defended_tle is None or intruder_tle is None:
            raise ValueError("TLE records not found for conjunction event")

        tca = self._tca_from_event(target_event)
        if tca is None:
            raise ValueError("Could not reconstruct TCA from event data")

        # Get covariance
        from sgp4.api import Satrec
        sat_d = Satrec.twoline2rv(defended_tle.line1, defended_tle.line2)
        sat_i = Satrec.twoline2rv(intruder_tle.line1, intruder_tle.line2)
        epoch_age_d = max(0.0, (now - defended_tle.updated_at).total_seconds() / 86400.0)
        epoch_age_i = max(0.0, (now - intruder_tle.updated_at).total_seconds() / 86400.0)
        cov_d = estimate_covariance_from_tle(sat_d.bstar, epoch_age_d)
        cov_i = estimate_covariance_from_tle(sat_i.bstar, epoch_age_i)

        # Pre-maneuver Pc is invariant across candidate maneuvers; compute once.
        if pre_pc_ref is None:
            pre_bplane = compute_bplane(tca)
            pre_cov_bp = project_covariance_to_bplane(
                cov_d.cov_3x3_pos,
                cov_i.cov_3x3_pos,
                pre_bplane,
            )
            pre_miss_bp = np.array([pre_bplane.b_dot_t_km, pre_bplane.b_dot_r_km])
            pre_pc_ref = compute_pc_foster(pre_miss_bp, pre_cov_bp).pc
        plan.pre_pc = pre_pc_ref

        # Build safety contexts for all upcoming events so final selection
        # avoids creating a new close approach with another intruder.
        event_contexts: list[_EventContext] = [
            _EventContext(
                event_id=target_event.id,
                intruder_norad_id=target_event.intruder_norad_id,
                intruder_tle=intruder_tle,
                tca=tca,
                cov_secondary=cov_i,
                pre_pc=float(pre_pc_ref or 0.0),
            )
        ]
        intruder_tle_cache: dict[int, TLERecord] = {target_event.intruder_norad_id: intruder_tle}

        for event in events[1:]:
            intr_tle = intruder_tle_cache.get(event.intruder_norad_id)
            if intr_tle is None:
                intr_tle = self.db.execute(
                    select(TLERecord).where(TLERecord.norad_id == event.intruder_norad_id)
                ).scalar_one_or_none()
                if intr_tle is not None:
                    intruder_tle_cache[event.intruder_norad_id] = intr_tle
            if intr_tle is None:
                logger.warning(
                    "Safety context skipped: event_id=%s intruder_tle_missing=%s",
                    event.id,
                    event.intruder_norad_id,
                )
                continue

            event_tca = self._tca_from_event(event)
            if event_tca is None:
                logger.warning("Safety context skipped: event_id=%s missing_tca", event.id)
                continue

            sat_i_ctx = Satrec.twoline2rv(intr_tle.line1, intr_tle.line2)
            epoch_age_i_ctx = max(0.0, (now - intr_tle.updated_at).total_seconds() / 86400.0)
            cov_i_ctx = estimate_covariance_from_tle(sat_i_ctx.bstar, epoch_age_i_ctx)

            pre_pc_ctx = event.pc_foster
            if pre_pc_ctx is None:
                try:
                    pre_bplane_ctx = compute_bplane(event_tca)
                    pre_cov_bp_ctx = project_covariance_to_bplane(
                        cov_d.cov_3x3_pos,
                        cov_i_ctx.cov_3x3_pos,
                        pre_bplane_ctx,
                    )
                    pre_miss_bp_ctx = np.array(
                        [pre_bplane_ctx.b_dot_t_km, pre_bplane_ctx.b_dot_r_km]
                    )
                    pre_pc_ctx = compute_pc_foster(pre_miss_bp_ctx, pre_cov_bp_ctx).pc
                except Exception:
                    pre_pc_ctx = float(pre_pc_ref or _ALLOWED_PC_BASE)

            event_contexts.append(
                _EventContext(
                    event_id=event.id,
                    intruder_norad_id=event.intruder_norad_id,
                    intruder_tle=intr_tle,
                    tca=event_tca,
                    cov_secondary=cov_i_ctx,
                    pre_pc=float(pre_pc_ctx or 0.0),
                )
            )

        if not event_contexts:
            raise ValueError("No valid conjunction contexts available for safety evaluation")

        # Filter delta-v ladder to max budget
        dv_ladder = [dv for dv in _DV_LADDER if dv <= max_delta_v_mps]
        if not dv_ladder or dv_ladder[-1] < max_delta_v_mps:
            dv_ladder.append(max_delta_v_mps)

        # Burn lead times to evaluate
        lead_times = [h for h in [6.0, 12.0, 24.0, 48.0] if h <= burn_window_hours]
        if not lead_times:
            lead_times = [burn_window_hours]

        # ---- Stage A: coarse grid search ----
        total_stage_a = len(BURN_DIRECTIONS) * len(dv_ladder) * len(lead_times)
        _update_progress(
            "stage_a",
            "Coarse search: evaluating maneuver grid",
            done=0,
            total=max(1, total_stage_a),
            force=True,
        )
        logger.info(
            "Stage A (coarse): directions=%s dv_steps=%s lead_times=%s",
            len(BURN_DIRECTIONS), len(dv_ladder), len(lead_times),
        )
        stage_a_start = perf_counter()
        candidates: list[_CandidateScore] = []
        best_score = float('inf')
        n_evaluated = 0

        for direction, rtn_unit in BURN_DIRECTIONS.items():
            for dv in dv_ladder:
                for lead_h in lead_times:
                    n_evaluated += 1
                    try:
                        result = simulate_maneuver(
                            defended_tle.line1, defended_tle.line2,
                            intruder_tle.line1, intruder_tle.line2,
                            tca, dv, direction, lead_h,
                            cov_d, cov_i,
                            pre_pc_override=pre_pc_ref,
                            propagation_step_seconds=_FAST_OPT_STEP_SECONDS,
                            post_tca_pad_hours=_FAST_OPT_POST_PAD_HOURS,
                            store_post_trajectory=False,
                        )
                        # Scoring: lower is better
                        # Normalize miss distance improvement (negative = good)
                        miss_improvement = -(result.post_miss_distance_km - tca.miss_distance_km)
                        score = (
                            -weight_miss * miss_improvement
                            + weight_dv * (dv / max_delta_v_mps)
                        )
                        # Penalize if Pc actually increased
                        if result.post_pc > (target_event.pc_foster or 1.0):
                            score += 10.0

                        candidate = _CandidateScore(
                            direction=direction,
                            dv_mps=dv,
                            burn_lead_hours=lead_h,
                            miss_distance_km=result.post_miss_distance_km,
                            pc=result.post_pc,
                            score=score,
                            rtn_vector=(rtn_unit * dv).tolist(),
                        )
                        candidates.append(candidate)

                        # Early-exit: track best score
                        if score < best_score:
                            best_score = score
                            logger.info(
                                "Stage A new best: dir=%s dv=%.2f lead=%.0fh miss=%.3f pc=%.2e score=%.4f",
                                direction, dv, lead_h,
                                result.post_miss_distance_km, result.post_pc, score,
                            )
                    except (ValueError, RuntimeError) as exc:
                        logger.debug("Stage A skip: dir=%s dv=%.2f err=%s", direction, dv, exc)
                        continue
                    if (n_evaluated % 25) == 0:
                        _update_progress("stage_a", done=n_evaluated)

        _update_progress("stage_a", done=n_evaluated, force=True)

        logger.info(
            "Stage A complete: evaluated=%s viable=%s elapsed_s=%.2f",
            n_evaluated, len(candidates), perf_counter() - stage_a_start,
        )

        if not candidates:
            raise ValueError("No viable maneuver candidates found in coarse search")

        # Sort by score and take top-3 for refinement
        candidates.sort(key=lambda c: c.score)
        top_candidates = candidates[:3]

        # ---- Stage B: local refinement ----
        _update_progress(
            "stage_b",
            "Refining top maneuver candidates",
            done=0,
            total=len(top_candidates),
            force=True,
        )
        logger.info("Stage B (refine): refining top %s candidates", len(top_candidates))
        stage_b_start = perf_counter()
        refined_candidates: list[_CandidateScore] = []

        for idx, cand in enumerate(top_candidates, start=1):
            try:
                refined = self._refine_candidate(
                    defended_tle, intruder_tle, tca,
                    cand, max_delta_v_mps, burn_window_hours,
                    weight_miss, weight_dv,
                    cov_d, cov_i,
                    pre_pc_ref,
                )
                if refined is not None:
                    refined_candidates.append(refined)
                    n_evaluated += 1
            except Exception as exc:
                logger.debug("Stage B refinement failed for %s: %s", cand.direction, exc)
            _update_progress("stage_b", done=idx, total=len(top_candidates))

        logger.info(
            "Stage B complete: elapsed_s=%.2f",
            perf_counter() - stage_b_start,
        )

        # Safety gate: choose the best candidate that stays safe across all
        # event contexts (not only the top-priority event).
        finalist_map: dict[tuple[str, int, int], _CandidateScore] = {}
        for cand in [*top_candidates, *refined_candidates]:
            key = (
                cand.direction,
                int(round(cand.dv_mps * 1000)),
                int(round(cand.burn_lead_hours * 100)),
            )
            prev = finalist_map.get(key)
            if prev is None or cand.score < prev.score:
                finalist_map[key] = cand
        finalists = sorted(finalist_map.values(), key=lambda c: c.score)

        safety_cache: dict[tuple[str, int, int], tuple[bool, float, float, float, int]] = {}

        def _evaluate_candidate_safety(
            candidate: _CandidateScore,
        ) -> tuple[bool, float, float, float, int]:
            key = (
                candidate.direction,
                int(round(candidate.dv_mps * 1000)),
                int(round(candidate.burn_lead_hours * 100)),
            )
            cached = safety_cache.get(key)
            if cached is not None:
                return cached

            hard_unsafe = False
            penalty = 0.0
            worst_miss_km = float("inf")
            worst_pc = 0.0
            evaluated = 0

            for ctx in event_contexts:
                try:
                    safety_result = simulate_maneuver(
                        defended_tle.line1, defended_tle.line2,
                        ctx.intruder_tle.line1, ctx.intruder_tle.line2,
                        ctx.tca,
                        candidate.dv_mps,
                        candidate.direction,
                        candidate.burn_lead_hours,
                        cov_d,
                        ctx.cov_secondary,
                        pre_pc_override=ctx.pre_pc,
                        propagation_step_seconds=_FAST_OPT_STEP_SECONDS,
                        post_tca_pad_hours=_FAST_OPT_POST_PAD_HOURS,
                        store_post_trajectory=False,
                    )
                except (ValueError, RuntimeError):
                    hard_unsafe = True
                    penalty += 100.0
                    continue

                evaluated += 1
                miss = float(safety_result.post_miss_distance_km)
                pc = float(safety_result.post_pc)
                worst_miss_km = min(worst_miss_km, miss)
                worst_pc = max(worst_pc, pc)

                if miss <= _HARD_COLLISION_MISS_KM or pc >= _HARD_COLLISION_PC:
                    hard_unsafe = True
                if miss < _SAFETY_MISS_FLOOR_KM:
                    penalty += (_SAFETY_MISS_FLOOR_KM - miss) * 6.0

                allowed_pc = max(_ALLOWED_PC_BASE, ctx.pre_pc * 1.15)
                if pc > allowed_pc:
                    penalty += min(25.0, (pc - allowed_pc) * 2.0e5)

            if evaluated == 0:
                hard_unsafe = True
                penalty += 1000.0

            result = (hard_unsafe, penalty, worst_miss_km, worst_pc, evaluated)
            safety_cache[key] = result
            return result

        winner: _CandidateScore | None = None
        winner_score = float("inf")
        winner_diag: tuple[float, float, int, float] | None = None
        for cand in finalists:
            hard_unsafe, penalty, worst_miss, worst_pc, evaluated = _evaluate_candidate_safety(cand)
            combined_score = cand.score + penalty
            logger.info(
                "Safety eval: dir=%s dv=%.3f lead=%.1fh base=%.4f penalty=%.3f worst_miss=%.3f worst_pc=%.2e evaluated=%s unsafe=%s",
                cand.direction,
                cand.dv_mps,
                cand.burn_lead_hours,
                cand.score,
                penalty,
                worst_miss,
                worst_pc,
                evaluated,
                hard_unsafe,
            )
            if hard_unsafe:
                continue
            if combined_score < winner_score:
                winner = cand
                winner_score = combined_score
                winner_diag = (worst_miss, worst_pc, evaluated, penalty)

        if winner is None:
            raise ValueError(
                "No collision-safe maneuver found within current limits. "
                "Try increasing max delta-v or burn window."
            )

        if winner_diag is not None:
            logger.info(
                "Winner safety summary: worst_miss=%.3f km worst_pc=%.2e contexts=%s penalty=%.3f",
                winner_diag[0],
                winner_diag[1],
                winner_diag[2],
                winner_diag[3],
            )

        # Run final simulation to get trajectories
        final_result = simulate_maneuver(
            defended_tle.line1, defended_tle.line2,
            intruder_tle.line1, intruder_tle.line2,
            tca, winner.dv_mps, winner.direction, winner.burn_lead_hours,
            cov_d, cov_i,
            pre_pc_override=pre_pc_ref,
            propagation_step_seconds=_FINAL_PATH_STEP_SECONDS,
            post_tca_pad_hours=_FINAL_PATH_POST_PAD_HOURS,
            store_post_trajectory=True,
        )
        _update_progress("finalizing", "Persisting optimization results", done=1, total=1, force=True)

        # Build canonical, time-ordered ECI trajectory polylines for map display.
        # Use the same sampling grid for nominal and deviated paths to avoid visual zig-zags.
        def _sanitize_path(samples: list[dict] | None) -> list[dict]:
            clean: list[dict] = []
            if not samples:
                return clean
            for sample in samples:
                t = sample.get("t")
                pos = sample.get("position_km")
                if not isinstance(t, str) or not isinstance(pos, (list, tuple)) or len(pos) != 3:
                    continue
                try:
                    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                except (TypeError, ValueError):
                    continue
                if not np.isfinite([x, y, z]).all():
                    continue
                clean.append({"t": t, "position_km": [x, y, z]})

            clean.sort(key=lambda s: s["t"])
            deduped: list[dict] = []
            for sample in clean:
                if deduped and sample["t"] == deduped[-1]["t"]:
                    deduped[-1] = sample
                else:
                    deduped.append(sample)
            return deduped

        def _parse_iso_utc_to_ts(value: str) -> float | None:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except Exception:
                return None

        def _clip_to_orbit_window(samples: list[dict], period_s: float, start_epoch: datetime) -> list[dict]:
            if len(samples) <= 2:
                return samples
            start_ts = start_epoch.timestamp()
            end_ts = start_ts + period_s
            clipped = []
            for sample in samples:
                ts = _parse_iso_utc_to_ts(sample["t"])
                if ts is None:
                    continue
                if start_ts <= ts <= end_ts:
                    clipped.append(sample)
            if len(clipped) >= 2:
                return clipped
            fallback_n = max(2, int(period_s / max(_FINAL_PATH_STEP_SECONDS, 1.0)) + 1)
            return samples[:fallback_n]

        nominal_traj = None
        nominal_hours = winner.burn_lead_hours + max(1.0, _FINAL_PATH_POST_PAD_HOURS)
        try:
            nominal_traj = apply_impulsive_burn_sgp4(
                defended_tle.line1,
                defended_tle.line2,
                final_result.burn_epoch,
                np.zeros(3, dtype=float),
                propagation_duration_hours=nominal_hours,
                step_seconds=_FINAL_PATH_STEP_SECONDS,
            )
        except Exception as exc:
            logger.warning("Nominal trajectory generation failed; falling back to pre_trajectory: %s", exc)

        current_path = _sanitize_path(nominal_traj)
        if not current_path:
            current_path = _sanitize_path(target_event.pre_trajectory)

        deviated_path = _sanitize_path(final_result.post_trajectory)

        # Keep one orbital revolution for clear, single-track rendering.
        mean_motion = float(defended_tle.mean_motion or 15.0)
        if mean_motion <= 0:
            mean_motion = 15.0
        orbit_period_s = float(np.clip(86400.0 / mean_motion, 3600.0, 10800.0))
        current_path = _clip_to_orbit_window(current_path, orbit_period_s, final_result.burn_epoch)
        deviated_path = _clip_to_orbit_window(deviated_path, orbit_period_s, final_result.burn_epoch)

        # Populate plan
        plan.burn_direction = winner.direction
        plan.burn_dv_mps = winner.dv_mps
        plan.burn_rtn_vector = winner.rtn_vector
        plan.burn_epoch = final_result.burn_epoch
        plan.post_miss_distance_km = final_result.post_miss_distance_km
        plan.post_pc = final_result.post_pc
        plan.fuel_cost_kg = final_result.fuel_cost_kg
        plan.current_path = current_path
        plan.deviated_path = deviated_path
        plan.candidates_evaluated = n_evaluated

        # Also persist to the conjunction event
        target_event.post_maneuver_delta_v_mps = winner.dv_mps
        target_event.post_maneuver_direction = winner.direction
        target_event.post_maneuver_burn_epoch = final_result.burn_epoch
        target_event.post_maneuver_pc = final_result.post_pc
        target_event.post_maneuver_miss_distance_km = final_result.post_miss_distance_km
        target_event.post_maneuver_fuel_cost_kg = final_result.fuel_cost_kg
        target_event.post_maneuver_trajectory = final_result.post_trajectory
        target_event.post_trajectory = final_result.post_trajectory
        target_event.post_miss_distance_km = final_result.post_miss_distance_km
        target_event.maneuver_confidence = "high" if winner.pc < 1e-5 else "medium"
        target_event.maneuver_optimizer_version = "v1"

        logger.info(
            "Optimization winner: dir=%s dv=%.3f m/s lead=%.0fh miss=%.3f→%.3f km pc=%.2e→%.2e fuel=%.3f kg",
            winner.direction, winner.dv_mps, winner.burn_lead_hours,
            tca.miss_distance_km, final_result.post_miss_distance_km,
            target_event.pc_foster or 0, final_result.post_pc,
            final_result.fuel_cost_kg,
        )

    def _refine_candidate(
        self,
        defended_tle: TLERecord,
        intruder_tle: TLERecord,
        tca: TCAResult,
        candidate: _CandidateScore,
        max_dv: float,
        max_lead_h: float,
        weight_miss: float,
        weight_dv: float,
        cov_d,
        cov_i,
        pre_pc_ref: float | None,
    ) -> Optional[_CandidateScore]:
        """Stage B: local continuous refinement using Nelder-Mead.

        Optimizes [dv_magnitude, burn_lead_hours] around a coarse candidate.
        """
        rtn_unit = BURN_DIRECTIONS[candidate.direction]

        def objective(params):
            dv, lead_h = params
            if dv < 0.01 or dv > max_dv or lead_h < 1.0 or lead_h > max_lead_h:
                return 100.0
            try:
                result = simulate_maneuver(
                    defended_tle.line1, defended_tle.line2,
                    intruder_tle.line1, intruder_tle.line2,
                    tca, float(dv), candidate.direction, float(lead_h),
                    cov_d, cov_i,
                    pre_pc_override=pre_pc_ref,
                    propagation_step_seconds=_FAST_OPT_STEP_SECONDS,
                    post_tca_pad_hours=_FAST_OPT_POST_PAD_HOURS,
                    store_post_trajectory=False,
                )
                miss_improvement = -(result.post_miss_distance_km - tca.miss_distance_km)
                score = -weight_miss * miss_improvement + weight_dv * (dv / max_dv)
                if result.post_pc > 1e-3:
                    score += 10.0
                return score
            except (ValueError, RuntimeError):
                return 100.0

        x0 = [candidate.dv_mps, candidate.burn_lead_hours]
        res = minimize(
            objective, x0, method="Nelder-Mead",
            options={"maxiter": 30, "xatol": 0.01, "fatol": 0.001},
        )

        if not res.success and res.fun >= candidate.score:
            return None

        opt_dv, opt_lead = float(res.x[0]), float(res.x[1])
        opt_dv = max(0.01, min(opt_dv, max_dv))
        opt_lead = max(1.0, min(opt_lead, max_lead_h))

        # Evaluate at refined point
        try:
            result = simulate_maneuver(
                defended_tle.line1, defended_tle.line2,
                intruder_tle.line1, intruder_tle.line2,
                tca, opt_dv, candidate.direction, opt_lead,
                cov_d, cov_i,
                pre_pc_override=pre_pc_ref,
                propagation_step_seconds=_FAST_OPT_STEP_SECONDS,
                post_tca_pad_hours=_FAST_OPT_POST_PAD_HOURS,
                store_post_trajectory=False,
            )
        except (ValueError, RuntimeError):
            return None

        return _CandidateScore(
            direction=candidate.direction,
            dv_mps=opt_dv,
            burn_lead_hours=opt_lead,
            miss_distance_km=result.post_miss_distance_km,
            pc=result.post_pc,
            score=res.fun,
            rtn_vector=(rtn_unit * opt_dv).tolist(),
        )
