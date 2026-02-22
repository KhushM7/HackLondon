"""Maneuver planning for collision avoidance.

Replaces the old AvoidanceSimulator with physically correct impulsive burns
in the RTN (Radial-Tangential-Normal) frame.

Features:
- Impulsive delta-v burns in 6 RTN directions
- Minimum delta-v optimization to achieve target Pc
- Fuel cost via Tsiolkovsky rocket equation
- Maneuver trade study across all directions

Reference:
    - Vallado (2013), Chapter 6 — Orbital Maneuvers
    - Tsiolkovsky rocket equation for fuel cost
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Literal

import numpy as np
from scipy.optimize import minimize_scalar
from sgp4.api import Satrec, jday

from .bplane_analysis import compute_bplane
from .covariance_handler import (
    CovarianceInfo,
    default_covariance,
    project_covariance_to_bplane,
)
from .orbital_constants import (
    GM_EARTH_KM3_S2,
    R_EARTH_KM,
    G0_M_S2,
    DEFAULT_ISP_S,
    DEFAULT_HBR_KM,
    PropagationConfig,
    eci_to_rtn,
    rtn_to_eci,
    classify_risk,
)
from .pc_calculator import compute_pc_foster
from .tca_finder import TCAResult, _sgp4_state_at, find_tca_sgp4

logger = logging.getLogger(__name__)

# Burn direction vectors in RTN frame
BURN_DIRECTIONS = {
    "prograde": np.array([0.0, 1.0, 0.0]),
    "retrograde": np.array([0.0, -1.0, 0.0]),
    "radial_plus": np.array([1.0, 0.0, 0.0]),
    "radial_minus": np.array([-1.0, 0.0, 0.0]),
    "normal_plus": np.array([0.0, 0.0, 1.0]),
    "normal_minus": np.array([0.0, 0.0, -1.0]),
}


@dataclass
class ManeuverResult:
    """Result of a collision avoidance maneuver computation.

    Attributes:
        delta_v_mps: Magnitude of the delta-v burn [m/s].
        direction: RTN direction name.
        burn_epoch: UTC epoch of the burn.
        pre_pc: Pre-maneuver probability of collision.
        post_pc: Post-maneuver probability of collision.
        post_miss_distance_km: Post-maneuver miss distance [km].
        post_tca_epoch: Post-maneuver TCA epoch.
        fuel_cost_kg: Fuel expenditure [kg] (Tsiolkovsky).
        post_trajectory: Post-maneuver trajectory samples.
        risk_level: Post-maneuver risk level.
    """
    delta_v_mps: float
    direction: str
    burn_epoch: datetime
    pre_pc: float
    post_pc: float
    post_miss_distance_km: float
    post_tca_epoch: datetime
    fuel_cost_kg: float
    post_trajectory: list[dict]
    risk_level: str


def tsiolkovsky_fuel_cost(
    delta_v_mps: float,
    mass_kg: float = 500.0,
    isp_s: float = DEFAULT_ISP_S,
) -> float:
    """Compute fuel expenditure using the Tsiolkovsky rocket equation.

    Δm = m₀ × (1 - e^(-Δv / (Isp × g₀)))

    Parameters:
        delta_v_mps: Delta-v magnitude [m/s].
        mass_kg: Initial satellite mass [kg].
        isp_s: Specific impulse [s].

    Returns:
        Fuel mass expended [kg].
    """
    ve = isp_s * G0_M_S2  # exhaust velocity [m/s]
    return mass_kg * (1.0 - np.exp(-delta_v_mps / ve))


def apply_impulsive_burn_sgp4(
    line1: str,
    line2: str,
    burn_epoch: datetime,
    delta_v_rtn_mps: np.ndarray,
    propagation_duration_hours: float = 72.0,
    step_seconds: float = 60.0,
) -> list[dict]:
    """Apply an impulsive burn to an SGP4 satellite and propagate the result.

    Since SGP4 doesn't support delta-v injection natively, we:
    1. Propagate to the burn epoch using SGP4.
    2. Extract state vector at burn epoch.
    3. Transform delta-v from RTN to ECI and add to velocity.
    4. Propagate new state using two-body + J2 (simplified Cowell for post-burn).

    Parameters:
        line1: TLE line 1.
        line2: TLE line 2.
        burn_epoch: UTC time of the burn.
        delta_v_rtn_mps: Delta-v vector in RTN frame [m/s].
        propagation_duration_hours: How long to propagate after burn [hours].
        step_seconds: Time step for output [s].

    Returns:
        List of trajectory sample dicts with position_km, velocity_kms, t.
    """
    sat = Satrec.twoline2rv(line1, line2)
    r0, v0 = _sgp4_state_at(sat, burn_epoch)

    # Convert delta-v from RTN to ECI
    rtn2eci = rtn_to_eci(r0, v0)
    dv_eci_kms = rtn2eci @ (delta_v_rtn_mps / 1000.0)  # m/s → km/s

    # Apply burn
    v0_new = v0 + dv_eci_kms

    # Check for deorbit risk
    r_mag = np.linalg.norm(r0)
    v_mag = np.linalg.norm(v0_new)
    energy = 0.5 * v_mag**2 - GM_EARTH_KM3_S2 / r_mag
    a = -GM_EARTH_KM3_S2 / (2.0 * energy) if energy != 0 else r_mag
    h_vec = np.cross(r0, v0_new)
    h_mag = np.linalg.norm(h_vec)
    e_mag = np.sqrt(max(0, 1.0 + 2.0 * energy * h_mag**2 / GM_EARTH_KM3_S2**2))
    perigee_km = a * (1.0 - e_mag) - R_EARTH_KM

    if perigee_km < 100.0:
        raise ValueError(
            f"Maneuver would result in deorbit (perigee altitude {perigee_km:.1f} km < 100 km)"
        )

    # Simple two-body + J2 propagation for post-burn trajectory
    from .cowell_propagator import propagate_cowell
    config = PropagationConfig(
        step_seconds=step_seconds,
        use_drag=False,
        use_srp=False,
        use_third_body=False,
    )
    duration_s = propagation_duration_hours * 3600.0
    times, positions, velocities = propagate_cowell(
        r0, v0_new, duration_s, config, step_seconds
    )

    samples = []
    for i in range(len(times)):
        t_dt = burn_epoch + timedelta(seconds=float(times[i]))
        samples.append({
            "t": t_dt.isoformat() + "Z",
            "position_km": positions[i].tolist(),
            "velocity_kms": velocities[i].tolist(),
        })

    return samples


def simulate_maneuver(
    line1_primary: str,
    line2_primary: str,
    line1_secondary: str,
    line2_secondary: str,
    tca: TCAResult,
    delta_v_mps: float,
    direction: str = "prograde",
    burn_lead_time_hours: float = 24.0,
    cov_primary: Optional[CovarianceInfo] = None,
    cov_secondary: Optional[CovarianceInfo] = None,
    hard_body_radius_km: float = DEFAULT_HBR_KM,
    mass_kg: float = 500.0,
    pre_pc_override: Optional[float] = None,
    propagation_step_seconds: float = 60.0,
    post_tca_pad_hours: float = 24.0,
    store_post_trajectory: bool = True,
) -> ManeuverResult:
    """Simulate a collision avoidance maneuver and compute post-maneuver Pc.

    Parameters:
        line1_primary: TLE line 1 for primary (defended) object.
        line2_primary: TLE line 2 for primary.
        line1_secondary: TLE line 1 for secondary (intruder).
        line2_secondary: TLE line 2 for secondary.
        tca: Pre-maneuver TCA result.
        delta_v_mps: Delta-v magnitude [m/s].
        direction: RTN direction name.
        burn_lead_time_hours: Time before TCA to execute burn [hours].
        cov_primary: Primary covariance (estimated if None).
        cov_secondary: Secondary covariance (estimated if None).
        hard_body_radius_km: Combined hard-body radius [km].
        mass_kg: Satellite mass [kg].

    Returns:
        ManeuverResult with post-maneuver analysis.
    """
    if direction not in BURN_DIRECTIONS:
        raise ValueError(f"Invalid direction '{direction}'. Must be one of {list(BURN_DIRECTIONS.keys())}")

    # Burn epoch
    burn_epoch = tca.tca_epoch - timedelta(hours=burn_lead_time_hours)
    if burn_epoch < datetime.utcnow():
        burn_epoch = datetime.utcnow() + timedelta(minutes=5)

    # Delta-v vector in RTN
    dv_rtn = BURN_DIRECTIONS[direction] * delta_v_mps

    # Apply burn and get post-burn trajectory
    # Only needs to span burn->TCA plus a post-TCA tail.
    propagation_hours = burn_lead_time_hours + max(1.0, post_tca_pad_hours)
    post_traj = apply_impulsive_burn_sgp4(
        line1_primary, line2_primary,
        burn_epoch, dv_rtn,
        propagation_duration_hours=propagation_hours,
        step_seconds=propagation_step_seconds,
    )

    # Propagate secondary to find new TCA
    sat2 = Satrec.twoline2rv(line1_secondary, line2_secondary)
    post_traj_np = np.array([s["position_km"] for s in post_traj])
    post_vel_np = np.array([s["velocity_kms"] for s in post_traj])
    times_np = np.array([(datetime.fromisoformat(s["t"].replace("Z", "")) - burn_epoch).total_seconds() for s in post_traj])

    # Get secondary positions at same times
    sec_pos = []
    sec_vel = []
    for s in post_traj:
        t_dt = datetime.fromisoformat(s["t"].replace("Z", ""))
        try:
            r2, v2 = _sgp4_state_at(sat2, t_dt)
            sec_pos.append(r2)
            sec_vel.append(v2)
        except RuntimeError:
            sec_pos.append(np.array([1e12, 0, 0]))
            sec_vel.append(np.zeros(3))

    sec_pos_np = np.array(sec_pos)
    sec_vel_np = np.array(sec_vel)

    # Find post-maneuver TCA
    from .tca_finder import find_tca_from_states
    post_tca = find_tca_from_states(
        times_np, post_traj_np, post_vel_np,
        sec_pos_np, sec_vel_np,
        burn_epoch,
    )

    if post_tca is None:
        post_miss = float(np.min(np.linalg.norm(post_traj_np - sec_pos_np, axis=1)))
        post_tca_epoch = tca.tca_epoch
        post_pc = 0.0
    else:
        post_miss = post_tca.miss_distance_km
        post_tca_epoch = post_tca.tca_epoch

        # Compute post-maneuver Pc
        bplane = compute_bplane(post_tca)
        cov_p = cov_primary or default_covariance()
        cov_s = cov_secondary or default_covariance()
        cov_bp = project_covariance_to_bplane(
            cov_p.cov_3x3_pos, cov_s.cov_3x3_pos, bplane
        )
        miss_bp = np.array([bplane.b_dot_t_km, bplane.b_dot_r_km])
        pc_result = compute_pc_foster(miss_bp, cov_bp, hard_body_radius_km)
        post_pc = pc_result.pc

    if pre_pc_override is None:
        # Pre-maneuver Pc
        pre_bplane = compute_bplane(tca)
        cov_p = cov_primary or default_covariance()
        cov_s = cov_secondary or default_covariance()
        pre_cov_bp = project_covariance_to_bplane(
            cov_p.cov_3x3_pos, cov_s.cov_3x3_pos, pre_bplane
        )
        pre_miss_bp = np.array([pre_bplane.b_dot_t_km, pre_bplane.b_dot_r_km])
        pre_pc_result = compute_pc_foster(pre_miss_bp, pre_cov_bp, hard_body_radius_km)
        pre_pc = pre_pc_result.pc
    else:
        pre_pc = pre_pc_override

    fuel = tsiolkovsky_fuel_cost(delta_v_mps, mass_kg)
    risk = classify_risk(post_pc)

    return ManeuverResult(
        delta_v_mps=delta_v_mps,
        direction=direction,
        burn_epoch=burn_epoch,
        pre_pc=pre_pc,
        post_pc=post_pc,
        post_miss_distance_km=post_miss,
        post_tca_epoch=post_tca_epoch,
        fuel_cost_kg=fuel,
        post_trajectory=post_traj if store_post_trajectory else [],
        risk_level=risk.value,
    )


def optimize_maneuver(
    line1_primary: str,
    line2_primary: str,
    line1_secondary: str,
    line2_secondary: str,
    tca: TCAResult,
    target_pc: float = 1e-4,
    direction: str = "prograde",
    max_dv_mps: float = 10.0,
    burn_lead_time_hours: float = 24.0,
    cov_primary: Optional[CovarianceInfo] = None,
    cov_secondary: Optional[CovarianceInfo] = None,
    hard_body_radius_km: float = DEFAULT_HBR_KM,
    mass_kg: float = 500.0,
) -> ManeuverResult:
    """Find the minimum delta-v that reduces Pc below a target threshold.

    Uses scalar optimization (Brent's method) over delta-v magnitude.

    Parameters:
        target_pc: Target maximum Pc (default 1e-4, NASA maneuver threshold).
        max_dv_mps: Maximum delta-v to consider [m/s].
        Other params: see simulate_maneuver.

    Returns:
        ManeuverResult with the optimal burn.
    """
    def pc_at_dv(dv: float) -> float:
        try:
            result = simulate_maneuver(
                line1_primary, line2_primary,
                line1_secondary, line2_secondary,
                tca, dv, direction, burn_lead_time_hours,
                cov_primary, cov_secondary, hard_body_radius_km, mass_kg,
            )
            return result.post_pc
        except Exception:
            return 1.0

    # Binary search: find smallest dv where Pc < target
    lo, hi = 0.01, max_dv_mps
    best_result = None

    # First check if max_dv achieves the target
    pc_max = pc_at_dv(hi)
    if pc_max >= target_pc:
        # Even max dv doesn't achieve target; return result at max dv
        logger.warning("Maximum delta-v (%.1f m/s) does not achieve target Pc", max_dv_mps)
        return simulate_maneuver(
            line1_primary, line2_primary,
            line1_secondary, line2_secondary,
            tca, hi, direction, burn_lead_time_hours,
            cov_primary, cov_secondary, hard_body_radius_km, mass_kg,
        )

    # Binary search for minimum dv
    for _ in range(20):  # ~1e-6 m/s precision
        mid = (lo + hi) / 2.0
        pc_mid = pc_at_dv(mid)
        if pc_mid < target_pc:
            hi = mid
        else:
            lo = mid

    optimal_dv = hi
    return simulate_maneuver(
        line1_primary, line2_primary,
        line1_secondary, line2_secondary,
        tca, optimal_dv, direction, burn_lead_time_hours,
        cov_primary, cov_secondary, hard_body_radius_km, mass_kg,
    )


def compute_trade_study(
    line1_primary: str,
    line2_primary: str,
    line1_secondary: str,
    line2_secondary: str,
    tca: TCAResult,
    dv_magnitudes_mps: Optional[list[float]] = None,
    burn_lead_time_hours: float = 24.0,
    cov_primary: Optional[CovarianceInfo] = None,
    cov_secondary: Optional[CovarianceInfo] = None,
    hard_body_radius_km: float = DEFAULT_HBR_KM,
    mass_kg: float = 500.0,
) -> list[dict]:
    """Compute maneuver trade study across all 6 RTN directions.

    Parameters:
        dv_magnitudes_mps: List of delta-v magnitudes to evaluate [m/s].
            Default: [0.1, 0.5, 1.0, 2.0, 5.0]
        Other params: see simulate_maneuver.

    Returns:
        List of dicts with direction, delta_v_mps, post_pc, post_miss_distance_km, fuel_cost_kg.
    """
    if dv_magnitudes_mps is None:
        dv_magnitudes_mps = [0.1, 0.5, 1.0, 2.0, 5.0]

    entries = []
    for direction in BURN_DIRECTIONS:
        for dv in dv_magnitudes_mps:
            try:
                result = simulate_maneuver(
                    line1_primary, line2_primary,
                    line1_secondary, line2_secondary,
                    tca, dv, direction, burn_lead_time_hours,
                    cov_primary, cov_secondary, hard_body_radius_km, mass_kg,
                )
                entries.append({
                    "direction": direction,
                    "delta_v_mps": dv,
                    "post_pc": result.post_pc,
                    "post_miss_distance_km": result.post_miss_distance_km,
                    "fuel_cost_kg": result.fuel_cost_kg,
                })
            except Exception as e:
                logger.warning("Trade study: %s @ %.1f m/s failed: %s", direction, dv, e)
                entries.append({
                    "direction": direction,
                    "delta_v_mps": dv,
                    "post_pc": float('nan'),
                    "post_miss_distance_km": float('nan'),
                    "fuel_cost_kg": tsiolkovsky_fuel_cost(dv, mass_kg),
                })

    return entries
