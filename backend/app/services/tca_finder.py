"""Time of Closest Approach (TCA) finder using root-finding.

Finds the precise time where d/dt |r_rel|^2 = 0 (i.e., r_rel · v_rel = 0)
using Brent's method for sub-second accuracy.

Reference: Vallado (2013), Chapter 10 — Close Approach Analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from sgp4.api import Satrec, jday


@dataclass
class TCAResult:
    """Result of a TCA (Time of Closest Approach) computation.

    Attributes:
        tca_epoch: UTC datetime at closest approach.
        miss_distance_km: Scalar miss distance [km].
        miss_vector_km: 3D miss vector (r_primary - r_secondary) at TCA [km].
        relative_speed_kms: Relative speed at TCA [km/s].
        r_primary_km: Primary position at TCA [km].
        v_primary_kms: Primary velocity at TCA [km/s].
        r_secondary_km: Secondary position at TCA [km].
        v_secondary_kms: Secondary velocity at TCA [km/s].
        tca_offset_seconds: Offset from start epoch [s].
    """
    tca_epoch: datetime
    miss_distance_km: float
    miss_vector_km: np.ndarray
    relative_speed_kms: float
    r_primary_km: np.ndarray
    v_primary_kms: np.ndarray
    r_secondary_km: np.ndarray
    v_secondary_kms: np.ndarray
    tca_offset_seconds: float


def _sgp4_state_at(sat: Satrec, dt: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Get position and velocity from SGP4 at a given datetime.

    Returns:
        (position [km], velocity [km/s]) in TEME frame.

    Raises:
        RuntimeError: If SGP4 propagation returns an error.
    """
    jd, fr = jday(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute,
        dt.second + dt.microsecond / 1e6,
    )
    err, pos, vel = sat.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 error code {err} at {dt.isoformat()}")
    return np.array(pos), np.array(vel)


def _relative_dot_product(
    sat1: Satrec, sat2: Satrec, epoch: datetime, t_offset: float
) -> float:
    """Compute r_rel · v_rel at epoch + t_offset seconds.

    This is the time derivative of |r_rel|^2 / 2, so its zero crossing
    corresponds to the closest approach.

    Parameters:
        sat1: Primary satellite SGP4 object.
        sat2: Secondary satellite SGP4 object.
        epoch: Reference epoch.
        t_offset: Time offset from epoch [seconds].

    Returns:
        Dot product r_rel · v_rel [km^2/s].
    """
    dt = epoch + timedelta(seconds=t_offset)
    r1, v1 = _sgp4_state_at(sat1, dt)
    r2, v2 = _sgp4_state_at(sat2, dt)
    r_rel = r1 - r2
    v_rel = v1 - v2
    return float(np.dot(r_rel, v_rel))


def _distance_at(
    sat1: Satrec, sat2: Satrec, epoch: datetime, t_offset: float
) -> float:
    """Compute distance between two satellites at epoch + t_offset."""
    dt = epoch + timedelta(seconds=t_offset)
    r1, _ = _sgp4_state_at(sat1, dt)
    r2, _ = _sgp4_state_at(sat2, dt)
    return float(np.linalg.norm(r1 - r2))


def find_tca_sgp4(
    line1_primary: str,
    line2_primary: str,
    line1_secondary: str,
    line2_secondary: str,
    start_epoch: Optional[datetime] = None,
    duration_hours: float = 168.0,
    coarse_step_seconds: float = 60.0,
    miss_distance_threshold_km: float = 50.0,
) -> Optional[TCAResult]:
    """Find the Time of Closest Approach between two TLE-defined objects.

    Algorithm:
    1. Coarse scan: compute distance at each time step over the window.
    2. Find intervals where r_rel · v_rel changes sign (approaching → receding).
    3. Refine each sign change using Brent's method to sub-second accuracy.
    4. Return the TCA with the smallest miss distance.

    Parameters:
        line1_primary: TLE line 1 for primary object.
        line2_primary: TLE line 2 for primary object.
        line1_secondary: TLE line 1 for secondary object.
        line2_secondary: TLE line 2 for secondary object.
        start_epoch: Start of search window (default: now).
        duration_hours: Search window length [hours].
        coarse_step_seconds: Step size for coarse scan [s].
        miss_distance_threshold_km: Only refine if coarse min distance < this [km].

    Returns:
        TCAResult with the closest approach, or None if no close approach found.
    """
    sat1 = Satrec.twoline2rv(line1_primary, line2_primary)
    sat2 = Satrec.twoline2rv(line1_secondary, line2_secondary)
    epoch = start_epoch or datetime.utcnow()

    total_seconds = duration_hours * 3600.0
    n_steps = int(total_seconds / coarse_step_seconds)

    # Coarse scan: find minimum distance and bracket sign changes
    best_tca: Optional[TCAResult] = None
    t_offsets = np.arange(0, n_steps + 1) * coarse_step_seconds
    distances = np.empty(len(t_offsets))
    dot_products = np.empty(len(t_offsets))

    for i, t in enumerate(t_offsets):
        try:
            distances[i] = _distance_at(sat1, sat2, epoch, t)
            dot_products[i] = _relative_dot_product(sat1, sat2, epoch, t)
        except RuntimeError:
            distances[i] = 1e12
            dot_products[i] = 0.0

    # Find coarse minimum
    coarse_min_idx = int(np.argmin(distances))
    coarse_min_dist = distances[coarse_min_idx]

    if coarse_min_dist > miss_distance_threshold_km:
        return None

    # Find all zero crossings of dot product (approaching → receding)
    sign_changes = []
    for i in range(len(dot_products) - 1):
        if dot_products[i] < 0 and dot_products[i + 1] >= 0:
            sign_changes.append((t_offsets[i], t_offsets[i + 1]))

    # Also check bracketed minimum without sign change
    if not sign_changes:
        # Fall back: bracket around the coarse minimum
        lo = max(0.0, t_offsets[max(0, coarse_min_idx - 1)])
        hi = min(total_seconds, t_offsets[min(len(t_offsets) - 1, coarse_min_idx + 1)])
        sign_changes.append((lo, hi))

    # Refine each candidate
    for t_lo, t_hi in sign_changes:
        try:
            # Brent's method to find where r_rel · v_rel = 0
            f_lo = _relative_dot_product(sat1, sat2, epoch, t_lo)
            f_hi = _relative_dot_product(sat1, sat2, epoch, t_hi)

            if f_lo * f_hi < 0:
                t_tca = brentq(
                    lambda t: _relative_dot_product(sat1, sat2, epoch, t),
                    t_lo, t_hi,
                    xtol=0.01,  # sub-second accuracy
                    rtol=1e-12,
                )
            else:
                # No proper bracket; use midpoint of coarse minimum
                t_tca = (t_lo + t_hi) / 2.0

            tca_dt = epoch + timedelta(seconds=t_tca)
            r1, v1 = _sgp4_state_at(sat1, tca_dt)
            r2, v2 = _sgp4_state_at(sat2, tca_dt)

            miss_vec = r1 - r2
            miss_dist = float(np.linalg.norm(miss_vec))
            rel_vel = v1 - v2
            rel_speed = float(np.linalg.norm(rel_vel))

            candidate = TCAResult(
                tca_epoch=tca_dt,
                miss_distance_km=miss_dist,
                miss_vector_km=miss_vec,
                relative_speed_kms=rel_speed,
                r_primary_km=r1,
                v_primary_kms=v1,
                r_secondary_km=r2,
                v_secondary_kms=v2,
                tca_offset_seconds=t_tca,
            )

            if best_tca is None or miss_dist < best_tca.miss_distance_km:
                best_tca = candidate

        except (RuntimeError, ValueError):
            continue

    return best_tca


def find_tca_from_states(
    times: np.ndarray,
    positions_primary: np.ndarray,
    velocities_primary: np.ndarray,
    positions_secondary: np.ndarray,
    velocities_secondary: np.ndarray,
    epoch: datetime,
) -> Optional[TCAResult]:
    """Find TCA from pre-computed state vector arrays.

    Uses interpolation and root-finding on r_rel · v_rel = 0.

    Parameters:
        times: Time offsets from epoch [s], shape (N,).
        positions_primary: Primary positions [km], shape (N, 3).
        velocities_primary: Primary velocities [km/s], shape (N, 3).
        positions_secondary: Secondary positions [km], shape (N, 3).
        velocities_secondary: Secondary velocities [km/s], shape (N, 3).
        epoch: Reference epoch.

    Returns:
        TCAResult or None.
    """
    r_rel = positions_primary - positions_secondary
    v_rel = velocities_primary - velocities_secondary

    distances = np.linalg.norm(r_rel, axis=1)
    dot_products = np.sum(r_rel * v_rel, axis=1)

    min_idx = int(np.argmin(distances))

    # Find sign changes
    best_tca = None
    for i in range(len(dot_products) - 1):
        if dot_products[i] < 0 and dot_products[i + 1] >= 0:
            # Linear interpolation for initial guess
            f0, f1 = dot_products[i], dot_products[i + 1]
            t0, t1 = times[i], times[i + 1]
            t_interp = t0 - f0 * (t1 - t0) / (f1 - f0)

            # Interpolate states at refined time
            alpha = (t_interp - t0) / (t1 - t0) if t1 != t0 else 0.5
            r1 = positions_primary[i] + alpha * (positions_primary[i + 1] - positions_primary[i])
            v1 = velocities_primary[i] + alpha * (velocities_primary[i + 1] - velocities_primary[i])
            r2 = positions_secondary[i] + alpha * (positions_secondary[i + 1] - positions_secondary[i])
            v2 = velocities_secondary[i] + alpha * (velocities_secondary[i + 1] - velocities_secondary[i])

            miss_vec = r1 - r2
            miss_dist = float(np.linalg.norm(miss_vec))
            rel_speed = float(np.linalg.norm(v1 - v2))

            candidate = TCAResult(
                tca_epoch=epoch + timedelta(seconds=float(t_interp)),
                miss_distance_km=miss_dist,
                miss_vector_km=miss_vec,
                relative_speed_kms=rel_speed,
                r_primary_km=r1,
                v_primary_kms=v1,
                r_secondary_km=r2,
                v_secondary_kms=v2,
                tca_offset_seconds=float(t_interp),
            )

            if best_tca is None or miss_dist < best_tca.miss_distance_km:
                best_tca = candidate

    if best_tca is None and len(times) > 0:
        # Fallback to coarse minimum
        idx = min_idx
        r1, v1 = positions_primary[idx], velocities_primary[idx]
        r2, v2 = positions_secondary[idx], velocities_secondary[idx]
        miss_vec = r1 - r2
        best_tca = TCAResult(
            tca_epoch=epoch + timedelta(seconds=float(times[idx])),
            miss_distance_km=float(distances[idx]),
            miss_vector_km=miss_vec,
            relative_speed_kms=float(np.linalg.norm(v1 - v2)),
            r_primary_km=r1,
            v_primary_kms=v1,
            r_secondary_km=r2,
            v_secondary_kms=v2,
            tca_offset_seconds=float(times[idx]),
        )

    return best_tca
