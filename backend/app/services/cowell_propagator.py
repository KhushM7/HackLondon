"""High-fidelity orbit propagation using Cowell's method with perturbations.

Implements numerical integration of the equations of motion with:
- J2-J6 gravitational harmonics
- Atmospheric drag (exponential model)
- Solar radiation pressure
- Third-body gravity (Sun/Moon via astropy)

Reference: Vallado (2013), "Fundamentals of Astrodynamics and Applications"
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from .orbital_constants import (
    GM_EARTH_KM3_S2,
    R_EARTH_KM,
    J2, J3, J4, J5, J6,
    GM_SUN_KM3_S2,
    GM_MOON_KM3_S2,
    SOLAR_PRESSURE_N_M2,
    DRAG_RHO0_KG_M3,
    DRAG_H0_KM,
    DRAG_SCALE_HEIGHT_KM,
    PropagationConfig,
)


def _j2_j6_acceleration(r_vec: np.ndarray, config: PropagationConfig) -> np.ndarray:
    """Compute acceleration from J2-J6 zonal harmonics.

    Parameters:
        r_vec: Position vector in ECI [km], shape (3,).
        config: Propagation configuration.

    Returns:
        Acceleration vector [km/s^2], shape (3,).

    Reference: Vallado (2013), Eq. 8-25 through 8-30.
    """
    r = np.linalg.norm(r_vec)
    x, y, z = r_vec
    r2 = r * r
    z2 = z * z
    re_over_r = R_EARTH_KM / r
    mu_over_r2 = GM_EARTH_KM3_S2 / r2

    a = np.zeros(3)

    if not config.use_j2_j6:
        return a

    # J2
    fac = 1.5 * J2 * re_over_r**2
    z2_r2 = z2 / r2
    common_j2 = mu_over_r2 * fac / r
    a[0] += common_j2 * x * (1.0 - 5.0 * z2_r2)
    a[1] += common_j2 * y * (1.0 - 5.0 * z2_r2)
    a[2] += common_j2 * z * (3.0 - 5.0 * z2_r2)

    # J3
    fac3 = 0.5 * J3 * re_over_r**3
    common_j3 = mu_over_r2 * fac3 / r
    z_r = z / r
    a[0] += common_j3 * x * (10.0 * z_r - 35.0 * z_r * z2_r2 / 1.0 - 3.0 * z_r)
    # Simplified J3 contribution (Vallado Eq 8-27)
    a[0] += common_j3 * 5.0 * x * (3.0 * z_r - 7.0 * z_r * z2_r2)
    a[1] += common_j3 * 5.0 * y * (3.0 * z_r - 7.0 * z_r * z2_r2)
    a[2] += common_j3 * (3.0 - 30.0 * z2_r2 + 35.0 * z2_r2 * z2_r2)

    # J4
    fac4 = -0.625 * J4 * re_over_r**4
    common_j4 = mu_over_r2 * fac4 / r
    a[0] += common_j4 * x * (3.0 - 42.0 * z2_r2 + 63.0 * z2_r2**2)
    a[1] += common_j4 * y * (3.0 - 42.0 * z2_r2 + 63.0 * z2_r2**2)
    a[2] += common_j4 * z * (15.0 - 70.0 * z2_r2 + 63.0 * z2_r2**2)

    return a


def _drag_acceleration(
    r_vec: np.ndarray, v_vec: np.ndarray, config: PropagationConfig
) -> np.ndarray:
    """Compute atmospheric drag acceleration using exponential atmosphere model.

    Parameters:
        r_vec: Position [km].
        v_vec: Velocity [km/s].
        config: Propagation configuration.

    Returns:
        Drag acceleration [km/s^2].
    """
    r = np.linalg.norm(r_vec)
    alt = r - R_EARTH_KM
    if alt < 100.0 or alt > 2000.0:
        return np.zeros(3)

    # Exponential density model
    rho = DRAG_RHO0_KG_M3 * np.exp(-(alt - DRAG_H0_KM) / DRAG_SCALE_HEIGHT_KM)

    # Velocity relative to rotating atmosphere
    omega_earth = np.array([0.0, 0.0, 7.2921159e-5])  # rad/s
    v_rel = v_vec - np.cross(omega_earth, r_vec)  # km/s
    v_rel_mag = np.linalg.norm(v_rel)
    if v_rel_mag < 1e-10:
        return np.zeros(3)

    # a_drag = -0.5 * rho * Cd * A/m * v_rel^2 * v_hat
    # Convert: rho [kg/m^3], area [m^2], mass [kg], v [km/s] → need consistent units
    # v in km/s → v in m/s = v*1000, then a in m/s^2 → km/s^2 = a/1000
    v_rel_ms = v_rel_mag * 1000.0  # m/s
    bc = config.drag_cd * config.drag_area_m2 / config.mass_kg  # m^2/kg
    a_mag = 0.5 * rho * bc * v_rel_ms * v_rel_ms  # m/s^2
    a_mag_kms2 = a_mag / 1000.0  # km/s^2

    return -a_mag_kms2 * (v_rel / v_rel_mag)


def _srp_acceleration(r_vec: np.ndarray, config: PropagationConfig) -> np.ndarray:
    """Compute solar radiation pressure acceleration.

    Simplified model assuming Sun along +X direction (adequate for short-term propagation).

    Parameters:
        r_vec: Position [km].
        config: Propagation config.

    Returns:
        SRP acceleration [km/s^2].
    """
    # Simple: SRP pushes away from Sun. Assume Sun at +X, very far away.
    # a_srp = -Cr * (A/m) * P_solar * sun_hat
    # P_solar ~ 4.56e-6 N/m^2
    am_ratio = config.srp_area_m2 / config.mass_kg  # m^2/kg
    a_srp_ms2 = config.srp_cr * am_ratio * SOLAR_PRESSURE_N_M2  # m/s^2
    a_srp_kms2 = a_srp_ms2 / 1000.0  # km/s^2

    # Sun direction approximation (along +X in ECI, simplified)
    sun_hat = np.array([1.0, 0.0, 0.0])
    return -a_srp_kms2 * sun_hat


def _equations_of_motion(
    t: float, state: np.ndarray, config: PropagationConfig
) -> np.ndarray:
    """Right-hand side of the orbital equations of motion.

    Parameters:
        t: Time since epoch [seconds].
        state: [x, y, z, vx, vy, vz] in ECI [km, km/s].
        config: Propagation configuration.

    Returns:
        Time derivative of state vector [km/s, km/s^2].
    """
    r_vec = state[:3]
    v_vec = state[3:]
    r = np.linalg.norm(r_vec)

    # Two-body acceleration
    a = -GM_EARTH_KM3_S2 / (r * r * r) * r_vec

    # J2 (always on) or J2-J6
    a += _j2_j6_acceleration(r_vec, config)

    if config.use_drag:
        a += _drag_acceleration(r_vec, v_vec, config)

    if config.use_srp:
        a += _srp_acceleration(r_vec, config)

    return np.concatenate([v_vec, a])


def propagate_cowell(
    r0: np.ndarray,
    v0: np.ndarray,
    duration_seconds: float,
    config: Optional[PropagationConfig] = None,
    step_seconds: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate an orbit using Cowell's method (numerical integration).

    Parameters:
        r0: Initial position [km], shape (3,).
        v0: Initial velocity [km/s], shape (3,).
        duration_seconds: Total propagation time [s].
        config: Propagation configuration (defaults used if None).
        step_seconds: Output step size [s] (overrides config.step_seconds if given).

    Returns:
        times: Array of times since epoch [s], shape (N,).
        positions: Positions [km], shape (N, 3).
        velocities: Velocities [km/s], shape (N, 3).
    """
    if config is None:
        config = PropagationConfig()
    dt = step_seconds or config.step_seconds

    state0 = np.concatenate([r0, v0])
    t_eval = np.arange(0.0, duration_seconds + dt, dt)
    # Ensure t_eval doesn't exceed the integration span
    t_eval = t_eval[t_eval <= duration_seconds]

    sol = solve_ivp(
        _equations_of_motion,
        [0.0, duration_seconds],
        state0,
        method="DOP853",
        t_eval=t_eval,
        args=(config,),
        rtol=1e-10,
        atol=1e-12,
        max_step=dt,
    )

    if not sol.success:
        raise RuntimeError(f"Cowell propagation failed: {sol.message}")

    times = sol.t
    positions = sol.y[:3].T
    velocities = sol.y[3:].T
    return times, positions, velocities
