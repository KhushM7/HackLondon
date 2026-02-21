"""Physical constants and configuration for orbital mechanics computations.

All constants are sourced from standard references (WGS-84, EGM2008, IAU).
Units: km, kg, seconds unless otherwise noted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# --- Earth constants (WGS-84 / EGM2008) ---
GM_EARTH_KM3_S2 = 398600.4418  # Earth gravitational parameter [km^3/s^2]
R_EARTH_KM = 6378.137  # Earth equatorial radius [km]
J2 = 1.08262668e-3  # Oblateness (EGM2008)
J3 = -2.53265648e-6  # EGM2008
J4 = -1.61962159e-6  # EGM2008
J5 = -2.27296082e-7  # EGM2008
J6 = 5.40681239e-7  # EGM2008
EARTH_ROTATION_RAD_S = 7.2921159e-5  # Earth rotation rate [rad/s]
EARTH_FLATTENING = 1.0 / 298.257223563  # WGS-84

# --- Sun/Moon constants ---
GM_SUN_KM3_S2 = 1.32712440018e11  # Sun gravitational parameter [km^3/s^2]
GM_MOON_KM3_S2 = 4902.800066  # Moon gravitational parameter [km^3/s^2]
AU_KM = 149597870.7  # Astronomical unit [km]

# --- Solar radiation pressure ---
SOLAR_FLUX_W_M2 = 1361.0  # Solar constant at 1 AU [W/m^2]
SPEED_OF_LIGHT_M_S = 299792458.0  # Speed of light [m/s]
# SRP acceleration constant: P = S/c in N/m^2, converted to km/s^2 per (m^2/kg)
SOLAR_PRESSURE_N_M2 = SOLAR_FLUX_W_M2 / SPEED_OF_LIGHT_M_S  # ~4.56e-6 N/m^2

# --- Atmospheric drag reference ---
DRAG_RHO0_KG_M3 = 3.614e-13  # Reference density at 700 km [kg/m^3] (NRLMSISE-00 approx)
DRAG_H0_KM = 700.0  # Reference altitude [km]
DRAG_SCALE_HEIGHT_KM = 88.667  # Scale height [km] (approximate for 400-800 km)

# --- Rocket equation ---
G0_M_S2 = 9.80665  # Standard gravity [m/s^2]
DEFAULT_ISP_S = 300.0  # Default specific impulse [s] (bipropellant)

# --- Operational thresholds (NASA CARA) ---
PC_THRESHOLD_RED = 1e-4  # Maneuver trigger
PC_THRESHOLD_YELLOW = 1e-5  # Elevated monitoring

# --- Default hard-body radius ---
DEFAULT_HBR_KM = 0.020  # 20 meters combined hard-body radius


class RiskLevel(Enum):
    """Pc-based risk classification per NASA/ESA operational standards."""
    GREEN = "green"    # Pc < 1e-5
    YELLOW = "yellow"  # 1e-5 <= Pc < 1e-4
    RED = "red"        # Pc >= 1e-4 — maneuver threshold


def classify_risk(pc: float) -> RiskLevel:
    """Classify probability of collision into operational risk level."""
    if pc >= PC_THRESHOLD_RED:
        return RiskLevel.RED
    if pc >= PC_THRESHOLD_YELLOW:
        return RiskLevel.YELLOW
    return RiskLevel.GREEN


@dataclass
class PropagationConfig:
    """Configuration for orbit propagation.

    Attributes:
        step_seconds: Integration time step [s].
        duration_hours: Total propagation window [hours].
        use_j2_j6: Enable J2-J6 gravitational harmonics.
        use_drag: Enable atmospheric drag perturbation.
        use_srp: Enable solar radiation pressure.
        use_third_body: Enable Sun/Moon third-body gravity.
        drag_area_m2: Satellite drag cross-section [m^2].
        drag_cd: Drag coefficient (dimensionless).
        srp_area_m2: SRP cross-section [m^2].
        srp_cr: Reflectivity coefficient (1.0=absorb, 2.0=reflect).
        mass_kg: Satellite mass [kg].
    """
    step_seconds: float = 60.0
    duration_hours: float = 168.0  # 7 days
    use_j2_j6: bool = True
    use_drag: bool = True
    use_srp: bool = True
    use_third_body: bool = True
    drag_area_m2: float = 10.0
    drag_cd: float = 2.2
    srp_area_m2: float = 10.0
    srp_cr: float = 1.5
    mass_kg: float = 500.0


def eci_to_rtn(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Compute ECI-to-RTN rotation matrix.

    Parameters:
        r_vec: Position vector in ECI [km], shape (3,).
        v_vec: Velocity vector in ECI [km/s], shape (3,).

    Returns:
        3x3 rotation matrix where rows are R-hat, T-hat, N-hat in ECI.
    """
    r_hat = r_vec / np.linalg.norm(r_vec)
    h_vec = np.cross(r_vec, v_vec)
    n_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(n_hat, r_hat)
    return np.array([r_hat, t_hat, n_hat])


def rtn_to_eci(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Compute RTN-to-ECI rotation matrix (transpose of ECI-to-RTN)."""
    return eci_to_rtn(r_vec, v_vec).T
