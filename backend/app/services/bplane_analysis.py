"""B-plane (impact parameter plane) analysis for conjunction geometry.

The B-plane is the plane perpendicular to the relative velocity vector at TCA,
passing through the secondary object. BdotT and BdotR characterize the
conjunction geometry per CCSDS 508.0-B-1.

Reference:
    - Vallado (2013), Section 10.4
    - CCSDS 508.0-B-1 Conjunction Data Message standard
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tca_finder import TCAResult


@dataclass
class BPlaneResult:
    """B-plane analysis output.

    Attributes:
        b_dot_t_km: Miss distance component along T-hat (in-plane) [km].
        b_dot_r_km: Miss distance component along R-hat (out-of-plane) [km].
        b_magnitude_km: Total B-plane miss distance |B| [km].
        theta_rad: B-plane angle [rad].
        t_hat: Unit vector along relative velocity (normal to B-plane).
        r_hat: R-hat unit vector in B-plane.
        n_hat: N-hat (=T direction in B-plane, along velocity projection).
        b_vector_km: B vector in ECI [km].
    """
    b_dot_t_km: float
    b_dot_r_km: float
    b_magnitude_km: float
    theta_rad: float
    t_hat: np.ndarray
    r_hat: np.ndarray
    n_hat: np.ndarray
    b_vector_km: np.ndarray


def compute_bplane(tca: TCAResult) -> BPlaneResult:
    """Compute B-plane parameters from TCA result.

    Constructs the B-plane coordinate frame and projects the miss vector.

    Algorithm:
    1. T_hat = v_rel / |v_rel| (unit relative velocity, normal to B-plane)
    2. Use Earth's pole (K_hat = [0,0,1]) as reference to build the frame:
       N_hat = T_hat × K_hat / |T_hat × K_hat| (handles most geometries)
       R_hat = T_hat × N_hat
    3. B_vector = miss_vector - (miss_vector · T_hat) * T_hat  (projection onto B-plane)
    4. BdotT = B_vector · N_hat, BdotR = B_vector · R_hat

    Parameters:
        tca: TCA result with positions and velocities at closest approach.

    Returns:
        BPlaneResult with B-plane geometry.

    Reference: Vallado (2013) Eq. 10-1 through 10-5.
    """
    v_rel = tca.v_primary_kms - tca.v_secondary_kms
    v_rel_mag = np.linalg.norm(v_rel)

    if v_rel_mag < 1e-12:
        # Degenerate case: objects co-moving
        return BPlaneResult(
            b_dot_t_km=0.0, b_dot_r_km=0.0, b_magnitude_km=0.0,
            theta_rad=0.0,
            t_hat=np.zeros(3), r_hat=np.zeros(3), n_hat=np.zeros(3),
            b_vector_km=np.zeros(3),
        )

    t_hat = v_rel / v_rel_mag

    # Reference direction: Earth's pole (K_hat)
    k_hat = np.array([0.0, 0.0, 1.0])

    # If T_hat is nearly aligned with K_hat, use alternate reference
    if abs(np.dot(t_hat, k_hat)) > 0.999:
        k_hat = np.array([1.0, 0.0, 0.0])

    n_hat_raw = np.cross(t_hat, k_hat)
    n_hat = n_hat_raw / np.linalg.norm(n_hat_raw)

    r_hat = np.cross(t_hat, n_hat)
    r_hat = r_hat / np.linalg.norm(r_hat)

    # Miss vector: from secondary to primary at TCA
    miss_vec = tca.r_primary_km - tca.r_secondary_km

    # Project miss vector onto B-plane (remove component along T_hat)
    b_vector = miss_vec - np.dot(miss_vec, t_hat) * t_hat

    # B-plane components
    b_dot_t = float(np.dot(b_vector, n_hat))  # "T" component in B-plane
    b_dot_r = float(np.dot(b_vector, r_hat))  # "R" component in B-plane
    b_mag = float(np.linalg.norm(b_vector))

    theta = float(np.arctan2(b_dot_r, b_dot_t))

    return BPlaneResult(
        b_dot_t_km=b_dot_t,
        b_dot_r_km=b_dot_r,
        b_magnitude_km=b_mag,
        theta_rad=theta,
        t_hat=t_hat,
        r_hat=r_hat,
        n_hat=n_hat,
        b_vector_km=b_vector,
    )
