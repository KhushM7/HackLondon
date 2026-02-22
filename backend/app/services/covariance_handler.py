"""Covariance matrix handling for conjunction analysis.

Handles:
- Covariance estimation from TLE quality (Vallado-Cefola model)
- Positive-definite validation and correction (Higham 1988)
- Projection to B-plane coordinates
- State transition matrix (STM) propagation

Reference:
    - Vallado (2013), Section 10.5
    - Higham (1988), "Computing a nearest symmetric positive semidefinite matrix"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .bplane_analysis import BPlaneResult
from .orbital_constants import GM_EARTH_KM3_S2

logger = logging.getLogger(__name__)
_PD_CORRECTION_WARN_COUNT = 0
_PD_CORRECTION_WARN_LIMIT = 5


@dataclass
class CovarianceInfo:
    """Covariance information for a conjunction object.

    Attributes:
        cov_6x6: Full 6x6 position-velocity covariance [km, km/s].
        cov_3x3_pos: 3x3 position-only covariance [km^2].
        source: Origin of the covariance ("cdm", "estimated", "default").
        was_corrected: True if positive-definite correction was applied.
    """
    cov_6x6: np.ndarray
    cov_3x3_pos: np.ndarray
    source: str
    was_corrected: bool = False


def estimate_covariance_from_tle(
    bstar: float,
    epoch_age_days: float,
    altitude_km: float = 400.0,
) -> CovarianceInfo:
    """Estimate position covariance from TLE quality indicators.

    Uses the Vallado-Cefola model: covariance scales with epoch age
    and BSTAR drag coefficient uncertainty.

    Parameters:
        bstar: BSTAR drag term from TLE [1/Earth radii].
        epoch_age_days: Age of TLE epoch in days.
        altitude_km: Approximate altitude [km] for scaling.

    Returns:
        CovarianceInfo with estimated covariance.

    Note: This is an estimate. The `source` field is set to "estimated"
    to distinguish from covariance provided by CDM.
    """
    # Base position uncertainty: ~1 km for fresh TLE, growing with age
    base_sigma_km = 1.0  # km RMS for a fresh TLE

    # Growth with epoch age (roughly quadratic in along-track)
    age_factor = 1.0 + 0.5 * epoch_age_days + 0.1 * epoch_age_days ** 2

    # BSTAR uncertainty contribution
    bstar_factor = 1.0 + abs(bstar) * 1e4

    sigma_pos = base_sigma_km * age_factor * bstar_factor

    # Along-track uncertainty is largest (~10x cross-track for LEO)
    sigma_along = sigma_pos * 3.0
    sigma_cross = sigma_pos * 1.0
    sigma_radial = sigma_pos * 0.5

    # Build diagonal 3x3 position covariance (in RSW/RTN frame, simplified)
    cov_3x3 = np.diag([sigma_radial**2, sigma_along**2, sigma_cross**2])

    # Velocity uncertainty from position uncertainty / orbital period
    orbital_period_s = 2.0 * np.pi * np.sqrt(
        (altitude_km + 6378.137) ** 3 / GM_EARTH_KM3_S2
    )
    sigma_vel = sigma_pos / orbital_period_s  # km/s

    cov_6x6 = np.zeros((6, 6))
    cov_6x6[:3, :3] = cov_3x3
    cov_6x6[3:, 3:] = np.diag([sigma_vel**2] * 3)

    return CovarianceInfo(
        cov_6x6=cov_6x6,
        cov_3x3_pos=cov_3x3,
        source="estimated",
    )


def default_covariance(sigma_pos_km: float = 1.0) -> CovarianceInfo:
    """Generate a default isotropic covariance when no data is available.

    Parameters:
        sigma_pos_km: Position uncertainty [km] (1-sigma).

    Returns:
        CovarianceInfo with default isotropic covariance.
    """
    cov_3x3 = np.eye(3) * sigma_pos_km**2
    cov_6x6 = np.zeros((6, 6))
    cov_6x6[:3, :3] = cov_3x3
    cov_6x6[3:, 3:] = np.eye(3) * (sigma_pos_km / 5600.0)**2  # ~LEO period scaling

    return CovarianceInfo(
        cov_6x6=cov_6x6,
        cov_3x3_pos=cov_3x3,
        source="default",
    )


def ensure_positive_definite(matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    """Ensure a matrix is symmetric positive definite.

    If the matrix is not PD, applies the Higham (1988) nearest PD correction.

    Parameters:
        matrix: Square symmetric matrix.

    Returns:
        (corrected_matrix, was_corrected) tuple.

    Reference: Higham (1988), "Computing a nearest symmetric positive semidefinite matrix"
    """
    global _PD_CORRECTION_WARN_COUNT

    # Symmetrize
    sym = (matrix + matrix.T) / 2.0

    # Fast path: avoid exception-heavy Cholesky probe in tight loops.
    eigvals = np.linalg.eigvalsh(sym)
    if float(np.min(eigvals)) > 0.0:
        return sym, False

    # Higham correction: eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.maximum(eigvals, 1e-10)
    corrected = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    corrected = (corrected + corrected.T) / 2.0  # re-symmetrize

    if _PD_CORRECTION_WARN_COUNT < _PD_CORRECTION_WARN_LIMIT:
        logger.warning(
            "Covariance matrix is not positive definite; applying Higham correction "
            "(%s/%s)",
            _PD_CORRECTION_WARN_COUNT + 1,
            _PD_CORRECTION_WARN_LIMIT,
        )
        if _PD_CORRECTION_WARN_COUNT + 1 == _PD_CORRECTION_WARN_LIMIT:
            logger.warning("Further covariance PD correction warnings will be suppressed for this process")
    _PD_CORRECTION_WARN_COUNT += 1

    return corrected, True


def project_covariance_to_bplane(
    cov_primary_3x3: np.ndarray,
    cov_secondary_3x3: np.ndarray,
    bplane: BPlaneResult,
) -> np.ndarray:
    """Project combined position covariance onto the B-plane.

    Parameters:
        cov_primary_3x3: Primary 3x3 position covariance in ECI [km^2].
        cov_secondary_3x3: Secondary 3x3 position covariance in ECI [km^2].
        bplane: B-plane analysis result.

    Returns:
        2x2 combined covariance in B-plane [BdotT, BdotR] coordinates [km^2].
    """
    # Combined covariance in ECI
    cov_combined = cov_primary_3x3 + cov_secondary_3x3

    # Rotation matrix from ECI to B-plane: rows are N_hat (T-component) and R_hat
    rotation = np.array([bplane.n_hat, bplane.r_hat])  # 2x3

    # Project: C_bplane = R @ C_combined @ R^T
    cov_bplane = rotation @ cov_combined @ rotation.T

    # Ensure positive definite
    cov_bplane, _ = ensure_positive_definite(cov_bplane)

    return cov_bplane


def covariance_from_stored(
    flat_list: Optional[list], source: str = "cdm"
) -> Optional[CovarianceInfo]:
    """Reconstruct CovarianceInfo from a flattened list stored in the database.

    Parameters:
        flat_list: 36-element list (6x6 flattened) or None.
        source: Source label.

    Returns:
        CovarianceInfo or None if input is None.
    """
    if flat_list is None:
        return None

    cov_6x6 = np.array(flat_list).reshape(6, 6)
    cov_6x6, was_corrected = ensure_positive_definite(cov_6x6)

    return CovarianceInfo(
        cov_6x6=cov_6x6,
        cov_3x3_pos=cov_6x6[:3, :3],
        source=source,
        was_corrected=was_corrected,
    )
