"""Probability of Collision (Pc) computation.

Implements three methods:
1. Foster (2D Gaussian) — the operational standard (NASA CARA)
2. Chan (Maximum Pc) — upper bound fallback
3. Monte Carlo — for validation and high-uncertainty cases

Reference:
    - Alfano (2005), "A Numerical Implementation of Spherical Object Collision Probability"
    - Foster & Estes (1992), original B-plane Pc method
    - Chan (1997), "Spacecraft Collision Probability"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import dblquad
from scipy.stats import norm

from .bplane_analysis import BPlaneResult
from .orbital_constants import DEFAULT_HBR_KM

logger = logging.getLogger(__name__)


@dataclass
class PcResult:
    """Probability of collision computation result.

    Attributes:
        pc: Probability of collision (dimensionless, 0-1).
        method: Computation method used ("foster", "chan", "monte_carlo").
        ci_low: Lower 95% CI (Monte Carlo only).
        ci_high: Upper 95% CI (Monte Carlo only).
    """
    pc: float
    method: str
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None


def compute_pc_foster(
    miss_vector_bplane: np.ndarray,
    cov_bplane_2x2: np.ndarray,
    hard_body_radius_km: float = DEFAULT_HBR_KM,
) -> PcResult:
    """Compute Pc using Foster's 2D Gaussian method.

    Integrates a bivariate normal PDF centered at the miss vector
    over a disk of radius HBR in the B-plane.

    Pc = ∬_{disk} f(x; μ, Σ) dA

    where:
        μ = [BdotT, BdotR] (miss vector in B-plane)
        Σ = combined covariance in B-plane (2x2)
        disk radius = HBR

    Parameters:
        miss_vector_bplane: [BdotT, BdotR] miss vector [km].
        cov_bplane_2x2: 2x2 combined covariance in B-plane [km^2].
        hard_body_radius_km: Combined hard-body radius [km].

    Returns:
        PcResult with Foster Pc value.

    Reference: Alfano (2005), Eq. 1-3.
    """
    mu = miss_vector_bplane
    cov = cov_bplane_2x2

    # Validate inputs
    det = np.linalg.det(cov)
    if det <= 0:
        logger.error("Foster Pc: covariance determinant <= 0, returning NaN")
        return PcResult(pc=float('nan'), method="foster")

    cov_inv = np.linalg.inv(cov)
    norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det))
    hbr = hard_body_radius_km

    def integrand(y: float, x: float) -> float:
        d = np.array([x - mu[0], y - mu[1]])
        exponent = -0.5 * d @ cov_inv @ d
        return norm_const * np.exp(exponent)

    try:
        pc, abserr = dblquad(
            integrand,
            -hbr, hbr,
            lambda x: -np.sqrt(max(0, hbr**2 - x**2)),
            lambda x: np.sqrt(max(0, hbr**2 - x**2)),
            epsabs=1e-14,
            epsrel=1e-10,
        )
    except Exception as e:
        logger.error("Foster Pc integration failed: %s", e)
        return PcResult(pc=float('nan'), method="foster")

    # Clamp to [0, 1]
    pc = max(0.0, min(1.0, pc))

    return PcResult(pc=pc, method="foster")


def compute_pc_chan(
    miss_distance_km: float,
    cov_bplane_2x2: np.ndarray,
    hard_body_radius_km: float = DEFAULT_HBR_KM,
) -> PcResult:
    """Compute maximum Pc using Chan's method.

    Provides an upper bound on Pc when full geometry is unavailable.
    Uses the maximum of the PDF at the miss distance times the HBR disk area.

    Pc_max = (HBR^2 / (2 * sigma_max * sigma_min)) * exp(-miss_dist^2 / (2 * sigma_max^2))

    Parameters:
        miss_distance_km: Scalar miss distance [km].
        cov_bplane_2x2: 2x2 covariance in B-plane [km^2].
        hard_body_radius_km: Combined HBR [km].

    Returns:
        PcResult with Chan maximum Pc.

    Reference: Chan (1997).
    """
    eigvals = np.linalg.eigvalsh(cov_bplane_2x2)
    eigvals = np.maximum(eigvals, 1e-20)  # avoid division by zero

    sigma_max = np.sqrt(max(eigvals))
    sigma_min = np.sqrt(min(eigvals))

    hbr = hard_body_radius_km
    u = miss_distance_km

    # Chan's formula: maximum probability over all orientations
    if sigma_max < 1e-15:
        return PcResult(pc=0.0, method="chan")

    pc = (hbr ** 2) / (2.0 * sigma_max * sigma_min) * np.exp(
        -u ** 2 / (2.0 * sigma_max ** 2)
    )

    pc = max(0.0, min(1.0, pc))
    return PcResult(pc=pc, method="chan")


def compute_pc_monte_carlo(
    miss_vector_bplane: np.ndarray,
    cov_bplane_2x2: np.ndarray,
    hard_body_radius_km: float = DEFAULT_HBR_KM,
    n_samples: int = 10000,
    seed: Optional[int] = None,
) -> PcResult:
    """Compute Pc via Monte Carlo sampling.

    Samples positions from the combined covariance distribution and counts
    the fraction that fall within the HBR disk.

    When GPU is enabled (ORBITGUARD_BACKEND=gpu), sampling and distance
    computation run on the GPU via CuPy for significant speedup at large
    sample counts.

    Parameters:
        miss_vector_bplane: [BdotT, BdotR] miss vector [km].
        cov_bplane_2x2: 2x2 combined covariance in B-plane [km^2].
        hard_body_radius_km: Combined HBR [km].
        n_samples: Number of Monte Carlo samples.
        seed: Random seed for reproducibility.

    Returns:
        PcResult with Monte Carlo Pc and 95% confidence interval.
    """
    from .compute_backend import get_xp, asnumpy, is_gpu_enabled

    xp = get_xp()

    try:
        if is_gpu_enabled():
            # GPU path: Cholesky decomposition + batch sampling on GPU
            mu = xp.asarray(miss_vector_bplane, dtype=xp.float64)
            cov = xp.asarray(cov_bplane_2x2, dtype=xp.float64)
            L = xp.linalg.cholesky(cov)
            rng = xp.random.default_rng(seed) if hasattr(xp.random, 'default_rng') else None
            if rng is not None:
                z = rng.standard_normal((n_samples, 2), dtype=xp.float64)
            else:
                if seed is not None:
                    xp.random.seed(seed)
                z = xp.random.standard_normal((n_samples, 2)).astype(xp.float64)
            samples = z @ L.T + mu
        else:
            # CPU path: use NumPy's multivariate_normal directly
            rng = np.random.default_rng(seed)
            samples = rng.multivariate_normal(miss_vector_bplane, cov_bplane_2x2, n_samples)
            samples = xp.asarray(samples)
    except np.linalg.LinAlgError:
        logger.error("Monte Carlo: covariance not valid for sampling")
        return PcResult(pc=float('nan'), method="monte_carlo")
    except Exception as exc:
        # CuPy may raise different exceptions for non-PD matrices
        if "positive" in str(exc).lower() or "cholesky" in str(exc).lower():
            logger.error("Monte Carlo: covariance not valid for sampling")
            return PcResult(pc=float('nan'), method="monte_carlo")
        raise

    # Count collisions (samples within HBR disk from origin)
    distances = xp.linalg.norm(samples, axis=1)
    n_collisions = int(xp.sum(distances < hard_body_radius_km))

    pc = n_collisions / n_samples

    # Wilson score interval for 95% CI
    z = 1.96
    n = n_samples
    p_hat = pc
    denom = 1.0 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    halfwidth = z * float(np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n)) / denom

    ci_low = max(0.0, center - halfwidth)
    ci_high = min(1.0, center + halfwidth)

    return PcResult(
        pc=pc,
        method="monte_carlo",
        ci_low=ci_low,
        ci_high=ci_high,
    )
