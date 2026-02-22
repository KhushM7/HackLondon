"""GPU/CPU backend abstraction for OrbitGuard.

Provides a thin layer over numpy/cupy so that numerical code can run
on either CPU or GPU with no source-level changes beyond calling
``get_xp()`` instead of importing ``numpy`` directly.

Configuration (any of these enables GPU):
    - Environment variable ``ORBITGUARD_GPU=1``
    - Environment variable ``ORBITGUARD_BACKEND=gpu``

If CuPy is not installed or no CUDA device is available the backend
silently falls back to NumPy and logs a one-time warning.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_gpu_available: bool | None = None  # lazy-init
_use_gpu: bool = False
_warned: bool = False


def _detect_gpu() -> bool:
    """Return True if CuPy is importable and at least one CUDA device exists."""
    try:
        import cupy  # noqa: F401
        cupy.cuda.Device(0).compute_capability  # probe device
        return True
    except Exception:
        return False


def _resolve_backend() -> bool:
    """Decide whether to use GPU based on env vars and hardware."""
    global _gpu_available, _use_gpu, _warned

    env_gpu = os.environ.get("ORBITGUARD_GPU", "0")
    env_backend = os.environ.get("ORBITGUARD_BACKEND", "cpu").lower()
    want_gpu = env_gpu == "1" or env_backend == "gpu"

    if _gpu_available is None:
        _gpu_available = _detect_gpu()

    if want_gpu and _gpu_available:
        _use_gpu = True
        if not _warned:
            logger.info("OrbitGuard compute backend: GPU (CuPy + CUDA)")
            _warned = True
    else:
        _use_gpu = False
        if want_gpu and not _gpu_available and not _warned:
            logger.warning(
                "GPU requested but CuPy/CUDA not available — falling back to CPU (NumPy)"
            )
            _warned = True
        elif not _warned:
            logger.info("OrbitGuard compute backend: CPU (NumPy)")
            _warned = True

    return _use_gpu


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_gpu_enabled() -> bool:
    """Return True if the GPU backend is active."""
    _resolve_backend()
    return _use_gpu


def get_xp() -> Any:
    """Return the array module to use (``numpy`` or ``cupy``).

    Usage::

        xp = get_xp()
        a = xp.array([1, 2, 3])
        b = xp.linalg.norm(a)
    """
    if is_gpu_enabled():
        import cupy
        return cupy
    return np


def asnumpy(arr: Any) -> np.ndarray:
    """Safely convert *arr* to a NumPy ndarray.

    If *arr* is already a NumPy array this is a no-op.  If it is a CuPy
    array it is transferred to host memory.
    """
    if isinstance(arr, np.ndarray):
        return arr
    # CuPy array
    try:
        import cupy
        if isinstance(arr, cupy.ndarray):
            return cupy.asnumpy(arr)
    except ImportError:
        pass
    return np.asarray(arr)


def to_device(arr: np.ndarray) -> Any:
    """Move a NumPy array to the current compute device.

    Returns a CuPy array when GPU is enabled, otherwise returns the
    input unchanged.
    """
    if is_gpu_enabled():
        import cupy
        return cupy.asarray(arr)
    return arr


# ---------------------------------------------------------------------------
# Eager initialization — call init_backend() after logging is configured
# to ensure the log message is visible at startup.
# ---------------------------------------------------------------------------

def init_backend() -> None:
    """Resolve and log the compute backend.

    Call this once at application startup (after ``logging.basicConfig``)
    so the one-time INFO/WARNING message is visible in the log output.
    """
    _resolve_backend()


def seed_rng(seed: int | None = None) -> Any:
    """Return a seeded random generator on the active device.

    On CPU this returns ``numpy.random.default_rng(seed)``.
    On GPU this returns a thin wrapper providing the same API subset
    needed by OrbitGuard (``multivariate_normal``).
    """
    if is_gpu_enabled():
        return _CupyRNG(seed)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# CuPy RNG wrapper (minimal API surface)
# ---------------------------------------------------------------------------

class _CupyRNG:
    """Wrap CuPy random functions to match the subset of
    ``numpy.random.Generator`` that OrbitGuard uses."""

    def __init__(self, seed: int | None = None):
        import cupy
        if seed is not None:
            cupy.random.seed(seed)
        self._xp = cupy

    def multivariate_normal(
        self, mean: Any, cov: Any, size: int | tuple[int, ...] = 1
    ) -> Any:
        """Draw from a multivariate normal on GPU.

        CuPy doesn't expose ``default_rng().multivariate_normal`` so we
        use Cholesky factorization of the covariance and transform
        standard normals:  samples = mean + L @ z,  where z ~ N(0, I).
        """
        xp = self._xp
        mean_d = xp.asarray(mean, dtype=xp.float64)
        cov_d = xp.asarray(cov, dtype=xp.float64)
        L = xp.linalg.cholesky(cov_d)  # lower triangular
        n = mean_d.shape[0]
        if isinstance(size, int):
            size = (size,)
        z = xp.random.standard_normal((*size, n), dtype=xp.float64)
        # samples = mean + z @ L^T
        return mean_d + z @ L.T
