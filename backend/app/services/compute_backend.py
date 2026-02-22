"""GPU/CPU compute backend abstraction for OrbitGuard.

Provides a unified array-library interface that transparently uses CuPy (GPU)
or NumPy (CPU) depending on configuration and availability.

Configuration:
    ORBITGUARD_BACKEND=cpu|gpu   (default: cpu)

Usage:
    from .compute_backend import get_xp, asnumpy, is_gpu_enabled, get_rng

    xp = get_xp()
    arr = xp.array([1, 2, 3])
    result = asnumpy(arr)  # always returns numpy array
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_backend: str = "cpu"
_cupy_available: bool = False
_initialized: bool = False


def _init_backend() -> None:
    """Initialize the compute backend (called once on first use)."""
    global _backend, _cupy_available, _initialized
    if _initialized:
        return
    _initialized = True

    requested = os.environ.get("ORBITGUARD_BACKEND", "cpu").lower().strip()

    if requested == "gpu":
        try:
            import cupy  # noqa: F401
            cupy.cuda.Device(0).compute_capability  # verify CUDA device exists
            _cupy_available = True
            _backend = "gpu"
            logger.info("OrbitGuard compute backend: GPU (CuPy %s)", cupy.__version__)
        except ImportError:
            logger.warning(
                "ORBITGUARD_BACKEND=gpu but CuPy is not installed; falling back to CPU. "
                "Install with: pip install cupy-cuda12x"
            )
            _backend = "cpu"
        except Exception as exc:
            logger.warning(
                "ORBITGUARD_BACKEND=gpu but CUDA device unavailable (%s); falling back to CPU",
                exc,
            )
            _backend = "cpu"
    else:
        _backend = "cpu"
        logger.info("OrbitGuard compute backend: CPU (NumPy)")


def get_xp() -> Any:
    """Return the active array library (numpy or cupy)."""
    _init_backend()
    if _backend == "gpu":
        import cupy
        return cupy
    return np


def is_gpu_enabled() -> bool:
    """Return True if the GPU backend is active."""
    _init_backend()
    return _backend == "gpu"


def asnumpy(arr: Any) -> np.ndarray:
    """Convert an array to a NumPy array regardless of backend.

    If the input is already a NumPy array, returns it unchanged.
    If it's a CuPy array, transfers to CPU.
    """
    if isinstance(arr, np.ndarray):
        return arr
    # CuPy arrays have a .get() method
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def get_rng(seed: int | None = None) -> Any:
    """Return a seeded random number generator for the active backend.

    For CPU: numpy.random.default_rng(seed)
    For GPU: cupy.random.default_rng(seed) if available, else cupy.random with seed
    """
    _init_backend()
    if _backend == "gpu":
        import cupy
        if hasattr(cupy.random, "default_rng"):
            return cupy.random.default_rng(seed)
        # Older CuPy versions: use global seed
        if seed is not None:
            cupy.random.seed(seed)
        return cupy.random
    return np.random.default_rng(seed)


def reset_backend() -> None:
    """Reset backend state (for testing only)."""
    global _backend, _cupy_available, _initialized
    _backend = "cpu"
    _cupy_available = False
    _initialized = False
