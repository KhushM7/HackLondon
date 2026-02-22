#!/usr/bin/env python
"""Compare CPU vs GPU results and timing for OrbitGuard compute paths.

Usage:
    # CPU-only (default)
    python scripts/compare_cpu_gpu.py

    # With GPU enabled
    ORBITGUARD_GPU=1 python scripts/compare_cpu_gpu.py

Reports:
    - Monte Carlo Pc values and timing on CPU vs GPU
    - Batch distance computation timing
    - TCA finder vectorized portion timing
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def benchmark_monte_carlo() -> None:
    """Benchmark Monte Carlo Pc on CPU, then GPU if available."""
    from app.services.pc_calculator import compute_pc_monte_carlo
    from app.services.compute import backend as be

    _banner("Monte Carlo Pc Benchmark")

    miss = np.array([0.01, 0.005])
    cov = np.eye(2) * 0.05**2
    hbr = 0.020
    n_samples = 100_000
    seed = 42

    # --- CPU run ---
    be._gpu_available = None
    be._use_gpu = False
    be._warned = False
    os.environ["ORBITGUARD_GPU"] = "0"
    os.environ["ORBITGUARD_BACKEND"] = "cpu"

    t0 = time.perf_counter()
    cpu_result = compute_pc_monte_carlo(miss, cov, hbr, n_samples=n_samples, seed=seed)
    cpu_time = time.perf_counter() - t0
    print(f"  CPU: Pc = {cpu_result.pc:.6f}  CI = [{cpu_result.ci_low:.6f}, {cpu_result.ci_high:.6f}]  time = {cpu_time:.4f}s")

    # --- GPU run (if available) ---
    be._gpu_available = None
    be._use_gpu = False
    be._warned = False
    os.environ["ORBITGUARD_GPU"] = "1"
    os.environ["ORBITGUARD_BACKEND"] = "gpu"

    if be._detect_gpu():
        t0 = time.perf_counter()
        gpu_result = compute_pc_monte_carlo(miss, cov, hbr, n_samples=n_samples, seed=seed)
        gpu_time = time.perf_counter() - t0
        print(f"  GPU: Pc = {gpu_result.pc:.6f}  CI = [{gpu_result.ci_low:.6f}, {gpu_result.ci_high:.6f}]  time = {gpu_time:.4f}s")
        print(f"  Speedup: {cpu_time / gpu_time:.1f}x")
        delta = abs(cpu_result.pc - gpu_result.pc)
        print(f"  Pc delta: {delta:.6f}")
    else:
        print("  GPU: not available (CuPy/CUDA not found)")

    # Reset
    os.environ.pop("ORBITGUARD_GPU", None)
    os.environ.pop("ORBITGUARD_BACKEND", None)
    be._gpu_available = None
    be._use_gpu = False
    be._warned = False


def benchmark_batch_distance() -> None:
    """Benchmark batch distance computation."""
    from app.services.compute.backend import get_xp, asnumpy
    from app.services.compute import backend as be

    _banner("Batch Distance Benchmark")

    rng = np.random.default_rng(0)
    n = 50_000
    a = rng.standard_normal((n, 3))
    b = rng.standard_normal((n, 3))

    # --- CPU ---
    be._gpu_available = None
    be._use_gpu = False
    be._warned = False
    os.environ["ORBITGUARD_GPU"] = "0"
    os.environ["ORBITGUARD_BACKEND"] = "cpu"

    t0 = time.perf_counter()
    cpu_dist = np.linalg.norm(a - b, axis=1)
    cpu_time = time.perf_counter() - t0
    print(f"  CPU: {n} distances in {cpu_time:.4f}s")

    # --- GPU ---
    be._gpu_available = None
    be._use_gpu = False
    be._warned = False
    os.environ["ORBITGUARD_GPU"] = "1"
    os.environ["ORBITGUARD_BACKEND"] = "gpu"

    if be._detect_gpu():
        be._resolve_backend()
        xp = get_xp()
        a_d = xp.asarray(a)
        b_d = xp.asarray(b)
        # Warm up
        _ = xp.linalg.norm(a_d - b_d, axis=1)

        t0 = time.perf_counter()
        gpu_dist = asnumpy(xp.linalg.norm(a_d - b_d, axis=1))
        gpu_time = time.perf_counter() - t0
        print(f"  GPU: {n} distances in {gpu_time:.4f}s")
        print(f"  Speedup: {cpu_time / gpu_time:.1f}x")
        max_delta = np.max(np.abs(cpu_dist - gpu_dist))
        print(f"  Max element delta: {max_delta:.2e}")
    else:
        print("  GPU: not available")

    os.environ.pop("ORBITGUARD_GPU", None)
    os.environ.pop("ORBITGUARD_BACKEND", None)
    be._gpu_available = None
    be._use_gpu = False
    be._warned = False


def benchmark_tca_norms() -> None:
    """Benchmark vectorized norm + dot product from TCA finder."""
    from app.services.compute.backend import get_xp, asnumpy
    from app.services.compute import backend as be

    _banner("TCA Vectorized Norms Benchmark")

    rng = np.random.default_rng(1)
    n = 10_000
    r_rel = rng.standard_normal((n, 3))
    v_rel = rng.standard_normal((n, 3))

    # --- CPU ---
    be._gpu_available = None
    be._use_gpu = False
    be._warned = False
    os.environ["ORBITGUARD_GPU"] = "0"

    t0 = time.perf_counter()
    cpu_norms = np.linalg.norm(r_rel, axis=1)
    cpu_dots = np.sum(r_rel * v_rel, axis=1)
    cpu_time = time.perf_counter() - t0
    print(f"  CPU: {n} norms + dots in {cpu_time:.6f}s")

    # --- GPU ---
    be._gpu_available = None
    be._use_gpu = False
    be._warned = False
    os.environ["ORBITGUARD_GPU"] = "1"
    os.environ["ORBITGUARD_BACKEND"] = "gpu"

    if be._detect_gpu():
        be._resolve_backend()
        xp = get_xp()
        r_d = xp.asarray(r_rel)
        v_d = xp.asarray(v_rel)
        _ = xp.linalg.norm(r_d, axis=1)  # warm up

        t0 = time.perf_counter()
        gpu_norms = asnumpy(xp.linalg.norm(r_d, axis=1))
        gpu_dots = asnumpy(xp.sum(r_d * v_d, axis=1))
        gpu_time = time.perf_counter() - t0
        print(f"  GPU: {n} norms + dots in {gpu_time:.6f}s")
        print(f"  Speedup: {cpu_time / gpu_time:.1f}x")
        print(f"  Max norm delta: {np.max(np.abs(cpu_norms - gpu_norms)):.2e}")
        print(f"  Max dot delta: {np.max(np.abs(cpu_dots - gpu_dots)):.2e}")
    else:
        print("  GPU: not available")

    os.environ.pop("ORBITGUARD_GPU", None)
    os.environ.pop("ORBITGUARD_BACKEND", None)
    be._gpu_available = None
    be._use_gpu = False
    be._warned = False


def main() -> None:
    from app.services.compute.backend import is_gpu_enabled
    _banner("OrbitGuard CPU vs GPU Comparison")
    print(f"  GPU enabled: {is_gpu_enabled()}")
    try:
        import cupy
        print(f"  CuPy version: {cupy.__version__}")
        print(f"  CUDA device: {cupy.cuda.Device(0).compute_capability}")
    except Exception:
        print("  CuPy: not available")

    benchmark_monte_carlo()
    benchmark_batch_distance()
    benchmark_tca_norms()

    _banner("Done")


if __name__ == "__main__":
    main()
