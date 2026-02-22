#!/usr/bin/env python3
"""Compare CPU vs GPU results for OrbitGuard numerical computations.

Runs the same Monte Carlo Pc and distance computations on both backends,
prints timing and result deltas.

Usage:
    python scripts/compare_cpu_gpu.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.compute_backend import reset_backend, get_xp, is_gpu_enabled
from app.services.pc_calculator import compute_pc_monte_carlo


def run_monte_carlo_benchmark(backend: str, n_samples: int = 100_000, seed: int = 42):
    """Run Monte Carlo Pc on a given backend and return result + timing."""
    os.environ["ORBITGUARD_BACKEND"] = backend
    reset_backend()

    miss = np.array([0.01, 0.005])
    cov = np.eye(2) * 0.05**2
    hbr = 0.020

    # Warm-up run
    compute_pc_monte_carlo(miss, cov, hbr, n_samples=100, seed=seed)

    # Timed run
    t0 = time.perf_counter()
    result = compute_pc_monte_carlo(miss, cov, hbr, n_samples=n_samples, seed=seed)
    elapsed = time.perf_counter() - t0

    return result, elapsed


def run_distance_benchmark(backend: str, n_points: int = 100_000):
    """Run vectorized distance computation on a given backend."""
    os.environ["ORBITGUARD_BACKEND"] = backend
    reset_backend()

    rng = np.random.default_rng(42)
    d_pos = rng.standard_normal((n_points, 3)) * 7000
    i_pos = d_pos + rng.standard_normal((n_points, 3)) * 10

    xp = get_xp()
    d_xp = xp.asarray(d_pos)
    i_xp = xp.asarray(i_pos)

    # Warm-up
    _ = xp.linalg.norm(d_xp - i_xp, axis=1)

    t0 = time.perf_counter()
    distances = xp.linalg.norm(d_xp - i_xp, axis=1)
    if hasattr(distances, "get"):
        distances = distances.get()
    elapsed = time.perf_counter() - t0

    return distances, elapsed


def main():
    print("=" * 60)
    print("OrbitGuard CPU vs GPU Comparison")
    print("=" * 60)

    n_samples = 500_000

    # --- Monte Carlo Benchmark ---
    print(f"\n--- Monte Carlo Pc ({n_samples:,} samples) ---")

    cpu_result, cpu_time = run_monte_carlo_benchmark("cpu", n_samples)
    print(f"  CPU: Pc = {cpu_result.pc:.8f}  "
          f"CI = [{cpu_result.ci_low:.8f}, {cpu_result.ci_high:.8f}]  "
          f"Time = {cpu_time:.4f}s")

    try:
        gpu_result, gpu_time = run_monte_carlo_benchmark("gpu", n_samples)
        gpu_active = is_gpu_enabled()
    except Exception as e:
        print(f"  GPU: Not available ({e})")
        gpu_active = False

    if gpu_active:
        print(f"  GPU: Pc = {gpu_result.pc:.8f}  "
              f"CI = [{gpu_result.ci_low:.8f}, {gpu_result.ci_high:.8f}]  "
              f"Time = {gpu_time:.4f}s")
        delta = abs(cpu_result.pc - gpu_result.pc)
        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        print(f"  Delta Pc = {delta:.2e}")
        print(f"  Speedup  = {speedup:.1f}x")
    else:
        print("  GPU: Skipped (CuPy/CUDA not available)")
        print("  Running CPU-only comparison with two different seeds instead:")
        r2, t2 = run_monte_carlo_benchmark("cpu", n_samples, seed=99)
        delta = abs(cpu_result.pc - r2.pc)
        print(f"  CPU (seed=99): Pc = {r2.pc:.8f}  Time = {t2:.4f}s")
        print(f"  Delta between seeds = {delta:.2e}")

    # --- Distance Benchmark ---
    n_points = 500_000
    print(f"\n--- Distance Computation ({n_points:,} points) ---")

    cpu_dist, cpu_dt = run_distance_benchmark("cpu", n_points)
    print(f"  CPU: Time = {cpu_dt:.4f}s")

    os.environ["ORBITGUARD_BACKEND"] = "gpu"
    reset_backend()
    if is_gpu_enabled():
        gpu_dist, gpu_dt = run_distance_benchmark("gpu", n_points)
        max_diff = float(np.max(np.abs(cpu_dist - gpu_dist)))
        speedup = cpu_dt / gpu_dt if gpu_dt > 0 else float("inf")
        print(f"  GPU: Time = {gpu_dt:.4f}s")
        print(f"  Max abs difference = {max_diff:.2e}")
        print(f"  Speedup = {speedup:.1f}x")
    else:
        print("  GPU: Skipped (CuPy/CUDA not available)")

    # Cleanup
    os.environ.pop("ORBITGUARD_BACKEND", None)
    reset_backend()

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
