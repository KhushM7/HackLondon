"""Tests for GPU/CPU compute backend and CPU-vs-GPU result parity.

These tests verify:
1. Backend selection and CPU fallback logic
2. Monte Carlo Pc produces same-shape outputs on CPU
3. Screening distance computation produces correct results
4. TCA state analysis produces correct results
"""
from __future__ import annotations

import os
import numpy as np
import pytest

from app.services.compute_backend import (
    get_xp,
    is_gpu_enabled,
    asnumpy,
    get_rng,
    reset_backend,
)


class TestBackendSelection:
    """Test backend initialization and selection logic."""

    def setup_method(self):
        reset_backend()

    def teardown_method(self):
        reset_backend()
        os.environ.pop("ORBITGUARD_BACKEND", None)

    def test_default_is_cpu(self):
        os.environ.pop("ORBITGUARD_BACKEND", None)
        reset_backend()
        xp = get_xp()
        assert xp is np
        assert not is_gpu_enabled()

    def test_explicit_cpu(self):
        os.environ["ORBITGUARD_BACKEND"] = "cpu"
        reset_backend()
        xp = get_xp()
        assert xp is np
        assert not is_gpu_enabled()

    def test_gpu_fallback_without_cupy(self):
        """When GPU requested but CuPy not installed, should fall back to CPU."""
        os.environ["ORBITGUARD_BACKEND"] = "gpu"
        reset_backend()
        # This will attempt to import cupy; if unavailable, falls back
        xp = get_xp()
        # Either cupy (if installed) or numpy (fallback)
        assert xp is not None
        # If cupy not available, should be numpy
        try:
            import cupy
            # CuPy is available — GPU may or may not be enabled depending on CUDA
        except ImportError:
            assert xp is np
            assert not is_gpu_enabled()

    def test_asnumpy_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = asnumpy(arr)
        assert isinstance(result, np.ndarray)
        assert result is arr  # should be the same object

    def test_asnumpy_from_list(self):
        result = asnumpy([1.0, 2.0])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_get_rng_returns_generator(self):
        rng = get_rng(seed=42)
        assert rng is not None
        # Should be able to generate random numbers
        if hasattr(rng, "standard_normal"):
            val = rng.standard_normal(10)
            assert len(val) == 10

    def test_get_rng_deterministic(self):
        rng1 = get_rng(seed=123)
        rng2 = get_rng(seed=123)
        if hasattr(rng1, "standard_normal"):
            a = rng1.standard_normal(100)
            b = rng2.standard_normal(100)
            np.testing.assert_array_equal(a, b)

    def test_reset_backend(self):
        os.environ["ORBITGUARD_BACKEND"] = "cpu"
        reset_backend()
        assert not is_gpu_enabled()
        # Reset and change
        os.environ["ORBITGUARD_BACKEND"] = "cpu"
        reset_backend()
        assert not is_gpu_enabled()


class TestMonteCarloCPUParity:
    """Test that Monte Carlo Pc produces consistent results on CPU."""

    def setup_method(self):
        os.environ["ORBITGUARD_BACKEND"] = "cpu"
        reset_backend()

    def teardown_method(self):
        os.environ.pop("ORBITGUARD_BACKEND", None)
        reset_backend()

    def test_mc_shape_and_type(self):
        from app.services.pc_calculator import compute_pc_monte_carlo
        miss = np.array([0.0, 0.0])
        cov = np.eye(2) * 0.05**2
        result = compute_pc_monte_carlo(miss, cov, 0.020, n_samples=1000, seed=42)
        assert isinstance(result.pc, float)
        assert 0.0 <= result.pc <= 1.0
        assert result.ci_low is not None
        assert result.ci_high is not None
        assert result.ci_low <= result.pc <= result.ci_high or result.pc == 0.0

    def test_mc_deterministic_with_seed(self):
        from app.services.pc_calculator import compute_pc_monte_carlo
        miss = np.array([0.01, 0.005])
        cov = np.eye(2) * 0.05**2

        r1 = compute_pc_monte_carlo(miss, cov, 0.020, n_samples=5000, seed=42)
        r2 = compute_pc_monte_carlo(miss, cov, 0.020, n_samples=5000, seed=42)
        assert r1.pc == r2.pc

    def test_mc_singular_covariance_handles_gracefully(self):
        from app.services.pc_calculator import compute_pc_monte_carlo
        miss = np.array([0.01, 0.01])
        cov = np.zeros((2, 2))  # singular
        result = compute_pc_monte_carlo(miss, cov, 0.020, n_samples=100, seed=42)
        # Should either return NaN (error) or a valid float (all samples at mean)
        assert isinstance(result.pc, float)
        assert result.method == "monte_carlo"

    def test_mc_zero_miss_high_pc(self):
        from app.services.pc_calculator import compute_pc_monte_carlo
        miss = np.array([0.0, 0.0])
        sigma = 0.01
        cov = np.eye(2) * sigma**2
        hbr = 0.020
        result = compute_pc_monte_carlo(miss, cov, hbr, n_samples=10000, seed=42)
        assert result.pc > 0.3, f"Expected high Pc for head-on, got {result.pc}"

    def test_mc_large_miss_low_pc(self):
        from app.services.pc_calculator import compute_pc_monte_carlo
        miss = np.array([10.0, 0.0])
        cov = np.eye(2) * 0.1**2
        result = compute_pc_monte_carlo(miss, cov, 0.020, n_samples=10000, seed=42)
        assert result.pc == 0.0, f"Expected ~0 Pc for large miss, got {result.pc}"


class TestScreeningDistanceParity:
    """Test that GPU-capable distance computation produces correct results."""

    def setup_method(self):
        os.environ["ORBITGUARD_BACKEND"] = "cpu"
        reset_backend()

    def teardown_method(self):
        os.environ.pop("ORBITGUARD_BACKEND", None)
        reset_backend()

    def test_distance_computation_correct(self):
        from app.services.compute_backend import get_xp, asnumpy
        xp = get_xp()

        d_pos = np.array([[7000.0, 0.0, 0.0], [7001.0, 1.0, 0.0], [7002.0, 2.0, 1.0]])
        i_pos = np.array([[7000.5, 0.1, 0.0], [7001.5, 1.1, 0.0], [7002.5, 2.1, 1.0]])

        d_xp = xp.asarray(d_pos)
        i_xp = xp.asarray(i_pos)
        distances_xp = xp.linalg.norm(d_xp - i_xp, axis=1)
        distances = asnumpy(distances_xp)

        expected = np.linalg.norm(d_pos - i_pos, axis=1)
        np.testing.assert_allclose(distances, expected, rtol=1e-12)

    def test_argmin_consistent(self):
        from app.services.compute_backend import get_xp
        xp = get_xp()
        dists = xp.asarray([5.0, 3.0, 1.0, 4.0, 2.0])
        assert int(xp.argmin(dists)) == 2


class TestTCAStateParity:
    """Test that GPU-capable TCA state analysis produces correct results."""

    def setup_method(self):
        os.environ["ORBITGUARD_BACKEND"] = "cpu"
        reset_backend()

    def teardown_method(self):
        os.environ.pop("ORBITGUARD_BACKEND", None)
        reset_backend()

    def test_distances_and_dots_match_numpy(self):
        from app.services.compute_backend import get_xp, asnumpy
        xp = get_xp()

        n = 100
        pos_p = np.random.default_rng(42).standard_normal((n, 3)) * 100
        pos_s = np.random.default_rng(43).standard_normal((n, 3)) * 100
        vel_p = np.random.default_rng(44).standard_normal((n, 3))
        vel_s = np.random.default_rng(45).standard_normal((n, 3))

        # Reference: pure numpy
        r_rel_np = pos_p - pos_s
        v_rel_np = vel_p - vel_s
        dist_np = np.linalg.norm(r_rel_np, axis=1)
        dot_np = np.sum(r_rel_np * v_rel_np, axis=1)

        # xp path
        r_rel = xp.asarray(pos_p) - xp.asarray(pos_s)
        v_rel = xp.asarray(vel_p) - xp.asarray(vel_s)
        dist_xp = asnumpy(xp.linalg.norm(r_rel, axis=1))
        dot_xp = asnumpy(xp.sum(r_rel * v_rel, axis=1))

        np.testing.assert_allclose(dist_xp, dist_np, rtol=1e-12)
        np.testing.assert_allclose(dot_xp, dot_np, rtol=1e-12)

    def test_find_tca_from_states_unchanged(self):
        """Ensure find_tca_from_states still works correctly with backend."""
        from datetime import datetime
        from app.services.tca_finder import find_tca_from_states

        n = 100
        times = np.arange(n) * 60.0
        epoch = datetime(2024, 2, 15, 12, 0, 0)

        pos1 = np.zeros((n, 3))
        pos1[:, 0] = 7000.0 + np.arange(n) * 0.1
        vel1 = np.tile([0.1 / 60.0, 0.0, 0.0], (n, 1))

        pos2 = np.zeros((n, 3))
        pos2[:, 0] = 7000.0 + 50 * 0.1
        pos2[:, 1] = -500.0 + np.arange(n) * 10.0
        vel2 = np.tile([0.0, 10.0 / 60.0, 0.0], (n, 1))

        result = find_tca_from_states(times, pos1, vel1, pos2, vel2, epoch)
        assert result is not None
        assert abs(result.tca_offset_seconds - 3000.0) < 120.0
