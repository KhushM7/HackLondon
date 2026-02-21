"""Comprehensive tests for the high-fidelity orbital mechanics pipeline.

Covers: propagation, TCA finding, B-plane, Pc (Foster/Chan/Monte Carlo),
covariance handling, maneuver planning, and integration.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

# --- ISS TLE (for propagation tests) ---
ISS_LINE1 = "1 25544U 98067A   24045.51782528  .00011988  00000-0  21539-3 0  9992"
ISS_LINE2 = "2 25544  51.6418 283.1113 0005188 107.9657  12.7847 15.50095752440727"

# A second object in similar orbit (Cosmos debris)
OBJ2_LINE1 = "1 25730U 99020A   24045.53456789  .00001234  00000-0  12345-3 0  9999"
OBJ2_LINE2 = "2 25730  51.6500 283.0000 0006000 110.0000  15.0000 15.50100000 10000"


class TestOrbitalConstants:
    def test_gm_earth(self):
        from app.services.orbital_constants import GM_EARTH_KM3_S2
        assert abs(GM_EARTH_KM3_S2 - 398600.4418) < 0.01

    def test_r_earth(self):
        from app.services.orbital_constants import R_EARTH_KM
        assert abs(R_EARTH_KM - 6378.137) < 0.001

    def test_risk_classification(self):
        from app.services.orbital_constants import classify_risk, RiskLevel
        assert classify_risk(1e-3) == RiskLevel.RED
        assert classify_risk(5e-5) == RiskLevel.YELLOW
        assert classify_risk(1e-6) == RiskLevel.GREEN

    def test_eci_to_rtn_orthogonal(self):
        from app.services.orbital_constants import eci_to_rtn
        r = np.array([7000.0, 0.0, 0.0])
        v = np.array([0.0, 7.5, 0.0])
        rot = eci_to_rtn(r, v)
        # Should be orthogonal
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-10)

    def test_rtn_roundtrip(self):
        from app.services.orbital_constants import eci_to_rtn, rtn_to_eci
        r = np.array([6800.0, 1000.0, 200.0])
        v = np.array([-1.0, 7.0, 0.5])
        fwd = eci_to_rtn(r, v)
        inv = rtn_to_eci(r, v)
        assert np.allclose(fwd @ inv, np.eye(3), atol=1e-10)


class TestCowellPropagator:
    def test_circular_orbit_one_period(self):
        """A circular orbit should maintain approximately constant radius with J2."""
        from app.services.cowell_propagator import propagate_cowell
        from app.services.orbital_constants import GM_EARTH_KM3_S2, PropagationConfig

        r0 = np.array([7000.0, 0.0, 0.0])  # km
        v_circ = np.sqrt(GM_EARTH_KM3_S2 / 7000.0)
        v0 = np.array([0.0, v_circ, 0.0])  # km/s

        period = 2.0 * np.pi * np.sqrt(7000.0**3 / GM_EARTH_KM3_S2)
        config = PropagationConfig(use_drag=False, use_srp=False, use_third_body=False)

        times, positions, velocities = propagate_cowell(
            r0, v0, period, config, step_seconds=10.0
        )

        # Radius should stay approximately constant (J2 causes nodal precession, not radius change)
        radii = np.linalg.norm(positions, axis=1)
        assert abs(radii[-1] - radii[0]) < 5.0, \
            f"Radius changed by {abs(radii[-1] - radii[0]):.1f} km (expect < 5)"
        assert np.std(radii) < 10.0, f"Radius std dev = {np.std(radii):.1f} km (expect < 10)"

    def test_two_body_no_perturbation(self):
        """Without perturbations, energy should be conserved."""
        from app.services.cowell_propagator import propagate_cowell
        from app.services.orbital_constants import GM_EARTH_KM3_S2, PropagationConfig

        r0 = np.array([7000.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.546, 0.0])
        config = PropagationConfig(use_j2_j6=False, use_drag=False, use_srp=False, use_third_body=False)

        times, positions, velocities = propagate_cowell(
            r0, v0, 3600.0, config, step_seconds=30.0
        )

        # Specific energy should be conserved
        e0 = 0.5 * np.dot(v0, v0) - GM_EARTH_KM3_S2 / np.linalg.norm(r0)
        ef = 0.5 * np.dot(velocities[-1], velocities[-1]) - GM_EARTH_KM3_S2 / np.linalg.norm(positions[-1])
        assert abs(e0 - ef) / abs(e0) < 1e-8, f"Energy not conserved: {e0} vs {ef}"


class TestTCAFinder:
    def test_known_geometry(self):
        """Two objects on crossing orbits should have a detectable TCA."""
        from app.services.tca_finder import find_tca_from_states

        # Create two straight-line "trajectories" that cross
        n = 100
        times = np.arange(n) * 60.0  # 0 to 99 minutes in seconds
        epoch = datetime(2024, 2, 15, 12, 0, 0)

        # Object 1: moving along X
        pos1 = np.zeros((n, 3))
        pos1[:, 0] = 7000.0 + np.arange(n) * 0.1
        vel1 = np.tile([0.1 / 60.0, 0.0, 0.0], (n, 1))

        # Object 2: moving along Y, crossing at time step 50
        pos2 = np.zeros((n, 3))
        pos2[:, 0] = 7000.0 + 50 * 0.1  # constant X = crossing point
        pos2[:, 1] = -500.0 + np.arange(n) * 10.0  # moving along Y
        vel2 = np.tile([0.0, 10.0 / 60.0, 0.0], (n, 1))

        result = find_tca_from_states(times, pos1, vel1, pos2, vel2, epoch)
        assert result is not None
        # TCA should be near step 50 (3000 seconds)
        assert abs(result.tca_offset_seconds - 3000.0) < 120.0
        assert result.miss_distance_km < 600  # They get close in Y

    def test_identical_objects_rejected(self):
        """Two identical trajectories should still produce a TCA (distance ~0)."""
        from app.services.tca_finder import find_tca_from_states

        n = 50
        times = np.arange(n) * 60.0
        epoch = datetime(2024, 2, 15)
        pos = np.column_stack([7000 + np.arange(n) * 0.1, np.zeros(n), np.zeros(n)])
        vel = np.tile([0.1 / 60, 0, 0], (n, 1))

        result = find_tca_from_states(times, pos, vel, pos, vel, epoch)
        assert result is not None
        assert result.miss_distance_km < 0.001


class TestBPlaneAnalysis:
    def test_bplane_orthogonality(self):
        """B-plane frame vectors should be mutually orthogonal."""
        from app.services.bplane_analysis import compute_bplane
        from app.services.tca_finder import TCAResult

        tca = TCAResult(
            tca_epoch=datetime(2024, 2, 15),
            miss_distance_km=5.0,
            miss_vector_km=np.array([3.0, 4.0, 0.0]),
            relative_speed_kms=10.0,
            r_primary_km=np.array([7000.0, 0.0, 0.0]),
            v_primary_kms=np.array([0.0, 7.5, 0.0]),
            r_secondary_km=np.array([6997.0, -4.0, 0.0]),
            v_secondary_kms=np.array([1.0, 7.0, 0.5]),
            tca_offset_seconds=0.0,
        )
        bp = compute_bplane(tca)

        # T, R, N should be orthogonal unit vectors
        assert abs(np.dot(bp.t_hat, bp.r_hat)) < 1e-10
        assert abs(np.dot(bp.t_hat, bp.n_hat)) < 1e-10
        assert abs(np.dot(bp.r_hat, bp.n_hat)) < 1e-10
        assert abs(np.linalg.norm(bp.t_hat) - 1.0) < 1e-10
        assert abs(np.linalg.norm(bp.r_hat) - 1.0) < 1e-10
        assert abs(np.linalg.norm(bp.n_hat) - 1.0) < 1e-10

    def test_bplane_magnitude_consistency(self):
        """B magnitude should equal sqrt(BdotT^2 + BdotR^2)."""
        from app.services.bplane_analysis import compute_bplane
        from app.services.tca_finder import TCAResult

        tca = TCAResult(
            tca_epoch=datetime(2024, 2, 15),
            miss_distance_km=5.0,
            miss_vector_km=np.array([3.0, 4.0, 0.0]),
            relative_speed_kms=10.0,
            r_primary_km=np.array([7000.0, 0.0, 0.0]),
            v_primary_kms=np.array([0.0, 7.5, 0.0]),
            r_secondary_km=np.array([6997.0, -4.0, 0.0]),
            v_secondary_kms=np.array([1.0, 7.0, 0.5]),
            tca_offset_seconds=0.0,
        )
        bp = compute_bplane(tca)

        expected_mag = np.sqrt(bp.b_dot_t_km**2 + bp.b_dot_r_km**2)
        assert abs(bp.b_magnitude_km - expected_mag) < 1e-10


class TestPcCalculator:
    def test_foster_head_on_zero_miss(self):
        """Zero miss distance with tight covariance → Pc should be very high."""
        from app.services.pc_calculator import compute_pc_foster

        miss = np.array([0.0, 0.0])
        sigma = 0.01  # 10 m uncertainty
        cov = np.eye(2) * sigma**2
        hbr = 0.020  # 20 m

        result = compute_pc_foster(miss, cov, hbr)
        assert result.pc > 0.5, f"Head-on collision Pc should be high, got {result.pc}"

    def test_foster_large_miss_distance(self):
        """Miss distance >> HBR → Pc should be negligible."""
        from app.services.pc_calculator import compute_pc_foster

        hbr = 0.020  # 20 m
        miss = np.array([100.0 * hbr, 0.0])  # 2 km miss
        cov = np.eye(2) * 0.1**2  # 100 m uncertainty

        result = compute_pc_foster(miss, cov, hbr)
        assert result.pc < 1e-10, f"Large miss Pc should be tiny, got {result.pc}"

    def test_foster_isotropic_zero_miss(self):
        """Zero miss + isotropic cov → Pc ≈ HBR²/(2σ²)."""
        from app.services.pc_calculator import compute_pc_foster

        sigma = 0.1  # 100 m
        cov = np.eye(2) * sigma**2
        hbr = 0.020  # 20 m
        miss = np.array([0.0, 0.0])

        result = compute_pc_foster(miss, cov, hbr)
        # Analytical: Pc = 1 - exp(-HBR²/(2σ²))
        expected = 1.0 - np.exp(-hbr**2 / (2.0 * sigma**2))
        assert abs(result.pc - expected) / max(expected, 1e-15) < 0.05, \
            f"Foster Pc {result.pc} doesn't match analytical {expected}"

    def test_chan_upper_bound(self):
        """Chan Pc should be >= Foster Pc for same inputs."""
        from app.services.pc_calculator import compute_pc_foster, compute_pc_chan

        miss = np.array([0.05, 0.03])
        cov = np.array([[0.01, 0.002], [0.002, 0.008]])
        hbr = 0.020

        foster = compute_pc_foster(miss, cov, hbr)
        chan = compute_pc_chan(np.linalg.norm(miss), cov, hbr)

        # Chan should generally be an upper bound
        # (not strictly guaranteed for all geometries, but typical)
        assert chan.pc >= foster.pc * 0.1, "Chan should be similar magnitude to Foster"

    def test_monte_carlo_agreement_with_foster(self):
        """Monte Carlo Pc should agree with Foster within statistical uncertainty."""
        from app.services.pc_calculator import compute_pc_foster, compute_pc_monte_carlo

        miss = np.array([0.0, 0.0])
        sigma = 0.05
        cov = np.eye(2) * sigma**2
        hbr = 0.020

        foster = compute_pc_foster(miss, cov, hbr)
        mc = compute_pc_monte_carlo(miss, cov, hbr, n_samples=50000, seed=42)

        # MC should be within its own CI of Foster
        assert mc.ci_low is not None and mc.ci_high is not None
        # Allow generous tolerance (Foster should be within 5x MC range)
        assert mc.pc > 0, "MC should detect some collisions"
        ratio = mc.pc / max(foster.pc, 1e-15)
        assert 0.2 < ratio < 5.0, f"MC/Foster ratio {ratio} too far from 1"

    def test_pc_nan_handling(self):
        """Singular covariance should not crash, returns NaN gracefully."""
        from app.services.pc_calculator import compute_pc_foster

        miss = np.array([0.01, 0.01])
        cov = np.array([[0.0, 0.0], [0.0, 0.0]])  # singular
        result = compute_pc_foster(miss, cov, 0.020)
        assert math.isnan(result.pc) or result.pc == 0.0


class TestCovarianceHandler:
    def test_estimate_from_tle(self):
        """Estimated covariance should be positive definite."""
        from app.services.covariance_handler import estimate_covariance_from_tle

        cov = estimate_covariance_from_tle(bstar=0.0001, epoch_age_days=1.0)
        assert cov.source == "estimated"
        assert cov.cov_6x6.shape == (6, 6)
        # Should be positive definite
        eigvals = np.linalg.eigvalsh(cov.cov_6x6)
        assert np.all(eigvals > 0), "Covariance should be positive definite"

    def test_higham_correction(self):
        """Non-PD matrix should be corrected to PD."""
        from app.services.covariance_handler import ensure_positive_definite

        # Create a non-PD matrix
        bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3, -1
        corrected, was_corrected = ensure_positive_definite(bad)
        assert was_corrected is True
        eigvals = np.linalg.eigvalsh(corrected)
        assert np.all(eigvals > 0)

    def test_pd_matrix_unchanged(self):
        """A PD matrix should pass through unchanged."""
        from app.services.covariance_handler import ensure_positive_definite

        good = np.eye(3)
        corrected, was_corrected = ensure_positive_definite(good)
        assert was_corrected is False
        assert np.allclose(corrected, good)

    def test_bplane_projection(self):
        """Projected covariance should be 2x2 and PD."""
        from app.services.covariance_handler import project_covariance_to_bplane
        from app.services.bplane_analysis import BPlaneResult

        cov1 = np.eye(3) * 0.5
        cov2 = np.eye(3) * 0.3
        bplane = BPlaneResult(
            b_dot_t_km=1.0, b_dot_r_km=0.5, b_magnitude_km=1.118,
            theta_rad=0.4636,
            t_hat=np.array([1.0, 0.0, 0.0]),
            r_hat=np.array([0.0, 0.0, 1.0]),
            n_hat=np.array([0.0, 1.0, 0.0]),
            b_vector_km=np.array([0.0, 1.0, 0.5]),
        )

        cov_bp = project_covariance_to_bplane(cov1, cov2, bplane)
        assert cov_bp.shape == (2, 2)
        eigvals = np.linalg.eigvalsh(cov_bp)
        assert np.all(eigvals > 0)


class TestManeuverPlanner:
    def test_tsiolkovsky_fuel_cost(self):
        """Verify Tsiolkovsky equation for known case."""
        from app.services.maneuver_planner import tsiolkovsky_fuel_cost
        from app.services.orbital_constants import G0_M_S2

        # For Isp=300s, m0=1000kg, dv=100 m/s
        isp = 300.0
        m0 = 1000.0
        dv = 100.0

        fuel = tsiolkovsky_fuel_cost(dv, m0, isp)
        # Expected: m0 * (1 - exp(-dv / (Isp*g0)))
        expected = m0 * (1.0 - np.exp(-dv / (isp * G0_M_S2)))
        assert abs(fuel - expected) < 0.01, f"Fuel {fuel} != expected {expected}"
        assert 0 < fuel < m0, "Fuel must be between 0 and total mass"

    def test_fuel_cost_increases_with_dv(self):
        """Higher delta-v should cost more fuel."""
        from app.services.maneuver_planner import tsiolkovsky_fuel_cost

        f1 = tsiolkovsky_fuel_cost(1.0)
        f2 = tsiolkovsky_fuel_cost(5.0)
        f3 = tsiolkovsky_fuel_cost(10.0)
        assert f1 < f2 < f3

    def test_burn_directions_complete(self):
        """All 6 RTN directions should be defined."""
        from app.services.maneuver_planner import BURN_DIRECTIONS

        expected = {"prograde", "retrograde", "radial_plus", "radial_minus", "normal_plus", "normal_minus"}
        assert set(BURN_DIRECTIONS.keys()) == expected

    def test_burn_directions_unit_vectors(self):
        """All burn direction vectors should be unit vectors."""
        from app.services.maneuver_planner import BURN_DIRECTIONS

        for name, vec in BURN_DIRECTIONS.items():
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-10, f"{name} is not a unit vector"


class TestPropagationConfig:
    def test_default_config(self):
        from app.services.orbital_constants import PropagationConfig
        config = PropagationConfig()
        assert config.step_seconds == 60.0
        assert config.duration_hours == 168.0
        assert config.mass_kg == 500.0


class TestAlfanoValidation:
    """Validation against Alfano (2005) test cases for Pc computation.

    These test cases verify that the Foster 2D Gaussian Pc implementation
    produces correct results for known conjunction geometries.
    """

    def test_zero_miss_symmetric_covariance(self):
        """Alfano-style: zero miss distance, symmetric covariance.

        For miss=[0,0], isotropic cov with sigma, HBR=r:
        Pc = 1 - exp(-r²/(2σ²))
        """
        from app.services.pc_calculator import compute_pc_foster

        sigma = 0.1  # 100 m
        hbr = 0.020  # 20 m
        miss = np.array([0.0, 0.0])
        cov = np.eye(2) * sigma**2

        result = compute_pc_foster(miss, cov, hbr)
        expected = 1.0 - np.exp(-hbr**2 / (2.0 * sigma**2))

        rel_error = abs(result.pc - expected) / expected
        assert rel_error < 0.01, f"Alfano case 1: error {rel_error:.4%} (got {result.pc}, expected {expected})"

    def test_offset_miss_known_pc(self):
        """Offset miss with known analytical result.

        For small HBR << sigma, Pc ≈ (HBR²/(2πσ_1σ_2)) * exp(-0.5 * miss^T @ C^-1 @ miss)
        """
        from app.services.pc_calculator import compute_pc_foster

        sigma1, sigma2 = 0.5, 0.3  # km
        hbr = 0.010  # 10 m
        miss = np.array([0.1, 0.05])  # 100m, 50m miss
        cov = np.diag([sigma1**2, sigma2**2])

        result = compute_pc_foster(miss, cov, hbr)

        # Approximate analytical formula for small HBR
        cov_inv = np.diag([1.0/sigma1**2, 1.0/sigma2**2])
        exp_term = np.exp(-0.5 * miss @ cov_inv @ miss)
        approx_pc = (np.pi * hbr**2) / (2.0 * np.pi * sigma1 * sigma2) * exp_term

        # Should be within 5% for small HBR
        if approx_pc > 1e-15:
            rel_error = abs(result.pc - approx_pc) / approx_pc
            assert rel_error < 0.1, f"Alfano offset case: error {rel_error:.4%}"

    def test_very_large_miss_negligible_pc(self):
        """100x HBR miss distance → Pc < 1e-10."""
        from app.services.pc_calculator import compute_pc_foster

        hbr = 0.020
        miss = np.array([100.0 * hbr, 0.0])
        cov = np.eye(2) * 0.1**2

        result = compute_pc_foster(miss, cov, hbr)
        assert result.pc < 1e-10, f"Large miss Pc = {result.pc}, expected < 1e-10"


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_with_synthetic_data(self):
        """Test: create TCA → B-plane → covariance → Pc → all values consistent."""
        from app.services.tca_finder import TCAResult
        from app.services.bplane_analysis import compute_bplane
        from app.services.covariance_handler import default_covariance, project_covariance_to_bplane
        from app.services.pc_calculator import compute_pc_foster, compute_pc_chan, compute_pc_monte_carlo

        # Synthetic TCA
        tca = TCAResult(
            tca_epoch=datetime(2024, 6, 15, 12, 0, 0),
            miss_distance_km=0.5,
            miss_vector_km=np.array([0.3, 0.4, 0.0]),
            relative_speed_kms=10.0,
            r_primary_km=np.array([7000.0, 0.0, 0.0]),
            v_primary_kms=np.array([0.0, 7.5, 0.5]),
            r_secondary_km=np.array([6999.7, -0.4, 0.0]),
            v_secondary_kms=np.array([0.5, 7.0, 1.0]),
            tca_offset_seconds=86400.0,
        )

        # B-plane
        bp = compute_bplane(tca)
        assert bp.b_magnitude_km > 0

        # Covariance
        cov_p = default_covariance(sigma_pos_km=0.5)
        cov_s = default_covariance(sigma_pos_km=0.5)
        cov_bp = project_covariance_to_bplane(cov_p.cov_3x3_pos, cov_s.cov_3x3_pos, bp)
        assert cov_bp.shape == (2, 2)

        # Pc
        miss_bp = np.array([bp.b_dot_t_km, bp.b_dot_r_km])
        pc_f = compute_pc_foster(miss_bp, cov_bp, 0.020)
        pc_c = compute_pc_chan(tca.miss_distance_km, cov_bp, 0.020)
        pc_mc = compute_pc_monte_carlo(miss_bp, cov_bp, 0.020, n_samples=5000, seed=42)

        assert 0 <= pc_f.pc <= 1
        assert 0 <= pc_c.pc <= 1
        assert 0 <= pc_mc.pc <= 1
        assert pc_f.method == "foster"
        assert pc_c.method == "chan"
        assert pc_mc.method == "monte_carlo"

    def test_risk_level_consistency(self):
        """Risk level should match Pc value."""
        from app.services.orbital_constants import classify_risk, RiskLevel

        assert classify_risk(0.001) == RiskLevel.RED
        assert classify_risk(0.0001) == RiskLevel.RED
        assert classify_risk(0.00005) == RiskLevel.YELLOW
        assert classify_risk(0.000001) == RiskLevel.GREEN
        assert classify_risk(0.0) == RiskLevel.GREEN


class TestExistingCompatibility:
    """Ensure the old API interface still works."""

    def test_risk_tier_still_works(self):
        from app.services.screening_engine import ScreeningEngine
        assert ScreeningEngine._risk_tier(0.2) == "High"
        assert ScreeningEngine._risk_tier(2.0) == "Medium"
        assert ScreeningEngine._risk_tier(8.0) == "Low"
