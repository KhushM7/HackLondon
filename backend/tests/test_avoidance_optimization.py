"""Tests for the avoidance optimization pipeline and API endpoints."""
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from app.models import AvoidancePlan, ConjunctionEvent, TLERecord
from app.schemas import (
    AvoidancePlanSummary,
    OptimizeAvoidanceRequest,
)
from app.services.avoidance_simulator import _eci_to_lla, _CandidateScore


class TestEciToLla:
    """Test ECI-to-LLA approximate conversion."""

    def test_equatorial_point(self):
        pos = [6778.0, 0.0, 0.0]  # ~400 km altitude on equator
        lla = _eci_to_lla(pos)
        assert len(lla) == 3
        assert abs(lla[0]) < 1.0  # lat near 0
        assert abs(lla[1]) < 1.0  # lon near 0
        assert 350 < lla[2] < 450  # alt ~400 km

    def test_polar_point(self):
        pos = [0.0, 0.0, 6778.0]  # over north pole
        lla = _eci_to_lla(pos)
        assert abs(lla[0] - 90.0) < 1.0  # lat ~90


class TestCandidateScore:
    """Test candidate scoring dataclass."""

    def test_basic_construction(self):
        c = _CandidateScore(
            direction="prograde",
            dv_mps=0.5,
            burn_lead_hours=24.0,
            miss_distance_km=5.0,
            pc=1e-6,
            score=-2.0,
            rtn_vector=[0.0, 0.5, 0.0],
        )
        assert c.direction == "prograde"
        assert c.dv_mps == 0.5
        assert c.score == -2.0


class TestAvoidancePlanModel:
    """Test AvoidancePlan SQLAlchemy model instantiation."""

    def test_defaults(self):
        plan = AvoidancePlan(asset_norad_id=25544, status="pending")
        assert plan.status == "pending"
        assert plan.asset_norad_id == 25544
        assert plan.burn_dv_mps is None
        assert plan.event_id is None


class TestOptimizeAvoidanceRequest:
    """Test request schema validation."""

    def test_defaults(self):
        req = OptimizeAvoidanceRequest()
        assert req.max_delta_v_mps == 5.0
        assert req.burn_window_hours == 48.0
        assert req.top_n_events == 3

    def test_custom_values(self):
        req = OptimizeAvoidanceRequest(
            max_delta_v_mps=2.0,
            burn_window_hours=12.0,
            weight_miss_distance=2.0,
            weight_delta_v=0.1,
            top_n_events=5,
        )
        assert req.max_delta_v_mps == 2.0
        assert req.top_n_events == 5

    def test_validation_bounds(self):
        with pytest.raises(Exception):
            OptimizeAvoidanceRequest(max_delta_v_mps=0.0)
        with pytest.raises(Exception):
            OptimizeAvoidanceRequest(burn_window_hours=0.5)


class TestAvoidancePlanSummary:
    """Test response schema construction."""

    def test_minimal(self):
        summary = AvoidancePlanSummary(
            id=1,
            asset_norad_id=25544,
            status="pending",
            created_at=datetime.utcnow(),
        )
        assert summary.status == "pending"
        assert summary.burn_dv_mps is None

    def test_completed(self):
        summary = AvoidancePlanSummary(
            id=1,
            asset_norad_id=25544,
            status="completed",
            event_id=42,
            burn_direction="prograde",
            burn_dv_mps=0.5,
            burn_rtn_vector=[0.0, 0.5, 0.0],
            burn_epoch=datetime.utcnow(),
            pre_miss_distance_km=0.8,
            post_miss_distance_km=5.2,
            pre_pc=1e-3,
            post_pc=1e-7,
            fuel_cost_kg=0.034,
            candidates_evaluated=42,
            optimization_elapsed_s=3.5,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        assert summary.status == "completed"
        assert summary.post_miss_distance_km > summary.pre_miss_distance_km
        assert summary.post_pc < summary.pre_pc
