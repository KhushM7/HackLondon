from app.services.screening_engine import ScreeningEngine


def test_risk_tier_thresholds():
    assert ScreeningEngine._risk_tier(0.2) == "High"
    assert ScreeningEngine._risk_tier(2.0) == "Medium"
    assert ScreeningEngine._risk_tier(8.0) == "Low"
