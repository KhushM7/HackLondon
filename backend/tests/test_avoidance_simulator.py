from app.services.avoidance_simulator import AvoidanceSimulator


def test_distance_helper():
    assert round(AvoidanceSimulator._distance([0, 0, 0], [3, 4, 0]), 6) == 5.0
