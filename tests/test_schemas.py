import pytest
from datetime import datetime
from pydantic import ValidationError
from mobility.schemas.traffic import TrafficSensorData

def test_valid_traffic_data():
    """Test that valid data passes validation."""
    data = {
        "sensor_id": "12345",
        "speed": 65.5,
        "timestamp": datetime.now()
    }
    sensor = TrafficSensorData(**data)
    assert sensor.speed == 65.5
    assert sensor.sensor_id == "12345"

def test_negative_speed_fails():
    """Test that negative speed raises a ValidationError."""
    data = {
        "sensor_id": "12345",
        "speed": -10.0,  # Invalid!
        "timestamp": datetime.now()
    }
    # We expect this block to raise an error. If it doesn't, the test fails.
    with pytest.raises(ValidationError):
        TrafficSensorData(**data)

def test_excessive_speed_fails():
    """Test that speed > 200 raises a ValidationError."""
    data = {
        "sensor_id": "12345",
        "speed": 250.0,  # Invalid!
        "timestamp": datetime.now()
    }
    with pytest.raises(ValidationError):
        TrafficSensorData(**data)