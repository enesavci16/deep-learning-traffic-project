from pydantic import BaseModel, Field, field_validator
from typing import Dict

class TrafficSnapshot(BaseModel):
    """
    Represents the state of the entire traffic network at a single point in time.
    """
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp")
    
    # Key = Sensor ID (e.g., "767541"), Value = Speed (mph)
    sensor_readings: Dict[str, float]

    @field_validator('sensor_readings')
    def validate_readings(cls, v):
        if not v:
            raise ValueError("Sensor readings cannot be empty")
        
        for sensor_id, speed in v.items():
            if speed < 0:
                raise ValueError(f"Sensor {sensor_id} reported negative speed: {speed}")
        return v