from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class TrafficSensorData(BaseModel):
    sensor_id: str
    timestamp: datetime
    # We add 'lt=200' (less than 200) directly in the Field for automatic validation
    speed: float = Field(..., ge=0, lt=200, description="Speed in mph")

    # Alternatively, you can keep your custom validator and expand it:
    @field_validator('speed')
    def check_speed_limits(cls, v):
        if v < 0:
            raise ValueError("Speed cannot be negative")
        if v > 200:
            raise ValueError("Speed is unrealistically high (> 200 mph)")
        return v    