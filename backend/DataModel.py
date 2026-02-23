from pydantic import BaseModel, Field
from typing import List

# Data model for water level data
class WaterLevelDataTest(BaseModel):
    water_level: List[float] = Field(title="Water Level", description="Data for Water Level over last period of time")
    forecast_steps: int = Field(title="Forecast Steps", description="Number of steps to predict")