from typing import Optional
from pydantic import BaseModel, Field


class TrainObservation(BaseModel):
    train_no: str
    journey_date: str
    station_code: str
    station_name: Optional[str] = None

    station_sequence: int

    delay: float = Field(
        description="Current observed delay in minutes"
    )

    previous_delay: float = 0.0
    delay_change: float = 0.0

    observed_at: Optional[str] = None


class ForecastStation(BaseModel):
    station_code: str
    station_name: Optional[str] = None
    station_sequence: int

    predicted_delay: float
    lower_delay: float
    upper_delay: float

    confidence: float


class ETAResponse(BaseModel):
    train_no: str
    journey_date: str

    current_station: str
    current_delay: float

    forecast_horizon: int

    stations: list[ForecastStation]

    model: str
    provider: str

    explanation: str
