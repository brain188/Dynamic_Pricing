from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LocationEnum(str, Enum):
    URBAN = "Urban"
    SUBURBAN = "Suburban"
    RURAL = "Rural"


class VehicleTypeEnum(str, Enum):
    ECONOMY = "Economy"
    PREMIUM = "Premium"


class LoyaltyLevelEnum(str, Enum):
    REGULAR = "Regular"
    SILVER = "Silver"
    GOLD = "Gold"


class RideFeatures(BaseModel):
    number_of_riders: int = Field(..., ge=1, le=10, description="Number of passengers")
    number_of_drivers: int = Field(..., ge=0, description="Number of available drivers")
    location_category: LocationEnum = Field(..., example="Urban")
    vehicle_type: VehicleTypeEnum = Field(..., example="Premium")
    customer_loyalty_status: LoyaltyLevelEnum = Field(..., example="Silver")
    number_of_past_rides: int = Field(..., ge=0, description="Customer history")
    average_rating: float = Field(..., ge=1.0, le=5.0, description="Customer rating")
    time_of_booking: datetime = Field(default_factory=datetime.now)
    expected_ride_duration: float = Field(
        ..., gt=0, description="Expected duration in minutes"
    )
    base_fare: float = Field(
        default=50.0, gt=0, description="Base fare for multiplier calculation"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "number_of_riders": 2,
                "number_of_drivers": 5,
                "location_category": "Urban",
                "vehicle_type": "Premium",
                "customer_loyalty_status": "Silver",
                "number_of_past_rides": 50,
                "average_rating": 4.8,
                "time_of_booking": "2024-04-09T15:00:00",
                "expected_ride_duration": 15.5,
            }
        }
    )


class PriceResponse(BaseModel):
    predicted_fare: float = Field(..., description="Final calculated fare")
    price_multiplier: float = Field(..., description="The dynamic multiplier applied")
    demand_supply_ratio: float = Field(..., description="Computed market tension")
    model_version: str = Field(..., description="Model ID used for inference")
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    request_id: str = Field(..., description="Unique ID for tracking")


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_version: str
    uptime_seconds: float
    timestamp: datetime


class MetricsResponse(BaseModel):
    total_predictions: int
    avg_latency_ms: float
    avg_fare: float
    last_retrain_date: datetime
