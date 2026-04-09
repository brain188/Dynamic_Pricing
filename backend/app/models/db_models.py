from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from backend.app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, unique=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Store full RideFeatures as JSONB for flexibilty and searchability on PostgreSQL
    input_features = Column(JSONB, nullable=False)

    # Prediction Results
    predicted_fare = Column(Float, nullable=False)
    price_multiplier = Column(Float)
    demand_supply_ratio = Column(Float)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)

    # Operational Metrics
    latency_ms = Column(Float)
    model_version = Column(String, index=True)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, index=True, nullable=False)
    model_type = Column(String, nullable=False)  # e.g., 'XGBoost', 'PPO'
    rmse = Column(Float)
    mae = Column(Float)
    r2 = Column(Float)
    deployed_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=False)

    # Additional metadata
    n_features = Column(Integer)
    path = Column(String)
