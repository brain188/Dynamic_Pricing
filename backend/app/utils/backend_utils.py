import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from backend.app.models.db_models import Prediction


def generate_request_id() -> str:
    """Returns a unique request ID."""
    return str(uuid.uuid4())


def log_prediction(db: Session, request_id: str, features, response, latency_ms: float):
    """
    Inserts a prediction log into the PostgreSQL database.
    """
    # Convert Pydantic features to dict for JSONB storage
    # Use mode='json' to ensure Enums and Datetimes are serialized to strings
    from enum import Enum

    try:
        if hasattr(features, "model_dump"):
            input_data = features.model_dump(mode="json")
        else:
            import json

            input_data = json.loads(features.json())
    except Exception:
        # Fallback to manual string conversion
        input_data = features.dict()
        for k, v in input_data.items():
            if isinstance(v, Enum):
                input_data[k] = v.value
            elif isinstance(v, datetime):
                input_data[k] = v.isoformat()

    db_log = Prediction(
        request_id=request_id,
        input_features=input_data,
        predicted_fare=response.predicted_fare,
        price_multiplier=response.price_multiplier,
        demand_supply_ratio=response.demand_supply_ratio,
        confidence_lower=response.confidence_lower,
        confidence_upper=response.confidence_upper,
        latency_ms=round(latency_ms, 2),
        model_version=response.model_version,
    )

    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log
