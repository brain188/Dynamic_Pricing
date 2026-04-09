from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.db_models import Prediction
from backend.app.models.schemas import MetricsResponse

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(db: Session = Depends(get_db)):
    """
    Aggregates prediction statistics from the database.
    """
    # 1. Query Aggregations
    stats = db.query(
        func.count(Prediction.id).label("total"),
        func.avg(Prediction.latency_ms).label("avg_latency"),
        func.avg(Prediction.predicted_fare).label("avg_fare"),
    ).first()

    # 2. Return Response
    return MetricsResponse(
        total_predictions=stats.total or 0,
        avg_latency_ms=round(stats.avg_latency or 0.0, 2),
        avg_fare=round(stats.avg_fare or 0.0, 2),
        last_retrain_date=datetime.now(),  # Mock for now
    )
