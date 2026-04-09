import time

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.schemas import PriceResponse, RideFeatures
from backend.app.utils.backend_utils import log_prediction

router = APIRouter()


@router.post("/predict", response_model=PriceResponse)
async def predict_price(
    features: RideFeatures, request: Request, db: Session = Depends(get_db)
):
    """
    Predicts the dynamic price and logs the result to the database.
    """
    start_time = time.perf_counter()

    # 1. Inference
    inference_service = request.app.state.inference_service
    prediction = inference_service.predict(features)

    latency_ms = (time.perf_counter() - start_time) * 1000

    # 2. Centralized Database Logging (using new JSONB schema)
    log_prediction(
        db=db,
        request_id=prediction.request_id,
        features=features,
        response=prediction,
        latency_ms=latency_ms,
    )

    return prediction
