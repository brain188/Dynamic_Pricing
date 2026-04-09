from datetime import datetime

from fastapi import APIRouter

from backend.app.models.schemas import HealthResponse

router = APIRouter()

# Store startup time for uptime calculation
START_TIME = datetime.now()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns the health status and metadata of the API and model.
    """
    uptime = (datetime.now() - START_TIME).total_seconds()

    return HealthResponse(
        status="healthy",
        model_version="xgboost_1.0.0",  # To be fetched from service
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.now(),
    )
