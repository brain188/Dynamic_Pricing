import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# from backend.app.database import Base, engine  # Removed: managed by Alembic
from backend.app.routes import health, metrics, predict
from backend.app.services.inference import InferenceService
from backend.app.utils.backend_utils import generate_request_id


# Structured JSON Logging Setup
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = record.latency_ms
        if hasattr(record, "status_code"):
            log_entry["status_code"] = record.status_code

        return json.dumps(log_entry)


# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("project_logger")
logger.setLevel(logging.INFO)

# File Handler for JSON logs
file_handler = logging.FileHandler("logs/project.log")
file_handler.setFormatter(JsonFormatter())
logger.addHandler(file_handler)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)

# Global rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    logger.info("Starting up FastAPI application...")

    # Tables are now managed by Alembic
    # 1. Instantiate Inference Service as a global singleton
    logger.info("Loading inference models...")
    app.state.inference_service = InferenceService()

    yield
    # --- Shutdown Logic ---
    logger.info("Shutting down application...")


app = FastAPI(
    title="Dynamic Pricing API",
    description="API for real-time dynamic pricing using ML/RL.",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiting setup
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for Request ID, Tracking, and JSON logging
@app.middleware("http")
async def add_request_id_and_log(request, call_next):
    request_id = generate_request_id()
    request.state.request_id = request_id

    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000

    # Inject X-Request-ID header
    response.headers["X-Request-ID"] = request_id

    # Log details
    logger.info(
        f"{request.method} {request.url.path}",
        extra={
            "request_id": request_id,
            "latency_ms": round(process_time, 2),
            "status_code": response.status_code,
        },
    )

    return response


# Root Endpoint
@app.get("/")
async def root():
    return {"message": "Dynamic Pricing API is running. Visit /docs for documentation."}


# Register Routers
app.include_router(predict.router, prefix="/api/v1", tags=["Inference"])
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(metrics.router, prefix="/api/v1", tags=["Metrics"])
