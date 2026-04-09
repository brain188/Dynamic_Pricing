import os

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.services.inference import InferenceService

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DB_URL")
# Use a specific schema for testing to avoid collisions with production data
TEST_SCHEMA = "test_pricing"


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """
    Creates a dedicated test schema on Supabase and initializes tables.
    """
    if not DATABASE_URL:
        pytest.skip("DB_URL not set in .env")

    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        # Create schema if it doesn't exist
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {TEST_SCHEMA}"))
        conn.commit()

    # Configure Base to use the test schema
    # Note: In a real production setup, we might use different approaches,
    # but here we force the metadata to use the test schema.
    for table in Base.metadata.tables.values():
        table.schema = TEST_SCHEMA

    # Create tables in the test schema
    Base.metadata.create_all(bind=engine)

    yield

    # Cleanup: Drop schema after session (optional, but good for clean slate)
    # with engine.connect() as conn:
    #     conn.execute(text(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE"))
    #     conn.commit()


@pytest.fixture
def db_session():
    """Returns a fresh database session for each test."""
    engine = create_engine(DATABASE_URL)
    # Ensure tables use test schema
    for table in Base.metadata.tables.values():
        table.schema = TEST_SCHEMA

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    """
    Returns a FastAPI TestClient with an overridden database dependency.
    """

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def inference_service():
    """Returns the real InferenceService instance for unit testing logic."""
    return InferenceService()


@pytest.fixture
def sample_ride_features():
    """Returns a valid RideFeatures object for testing."""
    from datetime import datetime

    from backend.app.models.schemas import (
        LocationEnum,
        LoyaltyLevelEnum,
        RideFeatures,
        VehicleTypeEnum,
    )

    return RideFeatures(
        number_of_riders=2,
        number_of_drivers=5,
        location_category=LocationEnum.URBAN,
        vehicle_type=VehicleTypeEnum.PREMIUM,
        customer_loyalty_status=LoyaltyLevelEnum.GOLD,
        number_of_past_rides=50,
        average_rating=4.8,
        time_of_booking=datetime.now(),
        expected_ride_duration=15.5,
    )
