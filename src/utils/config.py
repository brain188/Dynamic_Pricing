import logging  # noqa: F401
import os
from logging.handlers import RotatingFileHandler  # noqa: F401

# Directory Setup

# Define the base directory (two levels up from src/utils/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project Directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")

# List of all directories to be created
ALL_DIRS = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    VISUALIZATIONS_DIR,
    os.path.join(REPORTS_DIR, "results"),
    os.path.join(VISUALIZATIONS_DIR, "exploration"),  # Adjusted for EDA
    os.path.join(VISUALIZATIONS_DIR, "model_performance"),
    os.path.join(VISUALIZATIONS_DIR, "comparison"),
]

# Ensure directories exist
for directory in ALL_DIRS:
    os.makedirs(directory, exist_ok=True)

# Data Paths & Names

RAW_DATA_FILE = "dynamic_pricing.csv"
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, RAW_DATA_FILE)

PROCESSED_DATA_FILE = "dynamic_pricing_processed.csv"
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, PROCESSED_DATA_FILE)

# Data Column Names
TARGET_COLUMN = "Historical_Cost_of_Ride"
TIME_COLUMN = "Time_of_Booking"
FEATURE_COLUMNS = [
    "Number_of_Riders",
    "Number_of_Drivers",
    "Location_Category",
    "Customer_Loyalty_Status",
    "Number_of_Past_Rides",
    "Average_Ratings",
    "Vehicle_Type",
    "Expected_Ride_Duration",
]

# Modeling Parameters

RANDOM_STATE = 42
TEST_SIZE = 0.25
PROPENSITY_MODEL_NAME = "acceptance_propensity_model.pkl"

# Supervised Regression Models
MODEL_PARAMS = {
    "LinearRegression": {},
    "RandomForestRegressor": {
        "n_estimators": 150,
        "max_depth": 10,
        "random_state": RANDOM_STATE,
    },
    "XGBRegressor": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": RANDOM_STATE,
    },
}

# Logging Configuration

LOG_FILE = os.path.join(LOGS_DIR, "project_log.log")
