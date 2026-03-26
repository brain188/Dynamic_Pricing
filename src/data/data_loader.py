import pandas as pd
from pathlib import Path
from typing import Optional
import logging
import sys
import os

from src.utils.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

# Add the project root to the path for importing local modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

logger = logging.getLogger(__name__)

def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the raw ride-sharing data from the specified path.
    If no path is provided, it uses the RAW_DATA_PATH from the config.
    """
    path = file_path if file_path else RAW_DATA_PATH
    
    if not path:
        # This check is mostly redundant since RAW_DATA_PATH is defined, but good practice
        logger.error("Data path not specified in function call or config.")
        raise ValueError("Data path not specified in function call or config.")
    
    path = Path(path)
    if not path.exists():
        logger.error(f"File not found at: {path}. Ensure 'rides_raw.csv' is in 'data/raw'.")
        raise FileNotFoundError(f"File not found at: {path}.")

    logger.info(f"Loading data from: {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file at {path}: {e}")
        raise

def save_data(df: pd.DataFrame, file_path: Optional[str] = None):
    """
    Saves the DataFrame (typically processed data) to the specified path.
    If no path is provided, it uses the PROCESSED_DATA_PATH from the config.
    """
    path = file_path if file_path else PROCESSED_DATA_PATH
    
    if not path:
        logger.error("Save path not specified in function call or config.")
        raise ValueError("Save path not specified in function call or config.")
    
    path = Path(path)
    # Ensure directory exists before saving
    try:
        path.parent.mkdir(parents=True, exist_ok=True) 
        logger.info(f"Saving data to: {path}")
        df.to_csv(path, index=False)
        logger.info("Data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving data to {path}: {e}")
        raise