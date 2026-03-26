import joblib
import os
import logging
import numpy as np
from typing import Any, Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.config import MODELS_DIR, REPORTS_DIR, LOG_FILE

logger = logging.getLogger(__name__)


def setup_logging():
    """
    Sets up basic logging for the project.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete.")

def save_artifact(obj: Any, filename: str, subdir: str = MODELS_DIR):
    """
    S Saves a Python object (like a model or scaler) using joblib.
    """
    try:
        filepath = os.path.join(subdir, filename)
        os.makedirs(subdir, exist_ok=True)
        joblib.dump(obj, filepath)
        logger.info(f"Artifact successfully saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save artifact {filename}: {e}")
        return False

def load_artifact(filename: str, subdir: str = MODELS_DIR) -> Any:
    """
    Loads a Python object (like a model or scaler) using joblib.
    """
    filepath = os.path.join(subdir, filename)
    try:
        artifact = joblib.load(filepath)
        logger.info(f"Artifact successfully loaded from {filepath}")
        return artifact
    except FileNotFoundError:
        logger.error(f"Artifact not found at {filepath}")
        return None
    except Exception as e:
        logger.error(f"Failed to load artifact {filename}: {e}")
        return None

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates key regression metrics: RMSE, MAE, R2, and MAPE.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape  # Mean Absolute Percentage Error
    }
    
    return metrics

def log_metrics(model_name: str, metrics: Dict[str, float]):
    """Logs model performance metrics."""
    logger.info(f"--- Model Performance: {model_name} ---")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info("-" * 40)