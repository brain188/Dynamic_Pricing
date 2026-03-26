import os

import pandas as pd
import yaml

from src.utils.logger import logger


def load_config(config_path=None):
    """Load configuration from YAML file"""
    if config_path is None:
        # Resolve path relative to this file's location (src/utils/data_loader.py)
        # Root is two levels up from this file
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        config_path = os.path.join(base_dir, "src", "config", "config.yaml")

    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_project_root():
    """Helper to get the project root directory"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_raw():
    """Load raw data from the path specified in config"""
    config = load_config()
    raw_path_rel = config["paths"]["raw_data"]

    # Resolve relative to project root
    raw_path = os.path.join(get_project_root(), raw_path_rel)

    if not os.path.exists(raw_path):
        logger.error(f"Raw data file not found: {raw_path}")
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    return df


def save_processed(df, filename=None):
    """Save processed data to the path specified in config"""
    config = load_config()
    if filename is None:
        processed_path_rel = config["paths"]["processed_data"]
    else:
        processed_path_rel = os.path.join(
            os.path.dirname(config["paths"]["processed_data"]), filename
        )

    # Resolve relative to project root
    processed_path = os.path.join(get_project_root(), processed_path_rel)

    logger.info(f"Saving processed data to {processed_path}")
    df.to_csv(processed_path, index=False)
    return processed_path
