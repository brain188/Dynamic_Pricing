import numpy as np  # noqa: F401
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.utils.logger import logger


def get_time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """
    Configure and return a TimeSeriesSplit object to preserve temporal order.

    Args:
        n_splits: Number of cross-validation splits.

    Returns:
        TimeSeriesSplit: Configured sklearn object.
    """
    logger.info(f"Configuring TimeSeriesSplit with {n_splits} splits.")
    return TimeSeriesSplit(n_splits=n_splits)


def temporal_split(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple:
    """
    Split the data into training, validation, and test sets chronologically.

    Args:
        df: The DataFrame to split.
        train_frac: Fraction of data to use for training.
        val_frac: Fraction of data to use for validation.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    logger.info("Performing temporal split (no shuffle)...")

    # 1. Sort by Time_of_Booking ascending as per requirements
    # Note: This will be an alphabetical sort (Afternoon, Evening, Morning, Night)
    # unless categorical ordering was previously defined.
    df_sorted = df.sort_values(by="Time_of_Booking").reset_index(drop=True)

    # 2. Calculate slice points
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    # 3. Slice the data
    df_train = df_sorted.iloc[:train_end]
    df_val = df_sorted.iloc[train_end:val_end]
    df_test = df_sorted.iloc[val_end:]

    # 4. Print Documentation (using indices/Time_of_Booking since absolute
    # Date is unavailable)  # noqa: E501
    print("--- Temporal Split Summary ---")
    print(f"Total samples: {n}")
    print(f"Training:   {len(df_train)} samples ({train_frac*100:.0f}%)")
    print(f"Validation: {len(df_val)} samples ({val_frac*100:.0f}%)")
    print(
        f"Test:       {len(df_test)} samples ({(1 - train_frac - val_frac)*100:.0f}%)"
    )

    # Date range simulation/reporting
    if "Date" in df_sorted.columns or "Booking_Datetime" in df_sorted.columns:
        date_col = "Date" if "Date" in df_sorted.columns else "Booking_Datetime"
        print(
            f"Train Date Range: {df_train[date_col].min()} to "
            f"{df_train[date_col].max()}"  # noqa: E501
        )
        print(f"Val Date Range:   {df_val[date_col].min()} to {df_val[date_col].max()}")
        print(
            f"Test Date Range:  {df_test[date_col].min()} to {df_test[date_col].max()}"
        )

        # Assert no overlap
        assert (
            df_train[date_col].max() <= df_val[date_col].min()
        ), "Overlap between Train and Val"
        assert (
            df_val[date_col].max() <= df_test[date_col].min()
        ), "Overlap between Val and Test"
    else:
        print(
            "Note: Absolute datetime column not found. Splitting based on "
            "sorting of 'Time_of_Booking' and row order."  # noqa: E501
        )
        print(f"Train Indices: 0 to {train_end - 1}")
        print(f"Val Indices:   {train_end} to {val_end - 1}")
        print(f"Test Indices:  {val_end} to {n - 1}")

    logger.info("Temporal split completed successfully.")
    return df_train, df_val, df_test


if __name__ == "__main__":
    # Quick test
    import os

    from src.utils.data_loader import get_project_root

    processed_path = os.path.join(
        get_project_root(), "data", "processed", "dynamic_pricing_processed.csv"
    )
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        train, val, test = temporal_split(df)
        print("Check completed.")
    else:
        print(f"Processed data not found at {processed_path}")
