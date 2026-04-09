import itertools
import os
import sys
from datetime import datetime, timedelta

import joblib
import mlflow
import mlflow.prophet
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from prophet import Prophet  # noqa: E402
from prophet.diagnostics import cross_validation, performance_metrics  # noqa: E402

from src.utils.logger import logger  # noqa: E402


def prepare_prophet_df(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """
    Filters per location, aggregates to daily (ds, y) format, and handles gaps.
    For this project, we synthesize a 120-day timeline.
    """
    logger.info(f"Preparing Prophet data for location: {location}")

    # Filter for location
    loc_df = df[df["Location_Category"] == location].copy()

    # Map time categories to hours
    time_map = {
        "Morning": "09:00:00",
        "Afternoon": "15:00:00",
        "Evening": "18:00:00",
        "Night": "22:00:00",
    }

    # Synthesis: 120 days
    start_date = datetime(2023, 1, 1)
    new_rows = []

    for slot_name, time_str in time_map.items():
        slot_df = loc_df[loc_df["Time_of_Booking"] == slot_name].copy()
        ride_indices = slot_df.index.tolist()
        np.random.seed(42)  # Deterministic synthesis
        np.random.shuffle(ride_indices)

        for d in range(120):
            current_ds = (
                (start_date + timedelta(days=d)).strftime("%Y-%m-%d") + " " + time_str
            )
            # Partition rides across 120 days
            start_idx = (len(slot_df) * d) // 120
            end_idx = (len(slot_df) * (d + 1)) // 120
            count = end_idx - start_idx
            if count >= 0:
                new_rows.append({"ds": current_ds, "y": count})

    prophet_df = pd.DataFrame(new_rows)
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    # Handle gaps and reindex
    # Ensure all 120 days x 4 slots exist
    all_dates = []
    for d in range(120):
        for t in time_map.values():
            all_dates.append(
                (start_date + timedelta(days=d)).replace(
                    hour=int(t.split(":")[0]), minute=0, second=0
                )
            )

    full_index = pd.DataFrame({"ds": all_dates})
    prophet_df = pd.merge(full_index, prophet_df, on="ds", how="left").fillna(0)

    return prophet_df.sort_values("ds")


def fit_prophet(df_prophet: pd.DataFrame, location: str) -> Prophet:
    """
    Fits Prophet with multiplicative seasonality, holidays, and tuning.
    """
    logger.info(f"Fitting and Tuning Prophet for {location}")

    # Holidays
    holidays = pd.DataFrame(
        {
            "holiday": "dummy_peak",
            "ds": pd.to_datetime(["2023-01-01", "2023-02-14", "2023-04-01"]),
            "lower_window": 0,
            "upper_window": 1,
        }
    )

    # Hyperparameter Grid (Scaled down slightly for speed/data size)
    param_grid = {
        "changepoint_prior_scale": [0.01, 0.1, 0.5],
        "seasonality_prior_scale": [0.1, 1.0, 10.0],
    }

    # Generate all combinations
    all_params = [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]
    rmses = []

    # Manual Tuning with CV
    for params in all_params:
        m = Prophet(
            seasonality_mode="multiplicative",
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True,
            holidays=holidays,
            **params,
        )
        m.fit(df_prophet)

        # CV: initial=60 days, period=15 days, horizon=14 days
        # (reduced for 120-day data)
        df_cv = cross_validation(
            m,
            initial="60 days",
            period="15 days",
            horizon="14 days",
            parallel="threads",
        )
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p["rmse"].values[0])

    # Best params
    best_params = all_params[np.argmin(rmses)]
    logger.info(f"Best params for {location}: {best_params}")

    # Final fit with best params
    best_model = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        holidays=holidays,
        **best_params,
    )

    with mlflow.start_run(run_name=f"Prophet_Optimized_{location}"):
        best_model.fit(df_prophet)

        # Log params and metrics
        mlflow.log_params(best_params)
        df_cv = cross_validation(
            best_model,
            initial="60 days",
            period="15 days",
            horizon="14 days",
            parallel="threads",
        )
        df_p = performance_metrics(df_cv, rolling_window=1)

        available_metrics = ["rmse", "mae", "mape", "mdae", "smape", "coverage"]
        for metric in available_metrics:
            if metric in df_p.columns:
                mlflow.log_metric(metric, df_p[metric].values[0])

        mlflow.prophet.log_model(best_model, artifact_path=f"prophet_model_{location}")
        logger.info(
            f"Location {location} - Optimized RMSE: {df_p['rmse'].values[0]:.4f}"
        )

    return best_model


def generate_forecast(model: Prophet, periods=30) -> pd.DataFrame:
    """Returns forecast with ds, yhat, yhat_lower, yhat_upper."""
    future = model.make_future_dataframe(periods=periods, freq="6H")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def get_demand_for_datetime(model: Prophet, dt: datetime) -> float:
    """Look up or interpolate yhat for the given datetime."""
    future = pd.DataFrame({"ds": [dt]})
    forecast = model.predict(future)
    return max(0, forecast.iloc[0]["yhat"])


def main():
    processed_path = "data/processed/dynamic_pricing_processed.csv"
    df = pd.read_csv(processed_path)

    locations = df["Location_Category"].unique()
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    mlflow.set_experiment("Demand_Forecasting_Prophet_v2")

    all_demand_data = []  # To save for notebook visualization

    for loc in locations:
        prophet_df = prepare_prophet_df(df, loc)
        all_demand_data.append(prophet_df.assign(location=loc))

        model = fit_prophet(prophet_df, loc)

        # Save model
        model_path = os.path.join(models_dir, f"prophet_{loc}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Saved optimized model for {loc} to {model_path}")

    # Combine and save for notebook
    pd.concat(all_demand_data).to_csv("data/processed/demand_counts.csv", index=False)


if __name__ == "__main__":
    main()
