import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Attempt to import TargetEncoder (available in sklearn >= 1.3)
try:
    from sklearn.preprocessing import TargetEncoder
except ImportError:
    TargetEncoder = None


def build_feature_pipeline(
    df_train, df_val=None, df_test=None, target_col="Historical_Cost_of_Ride"
):
    """
    Composes all feature engineering steps in order.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # 1. Process Training Set
    df_train_proc = df_train.copy()
    df_train_proc = create_time_features(df_train_proc)
    df_train_proc = create_demand_supply_features(df_train_proc, is_training=True)
    df_train_proc = create_customer_features(df_train_proc)
    df_train_proc = create_categorical_features(df_train_proc, is_training=True)
    df_train_proc = create_interaction_features(df_train_proc)
    df_train_proc = scale_features(df_train_proc, is_training=True)

    y_train = df_train_proc[target_col]
    X_train = df_train_proc.drop(columns=[target_col])

    # 2. Process Validation Set
    X_val, y_val = None, None
    if df_val is not None:
        df_val_proc = df_val.copy()
        df_val_proc = create_time_features(df_val_proc)
        df_val_proc = create_demand_supply_features(df_val_proc, is_training=False)
        df_val_proc = create_customer_features(df_val_proc)
        df_val_proc = create_categorical_features(df_val_proc, is_training=False)
        df_val_proc = create_interaction_features(df_val_proc)
        df_val_proc = scale_features(df_val_proc, is_training=False)

        y_val = df_val_proc[target_col]
        X_val = df_val_proc.drop(columns=[target_col])

    # 3. Process Test Set
    X_test, y_test = None, None
    if df_test is not None:
        df_test_proc = df_test.copy()
        df_test_proc = create_time_features(df_test_proc)
        df_test_proc = create_demand_supply_features(df_test_proc, is_training=False)
        df_test_proc = create_customer_features(df_test_proc)
        df_test_proc = create_categorical_features(df_test_proc, is_training=False)
        df_test_proc = create_interaction_features(df_test_proc)
        df_test_proc = scale_features(df_test_proc, is_training=False)

        y_test = df_test_proc[target_col]
        X_test = df_test_proc.drop(columns=[target_col])

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from 'Time_of_Booking'.
    """
    time_mapping = {"Morning": 9, "Afternoon": 14, "Evening": 19, "Night": 2}
    df["hour"] = df["Time_of_Booking"].map(time_mapping).fillna(12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_night"] = (df["Time_of_Booking"] == "Night").astype(int)
    df["is_rush_hour"] = df["Time_of_Booking"].isin(["Morning", "Evening"]).astype(int)
    return df


def create_demand_supply_features(
    df: pd.DataFrame, is_training: bool = True
) -> pd.DataFrame:
    """
    Compute demand-supply related features with leakage prevention.
    """
    df["demand_supply_ratio"] = df["Number_of_Riders"] / (df["Number_of_Drivers"] + 1)
    df["driver_deficit"] = df["Number_of_Riders"] - df["Number_of_Drivers"]
    df["demand_surplus_flag"] = (df["demand_supply_ratio"] > 1.5).astype(int)

    lookup_path = os.path.join("data", "processed", "location_hour_lookup.csv")

    if is_training:
        loc_hour_stats = (
            df.groupby(["Location_Category", "hour"])
            .agg(
                {
                    "Number_of_Riders": "mean",
                    "Number_of_Drivers": "mean",
                    "demand_supply_ratio": "mean",
                }
            )
            .rename(
                columns={
                    "Number_of_Riders": "hist_avg_riders_loc_hour",
                    "Number_of_Drivers": "hist_avg_drivers_loc_hour",
                    "demand_supply_ratio": "hist_demand_supply_ratio_loc_hour",
                }
            )
            .reset_index()
        )
        os.makedirs(os.path.dirname(lookup_path), exist_ok=True)
        loc_hour_stats.to_csv(lookup_path, index=False)
    else:
        loc_hour_stats = (
            pd.read_csv(lookup_path) if os.path.exists(lookup_path) else pd.DataFrame()
        )

    return df.merge(loc_hour_stats, on=["Location_Category", "hour"], how="left")


def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute customer-related features.
    """
    loyalty_map = {"Regular": 0, "Silver": 1, "Gold": 2}
    df["loyalty_numeric"] = df["Customer_Loyalty_Status"].map(loyalty_map)
    df["ride_tenure_bucket"] = pd.cut(
        df["Number_of_Past_Rides"],
        bins=[-1, 5, 20, 50, 100, np.inf],
        labels=["0-5", "6-20", "21-50", "51-100", "100+"],
    )
    df["rating_bucket"] = pd.cut(
        df["Average_Ratings"], bins=[0, 2, 3, 4, 5], labels=["1-2", "2-3", "3-4", "4-5"]
    )
    df["is_new_user"] = (df["Number_of_Past_Rides"] < 3).astype(int)
    df["is_high_value"] = (
        (df["loyalty_numeric"] == 2) & (df["Average_Ratings"] > 4.5)
    ).astype(int)
    return df


def create_categorical_features(
    df: pd.DataFrame, is_training: bool = True
) -> pd.DataFrame:
    """
    Apply categorical encoding (One-Hot and Target Encoding).
    """
    ohe_path = os.path.join("models", "encoders", "ohe_encoder.pkl")
    te_path = os.path.join("models", "encoders", "target_encoder.pkl")
    os.makedirs(os.path.dirname(ohe_path), exist_ok=True)

    if is_training:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe.fit(df[["Vehicle_Type"]])
        joblib.dump(ohe, ohe_path)
    else:
        ohe = joblib.load(ohe_path)

    ohe_cols = [f"vehicle_{cat}" for cat in ohe.categories_[0]]
    df[ohe_cols] = ohe.transform(df[["Vehicle_Type"]])

    if is_training:
        if TargetEncoder:
            te = TargetEncoder(smooth="auto")
            te.fit(df[["Location_Category"]], df["Historical_Cost_of_Ride"])
            joblib.dump(te, te_path)
        else:
            means = df.groupby("Location_Category")["Historical_Cost_of_Ride"].mean()
            global_mean = df["Historical_Cost_of_Ride"].mean()
            joblib.dump((means, global_mean), te_path)

    if os.path.exists(te_path):
        te_obj = joblib.load(te_path)
        if hasattr(te_obj, "transform"):
            df["location_score"] = te_obj.transform(df[["Location_Category"]])
        else:
            means, global_mean = te_obj
            df["location_score"] = (
                df["Location_Category"].map(means).fillna(global_mean)
            )
    else:
        df["location_score"] = 0
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction terms between features.
    """
    if "vehicle_Premium" in df.columns:
        df["duration_x_vehicle_premium"] = (
            df["Expected_Ride_Duration"] * df["vehicle_Premium"]
        )
    df["demand_x_loyalty"] = df["demand_supply_ratio"] * df["loyalty_numeric"]
    df["rating_x_tenure"] = df["Average_Ratings"] * df["Number_of_Past_Rides"]
    df["location_x_rush_hour"] = df["location_score"] * df["is_rush_hour"]
    return df


def scale_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Apply RobustScaler to numeric features.
    """
    scaler_path = os.path.join("models", "scaler.pkl")
    numeric_cols = [
        "Number_of_Riders",
        "Number_of_Drivers",
        "Number_of_Past_Rides",
        "Average_Ratings",
        "Expected_Ride_Duration",
        "Historical_Cost_of_Ride",
    ]
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    if is_training:
        scaler = RobustScaler()
        scaler.fit(df[numeric_cols])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)

    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


if __name__ == "__main__":
    from src.utils.data_loader import load_raw

    df = load_raw()

    # Simulate a split (70/15/15)
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size : train_size + val_size]
    test_df = df.iloc[train_size + val_size :]

    X_train, X_val, X_test, y_train, y_val, y_test = build_feature_pipeline(
        train_df, val_df, test_df
    )

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape if X_val is not None else 'None'}")
    print(f"X_test shape: {X_test.shape if X_test is not None else 'None'}")

    # Save processed data (example just for training)
    processed_path = os.path.join("data", "processed", "dynamic_pricing_processed.csv")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    X_train.join(y_train).to_csv(processed_path, index=False)
    print(f"Final feature matrix (train) saved to {processed_path}")
