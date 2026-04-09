import os
import uuid

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from backend.app.models.schemas import PriceResponse, RideFeatures
from src.utils.features import (
    create_categorical_features,
    create_customer_features,
    create_demand_supply_features,
    create_interaction_features,
    create_time_features,
    scale_features,
)


class InferenceService:
    def __init__(self):
        """
        Initializes the inference service with models and utilities.
        """
        # Path resolution: backend/app/services/inference.py -> root/
        self.service_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(
            os.path.join(self.service_dir, "..", "..", "..")
        )
        self.models_dir = os.path.join(self.project_root, "models")
        self.registry_path = os.path.join(self.models_dir, "model_registry.json")

        # Load Registry
        if not os.path.exists(self.registry_path):
            raise FileNotFoundError(f"Model registry not found at {self.registry_path}")

        import json

        with open(self.registry_path, "r") as f:
            self.registry = json.load(f)

        self.best_model_info = self.registry["best_model"]
        self.model_version = self.best_model_info["version"]
        self.rmse = self.best_model_info.get("rmse", 0.25)

        # Load Model
        model_path = self.best_model_info["path"]
        # Registry stores absolute windows paths. Let's make it more robust.
        if not os.path.exists(model_path):
            # Try relative to models_dir if absolute fails
            filename = os.path.basename(model_path)
            model_path = os.path.join(self.models_dir, filename)

        self.model = joblib.load(model_path)

        # 3. Load RL PPO Model
        self.ppo_path = os.path.join(self.models_dir, "ppo_dynamic_pricing.zip")
        if os.path.exists(self.ppo_path):
            self.ppo_model = PPO.load(self.ppo_path)
        else:
            self.ppo_model = None

        # 4. Load Prophet Models (for RL demand forecast input)
        self.prophet_models = {}
        for loc in ["Urban", "Suburban", "Rural"]:
            p_path = os.path.join(self.models_dir, f"prophet_{loc}.pkl")
            if os.path.exists(p_path):
                self.prophet_models[loc] = joblib.load(p_path)

        # 5. Load other artifacts
        self.scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")

    def preprocess(self, features: RideFeatures) -> pd.DataFrame:
        """
        Converts Pydantic model to a processed DataFrame ready for prediction.
        Mirrors the training-time feature pipeline.
        """
        # 1. Convert to DataFrame
        # Map Pydantic field names to original CSV names expected by features.py
        data = {
            "Number_of_Riders": [features.number_of_riders],
            "Number_of_Drivers": [features.number_of_drivers],
            "Location_Category": [features.location_category.value],
            "Vehicle_Type": [features.vehicle_type.value],
            "Customer_Loyalty_Status": [features.customer_loyalty_status.value],
            "Number_of_Past_Rides": [features.number_of_past_rides],
            "Average_Ratings": [features.average_rating],
            "Time_of_Booking": [
                features.time_of_booking.strftime("%H:%M:%S")
            ],  # Placeholder, features.py maps via Morning/Night
            "Expected_Ride_Duration": [features.expected_ride_duration],
            "Historical_Cost_of_Ride": [0],  # Dummy for scaling
        }

        # Adjust Time_of_Booking to match categorical mapping in features.py
        hour = features.time_of_booking.hour
        if 5 <= hour < 12:
            data["Time_of_Booking"] = ["Morning"]
        elif 12 <= hour < 17:
            data["Time_of_Booking"] = ["Afternoon"]
        elif 17 <= hour < 22:
            data["Time_of_Booking"] = ["Evening"]
        else:
            data["Time_of_Booking"] = ["Night"]

        df = pd.DataFrame(data)

        # 2. Apply Pipeline Steps (Inference mode)
        df = create_time_features(df)
        df = create_demand_supply_features(df, is_training=False)
        df = create_customer_features(df)
        df = create_categorical_features(df, is_training=False)
        df = create_interaction_features(df)
        df = scale_features(df, is_training=False)

        # 3. Ensure exactly 26 features in correct order as per training
        feature_order = [
            "Number_of_Riders",
            "Number_of_Drivers",
            "Number_of_Past_Rides",
            "Average_Ratings",
            "Expected_Ride_Duration",
            "hour",
            "hour_sin",
            "hour_cos",
            "is_night",
            "is_rush_hour",
            "demand_supply_ratio",
            "driver_deficit",
            "demand_surplus_flag",
            "hist_avg_riders_loc_hour",
            "hist_avg_drivers_loc_hour",
            "hist_demand_supply_ratio_loc_hour",
            "loyalty_numeric",
            "is_new_user",
            "is_high_value",
            "vehicle_Economy",
            "vehicle_Premium",
            "location_score",
            "duration_x_vehicle_premium",
            "demand_x_loyalty",
            "rating_x_tenure",
            "location_x_rush_hour",
        ]

        X = df[feature_order]
        return X

    def predict(self, features: RideFeatures) -> PriceResponse:
        """
        Full prediction workflow.
        """
        # 1. Preprocess
        X = self.preprocess(features)

        # 2. Predict
        # model.predict returns the scaled price (since it was trained on scaled target)
        # OR it returns raw if we didn't scale target?
        # features.py line 220: df[numeric_cols] = scaler.transform(df[numeric_cols])
        # Historical_Cost_of_Ride was scaled. So predict() returns scaled value.
        scaled_prediction = self.model.predict(X)[0]

        # 3. Inverse Transform Prediction
        scaler = joblib.load(self.scaler_path)
        # Create a dummy array for inverse transform (must match scaler's n_features=6)
        # numeric_cols: [Riders, Drivers, PastRides, Ratings, Duration, Fare]
        dummy = np.zeros((1, 6))
        dummy[0, 5] = scaled_prediction  # Predicted fare index
        unscaled_prediction = scaler.inverse_transform(dummy)[0, 5]

        # 4. Compute Interval
        lower, upper = self._compute_prediction_interval(scaled_prediction)

        # Inverse transform intervals
        dummy[0, 5] = lower
        unscaled_lower = scaler.inverse_transform(dummy)[0, 5]
        dummy[0, 5] = upper
        unscaled_upper = scaler.inverse_transform(dummy)[0, 5]

        # 5. Compute Multiplier
        # Multiplier = Predicted Fare / Base Fare
        multiplier = unscaled_prediction / features.base_fare

        return PriceResponse(
            predicted_fare=round(float(unscaled_prediction), 2),
            price_multiplier=round(float(multiplier), 3),
            demand_supply_ratio=round(float(X["demand_supply_ratio"].iloc[0]), 3),
            model_version=self.model_version,
            confidence_lower=round(float(unscaled_lower), 2),
            confidence_upper=round(float(unscaled_upper), 2),
            request_id=str(uuid.uuid4()),
        )

    def predict_rl(self, features: RideFeatures) -> PriceResponse:
        """
        RL-based prediction using the PPO agent.
        """
        if self.ppo_model is None:
            raise RuntimeError("PPO model not loaded.")

        # 1. Preprocess to get Demand-Supply Ratio
        X = self.preprocess(features)
        ds_ratio = float(X["demand_supply_ratio"].iloc[0])

        # 2. Construct RL Observation: [ds_ratio, h_sin, h_cos,
        # dow_sin, dow_cos, forecast, loyalty, vehicle]
        # Same as src/simulator/env.py _get_obs()
        hour = features.time_of_booking.hour
        h_sin = np.sin(2 * np.pi * hour / 24)
        h_cos = np.cos(2 * np.pi * hour / 24)

        dow = features.time_of_booking.weekday()
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        # Demand forecast from Prophet
        loc = features.location_category.value
        forecast = 50.0  # Default fallback
        if loc in self.prophet_models:
            future = pd.DataFrame({"ds": [features.time_of_booking]})
            f_out = self.prophet_models[loc].predict(future)
            forecast = float(f_out.iloc[0]["yhat"])

        # Mappings
        loyalty_map = {"Regular": 0, "Silver": 1, "Gold": 2}
        loyalty_val = loyalty_map.get(features.customer_loyalty_status.value, 0)
        vehicle_val = 1 if features.vehicle_type.value == "Premium" else 0

        obs = np.array(
            [
                ds_ratio,
                h_sin,
                h_cos,
                dow_sin,
                dow_cos,
                forecast,
                loyalty_val,
                vehicle_val,
            ],
            dtype=np.float32,
        )

        # 3. Get Action from PPO
        action, _ = self.ppo_model.predict(obs, deterministic=True)
        multiplier = float(action[0])

        # 4. Compute Price
        predicted_fare = multiplier * features.base_fare

        return PriceResponse(
            predicted_fare=round(predicted_fare, 2),
            price_multiplier=round(multiplier, 3),
            demand_supply_ratio=round(ds_ratio, 3),
            model_version=f"{self.model_version}-PPO",
            confidence_lower=round(predicted_fare * 0.9, 2),  # Dummy interval for RL
            confidence_upper=round(predicted_fare * 1.1, 2),
            request_id=str(uuid.uuid4()),
        )

    def _compute_prediction_interval(self, prediction_scaled: float) -> tuple:
        """
        Computes a 95% confidence interval based on training RMSE.
        """
        z_score = 1.96
        margin_of_error = z_score * self.rmse
        return (
            prediction_scaled - margin_of_error,
            prediction_scaled + margin_of_error,
        )
