import os
import sys
from datetime import datetime, timedelta

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.linear_model import LogisticRegression

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.logger import logger  # noqa: E402


def fit_acceptance_beta(df: pd.DataFrame) -> float:
    """
    Fits a sigmoid acceptance model to capture price sensitivity.
    Since the dataset lacks rejection flags, we synthesize rejection data
    for prices significantly above the historical median.
    """
    logger.info("Fitting price acceptance model (beta)...")

    # Use Historical_Cost_of_Ride as the baseline for 'accepted' prices
    accepted_prices = df["Historical_Cost_of_Ride"].dropna().values

    # Create synthetic rejected prices (e.g., 20% to 150% higher)
    rejection_multiplier = np.random.uniform(1.2, 2.5, size=len(accepted_prices))
    rejected_prices = accepted_prices * rejection_multiplier

    # Prepare training data: diff from 'reference' price
    # (which is accepted_prices itself here)
    # For accepted cases, diff = 0 (or small noise)
    # For rejected cases, diff = rejected_prices - accepted_prices
    X_acc = np.zeros((len(accepted_prices), 1))
    X_rej = (rejected_prices - accepted_prices).reshape(-1, 1)

    X = np.vstack([X_acc, X_rej])
    y = np.concatenate([np.ones(len(accepted_prices)), np.zeros(len(rejected_prices))])

    # Fit Logistic Regression
    clf = LogisticRegression()
    clf.fit(X, y)

    # The coefficient beta represents sensitivity to price difference
    beta = -clf.coef_[0][
        0
    ]  # beta should be positive for exp(beta * diff) to decrease prob
    logger.info(f"Fitted beta: {beta:.6f}")
    return beta


class RideSharingEnv(gym.Env):
    """
    Custom Environment for Dynamic Pricing Simulation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, demand_models, supervised_model, config=None):
        super(RideSharingEnv, self).__init__()

        self.demand_models = demand_models  # Dict of {location: prophet_model}
        self.supervised_model = supervised_model  # XGBoost model
        self.config = config or {}

        # State: [ds_ratio, h_sin, h_cos, d_sin, d_cos,
        #         forecast, loyalty, vehicle]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # Action: Continuous multiplier [0.5, 3.0]
        self.action_space = spaces.Box(low=0.5, high=3.0, shape=(1,), dtype=np.float32)

        # Simulation parameters
        self.beta = self.config.get("beta", 0.1)
        self.locations = list(demand_models.keys())
        self.vehicle_types = [0, 1]  # 0: Economy, 1: Premium
        self.loyalty_levels = [0, 1, 2]  # 0: Regular, 1: Silver, 2: Gold

        # Current state
        self.current_hour = 0
        self.current_location = None
        self.current_vehicle_type = None
        self.current_loyalty = None
        self.start_date = datetime(2024, 1, 1)  # Start of simulation timeline

        # Cache for demand forecasts to avoid slow Prophet predict() calls in every step
        self.demand_cache = {}
        self._precalculate_demand()

    def _precalculate_demand(self):
        """Pre-calculates 24 hours of demand per location for the simulation."""
        logger.info("Pre-calculating demand forecasts for environment caching...")
        for loc, model in self.demand_models.items():
            self.demand_cache[loc] = {}
            for h in range(25):  # 0 to 24
                dt = self.start_date + timedelta(hours=h)
                future = pd.DataFrame({"ds": [dt]})
                forecast = model.predict(future)
                self.demand_cache[loc][h] = max(0, forecast.iloc[0]["yhat"])
        logger.info("Demand caching complete.")

    def _get_obs(self):
        # 1. Demand signal from Cache
        forecasted_demand = self.demand_cache[self.current_location][self.current_hour]

        # 2. Time features
        hour_sin = np.sin(2 * np.pi * self.current_hour / 24)
        hour_cos = np.cos(2 * np.pi * self.current_hour / 24)
        current_dt = self.start_date + timedelta(hours=self.current_hour)
        dow = current_dt.weekday()
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        # 3. Demand-Supply Ratio (Simulated based on hour)
        # Use a deterministic/seeded approach for check_env compatibility if needed
        # but for RL, stochasticity is expected.
        # We'll use self.np_random which is set by gym for consistency.
        base_ds = (
            1.2
            if 7 <= self.current_hour <= 10 or 17 <= self.current_hour <= 20
            else 0.8
        )
        demand_supply_ratio = self.np_random.normal(base_ds, 0.15)

        return np.array(
            [
                demand_supply_ratio,
                hour_sin,
                hour_cos,
                dow_sin,
                dow_cos,
                forecasted_demand,
                self.current_loyalty,
                self.current_vehicle_type,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_hour = 0
        self.current_location = self.np_random.choice(self.locations)
        self.current_vehicle_type = self.np_random.choice(self.vehicle_types)
        self.current_loyalty = self.np_random.choice(self.loyalty_levels)

        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        # ... logic stays same, uses _get_obs which is now fast ...
        base_fare_historical = 50.0
        multiplier = action[0]
        price = multiplier * base_fare_historical

        reference_price = base_fare_historical
        p_accept = 1.0 / (1.0 + np.exp(self.beta * (price - reference_price)))

        # Use self.np_random
        accepted = self.np_random.random() < p_accept

        forecasted_demand = self.demand_cache[self.current_location][self.current_hour]
        if accepted:
            reward = price * forecasted_demand
        else:
            reward = -5.0 * forecasted_demand  # Penalty for lost opportunity

        # 5. Transition
        self.current_hour += 1
        terminated = self.current_hour >= 24
        truncated = False

        observation = self._get_obs()
        info = {
            "fare": price,
            "acceptance_prob": p_accept,
            "accepted": accepted,
            "location": self.current_location,
            "hour": self.current_hour,
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(
            f"Time: {self.current_hour:02d}:00 | "
            f"Loc: {self.current_location} | "
            f"Vehicle: {self.current_vehicle_type} | "
            f"Loyalty: {self.current_loyalty}"
        )
