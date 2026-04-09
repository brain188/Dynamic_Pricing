import os
import sys
import traceback

# Ensure backend and project root are in path
sys.path.append(os.getcwd())

try:
    from datetime import datetime

    from backend.app.models.schemas import (
        LocationEnum,
        LoyaltyLevelEnum,
        RideFeatures,
        VehicleTypeEnum,
    )
    from backend.app.services.inference import InferenceService

    inf = InferenceService()
    f = RideFeatures(
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

    print("Testing XGBoost predict...")
    try:
        res = inf.predict(f)
        print(f"XGBoost Success: {res}")
    except Exception:
        print("XGBoost Failure:")
        traceback.print_exc()

    print("\nTesting PPO predict...")
    try:
        res = inf.predict_rl(f)
        print(f"PPO Success: {res}")
    except Exception:
        print("PPO Failure:")
        traceback.print_exc()

except Exception:
    traceback.print_exc()
