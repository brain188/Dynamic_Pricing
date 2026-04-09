import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


def main():
    # 1. Load processed data
    processed_path = "data/processed/dynamic_pricing_processed.csv"
    if not os.path.exists(processed_path):
        print(f"Error: {processed_path} not found.")
        return

    df = pd.read_csv(processed_path)
    print(f"Loaded {len(df)} records from {processed_path}")

    # 2. Synthesize Time Axis
    # Map time of booking to hours
    time_map = {
        "Morning": "09:00:00",
        "Afternoon": "15:00:00",
        "Evening": "18:00:00",
        "Night": "22:00:00",
    }

    # We'll assume the 1000 records represent 120 days (4 months)
    # Total timestamps: 120 * 4 = 480
    start_date = datetime(2023, 10, 1)  # Arbitrary start date

    # Shuffle indices to randomly assign records to one of the 480 timestamps
    # This prevents any accidental ordering bias in the raw data from being
    # interpreted as a trend, while still allowing us to form a time series.
    np.random.seed(42)
    indices = np.arange(len(df))
    np.random.shuffle(indices)

    # Assign a day_offset (0-13) and a slot (0-3) to each record
    df["day_offset"] = (indices % 56) // 4
    df["slot_index"] = (indices % 56) % 4

    # Map slot_index to Time_of_Booking to stay consistent
    # with original data if possible
    # But since we've already shuffled, we'll just use the time_map
    # for whatever slot index is.
    # Actually, the data already has Time_of_Booking. Let's use it!

    # We group rows by Location and Time_of_Booking,
    # then assign them sequentially to days.
    # This is better because it preserves the categorical distribution.

    new_rows = []
    for loc in df["Location_Category"].unique():
        loc_df = df[df["Location_Category"] == loc].copy()

        # Within each location, for each slot, we want to
        # distribute rides across 14 days
        for slot_name, time_str in time_map.items():
            slot_df = loc_df[loc_df["Time_of_Booking"] == slot_name].copy()

            # Divide these rides among 14 days
            # Each day gets roughly len(slot_df) // 14 rides
            ride_indices = slot_df.index.tolist()
            np.random.shuffle(ride_indices)

            for d in range(14):
                current_ds = (
                    (start_date + timedelta(days=d)).strftime("%Y-%m-%d")
                    + " "
                    + time_str
                )

                # Assign some rides to this day
                # We use a Poisson distribution to make it look realistic
                # (mean ~ len/14)
                # But to ensure we use all 1000, we'll just partition them.

                # Better yet: just partition
                start_idx = (len(slot_df) * d) // 14
                end_idx = (len(slot_df) * (d + 1)) // 14

                count = end_idx - start_idx
                if count > 0:
                    new_rows.append({"ds": current_ds, "location": loc, "y": count})

    demand_df = pd.DataFrame(new_rows)
    # Sort by ds
    demand_df["ds"] = pd.to_datetime(demand_df["ds"])
    demand_df = demand_df.sort_values("ds")

    # Save
    output_path = "data/processed/demand_counts.csv"
    demand_df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(demand_df)} timestamps to {output_path}")
    print(demand_df.head())


if __name__ == "__main__":
    main()
