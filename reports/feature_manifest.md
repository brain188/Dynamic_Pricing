# Feature Manifest - Dynamic Pricing ML System

This document outlines the features engineered during Phase 2 of the project.

## 1. Time-Based Features
| Feature Name | Description | Type | Logic / Formula | Scaling |
|--------------|-------------|------|-----------------|---------|
| `hour` | Proxy hour derived from `Time_of_Booking`. | Numeric | Morning=9, Afternoon=14, Evening=19, Night=2 | No |
| `hour_sin` | Cyclical encoding (sine) of `hour`. | Numeric | `sin(2 * pi * hour / 24)` | No |
| `hour_cos` | Cyclical encoding (cosine) of `hour`. | Numeric | `cos(2 * pi * hour / 24)` | No |
| `is_night` | Binary flag for night bookings. | Binary | `1` if `Time_of_Booking == 'Night'` else `0` | No |
| `is_rush_hour` | Binary flag for rush hour bookings. | Binary | `1` if `Time_of_Booking \in {Morning, Evening}` | No |

## 2. Demand–Supply Features
| Feature Name | Description | Type | Logic / Formula | Scaling |
|--------------|-------------|------|-----------------|---------|
| `demand_supply_ratio` | Current ratio of riders to drivers. | Numeric | `Number_of_Riders / (Number_of_Drivers + 1)` | No |
| `driver_deficit` | Gap between demand and supply. | Numeric | `Number_of_Riders - Number_of_Drivers` | No |
| `demand_surplus_flag` | Flag for high demand relative to supply. | Binary | `1` if `ratio > 1.5` else `0` | No |
| `hist_avg_riders_loc_hour` | Historical mean riders for location/hour. | Numeric | GroupBy mean (on 70% train split) | No |
| `hist_avg_drivers_loc_hour` | Historical mean drivers for location/hour. | Numeric | GroupBy mean (on 70% train split) | No |
| `hist_demand_supply_ratio_loc_hour` | Historical mean ratio for location/hour. | Numeric | GroupBy mean (on 70% train split) | No |

## 3. Customer Features
| Feature Name | Description | Type | Logic / Formula | Scaling |
|--------------|-------------|------|-----------------|---------|
| `loyalty_numeric` | Ordinal encoding of loyalty status. | Numeric | Regular=0, Silver=1, Gold=2 | No |
| `ride_tenure_bucket` | Binned number of past rides. | Categorical | `pd.cut` into 5 buckets | No |
| `rating_bucket` | Binned average ratings. | Categorical | `pd.cut` into 4 buckets | No |
| `is_new_user` | Flag for users with < 3 past rides. | Binary | `1` if `Past_Rides < 3` else `0` | No |
| `is_high_value` | Flag for high-loyalty, high-rating users. | Binary | `1` if `Gold` and `Rating > 4.5` else `0` | No |

## 4. Categorical Encodings
| Feature Name | Description | Type | Logic / Formula | Scaling |
|--------------|-------------|------|-----------------|---------|
| `vehicle_Economy` | One-hot encoded Vehicle Type. | Binary | `Vehicle_Type == 'Economy'` | No |
| `vehicle_Premium` | One-hot encoded Vehicle Type. | Binary | `Vehicle_Type == 'Premium'` | No |
| `location_score` | Target encoded Location Category. | Numeric | Mean `Historical_Cost_of_Ride` per Category | No |

## 5. Interaction Features
| Feature Name | Description | Type | Logic / Formula | Scaling |
|--------------|-------------|------|-----------------|---------|
| `duration_x_vehicle_premium` | Interaction between duration and premium status. | Numeric | `Expected_Ride_Duration * vehicle_Premium` | No |
| `demand_x_loyalty` | Interaction between demand and user loyalty. | Numeric | `demand_supply_ratio * loyalty_numeric` | No |
| `rating_x_tenure` | Interaction between rating and ride history. | Numeric | `Average_Ratings * Number_of_Past_Rides` | No |
| `location_x_rush_hour` | Interaction between location score and rush hour. | Numeric | `location_score * is_rush_hour` | No |

## 6. Scaled Features (RobustScaler)
*The following raw features were scaled using `RobustScaler` (Median/IQR) fitted on the training split.*
- `Number_of_Riders`
- `Number_of_Drivers`
- `Number_of_Past_Rides`
- `Average_Ratings`
- `Expected_Ride_Duration`
- `Historical_Cost_of_Ride`
