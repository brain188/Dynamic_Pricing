# Exploratory Data Analysis (EDA) Summary Report

## 1. Project Overview

This report summarizes the findings from the Exploratory Data Analysis (EDA) phase of the Dynamic Pricing project. The goal was to understand the dataset, verify data quality, and identify key drivers of ride pricing.

## 2. Data Profiling & Quality

- **Dataset Size**: 1,000 observations, 10 features.
- **Data Quality**:
  - No missing values were found across any columns.
  - No duplicate records detected.
  - `Time_of_Booking` is categorical (Night, Evening, Afternoon, Morning).
  - `Historical_Cost_of_Ride` is the target variable.

## 3. Univariate Analysis (Individual Features)

We analyzed the distribution of each variable to identify skewness and scale.

### Numeric Features

- **Historical Cost of Ride**: Shows a right-skewed distribution, suggesting most rides are lower-priced with some high-value outliers.
- **Number of Riders/Drivers**: Both are uniformly distributed within their respective ranges.
- **Expected Ride Duration**: Directly impacts cost; shows a wide range of values.

### Categorical Features

- **Location Category**: Balanced distribution across Urban, Suburban, and Rural.
- **Vehicle Type**: Even split between Economy and Premium.
- **Customer Loyalty Status**: Distribution among Silver, Regular, and Gold status.

## 4. Outlier Detection

Used **IQR (1.5x)** and **Z-score (>3)** methods to identify anomalies.

- **Findings**: Outliers were primarily found in `Historical_Cost_of_Ride` and `Expected_Ride_Duration`.
- **Decision**: Outliers were retained but flagged with a binary indicator `is_outlier_fare` to allow models to handle them specifically or for later clipping in feature engineering.
- **Grubbs' Test**: Confirmed the presence of statistically significant single outliers in the fare column.

## 5. Bivariate Analysis (Key Relationships)

This section explores how different factors influence the `Historical_Cost_of_Ride`.

### Demand vs. Supply

- **Demand-Supply Ratio**: This is a critical driver. As the ratio of `Number_of_Riders` to `Number_of_Drivers` increases, we see a clear upward trend in the average fare.
- **Correlation**: A moderate positive correlation exists between demand levels and pricing.

### Duration vs. Fare

- **Primary Driver**: `Expected_Ride_Duration` has the strongest linear relationship with `Historical_Cost_of_Ride` ($R^2 \approx 0.85$). This confirms that base pricing is heavily duration-dependent.

### Categorical Impacts

- **Vehicle Type**: Premium vehicles consistently command higher fares than Economy vehicles, as expected.
- **Location**: Rural rides tend to have higher variance in pricing, possibly due to longer durations.
- **Time of Day**: Afternoon and Evening rides show slightly higher demand and pricing compared to Morning/Night in this sample.

## 6. Correlation & Multicollinearity

- **Pearson Heatmap**: High correlation identified between `Expected_Ride_Duration` and `Historical_Cost_of_Ride`.
- **VIF Analysis**: No features showed VIF > 10, indicating that multicollinearity is not currently a major issue for linear modeling.

## 7. Proposed Features for Phase 2

Based on EDA, we recommend engineering:

1. **Demand_Supply_Ratio**: `Number_of_Riders / Number_of_Drivers`.
2. **Cost_Per_Minute**: `Historical_Cost_of_Ride / Expected_Ride_Duration`.
3. **Loyalty_Multiplier**: Interaction between `Customer_Loyalty_Status` and `Number_of_Past_Rides`.
4. **Time_Categorical_Encoding**: One-hot encoding for `Time_of_Booking`.

---
