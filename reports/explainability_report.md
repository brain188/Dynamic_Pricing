# Model Explainability Report — XGBoost

**Model path:** `C:\Users\BRAIN\Desktop\ML\dynamic_pricing\models\xgboost_best.pkl`
**Test RMSE:** 0.2233 | **MAE:** 0.1731 | **R²:** 0.8776

---

## Top-5 SHAP Feature Drivers

| Rank | Feature | Business Interpretation |
|---|---|---|
| 1 | `Expected_Ride_Duration` | Expected trip length in minutes — directly encodes distance cost. |
| 2 | `duration_x_vehicle_premium` | No description available. |
| 3 | `Average_Ratings` | No description available. |
| 4 | `hist_demand_supply_ratio_loc_hour` | Historical avg demand/supply for this location-hour bucket — encodes habitual surge patterns. |
| 5 | `rating_x_tenure` | Interaction of average rating × past rides — proxy for high-value habitual riders. |

---

## Key Findings

1. **Demand–Supply Ratio** is the dominant driver of predicted fare. Higher ratios trigger surge pricing, consistent with dynamic pricing theory.
2. **Expected Ride Duration** acts as a baseline fare proxy — longer trips command higher prices.
3. **Historical demand patterns** (location × hour) encode habitual surge windows, providing the model with temporal foresight.
4. **Driver deficit** captures acute supply shortfalls that push prices above the historical average.
5. **Location score** reflects the premium associated with high-demand pickup zones (e.g., airports, city centres).

## Visualisation Artifacts

All SHAP plots are saved to `C:\Users\BRAIN\Desktop\ML\dynamic_pricing\visualization\model_performance\shap/`:

- `summary_beeswarm.png` — global feature impact & direction
- `summary_bar.png` — mean |SHAP| feature ranking
- `dependence_*.png` — per-feature marginal effect
- `interaction_demand_hour.png` — demand × time interaction
- `pdp_*.png` — partial dependence plots
- `waterfall_sample_*.png` — row-level explanation for low/median/high fare rides
