"""
Phase 3.6 — Model Comparison script.
Evaluates all trained models on the held-out test set and saves artefacts.
"""

import os
import sys

# Allow `src.*` imports when running from the project root
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.utils.eval_utils import (  # noqa: E402
    check_statistical_significance,
    compute_mae,
    compute_mape,
    compute_r2,
    compute_rmse,
    plot_model_comparison,
    plot_residual_diagnostic,
)
from src.utils.train_utils import temporal_split  # noqa: E402

warnings.filterwarnings("ignore")

# ── Load and split (exact same 70/15/15 as training scripts) ─────────────────
df = pd.read_csv("data/processed/dynamic_pricing_processed.csv")
target = "Historical_Cost_of_Ride"

df_train, df_val, df_test = temporal_split(df)

y_test = df_test[target].reset_index(drop=True)
X_test = df_test.drop(columns=[target])
numeric_cols = X_test.select_dtypes(include=[np.number]).columns
X_test = X_test[numeric_cols].reset_index(drop=True)

print(f"Test set size: {X_test.shape}")

# ── Load all saved models ─────────────────────────────────────────────────────
model_paths = {
    "Linear Regression": "models/linear_best.pkl",
    "Random Forest": "models/rf_best.pkl",
    "XGBoost": "models/xgboost_best.pkl",
    "LightGBM": "models/lgbm_best.pkl",
}

models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        models[name] = joblib.load(path)
        print(f"Loaded: {name}")
    else:
        print(f"Model not found: {path}")

# ── Generate predictions and evaluate ────────────────────────────────────────
metrics_list = []
predictions = {}
errors_dict = {}

for name, model in models.items():
    if not hasattr(model, "predict"):
        continue

    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    errors_dict[name] = np.abs(y_test.values - y_pred)

    rmse = compute_rmse(y_test.values, y_pred)
    mae = compute_mae(y_test.values, y_pred)
    r2 = compute_r2(y_test.values, y_pred)
    mape = compute_mape(y_test.values, y_pred)

    metrics_list.append(
        {
            "model_name": name,
            "RMSE_test": round(rmse, 4),
            "MAE_test": round(mae, 4),
            "R2_test": round(r2, 4),
            "MAPE_test": round(mape, 4),
            "n_features": X_test.shape[1],
        }
    )
    print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

# ── Save metrics CSV ───────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(metrics_list)
os.makedirs("reports/results", exist_ok=True)
metrics_df.to_csv("reports/results/models_summary.csv", index=False)
print("\nSaved reports/results/models_summary.csv")

# ── Bar chart comparison ───────────────────────────────────────────────────────
plot_df = metrics_df.rename(
    columns={
        "model_name": "Model",
        "RMSE_test": "RMSE",
        "MAE_test": "MAE",
        "R2_test": "R2",
    }
)
os.makedirs("visualization/model_performance/comparison", exist_ok=True)
plot_model_comparison(
    plot_df,
    save_path=("visualization/model_performance/comparison/metrics_comparison.png"),
)
plt.close("all")
print("Saved metrics_comparison.png")

# ── Residual diagnostics ───────────────────────────────────────────────────────
for name in models.keys():
    safe_name = name.replace(" ", "_")
    save_path = f"visualization/model_performance/comparison/residuals_{safe_name}.png"
    plot_residual_diagnostic(y_test.values, predictions[name], name, save_path)
    plt.close("all")
    print(f"Saved residuals_{safe_name}.png")

# ── Statistical significance (Wilcoxon signed-rank) ──────────────────────────
p_values_df = check_statistical_significance(errors_dict)
plt.figure(figsize=(8, 6))
sns.heatmap(p_values_df, annot=True, cmap="coolwarm_r", vmin=0, vmax=0.1, fmt=".4f")
plt.title("Wilcoxon Signed-Rank Test p-values (Absolute Errors)")
plt.tight_layout()
sig_path = "visualization/model_performance/comparison/statistical_significance.png"
plt.savefig(sig_path, bbox_inches="tight")
plt.close("all")
print("Saved statistical_significance.png")

print("\n=== Phase 3.6 Complete ===")
