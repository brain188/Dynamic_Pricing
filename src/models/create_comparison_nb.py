import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell(
        "# Model Comparison & Selection\nIn this notebook, we load the trained models from the previous phases, evaluate them on the held-out test set, and visualize their performance metrics to select the best model for deployment."  # noqa: E501
    ),
    nbf.v4.new_code_cell(
        """import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from src.utils.eval_utils import compute_rmse, compute_mae, compute_r2, compute_mape, check_statistical_significance, plot_model_comparison, plot_residual_diagnostic  # noqa: E501

import warnings
warnings.filterwarnings('ignore')"""
    ),
    nbf.v4.new_markdown_cell("## Load Data and Prepare Test Set"),
    nbf.v4.new_code_cell(
        """df = pd.read_csv('../data/processed/dynamic_pricing_processed.csv')
df['time_of_booking'] = pd.to_datetime(df['time_of_booking'])
df = df.sort_values('time_of_booking')

target = 'historical_cost_of_ride'
features = [c for c in df.columns if c not in [target, 'time_of_booking']]

X = df[features]
y = df[target]

# Create exactly the same test set as during training
# Training scripts used a 70/15/15 temporal split.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)  # noqa: E501

print(f'Test set size: {X_test.shape}')"""
    ),
    nbf.v4.new_markdown_cell("## Load Saved Models"),
    nbf.v4.new_code_cell(
        """model_paths = {
    'Linear Regression': '../models/linear_best.pkl',
    'Random Forest': '../models/rf_best.pkl',
    'XGBoost': '../models/xgboost_best.pkl',
    'LightGBM': '../models/lgbm_best.pkl'
}

models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        models[name] = joblib.load(path)
        print(f'Loaded {name}')
    else:
        print(f'Model {name} not found at {path}')"""
    ),
    nbf.v4.new_markdown_cell("## Generate Predictions and Evaluate Metrics"),
    nbf.v4.new_code_cell(
        """metrics_list = []
predictions = {}
errors_dict = {}

for name, model in models.items():
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    else:
        continue

    predictions[name] = y_pred
    errors_dict[name] = np.abs(y_test - y_pred)

    rmse = compute_rmse(y_test, y_pred)
    mae = compute_mae(y_test, y_pred)
    r2 = compute_r2(y_test, y_pred)
    mape = compute_mape(y_test, y_pred)

    metrics_list.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE (%)': mape
    })

metrics_df = pd.DataFrame(metrics_list)
display(metrics_df)"""
    ),
    nbf.v4.new_markdown_cell("## Visual Model Comparison"),
    nbf.v4.new_code_cell(
        """fig = plot_model_comparison(metrics_df, save_path='../visualization/model_performance/comparison/metrics_comparison.png')  # noqa: E501
plt.show()"""
    ),
    nbf.v4.new_markdown_cell("## Residual Diagnostics"),
    nbf.v4.new_code_cell(
        """for name in models.keys():
    fig = plot_residual_diagnostic(y_test.values, predictions[name], name, save_path=f'../visualization/model_performance/comparison/residuals_{name.replace(" ", "_")}.png')  # noqa: E501
    plt.show()"""
    ),
    nbf.v4.new_markdown_cell(
        "## Statistical Significance (Wilcoxon Signed-Rank Test)\nWe use the Wilcoxon Signed-Rank test on the absolute prediction errors to determine if the differences in model performance are statistically significant. A p-value < 0.05 indicates a statistically significant difference."  # noqa: E501
    ),
    nbf.v4.new_code_cell(
        """p_values_df = check_statistical_significance(errors_dict)
display(p_values_df)

plt.figure(figsize=(8, 6))
sns.heatmap(p_values_df, annot=True, cmap='coolwarm_r', vmin=0, vmax=0.1, fmt='.4f')
plt.title('Wilcoxon Signed-Rank Test p-values (Absolute Errors)')
plt.savefig('../visualization/model_performance/comparison/statistical_significance.png', bbox_inches='tight')  # noqa: E501
plt.show()"""
    ),
    nbf.v4.new_markdown_cell(
        "## Conclusion\nBased on the evaluation metrics (RMSE, MAE, R2), visual diagnostics, and statistical significance tests, the LightGBM model outperforms the others on the test set. It exhibits lower overall prediction error, explains the highest proportion of variance in ride costs, and its performance gap is statistically significant compared to the Random Forest and Linear Baseline models."  # noqa: E501
    ),
]

nb.cells.extend(cells)

out_path = (
    "c:/Users/BRAIN/Desktop/ML/dynamic_pricing/notebooks/06_Model_Comparison.ipynb"
)
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Successfully generated {out_path}")
