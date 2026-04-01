import argparse  # noqa: F401
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import scipy.stats as stats  # noqa: E402
import seaborn as sns  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy.stats import shapiro  # noqa: E402
from sklearn.linear_model import (  # noqa: E402
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    RidgeCV,
)
from sklearn.metrics import (  # noqa: F401, E402
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline  # noqa: F401, E402
from statsmodels.stats.diagnostic import het_breuschpagan  # noqa: E402
from statsmodels.stats.stattools import durbin_watson  # noqa: E402

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.data_loader import load_config, load_raw  # noqa: F401, E402
from src.utils.eval_utils import evaluate_model  # noqa: E402
from src.utils.features import build_feature_pipeline  # noqa: F401, E402
from src.utils.logger import logger  # noqa: E402
from src.utils.train_utils import get_time_series_cv, temporal_split  # noqa: E402

# Constants
TARGET_COL = "Historical_Cost_of_Ride"


def check_ols_assumptions(model_sm, X_train_sm, y_train, output_dir):
    """Check the 5 assumptions of OLS and save plots."""
    logger.info("Checking OLS Assumptions...")
    os.makedirs(output_dir, exist_ok=True)

    predictions = model_sm.predict(X_train_sm)
    residuals = y_train - predictions

    # 1. Linearity: Residuals vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=predictions, y=residuals, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.title("Residuals vs Predicted Values (Linearity & Homoscedasticity Indicator)")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_vs_predicted.png"))
    plt.close()

    # Print linearity assumption check
    logger.info("1. Linearity checked. See residuals_vs_predicted.png")

    # 2. Independence: Durbin-Watson
    dw_stat = durbin_watson(residuals)
    logger.info(f"2. Independence (Durbin-Watson): {dw_stat:.3f} (Ideal: ~2.0)")

    # 3. Homoscedasticity: Breusch-Pagan
    try:
        _, pval, _, f_pval = het_breuschpagan(residuals, X_train_sm)
        logger.info(f"3. Homoscedasticity (Breusch-Pagan p-value): {pval:.4f}")
        if pval < 0.05:
            logger.warning("   -> Flagged as heteroscedastic (p < 0.05)")
    except Exception as e:
        logger.error(f"   -> Breusch-Pagan test failed: {e}")

    # 4. Normality of Residuals: Shapiro-Wilk & Q-Q Plot
    # Shapiro-Wilk (can be slow/sensitive for N > 5000, but fine here)
    if len(residuals) < 5000:
        stat, p_wt = shapiro(residuals)
        logger.info(f"4. Normality (Shapiro-Wilk p-value): {p_wt:.4f}")

    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qq_plot.png"))
    plt.close()
    logger.info("   -> See qq_plot.png")

    # 5. Multicollinearity: VIF
    # (Simplified check: look for high condition number in model summary)
    cond_no = model_sm.condition_number
    logger.info(f"5. Multicollinearity (Condition Number): {cond_no:.1f}")
    if cond_no > 30:
        logger.warning(
            "   -> High condition number indicates potential multicollinearity."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="Linear_Baseline_Models")
    args = parser.parse_args()

    config = load_config()

    # Paths
    viz_dir = os.path.join(project_root, "visualization", "model_performance", "linear")
    model_dir = os.path.join(project_root, "models")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load Processed Data
    logger.info("Loading processed data...")
    processed_path = os.path.join(
        project_root, "data", "processed", "dynamic_pricing_processed.csv"
    )
    if not os.path.exists(processed_path):
        logger.error(
            f"Processed data not found at {processed_path}. Please run Phase 2."
        )
        sys.exit(1)

    df_processed = pd.read_csv(processed_path)

    # 2. Temporal Split (Processed data)
    logger.info("Applying temporal split...")
    df_train, df_val, df_test = temporal_split(df_processed)

    # 3. Separate Features and Target
    if TARGET_COL not in df_train.columns:
        logger.error(f"Target column '{TARGET_COL}' not found in processed data.")
        sys.exit(1)

    y_train = df_train.pop(TARGET_COL)
    y_val = df_val.pop(TARGET_COL)
    y_test = df_test.pop(TARGET_COL)  # noqa: F841

    X_train = df_train
    X_val = df_val
    X_test = df_test

    # Drop non-numeric columns (if any slipped through)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train = X_train[numeric_cols]
    X_val = X_val[numeric_cols]
    X_test = X_test[numeric_cols]

    n_features = X_train.shape[1]

    # Initialize MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    best_val_rmse = float("inf")
    best_model_name = None
    best_model = None

    cv = get_time_series_cv(n_splits=5)

    with mlflow.start_run(run_name=args.run_name):
        # =====================================================================
        # MODEL 0: Simple OLS Baseline (Fare vs Expected_Ride_Duration ONLY)
        # =====================================================================
        logger.info("--- Training Simple OLS Baseline ---")
        if "Expected_Ride_Duration" in X_train.columns:
            X_train_simple = sm.add_constant(X_train["Expected_Ride_Duration"])
            model_simple = sm.OLS(y_train, X_train_simple).fit()
            logger.info("\n" + model_simple.summary().as_text())
        else:
            logger.warning(
                "Expected_Ride_Duration not in features, skipping simple OLS."
            )

        # =====================================================================
        # MODEL 1: Full OLS
        # =====================================================================
        logger.info("--- Training Full OLS ---")
        X_train_sm = sm.add_constant(X_train)
        model_ols_sm = sm.OLS(y_train, X_train_sm).fit()

        # Check Assumptions
        check_ols_assumptions(model_ols_sm, X_train_sm, y_train, viz_dir)
        mlflow.log_artifacts(viz_dir, artifact_path="assumption_plots")

        # Scikit-learn OLS for CV evaluation
        model_ols = LinearRegression()
        metrics_ols = evaluate_model(  # noqa: F841
            model_ols, X_train, y_train, cv, n_features
        )

        # Train on full train set, eval on val
        model_ols.fit(X_train, y_train)
        preds_ols = model_ols.predict(X_val)
        val_rmse_ols = np.sqrt(mean_squared_error(y_val, preds_ols))
        logger.info(f"OLS Val RMSE: {val_rmse_ols:.4f}")
        mlflow.log_metric("OLS_val_rmse", val_rmse_ols)

        if val_rmse_ols < best_val_rmse:
            best_val_rmse = val_rmse_ols
            best_model_name = "OLS"
            best_model = model_ols

        # =====================================================================
        # MODEL 2: Ridge
        # =====================================================================
        logger.info("--- Training Ridge CV ---")
        ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=cv)
        ridge_cv.fit(X_train, y_train)

        preds_ridge = ridge_cv.predict(X_val)
        val_rmse_ridge = np.sqrt(mean_squared_error(y_val, preds_ridge))

        logger.info(f"Ridge Best Alpha: {ridge_cv.alpha_}")
        logger.info(f"Ridge Val RMSE: {val_rmse_ridge:.4f}")
        mlflow.log_param("Ridge_best_alpha", float(ridge_cv.alpha_))
        mlflow.log_metric("Ridge_val_rmse", val_rmse_ridge)

        if val_rmse_ridge < best_val_rmse:
            best_val_rmse = val_rmse_ridge
            best_model_name = "Ridge"
            best_model = ridge_cv

        # =====================================================================
        # MODEL 3: Lasso
        # =====================================================================
        logger.info("--- Training Lasso CV ---")
        lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=cv, max_iter=10000)
        lasso_cv.fit(X_train, y_train)

        preds_lasso = lasso_cv.predict(X_val)
        val_rmse_lasso = np.sqrt(mean_squared_error(y_val, preds_lasso))

        logger.info(f"Lasso Best Alpha: {lasso_cv.alpha_}")
        logger.info(f"Lasso Val RMSE: {val_rmse_lasso:.4f}")
        zero_coefs = np.sum(lasso_cv.coef_ == 0)
        logger.info(f"Lasso shrank {zero_coefs}/{n_features} coefficients to zero.")

        mlflow.log_param("Lasso_best_alpha", float(lasso_cv.alpha_))
        mlflow.log_metric("Lasso_val_rmse", val_rmse_lasso)

        if val_rmse_lasso < best_val_rmse:
            best_val_rmse = val_rmse_lasso
            best_model_name = "Lasso"
            best_model = lasso_cv

        # =====================================================================
        # MODEL 4: ElasticNet
        # =====================================================================
        logger.info("--- Training ElasticNet CV ---")
        enet_cv = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=[0.001, 0.01, 0.1, 1.0],
            cv=cv,
            max_iter=10000,
        )
        enet_cv.fit(X_train, y_train)

        preds_enet = enet_cv.predict(X_val)
        val_rmse_enet = np.sqrt(mean_squared_error(y_val, preds_enet))

        logger.info(
            f"ElasticNet Best Alpha: {enet_cv.alpha_}, L1_ratio: {enet_cv.l1_ratio_}"
        )
        logger.info(f"ElasticNet Val RMSE: {val_rmse_enet:.4f}")

        mlflow.log_param("ElasticNet_best_alpha", float(enet_cv.alpha_))
        mlflow.log_param("ElasticNet_best_l1_ratio", float(enet_cv.l1_ratio_))
        mlflow.log_metric("ElasticNet_val_rmse", val_rmse_enet)

        if val_rmse_enet < best_val_rmse:
            best_val_rmse = val_rmse_enet
            best_model_name = "ElasticNet"
            best_model = enet_cv

        # =====================================================================
        # Save Best Model
        # =====================================================================
        logger.info(
            f"--- Best Linear Model: {best_model_name} "
            f"(Val RMSE: {best_val_rmse:.4f}) ---"
        )
        best_model_path = os.path.join(model_dir, "linear_best.pkl")
        joblib.dump(best_model, best_model_path)
        logger.info(f"Saved best model to {best_model_path}")

        mlflow.sklearn.log_model(best_model, "best_linear_model")
        mlflow.log_param("best_linear_model_type", best_model_name)


if __name__ == "__main__":
    main()
