import os
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # For environments without display
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV  # noqa: E402

# Add src to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.utils.data_loader import load_config  # noqa: E402
from src.utils.eval_utils import evaluate_model  # noqa: E402, F401
from src.utils.logger import logger  # noqa: E402
from src.utils.train_utils import get_time_series_cv, temporal_split  # noqa: E402


def plot_feature_importance(model, X_train, y_train, viz_dir):
    logger.info("Plotting Feature Importances...")

    # 1. Gini-based (Impurity) Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X_train.columns

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=importances[indices][:15],
        y=features[indices][:15],
        palette="viridis",
        hue=features[indices][:15],
    )
    plt.title("Top 15 Feature Importances (Impurity-based)")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    img_path1 = os.path.join(viz_dir, "feature_importance_impurity.png")
    plt.savefig(img_path1)
    plt.close()

    # 2. Permutation Importance
    logger.info("Computing Permutation Importance...")
    perm_importance = permutation_importance(
        model, X_train, y_train, n_repeats=5, random_state=42, n_jobs=-1
    )
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=perm_importance.importances_mean[sorted_idx][:15],
        y=features[sorted_idx][:15],
        palette="magma",
        hue=features[sorted_idx][:15],
    )
    plt.title("Top 15 Feature Importances (Permutation)")
    plt.xlabel("Mean Importance Decrease")
    plt.tight_layout()
    img_path2 = os.path.join(viz_dir, "feature_importance_permutation.png")
    plt.savefig(img_path2)
    plt.close()

    mlflow.log_artifact(img_path1)
    mlflow.log_artifact(img_path2)


def plot_learning_curve_rf(model, X_train, y_train, cv, viz_dir):
    from sklearn.model_selection import learning_curve

    logger.info("Plotting Learning Curve...")

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )

    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = -np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training RMSE")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="blue",
    )
    plt.plot(train_sizes, val_mean, "o-", color="green", label="Validation RMSE")
    plt.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="green"
    )

    plt.title("Learning Curve (Random Forest)")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    img_path = os.path.join(viz_dir, "learning_curve.png")
    plt.savefig(img_path)
    plt.close()
    mlflow.log_artifact(img_path)


def plot_oob_error(X_train, y_train, viz_dir):
    logger.info("Plotting OOB Error vs n_estimators...")

    # Range of `n_estimators` values to explore.
    min_estimators = 50
    max_estimators = 500

    ensemble_clfs = [
        (
            "RandomForestRegressor, max_features='sqrt'",
            RandomForestRegressor(
                warm_start=True, oob_score=True, max_features="sqrt", random_state=42
            ),
        )
    ]

    error_rate = {}

    for label, clf in ensemble_clfs:
        error_rate[label] = []
        for i in range(min_estimators, max_estimators + 1, 50):
            clf.set_params(n_estimators=i)
            clf.fit(X_train, y_train)

            # Record the OOB error for each `n_estimators=i` setting.
            # R2 score is default for regressor oob_score_, we convert to
            # 1 - R2 or compute MSE
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    plt.figure(figsize=(8, 6))
    for label, clf_err in error_rate.items():  # noqa: E501
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.title("OOB Error Rate vs. n_estimators")
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB Error (1 - R^2)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()

    img_path = os.path.join(viz_dir, "oob_error.png")
    plt.savefig(img_path)
    plt.close()
    mlflow.log_artifact(img_path)


def main():
    config = load_config()  # noqa: F841
    TARGET_COL = "Historical_Cost_of_Ride"

    viz_dir = os.path.join(project_root, "visualization", "model_performance", "rf")
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

    # 2. Temporal Split
    logger.info("Applying temporal split...")
    df_train, df_val, df_test = temporal_split(df_processed)

    if TARGET_COL not in df_train.columns:
        logger.error(f"Target column '{TARGET_COL}' not found in processed data.")
        sys.exit(1)

    y_train = df_train.pop(TARGET_COL)
    y_val = df_val.pop(TARGET_COL)
    y_test = df_test.pop(TARGET_COL)  # noqa: F841

    X_train = df_train
    X_val = df_val
    X_test = df_test

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train = X_train[numeric_cols]
    X_val = X_val[numeric_cols]
    X_test = X_test[numeric_cols]

    tscv = get_time_series_cv(n_splits=5)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Random_Forest_Models")

    with mlflow.start_run(run_name="RF_Baseline_and_Tuning"):
        # 3. Baseline RF
        logger.info("--- Training Baseline Random Forest ---")
        rf_base = RandomForestRegressor(random_state=42)
        rf_base.fit(X_train, y_train)
        preds_base = rf_base.predict(X_val)
        val_rmse_base = np.sqrt(mean_squared_error(y_val, preds_base))
        logger.info(f"Baseline RF Val RMSE: {val_rmse_base:.4f}")
        mlflow.log_metric("Baseline_Val_RMSE", val_rmse_base)

        # 4. Hyperparameter Tuning
        logger.info("--- Hyperparameter Tuning (RandomizedSearchCV) ---")
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.3, 0.5],
        }

        rf_tune = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=rf_tune,
            param_distributions=param_dist,
            n_iter=20,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        random_search.fit(X_train, y_train)
        best_rf = random_search.best_estimator_

        logger.info(f"Best Parameters: {random_search.best_params_}")
        for param_name, param_val in random_search.best_params_.items():
            mlflow.log_param(f"best_{param_name}", param_val)

        preds_best = best_rf.predict(X_val)
        val_rmse_best = np.sqrt(mean_squared_error(y_val, preds_best))
        val_mae_best = mean_absolute_error(y_val, preds_best)
        val_r2_best = r2_score(y_val, preds_best)

        logger.info(
            f"Tuned RF Val RMSE: {val_rmse_best:.4f}, MAE: {val_mae_best:.4f}, "
            f"R2: {val_r2_best:.4f}"
        )
        mlflow.log_metric("Tuned_Val_RMSE", val_rmse_best)
        mlflow.log_metric("Tuned_Val_MAE", val_mae_best)
        mlflow.log_metric("Tuned_Val_R2", val_r2_best)

        # 5. Visualizations
        plot_feature_importance(best_rf, X_train, y_train, viz_dir)
        plot_learning_curve_rf(best_rf, X_train, y_train, tscv, viz_dir)
        plot_oob_error(X_train, y_train, viz_dir)

        # 6. Save Model
        best_model_path = os.path.join(model_dir, "rf_best.pkl")
        joblib.dump(best_rf, best_model_path)
        logger.info(f"Saved best RF model to {best_model_path}")

        # Save to mlflow as well
        mlflow.sklearn.log_model(best_rf, "random_forest_model")


if __name__ == "__main__":
    main()
