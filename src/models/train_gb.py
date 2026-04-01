import os
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import lightgbm as lgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import optuna  # noqa: E402
import seaborn as sns  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor  # noqa: E402

# Add src to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.utils.data_loader import load_config  # noqa: E402
from src.utils.eval_utils import evaluate_model  # noqa: E402, F401
from src.utils.logger import logger  # noqa: E402
from src.utils.train_utils import get_time_series_cv, temporal_split  # noqa: E402


def plot_loss_curves(evals_result, model_name, viz_dir):
    logger.info(f"Plotting Loss Curves for {model_name}...")
    plt.figure(figsize=(10, 6))

    if model_name == "xgb":
        train_rmse = evals_result["validation_0"]["rmse"]
        val_rmse = evals_result["validation_1"]["rmse"]
    else:  # lightgbm
        train_rmse = evals_result["training"]["rmse"]
        val_rmse = evals_result["valid_1"]["rmse"]

    epochs = len(train_rmse)
    x_axis = range(0, epochs)

    plt.plot(x_axis, train_rmse, label="Train RMSE")
    plt.plot(x_axis, val_rmse, label="Validation RMSE")
    plt.legend()
    plt.title(f"{model_name.upper()} Loss Curves (RMSE vs Boosting Rounds)")
    plt.xlabel("Boosting Rounds")
    plt.ylabel("RMSE")
    plt.grid(True)

    img_path = os.path.join(viz_dir, f"{model_name}_loss_curves.png")
    plt.savefig(img_path)
    plt.close()

    mlflow.log_artifact(img_path)


def plot_feature_importance(model, X_train, model_name, viz_dir):
    logger.info(f"Plotting Feature Importances for {model_name}...")
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
    plt.title(f"Top 15 Feature Importances ({model_name.upper()})")
    plt.xlabel("Importance")
    plt.tight_layout()
    img_path = os.path.join(viz_dir, f"{model_name}_feature_importance.png")
    plt.savefig(img_path)
    plt.close()

    mlflow.log_artifact(img_path)


def optimize_xgb(X_train, y_train, tscv, n_trials=50):
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "importance_type": trial.suggest_categorical(
                "importance_type", ["weight", "gain", "cover"]
            ),
        }

        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

            model = XGBRegressor(**params, early_stopping_rounds=50, random_state=42)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            preds = model.predict(X_va)
            rmse = np.sqrt(mean_squared_error(y_va, preds))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def optimize_lgb(X_train, y_train, tscv, n_trials=50):
    def objective(trial):
        params = {
            "objective": "regression",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "importance_type": trial.suggest_categorical(
                "importance_type", ["split", "gain"]
            ),
            "verbose": -1,
        }

        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

            model = LGBMRegressor(**params, random_state=42)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )

            preds = model.predict(X_va)
            rmse = np.sqrt(mean_squared_error(y_va, preds))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    study = optuna.create_study(direction="minimize")
    # Set logging level to ERROR to reduce LightGBM verbosity in Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def main():
    config = load_config()  # noqa: F841
    TARGET_COL = "Historical_Cost_of_Ride"

    viz_dir = os.path.join(project_root, "visualization", "model_performance", "gb")
    model_dir = os.path.join(project_root, "models")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logger.info("Loading processed data...")
    processed_path = os.path.join(
        project_root, "data", "processed", "dynamic_pricing_processed.csv"
    )
    if not os.path.exists(processed_path):
        logger.error(
            f"Processed data not found at {processed_path}. Please run previous phases."
        )
        sys.exit(1)

    df_processed = pd.read_csv(processed_path)

    logger.info("Applying temporal split...")
    df_train, df_val, df_test = temporal_split(df_processed)

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
    mlflow.set_experiment("Gradient_Boosting_Models")

    n_trials = 50

    # ------------------ XGBOOST ------------------
    with mlflow.start_run(run_name="XGBoost_Tuning"):
        logger.info("--- Optimizing XGBoost ---")
        best_xgb_params = optimize_xgb(X_train, y_train, tscv, n_trials=n_trials)
        logger.info(f"Best XGB Params: {best_xgb_params}")

        # Log params
        mlflow.log_params({f"xgb_{k}": v for k, v in best_xgb_params.items()})

        # Train final model on X_train, eval on X_val for loss curves
        final_xgb = XGBRegressor(
            **best_xgb_params, early_stopping_rounds=50, random_state=42
        )
        final_xgb.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        preds_val = final_xgb.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, preds_val))
        val_mae = mean_absolute_error(y_val, preds_val)
        val_r2 = r2_score(y_val, preds_val)

        logger.info(
            f"Final XGB Val RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}"
        )
        mlflow.log_metric("Val_RMSE", val_rmse)
        mlflow.log_metric("Val_MAE", val_mae)
        mlflow.log_metric("Val_R2", val_r2)

        # Plot Loss Curves and Feature Importances
        evals_result = final_xgb.evals_result()
        plot_loss_curves(evals_result, "xgb", viz_dir)
        plot_feature_importance(final_xgb, X_train, "xgb", viz_dir)

        # Save Model
        xgb_path = os.path.join(model_dir, "xgboost_best.pkl")
        joblib.dump(final_xgb, xgb_path)
        logger.info(f"Saved best XGBoost model to {xgb_path}")
        mlflow.sklearn.log_model(final_xgb, "xgboost_model")

    # ------------------ LIGHTGBM ------------------
    with mlflow.start_run(run_name="LightGBM_Tuning"):
        logger.info("--- Optimizing LightGBM ---")
        best_lgb_params = optimize_lgb(X_train, y_train, tscv, n_trials=n_trials)
        logger.info(f"Best LGB Params: {best_lgb_params}")

        # Log params
        mlflow.log_params({f"lgb_{k}": v for k, v in best_lgb_params.items()})

        # Train final model
        final_lgb = LGBMRegressor(**best_lgb_params, random_state=42, verbose=-1)
        final_lgb.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.record_evaluation({}),
            ],
        )

        preds_val = final_lgb.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, preds_val))
        val_mae = mean_absolute_error(y_val, preds_val)
        val_r2 = r2_score(y_val, preds_val)

        logger.info(
            f"Final LGB Val RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}"
        )
        mlflow.log_metric("Val_RMSE", val_rmse)
        mlflow.log_metric("Val_MAE", val_mae)
        mlflow.log_metric("Val_R2", val_r2)

        # Extract evals result specifically for LightGBM
        # LightGBM stores it in model.evals_result_ after record_evaluation callback
        if hasattr(final_lgb, "evals_result_"):
            plot_loss_curves(final_lgb.evals_result_, "lgb", viz_dir)
        else:
            logger.warning("Could not extract evals_result from LightGBM for plotting.")

        plot_feature_importance(final_lgb, X_train, "lgb", viz_dir)

        # Save Model
        lgb_path = os.path.join(model_dir, "lgbm_best.pkl")
        joblib.dump(final_lgb, lgb_path)
        logger.info(f"Saved best LightGBM model to {lgb_path}")
        mlflow.sklearn.log_model(final_lgb, "lgbm_model")


if __name__ == "__main__":
    main()
