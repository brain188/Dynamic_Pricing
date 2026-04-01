import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate

from src.utils.logger import logger


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared"""
    return r2_score(y_true, y_pred)


def compute_adj_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """Adjusted R-squared: 1 - (1-R2)(n-1)/(n-p-1)"""
    r2 = compute_r2(y_true, y_pred)
    n = len(y_true)
    p = n_features
    if n - p - 1 <= 0:
        logger.warning("Degrees of freedom <= 0; returning raw R2.")
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def evaluate_model(model, X, y, cv, n_features: int) -> dict:
    """
    Run cross-validation and return metrics summary.

    Args:
        model: The fitted model or pipeline to evaluate.
        X: Feature matrix.
        y: Target vector.
        cv: Cross-validation strategy.
        n_features: Number of features used (for Adjusted R2).

    Returns:
        dict: Mean and std of metrics across folds.
    """
    logger.info(f"Evaluating model: {type(model).__name__}")

    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }

    cv_results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=False
    )

    # Calculate RMSE from MSE
    rmse_scores = np.sqrt(-cv_results["test_mse"])
    mae_scores = -cv_results["test_mae"]
    r2_scores = cv_results["test_r2"]

    metrics_summary = {
        "RMSE": (np.mean(rmse_scores), np.std(rmse_scores)),
        "MAE": (np.mean(mae_scores), np.std(mae_scores)),
        "R2": (np.mean(r2_scores), np.std(r2_scores)),
    }

    logger.info(
        f"CV Metrics: RMSE={metrics_summary['RMSE'][0]:.4f} "
        f"(+/- {metrics_summary['RMSE'][1]:.4f})"
    )
    return metrics_summary


def check_statistical_significance(
    errors_dict: dict, alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform Wilcoxon signed-rank test on absolute errors between pairs of models.

    Args:
        errors_dict: Dictionary mapping model names to their array of absolute errors.
        alpha: Significance level.

    Returns:
        pd.DataFrame: Matrix of p-values for model pairwise comparisons.
    """
    models = list(errors_dict.keys())
    n_models = len(models)

    p_values = np.zeros((n_models, n_models))
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i == j:
                p_values[i, j] = 1.0
            else:
                err1 = errors_dict[model1]
                err2 = errors_dict[model2]

                # Check for identical errors to avoid ValueError
                if np.array_equal(err1, err2):
                    p_values[i, j] = 1.0
                else:
                    _, p_val = stats.wilcoxon(err1, err2)
                    p_values[i, j] = p_val

    df_pvals = pd.DataFrame(p_values, index=models, columns=models)
    return df_pvals


def plot_model_comparison(metrics_df: pd.DataFrame, save_path: str = None):
    """
    Generate bar charts with error bars for model comparison (RMSE, MAE, R2).

    Args:
        metrics_df: DataFrame containing model performance metrics.
                    Must contain columns: 'Model', 'RMSE', 'MAE', 'R2'.
        save_path: Optional path to save the plot.
    """
    metrics_to_plot = ["RMSE", "MAE", "R2"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(x="Model", y=metric, data=metrics_df, ax=axes[i], palette="viridis")
        axes[i].set_title(f"Model Comparison: {metric}")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved model comparison plot to {save_path}")

    return fig


def plot_residual_diagnostic(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: str = None
):
    """
    Create actual vs. predicted and residual diagnostic plots.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        model_name: Name of the model.
        save_path: Optional path to save the plot.
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Actual vs Predicted
    axes[0].scatter(y_pred, y_true, alpha=0.3, color="blue")
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    axes[0].set_xlabel("Predicted Fare")
    axes[0].set_ylabel("Actual Fare")
    axes[0].set_title(f"{model_name}: Actual vs Predicted")

    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.3, color="red")
    axes[1].axhline(y=0, color="black", linestyle="--")
    axes[1].set_xlabel("Predicted Fare")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title(f"{model_name}: Residuals vs Predicted")

    # Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title(f"{model_name}: Q-Q Plot of Residuals")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved residual diagnostic plot for {model_name} to {save_path}")

    return fig


def generate_shap_plots(
    model, X_train: pd.DataFrame, X_test: pd.DataFrame, output_dir: str
) -> None:
    """
    Generate and save a full suite of SHAP explainability plots for a tree-based model.

    Produces:
        - summary_beeswarm.png  : SHAP beeswarm summary (global impact + dir)
        - summary_bar.png       : SHAP bar chart (mean |SHAP|, global ranking)
        - dependence_{feat}.png : Dependence plots for top-5 features
        - interaction_demand_hour.png : SHAP interaction values
          (demand_supply_ratio x hour_sin)  # noqa: E501
        - pdp_{feat}.png        : Partial Dependence Plots for key features
        - waterfall_sample_{n}.png : Waterfall plots for
          low / median / high fare samples  # noqa: E501

    Args:
        model: A fitted tree-based model (XGBoost, LightGBM, Random Forest).
        X_train: Training feature matrix (used to build background dataset
                 for KernelExplainer fallback).
        X_test: Test feature matrix to compute SHAP values on.
        output_dir: Directory where all plots are saved.
    """
    import shap

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating SHAP plots → {output_dir}")

    # ── 1. Explainer ────────────────────────────────────────────────────────────
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)  # Explanation object (shap >= 0.40)
        shap_matrix = shap_values.values  # (n_samples, n_features)
    except Exception as e:
        logger.warning(
            f"TreeExplainer failed ({e}); falling back to KernelExplainer (slower)."
        )
        background = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_matrix = explainer.shap_values(X_test)
        shap_values = shap_matrix  # raw array for older API

    feature_names = list(X_test.columns)
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    top5_idx = np.argsort(mean_abs_shap)[::-1][:5]
    top5_features = [feature_names[i] for i in top5_idx]
    logger.info(f"Top-5 SHAP features: {top5_features}")

    # ── 2. Beeswarm summary ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    plt.tight_layout()
    beeswarm_path = os.path.join(output_dir, "summary_beeswarm.png")
    plt.savefig(beeswarm_path, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved {beeswarm_path}")

    # ── 3. Bar summary ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "summary_bar.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved {bar_path}")

    # ── 4. Dependence plots for top-5 features ─────────────────────────────────
    for feat in top5_features:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feat, shap_matrix, X_test, interaction_index="auto", ax=ax, show=False
        )
        plt.tight_layout()
        safe_feat = feat.replace(" ", "_").replace("/", "_")
        dep_path = os.path.join(output_dir, f"dependence_{safe_feat}.png")
        plt.savefig(dep_path, bbox_inches="tight")
        plt.close("all")
        logger.info(f"Saved {dep_path}")

    # ── 5. SHAP interaction (demand_supply_ratio × hour_sin) ───────────────────
    if "demand_supply_ratio" in feature_names and "hour_sin" in feature_names:
        try:
            shap_interact = explainer.shap_interaction_values(X_test)
            feat_a = feature_names.index("demand_supply_ratio")
            feat_b = feature_names.index("hour_sin")
            interact_vals = shap_interact[:, feat_a, feat_b]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(
                X_test["demand_supply_ratio"],
                interact_vals,
                c=X_test["hour_sin"],
                cmap="coolwarm",
                alpha=0.5,
            )
            ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
            ax.set_xlabel("demand_supply_ratio")
            ax.set_ylabel("SHAP interaction\n(demand_supply_ratio × hour_sin)")
            ax.set_title("SHAP Interaction: demand_supply_ratio × hour_sin")
            plt.colorbar(ax.collections[0], ax=ax, label="hour_sin")
            plt.tight_layout()
            interact_path = os.path.join(output_dir, "interaction_demand_hour.png")
            plt.savefig(interact_path, bbox_inches="tight")
            plt.close("all")
            logger.info(f"Saved {interact_path}")
        except Exception as e:
            logger.warning(f"SHAP interaction computation skipped: {e}")
    else:
        logger.warning(
            "demand_supply_ratio or hour_sin not in feature set; "
            "skipping interaction plot."  # noqa: E501
        )

    # ── 6. Partial Dependence Plots ────────────────────────────────────────────
    pdp_features = [
        f
        for f in ["demand_supply_ratio", "Expected_Ride_Duration", "hour_sin"]
        if f in feature_names
    ]
    for feat in pdp_features:
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.partial_dependence_plot(
                feat,
                model.predict,
                X_test,
                ice=False,
                model_expected_value=True,
                feature_expected_value=True,
                ax=ax,
                show=False,
            )
            plt.tight_layout()
            safe_feat = feat.replace(" ", "_").replace("/", "_")
            pdp_path = os.path.join(output_dir, f"pdp_{safe_feat}.png")
            plt.savefig(pdp_path, bbox_inches="tight")
            plt.close("all")
            logger.info(f"Saved {pdp_path}")
        except Exception as e:
            logger.warning(f"PDP for '{feat}' skipped: {e}")

    # ── 7. Waterfall plots: low / median / high fare samples ───────────────────
    y_pred_all = model.predict(X_test)
    low_idx = int(np.argmin(y_pred_all))
    median_idx = int(np.argsort(y_pred_all)[len(y_pred_all) // 2])
    high_idx = int(np.argmax(y_pred_all))

    for label, idx in [("low", low_idx), ("median", median_idx), ("high", high_idx)]:
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.waterfall_plot(shap_values[idx], show=False)
            plt.title(f"SHAP Waterfall — {label.capitalize()} Fare Sample (idx={idx})")
            plt.tight_layout()
            wf_path = os.path.join(output_dir, f"waterfall_sample_{label}.png")
            plt.savefig(wf_path, bbox_inches="tight")
            plt.close("all")
            logger.info(f"Saved {wf_path}")
        except Exception as e:
            logger.warning(f"Waterfall plot for '{label}' sample skipped: {e}")

    logger.info("SHAP plot generation complete.")
    return top5_features
