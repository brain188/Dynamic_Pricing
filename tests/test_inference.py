import pandas as pd
import pytest

from backend.app.models.schemas import PriceResponse


def test_inference_preprocess(inference_service, sample_ride_features):
    """
    Test the preprocessing step of the inference service.
    """
    df = inference_service.preprocess(sample_ride_features)

    assert isinstance(df, pd.DataFrame)
    # The processed dataframe should have exactly the number of features
    # the model expects. In our trained model, it was 26 features.
    assert df.shape[1] == 26
    assert "demand_supply_ratio" in df.columns
    assert "hour_sin" in df.columns


def test_inference_predict_xgboost(inference_service, sample_ride_features):
    """
    Test the full XGBoost prediction logic.
    """
    response = inference_service.predict(sample_ride_features)

    assert isinstance(response, PriceResponse)
    assert response.predicted_fare > 0
    assert 0.5 <= response.price_multiplier <= 3.0
    assert (
        response.confidence_lower
        <= response.predicted_fare
        <= response.confidence_upper
    )


def test_inference_predict_ppo(inference_service, sample_ride_features):
    """
    Test the RL PPO prediction logic.
    """
    if inference_service.ppo_model is None:
        pytest.skip("PPO model not found in models/ directory")

    response = inference_service.predict_rl(sample_ride_features)

    assert isinstance(response, PriceResponse)
    assert response.predicted_fare > 0
    # PPO base_fare was 50 in trainer, here it depends on sample_ride_features.base_fare
    # But usually it should return a multiplier in [0.5, 3.0]
    assert 0.5 <= response.price_multiplier <= 3.1  # Small buffer
    assert "-PPO" in response.model_version


def test_inference_prediction_interval(inference_service):
    """
    Test the internal confidence interval calculation.
    """
    lower, upper = inference_service._compute_prediction_interval(0.5)
    assert lower < 0.5 < upper
    assert upper - lower == pytest.approx(1.96 * 2 * inference_service.rmse)
