from backend.app.models.schemas import PriceResponse


def test_predict_success(client, sample_ride_features):
    """
    Test a successful prediction request.
    """
    response = client.post(
        "/api/v1/predict", json=sample_ride_features.model_dump(mode="json")
    )
    assert response.status_code == 200

    data = response.json()
    assert "predicted_fare" in data
    assert "request_id" in data
    assert "X-Request-ID" in response.headers

    # Validate against Pydantic schema
    PriceResponse(**data)


def test_predict_validation_error(client):
    """
    Test prediction with invalid input data (e.g., negative riders).
    """
    invalid_data = {
        "number_of_riders": -1,  # Should be >= 1
        "number_of_drivers": 5,
        "location_category": "Urban",
        "vehicle_type": "Premium",
        "customer_loyalty_status": "Gold",
        "number_of_past_rides": 50,
        "average_rating": 4.8,
        "time_of_booking": "2026-04-09T17:00:00",
        "expected_ride_duration": 15.5,
    }
    response = client.post("/api/v1/predict", json=invalid_data)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_missing_field(client):
    """
    Test prediction with missing required fields.
    """
    incomplete_data = {"number_of_riders": 2}
    response = client.post("/api/v1/predict", json=incomplete_data)
    assert response.status_code == 422


def test_predict_invalid_enum(client, sample_ride_features):
    """
    Test prediction with invalid enum values.
    """
    data = sample_ride_features.model_dump(mode="json")
    data["location_category"] = "Mars"  # Invalid location

    response = client.post("/api/v1/predict", json=data)
    assert response.status_code == 422
