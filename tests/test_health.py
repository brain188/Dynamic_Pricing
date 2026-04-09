from backend.app.models.schemas import HealthResponse


def test_health_check(client):
    """
    Test the GET /health endpoint.
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data
    assert "uptime_seconds" in data
    assert "timestamp" in data

    # Validate against Pydantic schema
    HealthResponse(**data)
