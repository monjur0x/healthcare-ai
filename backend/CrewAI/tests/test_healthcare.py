"""Test suite for Healthcare AI Backend."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
import io


client = TestClient(app)


class TestHealthEndpoints:
    """Test health and root endpoints."""

    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["status"] == "running"

    def test_health(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestCSVProcessing:
    """Test CSV data processing."""

    def test_preprocess_csv_valid(self):
        """Test preprocessing valid CSV data."""
        from app.utils.preprocessing import preprocess_csv_data

        csv_content = """age,bmi,blood_pressure_systolic,glucose
45,28.5,135,110
52,31.2,145,125
38,24.8,118,95"""

        result = preprocess_csv_data(csv_content)
        assert result["success"] is True
        assert result["statistics"]["total_rows"] == 3
        assert "key_metrics" in result

    def test_preprocess_csv_empty(self):
        """Test preprocessing empty CSV."""
        from app.utils.preprocessing import preprocess_csv_data

        result = preprocess_csv_data("")
        assert result["success"] is False


class TestRiskCalculation:
    """Test risk assessment calculations."""

    def test_low_risk(self):
        """Test low risk calculation."""
        from app.utils.risk import calculate_risk_score

        result = calculate_risk_score(
            age=30,
            bmi=22.0,
            blood_pressure_systolic=115,
            glucose=90,
            cholesterol=180
        )
        assert result["risk_category"] == "low"
        assert result["risk_score"] < 0.3

    def test_high_risk(self):
        """Test high risk calculation."""
        from app.utils.risk import calculate_risk_score

        result = calculate_risk_score(
            age=70,
            bmi=35.0,
            blood_pressure_systolic=165,
            glucose=180,
            cholesterol=280,
            smoking_status="smoker"
        )
        assert result["risk_category"] in ["high", "very_high"]
        assert result["risk_score"] > 0.5


class TestAPIValidation:
    """Test API input validation."""

    def test_missing_files(self):
        """Test request without required files."""
        response = client.post(
            "/api/run-healthcare-crew",
            data={
                "patient_name": "Test Patient",
                "patient_id": "P123",
                "patient_age": 45
            }
        )
        assert response.status_code == 400

    def test_invalid_age(self):
        """Test request with invalid age."""
        csv_content = b"age,bmi\n45,25"
        response = client.post(
            "/api/run-healthcare-crew",
            data={
                "patient_name": "Test Patient",
                "patient_id": "P123",
                "patient_age": -5
            },
            files={"csv_file": ("test.csv", io.BytesIO(csv_content), "text/csv")}
        )
        # Should fail validation
        assert response.status_code in [400, 422]
