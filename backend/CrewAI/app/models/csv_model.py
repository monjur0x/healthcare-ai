"""CSV-based prediction model for healthcare data."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from typing import Any
import logging

logger = logging.getLogger(__name__)


class CSVModel:
    """Machine learning model for CSV-based disease prediction."""

    # Feature columns expected in CSV
    EXPECTED_FEATURES = [
        "age", "bmi", "blood_pressure_systolic", "blood_pressure_diastolic",
        "glucose", "heart_rate", "creatinine", "hemoglobin",
        "cholesterol", "white_blood_cells", "platelet_count"
    ]

    # Optional features
    OPTIONAL_FEATURES = [
        "medication_count", "chronic_conditions", "family_history_score",
        "smoking_status", "alcohol_consumption", "exercise_frequency"
    ]

    def __init__(self):
        """Initialize the CSV prediction model."""
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.is_trained = False
        self.feature_names = []
        self._initialize_mock_model()

    def _initialize_mock_model(self):
        """Initialize with synthetic data for demonstration."""
        np.random.seed(42)
        n_samples = 1000

        # Generate synthetic training data
        X = np.column_stack([
            np.random.uniform(18, 90, n_samples),      # age
            np.random.uniform(16, 45, n_samples),       # bmi
            np.random.uniform(90, 180, n_samples),      # systolic BP
            np.random.uniform(60, 120, n_samples),      # diastolic BP
            np.random.uniform(70, 200, n_samples),      # glucose
            np.random.uniform(50, 120, n_samples),      # heart rate
            np.random.uniform(0.5, 3.0, n_samples),    # creatinine
            np.random.uniform(10, 18, n_samples),       # hemoglobin
            np.random.uniform(100, 300, n_samples),     # cholesterol
            np.random.uniform(3000, 15000, n_samples),  # WBC
            np.random.uniform(150000, 400000, n_samples) # platelets
        ])

        # Generate labels based on simple rules
        y = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            risk_score = 0
            if X[i, 0] > 60: risk_score += 1
            if X[i, 1] > 30: risk_score += 1
            if X[i, 2] > 140: risk_score += 1
            if X[i, 4] > 126: risk_score += 2
            if X[i, 6] > 1.5: risk_score += 1
            if X[i, 8] > 240: risk_score += 1

            if risk_score >= 3:
                y[i] = 2  # High risk
            elif risk_score >= 1:
                y[i] = 1  # Medium risk
            else:
                y[i] = 0  # Low risk

        self.feature_names = self.EXPECTED_FEATURES[:X.shape[1]]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info("CSV model initialized with synthetic data")

    def extract_features(self, csv_data: pd.DataFrame) -> dict[str, Any]:
        """Extract and validate features from CSV data."""
        features = {}
        for col in self.EXPECTED_FEATURES:
            if col in csv_data.columns:
                features[col] = float(csv_data[col].mean())
            else:
                features[col] = self._get_default_value(col)

        # Add optional features if present
        for col in self.OPTIONAL_FEATURES:
            if col in csv_data.columns:
                features[col] = float(csv_data[col].mean())

        return features

    def _get_default_value(self, feature: str) -> float:
        """Get default value for missing features."""
        defaults = {
            "age": 45.0, "bmi": 25.0, "blood_pressure_systolic": 120.0,
            "blood_pressure_diastolic": 80.0, "glucose": 100.0,
            "heart_rate": 72.0, "creatinine": 1.0, "hemoglobin": 14.0,
            "cholesterol": 200.0, "white_blood_cells": 7000.0,
            "platelet_count": 250000.0
        }
        return defaults.get(feature, 0.0)

    def predict(self, csv_data: pd.DataFrame) -> dict[str, Any]:
        """Make prediction based on CSV data."""
        features = self.extract_features(csv_data)
        feature_array = np.array([features[f] for f in self.EXPECTED_FEATURES]).reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_array)

        prediction = self.model.predict(feature_scaled)[0]
        probabilities = self.model.predict_proba(feature_scaled)[0]

        risk_levels = {0: "low", 1: "medium", 2: "high"}
        risk_labels = {
            0: "Low risk - Continue healthy lifestyle",
            1: "Medium risk - Lifestyle modifications recommended",
            2: "High risk - Medical intervention recommended"
        }

        # Map predictions to conditions
        conditions = self._map_to_conditions(features, prediction)

        return {
            "risk_level": risk_levels.get(prediction, "unknown"),
            "confidence": float(np.max(probabilities)),
            "risk_score": float(np.mean([features.get("glucose", 100) / 200,
                                         features.get("blood_pressure_systolic", 120) / 200,
                                         features.get("bmi", 25) / 40])),
            "primary_condition": conditions["primary"],
            "secondary_conditions": conditions["secondary"],
            "abnormal_biomarkers": self._identify_abnormal(features),
            "risk_assessment": risk_labels.get(prediction, "Unknown risk level"),
            "features_used": features
        }

    def _map_to_conditions(self, features: dict, prediction: int) -> dict:
        """Map features to potential conditions."""
        conditions = {"primary": "General Health Assessment", "secondary": []}

        if features.get("glucose", 100) > 126:
            conditions["primary"] = "Type 2 Diabetes Mellitus"
            conditions["secondary"].append("Metabolic Syndrome")
        elif features.get("blood_pressure_systolic", 120) > 140:
            conditions["primary"] = "Hypertension"
            conditions["secondary"].append("Cardiovascular Risk")
        elif features.get("bmi", 25) > 30:
            conditions["primary"] = "Obesity"
            conditions["secondary"].append("Metabolic Risk")
        elif features.get("cholesterol", 200) > 240:
            conditions["primary"] = "Hyperlipidemia"
            conditions["secondary"].append("Cardiovascular Risk")
        elif features.get("creatinine", 1.0) > 1.5:
            conditions["primary"] = "Chronic Kidney Disease Risk"
            conditions["secondary"].append("Renal Function Impairment")

        return conditions

    def _identify_abnormal(self, features: dict) -> list[dict]:
        """Identify abnormal biomarker values."""
        abnormal = []
        thresholds = {
            "glucose": (70, 126, "mg/dL"),
            "blood_pressure_systolic": (90, 140, "mmHg"),
            "blood_pressure_diastolic": (60, 90, "mmHg"),
            "bmi": (18.5, 30, "kg/m²"),
            "heart_rate": (60, 100, "bpm"),
            "creatinine": (0.6, 1.5, "mg/dL"),
            "hemoglobin": (12, 17, "g/dL"),
            "cholesterol": (100, 240, "mg/dL")
        }

        for biomarker, (low, high, unit) in thresholds.items():
            value = features.get(biomarker)
            if value is not None:
                if value < low:
                    abnormal.append({
                        "biomarker": biomarker,
                        "value": value,
                        "status": "low",
                        "normal_range": f"{low}-{high} {unit}"
                    })
                elif value > high:
                    abnormal.append({
                        "biomarker": biomarker,
                        "value": value,
                        "status": "high",
                        "normal_range": f"{low}-{high} {unit}"
                    })

        return abnormal
