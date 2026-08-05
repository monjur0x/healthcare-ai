"""Prediction fusion module for combining CSV and image model outputs."""

from typing import Any
import logging

logger = logging.getLogger(__name__)


class PredictionFusion:
    """Fuses predictions from multiple models into a unified diagnosis."""

    def __init__(self, csv_weight: float = 0.5, image_weight: float = 0.5):
        """Initialize fusion weights."""
        self.csv_weight = csv_weight
        self.image_weight = image_weight

    def fuse_predictions(
        self,
        csv_prediction: dict[str, Any] | None,
        image_prediction: dict[str, Any] | None,
        input_type: str
    ) -> dict[str, Any]:
        """Fuse predictions from available models based on input type."""
        if input_type == "csv" and csv_prediction:
            return self._csv_only_fusion(csv_prediction)
        elif input_type == "image" and image_prediction:
            return self._image_only_fusion(image_prediction)
        elif input_type == "csv_image" and csv_prediction and image_prediction:
            return self._combined_fusion(csv_prediction, image_prediction)
        else:
            return self._default_prediction()

    def _csv_only_fusion(self, csv_pred: dict) -> dict:
        """Generate fused prediction from CSV data only."""
        risk_score = csv_pred.get("risk_score", 0.5)
        conditions = csv_pred.get("primary_condition", "General Assessment")

        return {
            "primary_diagnosis": conditions,
            "secondary_diagnosis": ", ".join(csv_pred.get("secondary_conditions", [])) or "None identified",
            "confidence": round(min(csv_pred.get("confidence", 0.7), 0.95), 2),
            "severity": self._calculate_severity(risk_score),
            "risk_level": csv_pred.get("risk_level", "unknown"),
            "risk_score": round(risk_score, 3),
            "abnormal_biomarkers": csv_pred.get("abnormal_biomarkers", []),
            "model_used": "csv_analysis",
            "fusion_method": "csv_only"
        }

    def _image_only_fusion(self, image_pred: dict) -> dict:
        """Generate fused prediction from image data only."""
        confidence = image_pred.get("confidence", 0.8)

        return {
            "primary_diagnosis": image_pred.get("primary_finding", "Image Analysis Complete"),
            "secondary_diagnosis": "Differential diagnosis recommended",
            "confidence": round(min(confidence, 0.95), 2),
            "severity": self._image_severity(image_pred),
            "risk_level": self._image_risk_level(confidence),
            "risk_score": round(1 - confidence, 3),
            "image_type": image_pred.get("image_type", "unknown"),
            "anatomical_region": image_pred.get("anatomical_region", "Unknown"),
            "abnormalities": image_pred.get("abnormalities", []),
            "model_used": "image_analysis",
            "fusion_method": "image_only"
        }

    def _combined_fusion(self, csv_pred: dict, image_pred: dict) -> dict:
        """Fuse predictions from both CSV and image models."""
        csv_confidence = csv_pred.get("confidence", 0.7)
        image_confidence = image_pred.get("confidence", 0.8)

        # Weighted confidence
        fused_confidence = (
            self.csv_weight * csv_confidence +
            self.image_weight * image_confidence
        )

        # Combine conditions
        csv_condition = csv_pred.get("primary_condition", "")
        image_finding = image_pred.get("primary_finding", "")

        # Determine primary diagnosis (prioritize abnormal findings)
        if "Normal" not in image_finding and image_confidence > 0.8:
            primary = f"{csv_condition} with {image_finding}"
        elif csv_pred.get("risk_level") == "high":
            primary = csv_condition
        else:
            primary = csv_condition if csv_condition else image_finding

        return {
            "primary_diagnosis": primary,
            "secondary_diagnosis": ", ".join(csv_pred.get("secondary_conditions", [])) or "None identified",
            "confidence": round(min(fused_confidence, 0.95), 2),
            "severity": self._calculate_severity(
                (csv_pred.get("risk_score", 0.5) + (1 - image_confidence)) / 2
            ),
            "risk_level": self._combined_risk_level(csv_pred.get("risk_level", "low"), image_confidence),
            "risk_score": round(
                (csv_pred.get("risk_score", 0.5) * self.csv_weight +
                 (1 - image_confidence) * self.image_weight), 3
            ),
            "abnormal_biomarkers": csv_pred.get("abnormal_biomarkers", []),
            "image_findings": image_pred.get("findings", []),
            "model_used": "csv_image_fusion",
            "fusion_method": "combined_weighted"
        }

    def _calculate_severity(self, risk_score: float) -> str:
        """Calculate severity based on risk score."""
        if risk_score < 0.3:
            return "mild"
        elif risk_score < 0.6:
            return "moderate"
        elif risk_score < 0.8:
            return "severe"
        return "critical"

    def _image_severity(self, image_pred: dict) -> str:
        """Determine severity from image findings."""
        if "Normal" in image_pred.get("primary_finding", ""):
            return "mild"
        return "moderate"

    def _image_risk_level(self, confidence: float) -> str:
        """Determine risk level from image confidence."""
        if confidence > 0.9:
            return "low"
        elif confidence > 0.7:
            return "medium"
        return "high"

    def _combined_risk_level(self, csv_risk: str, image_confidence: float) -> str:
        """Combine risk levels from both sources."""
        risk_weights = {"low": 0, "medium": 1, "high": 2}
        csv_weight_val = risk_weights.get(csv_risk, 1)
        image_weight_val = 0 if image_confidence > 0.9 else (1 if image_confidence > 0.7 else 2)

        combined = (csv_weight_val * self.csv_weight + image_weight_val * self.image_weight)

        if combined < 0.5:
            return "low"
        elif combined < 1.2:
            return "medium"
        return "high"

    def _default_prediction(self) -> dict:
        """Return default prediction when no data is available."""
        return {
            "primary_diagnosis": "Insufficient Data for Diagnosis",
            "secondary_diagnosis": "Additional clinical data required",
            "confidence": 0.0,
            "severity": "unknown",
            "risk_level": "unknown",
            "risk_score": 0.0,
            "model_used": "none",
            "fusion_method": "default"
        }
