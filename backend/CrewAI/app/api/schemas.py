"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class InputType(str, Enum):
    """Supported input types."""
    CSV = "csv"
    IMAGE = "image"
    CSV_IMAGE = "csv_image"


class PatientInfo(BaseModel):
    """Patient information model."""
    name: str = Field(..., description="Patient full name")
    id: str = Field(..., description="Patient ID")
    age: int = Field(..., ge=0, le=150, description="Patient age")


class Prediction(BaseModel):
    """Disease prediction model."""
    primary_diagnosis: str = Field(default="", description="Primary diagnosis")
    secondary_diagnosis: str = Field(default="", description="Secondary diagnosis")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    severity: str = Field(default="unknown", description="Severity level")
    risk_level: str = Field(default="unknown", description="Risk level")


class Evidence(BaseModel):
    """Clinical evidence model."""
    source: str = Field(default="", description="Evidence source")
    summary: str = Field(default="", description="Evidence summary")
    reference: str = Field(default="", description="Reference link/citation")


class HealthcareResponse(BaseModel):
    """Complete healthcare analysis response."""
    patient: PatientInfo
    input_type: str
    patient_summary: str
    prediction: Prediction
    clinical_findings: list[str] = Field(default_factory=list)
    image_findings: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    follow_up: list[str] = Field(default_factory=list)
    monitoring_plan: list[str] = Field(default_factory=list)
    explanation: str
    limitations: str
    doctor_notice: str = "This report is AI-assisted. Final diagnosis must be made by a licensed physician."

    class Config:
        json_schema_extra = {
            "example": {
                "patient": {"name": "John Doe", "id": "P12345", "age": 45},
                "input_type": "csv",
                "patient_summary": "Patient presents with elevated biomarkers...",
                "prediction": {
                    "primary_diagnosis": "Type 2 Diabetes",
                    "secondary_diagnosis": "Hypertension",
                    "confidence": 0.87,
                    "severity": "moderate",
                    "risk_level": "medium"
                },
                "clinical_findings": ["Elevated glucose levels", "High BMI"],
                "image_findings": [],
                "evidence": [{"source": "WHO", "summary": "...", "reference": "..."}],
                "recommendations": ["Lifestyle modifications", "Metformin therapy"],
                "follow_up": ["Follow-up in 3 months"],
                "monitoring_plan": ["Monthly glucose monitoring"],
                "explanation": "Based on elevated glucose and BMI...",
                "limitations": "AI analysis has limitations...",
                "doctor_notice": "This report is AI-assisted..."
            }
        }
