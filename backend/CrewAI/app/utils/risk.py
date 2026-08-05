"""Risk assessment utilities."""

from typing import Any
import logging

logger = logging.getLogger(__name__)


def calculate_risk_score(
    age: int = 45,
    bmi: float = 25.0,
    blood_pressure_systolic: float = 120.0,
    glucose: float = 100.0,
    cholesterol: float = 200.0,
    smoking_status: str = "non_smoker",
    family_history: bool = False,
    **kwargs
) -> dict[str, Any]:
    """Calculate comprehensive risk score.

    Args:
        age: Patient age
        bmi: Body Mass Index
        blood_pressure_systolic: Systolic blood pressure
        glucose: Fasting glucose level
        cholesterol: Total cholesterol
        smoking_status: Smoking status
        family_history: Family history of chronic disease

    Returns:
        Risk assessment dictionary
    """
    risk_factors = []
    risk_score = 0.0

    # Age risk
    if age > 65:
        risk_score += 0.25
        risk_factors.append("Advanced age (>65)")
    elif age > 50:
        risk_score += 0.15
        risk_factors.append("Age >50")
    elif age > 40:
        risk_score += 0.05

    # BMI risk
    if bmi > 35:
        risk_score += 0.20
        risk_factors.append(f"Severe obesity (BMI: {bmi:.1f})")
    elif bmi > 30:
        risk_score += 0.15
        risk_factors.append(f"Obesity (BMI: {bmi:.1f})")
    elif bmi > 25:
        risk_score += 0.05

    # Blood pressure risk
    if blood_pressure_systolic > 160:
        risk_score += 0.20
        risk_factors.append(f"Severe hypertension (BP: {blood_pressure_systolic:.0f})")
    elif blood_pressure_systolic > 140:
        risk_score += 0.15
        risk_factors.append(f"Hypertension (BP: {blood_pressure_systolic:.0f})")
    elif blood_pressure_systolic > 130:
        risk_score += 0.08

    # Glucose risk
    if glucose > 200:
        risk_score += 0.20
        risk_factors.append(f"Severe hyperglycemia (Glucose: {glucose:.0f})")
    elif glucose > 126:
        risk_score += 0.15
        risk_factors.append(f"Diabetes range (Glucose: {glucose:.0f})")
    elif glucose > 100:
        risk_score += 0.05

    # Cholesterol risk
    if cholesterol > 300:
        risk_score += 0.15
        risk_factors.append(f"Severe hyperlipidemia (Cholesterol: {cholesterol:.0f})")
    elif cholesterol > 240:
        risk_score += 0.10
        risk_factors.append(f"High cholesterol (Cholesterol: {cholesterol:.0f})")
    elif cholesterol > 200:
        risk_score += 0.03

    # Smoking risk
    if smoking_status == "smoker":
        risk_score += 0.15
        risk_factors.append("Current smoker")
    elif smoking_status == "former":
        risk_score += 0.05
        risk_factors.append("Former smoker")

    # Family history risk
    if family_history:
        risk_score += 0.10
        risk_factors.append("Positive family history")

    # Cap risk score at 1.0
    risk_score = min(risk_score, 1.0)

    # Determine risk category
    if risk_score < 0.2:
        risk_category = "low"
        risk_color = "green"
    elif risk_score < 0.4:
        risk_category = "mild"
        risk_color = "yellow"
    elif risk_score < 0.6:
        risk_category = "moderate"
        risk_color = "orange"
    elif risk_score < 0.8:
        risk_category = "high"
        risk_color = "red"
    else:
        risk_category = "very_high"
        risk_color = "dark_red"

    # Generate monitoring recommendations
    monitoring_schedule = generate_monitoring_schedule(risk_category, risk_factors)

    return {
        "risk_score": round(risk_score, 3),
        "risk_category": risk_category,
        "risk_color": risk_color,
        "risk_factors": risk_factors,
        "risk_percentage": round(risk_score * 100, 1),
        "monitoring_schedule": monitoring_schedule,
        "risk_stratification": {
            "cardiovascular": assess_cardiovascular_risk(age, blood_pressure_systolic, cholesterol, smoking_status),
            "metabolic": assess_metabolic_risk(bmi, glucose),
            "overall": risk_category
        }
    }


def assess_cardiovascular_risk(
    age: int,
    bp: float,
    cholesterol: float,
    smoking: str
) -> dict[str, Any]:
    """Assess cardiovascular risk."""
    score = 0

    # Simplified Framingham-like scoring
    if age > 70: score += 4
    elif age > 60: score += 3
    elif age > 50: score += 2
    elif age > 40: score += 1

    if bp > 160: score += 3
    elif bp > 140: score += 2
    elif bp > 130: score += 1

    if cholesterol > 280: score += 3
    elif cholesterol > 240: score += 2
    elif cholesterol > 200: score += 1

    if smoking == "smoker": score += 2
    elif smoking == "former": score += 1

    if score >= 7:
        risk = "high"
    elif score >= 4:
        risk = "moderate"
    else:
        risk = "low"

    return {"score": score, "risk": risk}


def assess_metabolic_risk(bmi: float, glucose: float) -> dict[str, Any]:
    """Assess metabolic risk."""
    score = 0

    if bmi > 35: score += 3
    elif bmi > 30: score += 2
    elif bmi > 25: score += 1

    if glucose > 200: score += 3
    elif glucose > 126: score += 2
    elif glucose > 100: score += 1

    if score >= 5:
        risk = "high"
    elif score >= 3:
        risk = "moderate"
    else:
        risk = "low"

    return {"score": score, "risk": risk}


def generate_monitoring_schedule(risk_category: str, risk_factors: list[str]) -> list[dict]:
    """Generate monitoring schedule based on risk category."""
    schedule = []

    base_schedules = {
        "low": [
            {"test": "Annual physical examination", "frequency": "Yearly"},
            {"test": "Blood pressure check", "frequency": "Annually"},
            {"test": "Lipid panel", "frequency": "Every 5 years"},
            {"test": "Blood glucose", "frequency": "Every 3 years"}
        ],
        "mild": [
            {"test": "Physical examination", "frequency": "Every 6-12 months"},
            {"test": "Blood pressure monitoring", "frequency": "Every 6 months"},
            {"test": "Lipid panel", "frequency": "Every 2-3 years"},
            {"test": "Blood glucose", "frequency": "Annually"}
        ],
        "moderate": [
            {"test": "Medical consultation", "frequency": "Every 3-6 months"},
            {"test": "Blood pressure monitoring", "frequency": "Monthly"},
            {"test": "Lipid panel", "frequency": "Every 6-12 months"},
            {"test": "Blood glucose", "frequency": "Every 3-6 months"},
            {"test": "Kidney function tests", "frequency": "Every 6 months"}
        ],
        "high": [
            {"test": "Medical consultation", "frequency": "Monthly"},
            {"test": "Blood pressure monitoring", "frequency": "Weekly"},
            {"test": "Comprehensive metabolic panel", "frequency": "Monthly"},
            {"test": "HbA1c", "frequency": "Every 3 months"},
            {"test": "Cardiac evaluation", "frequency": "As recommended"}
        ],
        "very_high": [
            {"test": "Immediate medical consultation", "frequency": "Urgent"},
            {"test": "Continuous monitoring recommended", "frequency": "Daily"},
            {"test": "Frequent lab work", "frequency": "Weekly"},
            {"test": "Specialist referral", "frequency": "As needed"}
        ]
    }

    schedule = base_schedules.get(risk_category, base_schedules["moderate"])

    # Add specific tests based on risk factors
    risk_factor_lower = [f.lower() for f in risk_factors]

    if any("glucose" in f or "diabetes" in f for f in risk_factor_lower):
        schedule.append({"test": "HbA1c", "frequency": "Every 3 months"})

    if any("cholesterol" in f or "lipid" in f for f in risk_factor_lower):
        schedule.append({"test": "Comprehensive lipid panel", "frequency": "Every 6 months"})

    if any("kidney" in f or "creatinine" in f for f in risk_factor_lower):
        schedule.append({"test": "Kidney function panel", "frequency": "Every 3 months"})

    return schedule
