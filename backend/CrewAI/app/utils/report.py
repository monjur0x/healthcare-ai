"""Report generation utilities."""

from typing import Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def generate_report(
    patient_info: dict,
    prediction: dict,
    evidence: list[dict],
    recommendations: list[str],
    risk_assessment: dict,
    clinical_findings: list[str] = None,
    image_findings: list[str] = None,
    **kwargs
) -> dict[str, Any]:
    """Generate comprehensive healthcare report.

    Args:
        patient_info: Patient information
        prediction: Disease prediction results
        evidence: Retrieved clinical evidence
        recommendations: Treatment recommendations
        risk_assessment: Risk assessment results
        clinical_findings: Clinical findings from CSV analysis
        image_findings: Findings from image analysis

    Returns:
        Complete healthcare report dictionary
    """
    try:
        # Build the report structure
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_version": "1.0",
                "ai_model": "HealthcareCrew v1.0",
                "disclaimer": "This report is AI-assisted and should be reviewed by a licensed physician."
            },
            "patient_summary": generate_patient_summary(patient_info, prediction, risk_assessment),
            "clinical_summary": {
                "findings": clinical_findings or [],
                "image_findings": image_findings or [],
                "diagnosis": prediction.get("primary_diagnosis", "Not determined"),
                "confidence": prediction.get("confidence", 0.0),
                "severity": prediction.get("severity", "unknown"),
                "risk_level": prediction.get("risk_level", "unknown")
            },
            "evidence_summary": {
                "total_sources": len(evidence),
                "key_findings": extract_key_findings(evidence),
                "clinical_references": evidence[:5]  # Top 5 references
            },
            "recommendations_summary": {
                "total_recommendations": len(recommendations),
                "priority_actions": get_priority_actions(recommendations),
                "all_recommendations": recommendations
            },
            "risk_summary": {
                "overall_risk": risk_assessment.get("risk_category", "unknown"),
                "risk_score": risk_assessment.get("risk_score", 0),
                "key_risk_factors": risk_assessment.get("risk_factors", []),
                "monitoring_frequency": get_monitoring_frequency(risk_assessment.get("risk_category", "moderate"))
            },
            "follow_up_plan": generate_follow_up_plan(prediction, risk_assessment),
            "limitations": generate_limitations_disclaimer()
        }

        logger.info(f"Generated report for patient {patient_info.get('name', 'Unknown')}")
        return report

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return {
            "error": str(e),
            "patient_summary": "Error generating report",
            "limitations": "Report generation encountered errors"
        }


def generate_patient_summary(
    patient_info: dict,
    prediction: dict,
    risk_assessment: dict
) -> str:
    """Generate patient summary text."""
    name = patient_info.get("name", "Patient")
    age = patient_info.get("age", "Unknown")
    diagnosis = prediction.get("primary_diagnosis", "under evaluation")
    risk = risk_assessment.get("risk_category", "unknown")

    summary = (
        f"Patient {name}, aged {age} years, has been evaluated. "
        f"Primary assessment indicates {diagnosis}. "
        f"Overall risk stratification: {risk}. "
    )

    risk_factors = risk_assessment.get("risk_factors", [])
    if risk_factors:
        summary += f"Key risk factors identified: {', '.join(risk_factors[:3])}. "

    summary += "Comprehensive analysis and recommendations follow."
    return summary


def extract_key_findings(evidence: list[dict]) -> list[str]:
    """Extract key findings from evidence list."""
    findings = []
    for item in evidence[:3]:
        source = item.get("source", "Unknown")
        summary = item.get("summary", "")
        if summary:
            findings.append(f"According to {source}: {summary[:100]}...")
    return findings


def get_priority_actions(recommendations: list[str]) -> list[str]:
    """Extract priority actions from recommendations."""
    priority_keywords = ["immediately", "urgent", "emergency", "critical", "priority"]
    priority_actions = []

    for rec in recommendations:
        if any(keyword in rec.lower() for keyword in priority_keywords):
            priority_actions.append(rec)

    if not priority_actions and recommendations:
        priority_actions = recommendations[:2]

    return priority_actions


def get_monitoring_frequency(risk_category: str) -> str:
    """Get monitoring frequency based on risk category."""
    frequencies = {
        "low": "Annual",
        "mild": "Every 6-12 months",
        "moderate": "Every 3-6 months",
        "high": "Monthly",
        "very_high": "Weekly"
    }
    return frequencies.get(risk_category, "As clinically indicated")


def generate_follow_up_plan(prediction: dict, risk_assessment: dict) -> list[dict]:
    """Generate follow-up plan."""
    plan = []
    risk_category = risk_assessment.get("risk_category", "moderate")

    # Standard follow-ups
    plan.append({
        "action": "Primary care follow-up",
        "timeframe": "Within 2 weeks" if risk_category in ["high", "very_high"] else "Within 4 weeks",
        "purpose": "Review findings and treatment plan"
    })

    # Condition-specific follow-ups
    diagnosis = prediction.get("primary_diagnosis", "").lower()

    if "diabetes" in diagnosis or "glucose" in diagnosis:
        plan.append({
            "action": "HbA1c testing",
            "timeframe": "3 months",
            "purpose": "Monitor glycemic control"
        })

    if "hypertension" in diagnosis or "blood pressure" in diagnosis:
        plan.append({
            "action": "Blood pressure monitoring",
            "timeframe": "1 month",
            "purpose": "Track BP response to treatment"
        })

    if "kidney" in diagnosis or "renal" in diagnosis:
        plan.append({
            "action": "Kidney function panel",
            "timeframe": "3 months",
            "purpose": "Monitor renal function"
        })

    # Default follow-up if no specific conditions
    if len(plan) == 1:
        plan.append({
            "action": "Comprehensive lab panel",
            "timeframe": "3 months",
            "purpose": "Monitor overall health status"
        })

    return plan


def generate_limitations_disclaimer() -> str:
    """Generate limitations disclaimer."""
    return (
        "This AI-generated report has several limitations: "
        "1) It is based on the data provided and may not capture the complete clinical picture. "
        "2) AI predictions are probabilistic and should be validated by clinical judgment. "
        "3) This report does not replace professional medical advice or diagnosis. "
        "4) Always consult a licensed healthcare provider for medical decisions. "
        "5) Emergency situations require immediate professional medical attention, not AI analysis."
    )
