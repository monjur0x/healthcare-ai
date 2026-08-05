"""FastAPI routes for healthcare analysis API."""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import logging

from .schemas import HealthcareResponse, PatientInfo, Prediction, Evidence
from ..crew.crew import HealthcareCrew
from ..config import settings
from ..utils.preprocessing import preprocess_csv_data

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/run-healthcare-crew", response_model=HealthcareResponse)
async def run_healthcare_crew(
    csv_file: Optional[UploadFile] = File(None, description="CSV file with patient health data"),
    medical_image: Optional[UploadFile] = File(None, description="Medical image (optional)"),
    patient_name: str = Form(..., description="Patient full name"),
    patient_id: str = Form(..., description="Patient ID"),
    patient_age: int = Form(..., description="Patient age"),
    notes: Optional[str] = Form(None, description="Additional clinical notes")
):
    """Run the HealthcareCrew analysis pipeline.

    This endpoint accepts patient data via multipart/form-data and runs
    the complete multi-agent healthcare analysis pipeline.

    Args:
        csv_file: CSV file containing patient health metrics
        medical_image: Optional medical image for analysis
        patient_name: Patient's full name
        patient_id: Unique patient identifier
        patient_age: Patient's age in years
        notes: Additional clinical notes

    Returns:
        HealthcareResponse with complete analysis results
    """
    try:
        logger.info(f"Processing request for patient: {patient_name} (ID: {patient_id})")

        # Validate inputs
        if not csv_file and not medical_image:
            raise HTTPException(
                status_code=400,
                detail="At least one input file (CSV or medical image) is required"
            )

        # Determine input type
        input_type = determine_input_type(csv_file, medical_image)

        # Process CSV file if provided
        csv_summary = ""
        csv_prediction = None
        if csv_file:
            csv_content = await csv_file.read()
            csv_result = preprocess_csv_data(csv_content)
            csv_summary = format_csv_summary(csv_result)
            csv_prediction = csv_result.get("key_metrics", {})

        # Process medical image if provided
        image_summary = ""
        image_prediction = None
        if medical_image:
            image_metadata = await extract_image_metadata(medical_image)
            image_summary = f"Medical image uploaded: {medical_image.content_type}"
            image_prediction = {"image_type": medical_image.content_type, "status": "received"}

        # Prepare context for the crew
        context = {
            "patient_info": {
                "name": patient_name,
                "id": patient_id,
                "age": patient_age,
                "notes": notes or ""
            },
            "input_type": input_type,
            "csv_summary": csv_summary,
            "image_summary": image_summary,
            "csv_prediction": csv_prediction,
            "image_prediction": image_prediction
        }

        # Initialize and run the HealthcareCrew
        crew = HealthcareCrew(context)
        result = crew.run()

        logger.info(f"Analysis completed for patient: {patient_name}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


def determine_input_type(
    csv_file: Optional[UploadFile],
    medical_image: Optional[UploadFile]
) -> str:
    """Determine the input type based on uploaded files."""
    if csv_file and medical_image:
        return "csv_image"
    elif csv_file:
        return "csv"
    elif medical_image:
        return "image"
    return "unknown"


def format_csv_summary(csv_result: dict) -> str:
    """Format CSV analysis results into summary text."""
    if not csv_result.get("success"):
        return f"CSV processing error: {csv_result.get('error', 'Unknown error')}"

    stats = csv_result.get("statistics", {})
    metrics = csv_result.get("key_metrics", {})
    conditions = csv_result.get("potential_conditions", [])

    summary_parts = [
        f"CSV Analysis: {stats.get('total_rows', 0)} rows, {stats.get('total_columns', 0)} columns",
        f"Data Quality Score: {csv_result.get('quality_score', 0):.1%}",
    ]

    if metrics:
        summary_parts.append("Key Metrics Identified:")
        for metric, data in list(metrics.items())[:5]:
            if isinstance(data, dict):
                summary_parts.append(f"  - {metric}: {data.get('value', 'N/A')}")

    if conditions:
        summary_parts.append("Potential Conditions:")
        for condition in conditions[:3]:
            summary_parts.append(f"  - {condition.get('condition', 'Unknown')} "
                               f"(Confidence: {condition.get('confidence', 0):.1%})")

    return "\n".join(summary_parts)


async def extract_image_metadata(upload_file: UploadFile) -> dict:
    """Extract metadata from uploaded medical image."""
    content = await upload_file.read()

    metadata = {
        "filename": upload_file.filename,
        "content_type": upload_file.content_type,
        "size_bytes": len(content),
        "dimensions": "Unknown",  # Would use PIL to get actual dimensions
        "format": upload_file.content_type.split("/")[-1] if upload_file.content_type else "unknown"
    }

    return metadata
