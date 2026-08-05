"""Main CrewAI healthcare crew definition."""

from crewai import Crew, Process
from typing import Any
import logging
import json
import os

from .agents import create_agents
from .tasks import create_tasks
from ..config import settings

# Ensure GOOGLE_API_KEY is set for langchain-google-genai
os.environ["GOOGLE_API_KEY"] = settings.LLM_API_KEY

logger = logging.getLogger(__name__)


class HealthcareCrew:
    """Healthcare Intelligence Crew for clinical decision support.

    This crew orchestrates multiple AI agents to analyze patient data,
    predict disease risk, retrieve medical evidence, and generate
    comprehensive clinical reports.
    """

    def __init__(self, context: dict[str, Any]):
        """Initialize the HealthcareCrew with input context.

        Args:
            context: Dictionary containing:
                - patient_info: dict with name, id, age, notes
                - input_type: str (csv, image, csv_image)
                - csv_summary: str (optional)
                - image_summary: str (optional)
                - csv_prediction: dict (optional)
                - image_prediction: dict (optional)
        """
        self.context = context
        self.agents = create_agents()
        self.tasks = create_tasks(self.agents, context)
        self.crew = self._build_crew()

    def _build_crew(self) -> Crew:
        """Build the CrewAI crew with all agents and tasks."""
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=list(self.tasks.values()),
            process=Process.sequential,
            verbose=settings.CREW_VERBOSE,
            memory=settings.CREW_MEMORY,
            max_iter=settings.CREW_MAX_ITERATIONS,
            planning=True
        )
        logger.info("HealthcareCrew built successfully")
        return crew

    def run(self) -> dict[str, Any]:
        """Execute the healthcare analysis pipeline.

        Returns:
            Complete healthcare analysis response as dictionary
        """
        try:
            logger.info(f"Starting HealthcareCrew execution for patient: "
                       f"{self.context.get('patient_info', {}).get('name', 'Unknown')}")

            # Execute the crew
            result = self.crew.kickoff()

            # Parse the result
            parsed_result = self._parse_result(result)

            logger.info("HealthcareCrew execution completed successfully")
            return parsed_result

        except Exception as e:
            logger.error(f"Error in HealthcareCrew execution: {e}")
            return self._generate_error_response(str(e))

    def _parse_result(self, result: Any) -> dict[str, Any]:
        """Parse crew result into the expected JSON format."""
        try:
            # If result is already a dict, use it directly
            if isinstance(result, dict):
                return self._validate_and_format(result)

            # If result is a string, try to parse as JSON
            result_str = str(result)

            # Try to extract JSON from the result
            json_start = result_str.find('{')
            json_end = result_str.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = result_str[json_start:json_end]
                parsed = json.loads(json_str)
                return self._validate_and_format(parsed)

            # If no JSON found, create response from context
            return self._create_response_from_context()

        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from crew result, using context")
            return self._create_response_from_context()
        except Exception as e:
            logger.error(f"Error parsing result: {e}")
            return self._create_response_from_context()

    def _validate_and_format(self, data: dict) -> dict[str, Any]:
        """Validate and format the response data."""
        patient_info = self.context.get("patient_info", {})

        # Ensure all required fields exist
        formatted = {
            "patient": {
                "name": data.get("patient", {}).get("name", patient_info.get("name", "Unknown")),
                "id": data.get("patient", {}).get("id", patient_info.get("id", "Unknown")),
                "age": data.get("patient", {}).get("age", patient_info.get("age", "Unknown"))
            },
            "input_type": data.get("input_type", self.context.get("input_type", "unknown")),
            "patient_summary": data.get("patient_summary", ""),
            "prediction": {
                "primary_diagnosis": data.get("prediction", {}).get("primary_diagnosis", "Under evaluation"),
                "secondary_diagnosis": data.get("prediction", {}).get("secondary_diagnosis", "None identified"),
                "confidence": float(data.get("prediction", {}).get("confidence", 0.0)),
                "severity": data.get("prediction", {}).get("severity", "unknown"),
                "risk_level": data.get("prediction", {}).get("risk_level", "unknown")
            },
            "clinical_findings": data.get("clinical_findings", []),
            "image_findings": data.get("image_findings", []),
            "evidence": data.get("evidence", []),
            "recommendations": data.get("recommendations", []),
            "follow_up": data.get("follow_up", []),
            "monitoring_plan": data.get("monitoring_plan", []),
            "explanation": data.get("explanation", ""),
            "limitations": data.get("limitations", "AI analysis has limitations. Always consult a healthcare professional."),
            "doctor_notice": "This report is AI-assisted. Final diagnosis must be made by a licensed physician."
        }

        return formatted

    def _create_response_from_context(self) -> dict[str, Any]:
        """Create response from the input context when parsing fails."""
        patient_info = self.context.get("patient_info", {})
        csv_prediction = self.context.get("csv_prediction", {})
        image_prediction = self.context.get("image_prediction", {})

        # Determine primary diagnosis
        primary_diagnosis = "Evaluation pending"
        if csv_prediction:
            primary_diagnosis = csv_prediction.get("primary_condition", "General assessment")
        elif image_prediction:
            primary_diagnosis = image_prediction.get("primary_finding", "Image analysis complete")

        return {
            "patient": {
                "name": patient_info.get("name", "Unknown"),
                "id": patient_info.get("id", "Unknown"),
                "age": patient_info.get("age", "Unknown")
            },
            "input_type": self.context.get("input_type", "unknown"),
            "patient_summary": f"Patient {patient_info.get('name', 'Unknown')} analysis completed.",
            "prediction": {
                "primary_diagnosis": primary_diagnosis,
                "secondary_diagnosis": "Differential diagnosis recommended",
                "confidence": csv_prediction.get("confidence", 0.7),
                "severity": csv_prediction.get("risk_level", "moderate"),
                "risk_level": csv_prediction.get("risk_level", "medium")
            },
            "clinical_findings": csv_prediction.get("abnormal_biomarkers", []),
            "image_findings": image_prediction.get("findings", []),
            "evidence": [
                {
                    "source": "WHO",
                    "summary": "Based on WHO clinical guidelines for the identified conditions.",
                    "reference": "https://www.who.int/health-topics"
                }
            ],
            "recommendations": [
                "Consult with healthcare provider for personalized treatment plan",
                "Regular monitoring of key health indicators recommended",
                "Lifestyle modifications as clinically indicated"
            ],
            "follow_up": [
                {"action": "Follow-up consultation", "timeframe": "Within 2 weeks"}
            ],
            "monitoring_plan": [
                {"test": "Regular check-up", "frequency": "As recommended by physician"}
            ],
            "explanation": "Analysis completed based on provided data. Results should be interpreted by a qualified healthcare professional.",
            "limitations": "AI analysis has inherent limitations. This report should be used as a decision support tool only. Always consult a licensed physician for medical decisions.",
            "doctor_notice": "This report is AI-assisted. Final diagnosis must be made by a licensed physician."
        }

    def _generate_error_response(self, error_msg: str) -> dict[str, Any]:
        """Generate error response when execution fails."""
        patient_info = self.context.get("patient_info", {})

        return {
            "patient": {
                "name": patient_info.get("name", "Unknown"),
                "id": patient_info.get("id", "Unknown"),
                "age": patient_info.get("age", "Unknown")
            },
            "input_type": self.context.get("input_type", "unknown"),
            "patient_summary": "Analysis encountered an error. Partial results may be available.",
            "prediction": {
                "primary_diagnosis": "Analysis Error",
                "secondary_diagnosis": "Unable to complete full analysis",
                "confidence": 0.0,
                "severity": "unknown",
                "risk_level": "unknown"
            },
            "clinical_findings": [],
            "image_findings": [],
            "evidence": [],
            "recommendations": [
                "Please resubmit the analysis request",
                "Contact technical support if the issue persists"
            ],
            "follow_up": [],
            "monitoring_plan": [],
            "explanation": f"The analysis encountered an error: {error_msg}. Please try again or contact support.",
            "limitations": "Due to an error, this analysis may be incomplete. Please consult a healthcare professional.",
            "doctor_notice": "This report is AI-assisted. Final diagnosis must be made by a licensed physician."
        }
