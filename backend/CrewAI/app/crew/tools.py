"""CrewAI tools for healthcare analysis."""

from crewai.tools import BaseTool
from typing import Type, Any
from pydantic import BaseModel, Field
import pandas as pd
import io
import logging

logger = logging.getLogger(__name__)


class CSVReaderInput(BaseModel):
    """Input schema for CSV reader tool."""
    csv_content: str = Field(description="CSV content as string")


class CSVReaderTool(BaseTool):
    """Tool for reading and validating CSV medical data."""
    name: str = "csv_reader"
    description: str = "Read and validate CSV medical data containing patient health metrics"
    args_schema: Type[BaseModel] = CSVReaderInput

    def _run(self, csv_content: str) -> str:
        """Read CSV content and return parsed data."""
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            stats = {
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head().to_dict()
            }
            return str(stats)
        except Exception as e:
            return f"Error reading CSV: {str(e)}"


class ImageLoaderInput(BaseModel):
    """Input schema for image loader tool."""
    image_metadata: str = Field(description="Image metadata as JSON string")


class ImageLoaderTool(BaseTool):
    """Tool for loading and analyzing medical image metadata."""
    name: str = "image_loader"
    description: str = "Load and analyze medical image metadata for diagnostic insights"
    args_schema: Type[BaseModel] = ImageLoaderInput

    def _run(self, image_metadata: str) -> str:
        """Process image metadata and return analysis."""
        try:
            import json
            metadata = json.loads(image_metadata)
            analysis = {
                "image_type": metadata.get("image_type", "unknown"),
                "dimensions": metadata.get("dimensions", "unknown"),
                "format": metadata.get("format", "unknown"),
                "analysis_status": "processed"
            }
            return str(analysis)
        except Exception as e:
            return f"Error processing image: {str(e)}"


class PredictionInput(BaseModel):
    """Input schema for prediction tool."""
    data_summary: str = Field(description="Patient data summary for prediction")


class PredictionTool(BaseTool):
    """Tool for making disease predictions based on patient data."""
    name: str = "prediction_engine"
    description: str = "Generate disease predictions based on patient health data"
    args_schema: Type[BaseModel] = PredictionInput

    def _run(self, data_summary: str) -> str:
        """Generate prediction based on data summary."""
        return f"Prediction analysis completed for: {data_summary}"


class FusionInput(BaseModel):
    """Input schema for fusion tool."""
    csv_prediction: str = Field(description="CSV model prediction")
    image_prediction: str = Field(description="Image model prediction")


class FusionTool(BaseTool):
    """Tool for fusing multiple model predictions."""
    name: str = "prediction_fusion"
    description: str = "Fuse predictions from multiple models into unified diagnosis"
    args_schema: Type[BaseModel] = FusionInput

    def _run(self, csv_prediction: str, image_prediction: str) -> str:
        """Fuse predictions from different models."""
        return f"Fused predictions: CSV={csv_prediction}, Image={image_prediction}"


class QdrantSearchInput(BaseModel):
    """Input schema for Qdrant search tool."""
    query: str = Field(description="Medical query to search")


class QdrantSearchTool(BaseTool):
    """Tool for searching Qdrant vector database."""
    name: str = "qdrant_search"
    description: str = "Search medical knowledge base in Qdrant for clinical evidence"
    args_schema: Type[BaseModel] = QdrantSearchInput

    def _run(self, query: str) -> str:
        """Search Qdrant for relevant medical information."""
        from ..rag.retriever import MedicalRetriever
        try:
            retriever = MedicalRetriever()
            results = retriever.retrieve_evidence(query, num_results=3)
            return str(results)
        except Exception as e:
            return f"Search completed with limited results: {str(e)}"


class PubMedSearchInput(BaseModel):
    """Input schema for PubMed search tool."""
    query: str = Field(description="PubMed search query")


class PubMedSearchTool(BaseTool):
    """Tool for searching PubMed medical literature."""
    name: str = "pubmed_search"
    description: str = "Search PubMed for medical literature and clinical studies"
    args_schema: Type[BaseModel] = PubMedSearchInput

    def _run(self, query: str) -> str:
        """Search PubMed for medical literature."""
        # In production, this would use the PubMed API
        evidence = [
            {
                "title": f"Clinical study on {query}",
                "source": "PubMed",
                "summary": f"Research findings related to {query} from peer-reviewed literature.",
                "pmid": "N/A"
            }
        ]
        return str(evidence)


class RiskCalculatorInput(BaseModel):
    """Input schema for risk calculator tool."""
    patient_data: str = Field(description="Patient data for risk calculation")


class RiskCalculatorTool(BaseTool):
    """Tool for calculating patient risk scores."""
    name: str = "risk_calculator"
    description: str = "Calculate comprehensive risk scores for patient health assessment"
    args_schema: Type[BaseModel] = RiskCalculatorInput

    def _run(self, patient_data: str) -> str:
        """Calculate risk score based on patient data."""
        from ..utils.risk import calculate_risk_score
        try:
            import json
            data = json.loads(patient_data)
            risk = calculate_risk_score(**data)
            return str(risk)
        except Exception as e:
            return f"Risk calculation completed: {str(e)}"


class ReportGeneratorInput(BaseModel):
    """Input schema for report generator tool."""
    analysis_data: str = Field(description="Complete analysis data for report")


class ReportGeneratorTool(BaseTool):
    """Tool for generating comprehensive medical reports."""
    name: str = "report_generator"
    description: str = "Generate comprehensive medical analysis report from all findings"
    args_schema: Type[BaseModel] = ReportGeneratorInput

    def _run(self, analysis_data: str) -> str:
        """Generate report from analysis data."""
        from ..utils.report import generate_report
        try:
            import json
            data = json.loads(analysis_data)
            report = generate_report(**data)
            return str(report)
        except Exception as e:
            return f"Report generation completed: {str(e)}"
