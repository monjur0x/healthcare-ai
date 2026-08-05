"""CrewAI agents for healthcare analysis pipeline."""

from crewai import Agent
from ..config import settings
from .tools import (
    CSVReaderTool, ImageLoaderTool, PredictionTool, FusionTool,
    QdrantSearchTool, PubMedSearchTool, RiskCalculatorTool,
    ReportGeneratorTool
)


def create_agents() -> dict[str, Agent]:
    """Create all healthcare analysis agents.

    Returns:
        Dictionary of named agents
    """
    # Common agent configuration
    common_kwargs = {
        "verbose": settings.CREW_VERBOSE,
        "allow_delegation": False,
        "max_iter": settings.CREW_MAX_ITERATIONS,
    }

    # Agent 1: Patient Analysis Agent
    patient_analyst = Agent(
        role="Clinical Data Analyst",
        goal="Understand and validate patient information from uploaded CSV files and medical images. "
             "Create a comprehensive structured patient summary.",
        backstory="You are an experienced clinical data analyst with expertise in processing "
                  "medical records and patient data. You excel at identifying key health metrics, "
                  "validating data quality, and creating structured summaries for downstream analysis.",
        tools=[CSVReaderTool(), ImageLoaderTool()],
        llm=f"google/{settings.LLM_MODEL}",
        **common_kwargs
    )

    # Agent 2: Disease Prediction Agent
    disease_predictor = Agent(
        role="Clinical Prediction Specialist",
        goal="Predict disease risk based on patient data from CSV analysis and image findings. "
             "Fuse predictions from multiple sources and provide accurate risk assessment.",
        backstory="You are a clinical prediction specialist with deep knowledge in medical "
                  "diagnostics. You combine statistical analysis with clinical knowledge to "
                  "provide accurate disease risk predictions and severity assessments.",
        tools=[PredictionTool(), FusionTool()],
        llm=f"google/{settings.LLM_MODEL}",
        **common_kwargs
    )

    # Agent 3: Medical RAG Agent
    medical_researcher = Agent(
        role="Clinical Research Specialist",
        goal="Retrieve and synthesize clinical evidence from authoritative medical sources "
             "including WHO, CDC, NIH, PubMed, and clinical guidelines.",
        backstory="You are a medical research specialist who excels at finding and synthesizing "
                  "evidence-based medicine from authoritative sources. You never hallucinate "
                  "information and always provide verifiable references.",
        tools=[QdrantSearchTool(), PubMedSearchTool()],
        llm=f"google/{settings.LLM_MODEL}",
        **common_kwargs
    )

    # Agent 4: Treatment Recommendation Agent
    treatment_planner = Agent(
        role="Treatment Planner",
        goal="Generate evidence-based treatment recommendations including medications, "
             "lifestyle modifications, diagnostic tests, and follow-up plans.",
        backstory="You are a treatment planning specialist who creates comprehensive, "
                  "evidence-based treatment plans. You always emphasize that recommendations "
                  "must be reviewed and approved by a licensed physician.",
        tools=[QdrantSearchTool()],
        llm=f"google/{settings.LLM_MODEL}",
        **common_kwargs
    )

    # Agent 5: Explainability Agent
    explainability_expert = Agent(
        role="Medical AI Explainer",
        goal="Provide clear, understandable explanations of why specific predictions were made, "
             "including analysis of biomarkers for CSV data and suspicious findings for images.",
        backstory="You are a medical AI explainability expert who bridges the gap between "
                  "complex AI predictions and patient/physician understanding. You explain "
                  "technical findings in accessible language.",
        tools=[],
        llm=f"google/{settings.LLM_MODEL}",
        **common_kwargs
    )

    # Agent 6: Risk Monitoring Agent
    risk_monitor = Agent(
        role="Risk Assessment Specialist",
        goal="Estimate future patient risk trajectory and create comprehensive monitoring "
             "plans with appropriate alert thresholds.",
        backstory="You are a risk assessment specialist focused on preventive healthcare. "
                  "You excel at predicting disease progression and creating personalized "
                  "monitoring schedules for optimal patient outcomes.",
        tools=[RiskCalculatorTool()],
        llm=f"google/{settings.LLM_MODEL}",
        **common_kwargs
    )

    # Agent 7: Clinical Report Agent
    report_writer = Agent(
        role="Medical Report Writer",
        goal="Merge all previous outputs into a comprehensive, well-structured clinical "
             "report in the required JSON format with all necessary disclaimers.",
        backstory="You are a medical report writing specialist who creates clear, comprehensive, "
                  "and professionally formatted clinical reports. You ensure all findings are "
                  "properly documented and all disclaimers are included.",
        tools=[ReportGeneratorTool()],
        llm=f"google/{settings.LLM_MODEL}",
        **common_kwargs
    )

    return {
        "patient_analyst": patient_analyst,
        "disease_predictor": disease_predictor,
        "medical_researcher": medical_researcher,
        "treatment_planner": treatment_planner,
        "explainability_expert": explainability_expert,
        "risk_monitor": risk_monitor,
        "report_writer": report_writer
    }
