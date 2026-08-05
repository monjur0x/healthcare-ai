"""CrewAI tasks for healthcare analysis pipeline."""

from crewai import Task
from typing import Any


def create_tasks(agents: dict, context: dict[str, Any]) -> dict[str, Task]:
    """Create tasks for the healthcare analysis crew.

    Args:
        agents: Dictionary of CrewAI agents
        context: Input context containing patient data and analysis results

    Returns:
        Dictionary of named tasks
    """
    patient_info = context.get("patient_info", {})
    input_type = context.get("input_type", "csv")
    csv_summary = context.get("csv_summary", "")
    image_summary = context.get("image_summary", "")

    # Task 1: Patient Analysis
    task_patient_analysis = Task(
        description=f"""Analyze the following patient information and create a comprehensive patient summary.

Patient Information:
- Name: {patient_info.get('name', 'Unknown')}
- ID: {patient_info.get('id', 'Unknown')}
- Age: {patient_info.get('age', 'Unknown')}
- Notes: {patient_info.get('notes', 'None')}

Input Type: {input_type}

CSV Data Summary:
{csv_summary}

Medical Image Summary:
{image_summary}

Your task:
1. Validate all patient data for completeness
2. Identify key health metrics from the data
3. Create a structured patient summary
4. Flag any data quality issues
5. Return a JSON-formatted patient summary with:
   - patient_info: validated patient details
   - data_quality: assessment of input data
   - key_metrics: identified health indicators
   - clinical_flags: any concerning values""",
        expected_output="A comprehensive patient summary in JSON format with validated data and key metrics",
        agent=agents["patient_analyst"]
    )

    # Task 2: Disease Prediction
    task_disease_prediction = Task(
        description=f"""Based on the patient analysis, predict disease risk and create a diagnostic assessment.

Input Type: {input_type}
Patient Age: {patient_info.get('age', 'Unknown')}

Your task:
1. Analyze the patient data for disease indicators
2. If CSV data is available, identify abnormal biomarkers
3. If image data is available, interpret imaging findings
4. Combine predictions for a unified diagnosis
5. Provide:
   - primary_diagnosis: main condition identified
   - secondary_diagnosis: secondary conditions
   - confidence: prediction confidence score (0-1)
   - severity: mild/moderate/severe/critical
   - risk_level: low/medium/high
   - risk_factors: contributing factors""",
        expected_output="Disease prediction with diagnosis, confidence, severity and risk assessment",
        agent=agents["disease_predictor"],
        context=[task_patient_analysis]
    )

    # Task 3: Medical RAG Evidence Retrieval
    task_evidence_retrieval = Task(
        description=f"""Search medical knowledge bases for relevant clinical evidence.

Primary Diagnosis: Based on patient analysis and predictions
Patient Age: {patient_info.get('age', 'Unknown')}

Your task:
1. Search for clinical guidelines related to the diagnosis
2. Retrieve evidence from WHO, CDC, NIH, PubMed
3. Find relevant clinical recommendations
4. Verify all sources are authoritative
5. Return:
   - evidence: list of relevant findings with source and reference
   - clinical_guidelines: relevant treatment guidelines
   - research_citations: supporting research papers
   - evidence_quality: assessment of evidence strength

IMPORTANT: Never hallucinate sources. Only return verifiable evidence.""",
        expected_output="Clinical evidence with verifiable sources and references",
        agent=agents["medical_researcher"],
        context=[task_patient_analysis, task_disease_prediction]
    )

    # Task 4: Treatment Recommendations
    task_treatment = Task(
        description=f"""Generate comprehensive treatment recommendations based on the diagnosis and evidence.

Patient Age: {patient_info.get('age', 'Unknown')}
Input Type: {input_type}

Your task:
1. Review diagnosis and clinical evidence
2. Create evidence-based treatment recommendations
3. Include:
   - Medication suggestions (with dosages and frequency)
   - Lifestyle modifications
   - Recommended diagnostic tests
   - Follow-up schedule
   - Referral recommendations if needed
   - Emergency warning signs

CRITICAL: Always include the disclaimer that these recommendations must be reviewed by a licensed physician before implementation.""",
        expected_output="Comprehensive treatment plan with medication, lifestyle, and follow-up recommendations",
        agent=agents["treatment_planner"],
        context=[task_patient_analysis, task_disease_prediction, task_evidence_retrieval]
    )

    # Task 5: Explainability Analysis
    task_explanation = Task(
        description=f"""Explain the AI predictions and findings in understandable terms.

Input Type: {input_type}
Patient Age: {patient_info.get('age', 'Unknown')}

Your task:
1. Explain why specific predictions were made
2. If CSV data was analyzed:
   - Explain which biomarkers were abnormal
   - Describe what each abnormality means
   - Provide context for risk levels
3. If image was analyzed:
   - Describe suspicious findings
   - Explain anatomical significance
   - Clarify image interpretation
4. Provide:
   - explanation: clear explanation of all findings
   - biomarker_analysis: explanation of each abnormal marker (if CSV)
   - imaging_interpretation: explanation of image findings (if applicable)
   - clinical_significance: why these findings matter""",
        expected_output="Clear explanation of AI predictions and clinical findings",
        agent=agents["explainability_expert"],
        context=[task_patient_analysis, task_disease_prediction]
    )

    # Task 6: Risk Monitoring Plan
    task_risk_monitoring = Task(
        description=f"""Create a comprehensive risk monitoring and follow-up plan.

Patient Age: {patient_info.get('age', 'Unknown')}
Input Type: {input_type}

Your task:
1. Assess current and future risk trajectory
2. Create monitoring schedule based on risk level
3. Define alert thresholds for critical values
4. Include:
   - risk_category: overall risk stratification
   - monitoring_schedule: recommended follow-up tests
   - alert_thresholds: values requiring immediate attention
   - preventive_measures: lifestyle and health recommendations
   - screening_recommendations: age-appropriate screenings""",
        expected_output="Risk monitoring plan with schedule and alert thresholds",
        agent=agents["risk_monitor"],
        context=[task_patient_analysis, task_disease_prediction, task_treatment]
    )

    # Task 7: Clinical Report Generation
    task_report_generation = Task(
        description=f"""Compile all findings into a comprehensive clinical report in JSON format.

Patient Information:
- Name: {patient_info.get('name', 'Unknown')}
- ID: {patient_info.get('id', 'Unknown')}
- Age: {patient_info.get('age', 'Unknown')}

Input Type: {input_type}

Your task:
1. Merge all previous task outputs
2. Structure data in the required JSON format:
   {{
     "patient": {{"name": "", "id": "", "age": ""}},
     "input_type": "",
     "patient_summary": "",
     "prediction": {{
       "primary_diagnosis": "",
       "secondary_diagnosis": "",
       "confidence": 0,
       "severity": "",
       "risk_level": ""
     }},
     "clinical_findings": [],
     "image_findings": [],
     "evidence": [{{"source": "", "summary": "", "reference": ""}}],
     "recommendations": [],
     "follow_up": [],
     "monitoring_plan": [],
     "explanation": "",
     "limitations": "",
     "doctor_notice": "This report is AI-assisted. Final diagnosis must be made by a licensed physician."
   }}

3. Ensure all fields are properly populated
4. Include all required disclaimers
5. Validate JSON structure""",
        expected_output="Complete clinical report in the specified JSON format",
        agent=agents["report_writer"],
        context=[
            task_patient_analysis, task_disease_prediction,
            task_evidence_retrieval, task_treatment,
            task_explanation, task_risk_monitoring
        ]
    )

    return {
        "patient_analysis": task_patient_analysis,
        "disease_prediction": task_disease_prediction,
        "evidence_retrieval": task_evidence_retrieval,
        "treatment_recommendation": task_treatment,
        "explanation": task_explanation,
        "risk_monitoring": task_risk_monitoring,
        "report_generation": task_report_generation
    }
