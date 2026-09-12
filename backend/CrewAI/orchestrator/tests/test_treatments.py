"""
Tests for evidence-grounded, model-graded treatments (P2.3).

Playbook candidates are ranked toward the patient's model-derived
drivers and cited against retrieved evidence; candidates with no
retrieved support keep a playbook-only label instead of a citation.
Without evidence or drivers the legacy playbook order applies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from CrewAI.orchestrator.crew import ClinicalCrew
from CrewAI.orchestrator.schemas import (
    EvidenceItem,
    PatientInfo,
    PredictionResult,
    RiskResult,
)
from CrewAI.orchestrator.services import build_treatment_recommendations
from CrewAI.orchestrator.treatments import grade_recommendations
from models import TabularClassifier
from rag import HashingEmbedder, RAGPipeline

EVIDENCE = [
    EvidenceItem(
        document_id="diabetes.txt",
        source="protocols",
        score=0.9,
        text="diabetes mellitus is managed with metformin, lifestyle "
        "changes, and regular glucose monitoring",
        topics=["diabetes"],
    )
]


def _prediction() -> PredictionResult:
    return PredictionResult(
        predicted_class="1",
        probabilities={"0": 0.2, "1": 0.8},
        confidence=0.8,
        model_name="logistic",
        disease="diabetes",
        positive_probability=0.8,
        negative_probability=0.2,
    )


def _risk() -> RiskResult:
    return RiskResult(risk_score=0.6, risk_level="medium")


def test_grade_ranks_driver_matched_candidate_first() -> None:
    graded = grade_recommendations(
        [
            "Confirm the diagnosis on a separate day.",
            "Monitor HbA1c every 3-6 months until stable at target.",
        ],
        [],
        drivers=["hba1c", "glucose"],
    )
    assert graded[0].text.startswith("Monitor HbA1c")
    assert graded[0].driver_hits == ("hba1c",)
    assert graded[0].grade > graded[1].grade


def test_grade_cites_supporting_evidence() -> None:
    graded = grade_recommendations(
        ["First-line management includes metformin per clinician."],
        EVIDENCE,
    )
    assert graded[0].supported
    assert graded[0].evidence_ids == ("diabetes.txt",)
    assert graded[0].annotated().endswith("[evidence: diabetes.txt]")


def test_grade_labels_unsupported_playbook_only() -> None:
    graded = grade_recommendations(
        ["Screen for diabetic complications at baseline."],
        EVIDENCE,
    )
    assert not graded[0].supported
    assert graded[0].annotated().endswith("[playbook-only: no retrieved evidence]")


def test_builder_legacy_order_without_inputs() -> None:
    recs, _ = build_treatment_recommendations(_prediction(), _risk())
    assert recs[0].startswith("First-line management")
    assert all("[" not in rec for rec in recs)


def test_builder_grades_with_evidence_and_drivers() -> None:
    recs, monitoring = build_treatment_recommendations(
        _prediction(), _risk(), evidence=EVIDENCE, drivers=["hba1c"]
    )
    assert monitoring == []
    # Driver-matched HbA1c monitoring outranks playbook-first metformin.
    assert recs[0].startswith("Monitor HbA1c")
    assert any("[evidence: diabetes.txt]" in rec for rec in recs)
    assert any("[playbook-only:" in rec for rec in recs)


def test_builder_grades_with_drivers_only() -> None:
    recs, _ = build_treatment_recommendations(_prediction(), _risk(), drivers=["hba1c"])
    assert recs[0].startswith("Monitor HbA1c")
    assert all("[playbook-only:" in rec for rec in recs)


@pytest.fixture
def crew_model() -> TabularClassifier:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(60, 2))
    frame = pd.DataFrame(x * 2.0, columns=["glucose", "bmi"])
    labels = pd.Series((frame["glucose"] > 0).astype(int))
    return TabularClassifier(model_name="logistic").fit(frame, labels)


@pytest.fixture
def crew_background(crew_model) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(20, 2))
    return pd.DataFrame(x * 2.0, columns=["glucose", "bmi"])


@pytest.fixture
def crew_pipeline() -> RAGPipeline:
    rag = RAGPipeline(embedder=HashingEmbedder(dims=64))
    rag.ingest_texts(
        [
            "diabetes mellitus is managed with metformin, lifestyle "
            "changes, and regular glucose monitoring"
        ]
    )
    return rag


def _agent4_output(crew: ClinicalCrew) -> dict:
    crew.run_analysis()
    return next(
        s.output_data
        for s in crew.crew_trace.steps
        if isinstance(s.output_data, dict) and "graded" in s.output_data
    )


def test_crew_agent4_grades_with_model_and_evidence(
    crew_model, crew_background, crew_pipeline
) -> None:
    crew = ClinicalCrew(
        patient=PatientInfo(id="p-treat"),
        model=crew_model,
        features={"glucose": 4.0, "bmi": 0.0},
        background=crew_background,
        rag_pipeline=crew_pipeline,
        markers={"glucose": 200.0},
    )
    output = _agent4_output(crew)
    assert output["graded"] is True
    assert output["drivers"] and output["drivers"][0] == "glucose"


def test_crew_agent4_ungraded_without_inputs() -> None:
    crew = ClinicalCrew(patient=PatientInfo(id="p-plain"))
    output = _agent4_output(crew)
    assert output["graded"] is False
    assert output["drivers"] == []
