"""
Baseline comparison study (paper §13).

Runs the five configurations of the research proposal against each
shipped dataset and writes a markdown results table:

    1. Centralized ML
    2. Federated-only (FedAvg)
    3. Federated + RAG
    4. Federated + Multi-Agent
    5. Proposed: Federated + Multi-Agent + RAG + n8n

No metric is reimplemented here — every number comes from the existing
metric modules:

- ``evaluation.metrics.classification_metrics`` for the held-out
  Accuracy / Precision / Recall / F1 / ROC-AUC set.
- ``federated.metrics`` (``FederatedMetrics``, ``parameter_set_bytes``,
  ``convergence_round``) for communication cost and convergence, as
  recorded by the synchronous ``FedAvgServer`` inside
  ``AnalysisService.train(federated=True)``.
- ``rag.metrics.rag_quality_metrics`` for the RAGAS-style generation
  quality block.
- ``CrewAI.orchestrator.metrics.compute_agent_metrics`` for the
  multi-agent block.

Design notes (pilot scale — see the generated Findings section):

- The held-out split is reproduced deterministically for every baseline
  and dataset with ``test_size=0.25`` and ``seed=42`` so all
  classification numbers are directly comparable.
- The RAG layer is evaluated on a small fixed query set per dataset
  (5 queries, written literally below). The "answer" graded by
  ``rag_quality_metrics`` is a reference clinical answer written in this
  script and grounded in the per-dataset corpus, so faithfulness /
  answer-relevancy are meaningful rather than trivially 1.0.
- Agent metrics are computed from the deterministic (LLM-free) crew
  outputs over a handful of sample patient rows drawn from each test
  split. "Task outputs" are the five sections of each assembled
  :class:`ClinicalReport` (summary, prediction, risk, evidence context,
  recommendations).
- RAG / multi-agent layers do not change the trained model, so the
  classification block of baselines 2-5 is the federated model's numbers
  (reported once, reused). Baselines that do not measure a metric report
  ``n/a`` explicitly.
- n8n adds no new metric: it is the orchestration layer already
  exercised live in ``docs/CHANGELOG.md``'s verification entries. The
  Proposed row is the union of classification + RAG + agent metrics.

Usage (run from ``backend/``):

    DATASET_DIR=/path/to/datasets python scripts/baseline_study.py

The dataset directory defaults to ``DATASET_DIR`` or ``.``; ``--out``
defaults to ``docs/BASELINE_STUDY_RESULTS.md``. The table is printed to
stdout and written to ``--out``. A hand-written ``## Findings`` section
already present in the output file is preserved across runs.
"""

from __future__ import annotations

import argparse
import os
import sys

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent
_REPO_DIR = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

#: Default output path anchored to the repository docs directory so the
#: script produces ``docs/BASELINE_STUDY_RESULTS.md`` regardless of CWD.
_DEFAULT_OUT = _REPO_DIR / "docs" / "BASELINE_STUDY_RESULTS.md"

from api.services import (  # noqa: E402
    PRESETS,
    AnalysisService,
    TrainResult,
    prepare_tabular_data,
)
from CrewAI.orchestrator import (  # noqa: E402
    AgentMetrics,
    ClinicalReport,
    PatientInfo,
    compute_agent_metrics,
)
from evaluation import (  # noqa: E402
    ClassificationMetrics,
    classification_metrics,
)
from federated import FederatedMetrics, parameter_set_bytes  # noqa: E402
from preprocessing.logger import get_logger  # noqa: E402
from rag import (  # noqa: E402
    RAGPipeline,
    RAGQualityMetrics,
    rag_quality_metrics,
)
from rag.documents import Document  # noqa: E402

logger = get_logger(__name__)

#: Human-readable condition names for the results tables.
DATASET_DISPLAY_NAMES: dict[str, str] = {
    "diabetes": "Diabetes (PIMA, type-2 risk)",
    "heart": "Heart disease (UCI)",
    "kidney": "Chronic kidney disease",
    "sepsis": "Sepsis (synthetic ICU)",
}

#: Federated run used by every federated baseline.
FEDERATED_CLIENTS = 3
FEDERATED_ROUNDS = 5

#: Reproducibility constants for the shared held-out split.
TEST_SIZE = 0.25
SEED = 42

#: Number of sample patient rows drawn from each test split for the
#: multi-agent baselines.
N_PATIENTS = 5

#: Retrieval depth used for the RAG evaluation.
RAG_TOP_K = 5


@dataclass(frozen=True)
class RagItem:
    """
    One RAG evaluation query with its ground-truth and reference answer.

    Parameters
    ----------
    query : str
        The clinical question asked.
    relevant_ids : list[str]
        Ground-truth relevant document ids in the per-dataset corpus.
    reference_answer : str
        Reference clinical answer grounded in the corpus (the "generated"
        answer graded by ``rag_quality_metrics``).
    """

    query: str
    relevant_ids: list[str]
    reference_answer: str


#: Per-dataset corpora as ``(document id, source, text)`` triples. These
#: literal clinical statements stand in for the PubMed / WHO / NICE /
#: protocol knowledge base the RAG layer would normally ingest.
RAG_CORPORA: dict[str, list[tuple[str, str, str]]] = {
    "diabetes": [
        (
            "diabetes_management",
            "guidelines",
            "Type 2 diabetes mellitus is managed with metformin as "
            "first-line pharmacotherapy combined with lifestyle "
            "modification including dietary change, weight control, and "
            "regular physical activity.",
        ),
        (
            "diabetes_glucose",
            "guidelines",
            "Persistent hyperglycemia drives microvascular complications "
            "such as retinopathy, nephropathy, and neuropathy; the HbA1c "
            "target for most adults is below seven percent.",
        ),
        (
            "diabetes_risk",
            "guidelines",
            "Major risk factors for type 2 diabetes include obesity, "
            "physical inactivity, a family history of diabetes, and "
            "elevated fasting plasma glucose.",
        ),
        (
            "diabetes_monitoring",
            "guidelines",
            "Routine monitoring for diabetic patients includes HbA1c "
            "every three months, annual lipid panels, blood pressure "
            "checks, and annual eye and foot examinations.",
        ),
    ],
    "heart": [
        (
            "heart_risk",
            "guidelines",
            "Established modifiable risk factors for coronary heart "
            "disease include hypertension, dyslipidemia, diabetes, "
            "obesity, smoking, and physical inactivity.",
        ),
        (
            "heart_management",
            "guidelines",
            "Management of heart disease involves blood pressure "
            "control, lipid-lowering therapy such as statins, lifestyle "
            "change, and revascularization in selected patients.",
        ),
        (
            "heart_diagnosis",
            "guidelines",
            "Diagnosis of coronary artery disease combines clinical "
            "symptoms, an electrocardiogram, cardiac biomarkers, and "
            "imaging such as echocardiography or coronary angiography.",
        ),
        (
            "heart_secondary",
            "guidelines",
            "Secondary prevention after myocardial infarction includes "
            "aspirin, statins, beta blockers, blood pressure control, "
            "cardiac rehabilitation, and smoking cessation.",
        ),
    ],
    "kidney": [
        (
            "ckd_definition",
            "guidelines",
            "Chronic kidney disease is defined by a reduced glomerular "
            "filtration rate or markers of kidney damage that persist for "
            "more than three months.",
        ),
        (
            "ckd_risk",
            "guidelines",
            "Risk factors for chronic kidney disease include diabetes, "
            "hypertension, older age, and a family history of kidney "
            "disease.",
        ),
        (
            "ckd_management",
            "guidelines",
            "Management of chronic kidney disease focuses on blood "
            "pressure control, glycemic control, reducing proteinuria, "
            "and avoiding nephrotoxic drugs.",
        ),
        (
            "ckd_labs",
            "guidelines",
            "Laboratory evaluation of kidney disease includes serum "
            "creatinine, estimated GFR, urine albumin, and a complete "
            "urinalysis.",
        ),
    ],
    "sepsis": [
        (
            "sepsis_definition",
            "guidelines",
            "Sepsis is life-threatening organ dysfunction caused by a "
            "dysregulated host response to infection.",
        ),
        (
            "sepsis_recognition",
            "guidelines",
            "Recognition of sepsis relies on clinical criteria such as "
            "suspected infection with organ dysfunction measured by SOFA "
            "or qSOFA.",
        ),
        (
            "sepsis_treatment",
            "guidelines",
            "Treatment of sepsis includes broad-spectrum antibiotics "
            "within one hour of recognition, source control, and "
            "intravenous fluids.",
        ),
        (
            "sepsis_hour1",
            "guidelines",
            "The first-hour bundle for sepsis delivers antibiotics, "
            "lactate measurement, blood cultures, and fluids promptly.",
        ),
    ],
}

#: Per-dataset RAG queries referencing the condition each preset predicts.
RAG_EVALUATION: dict[str, list[RagItem]] = {
    "diabetes": [
        RagItem(
            query="What is the first-line treatment for type 2 diabetes?",
            relevant_ids=["diabetes_management"],
            reference_answer="Metformin combined with lifestyle "
            "modification including dietary change and regular physical "
            "activity is the first-line approach for type 2 diabetes.",
        ),
        RagItem(
            query="Which complications follow persistent hyperglycemia?",
            relevant_ids=["diabetes_glucose"],
            reference_answer="Persistent hyperglycemia leads to "
            "microvascular complications such as retinopathy, "
            "nephropathy, and neuropathy, with an HbA1c target below "
            "seven percent for most adults.",
        ),
        RagItem(
            query="What are the main risk factors for developing type 2 diabetes?",
            relevant_ids=["diabetes_risk"],
            reference_answer="Major risk factors for type 2 diabetes are "
            "obesity, physical inactivity, a family history of diabetes, "
            "and elevated fasting plasma glucose.",
        ),
        RagItem(
            query="How should a diabetic patient be monitored?",
            relevant_ids=["diabetes_monitoring"],
            reference_answer="Monitoring includes HbA1c every three "
            "months, annual lipid panels, blood pressure checks, and "
            "annual eye and foot examinations.",
        ),
        RagItem(
            query="What is the role of metformin in diabetes management?",
            relevant_ids=["diabetes_management", "diabetes_glucose"],
            reference_answer="Metformin is the first-line pharmacotherapy "
            "for type 2 diabetes and works alongside glucose-lowering "
            "lifestyle changes to keep HbA1c below target.",
        ),
    ],
    "heart": [
        RagItem(
            query="Which factors increase the risk of coronary heart disease?",
            relevant_ids=["heart_risk"],
            reference_answer="Modifiable risk factors for coronary heart "
            "disease include hypertension, dyslipidemia, diabetes, "
            "obesity, smoking, and physical inactivity.",
        ),
        RagItem(
            query="How is heart disease managed?",
            relevant_ids=["heart_management"],
            reference_answer="Management combines blood pressure control, "
            "lipid-lowering statins, lifestyle change, and "
            "revascularization for selected patients.",
        ),
        RagItem(
            query="What tests are used to diagnose coronary artery disease?",
            relevant_ids=["heart_diagnosis"],
            reference_answer="Diagnosis combines clinical symptoms, an "
            "electrocardiogram, cardiac biomarkers, and imaging such as "
            "echocardiography or coronary angiography.",
        ),
        RagItem(
            query="What does secondary prevention after a heart attack include?",
            relevant_ids=["heart_secondary"],
            reference_answer="Secondary prevention includes aspirin, "
            "statins, beta blockers, blood pressure control, cardiac "
            "rehabilitation, and smoking cessation.",
        ),
        RagItem(
            query="Why do statins matter in heart disease?",
            relevant_ids=["heart_management", "heart_secondary"],
            reference_answer="Statins lower lipids as part of heart "
            "disease management and continue in secondary prevention "
            "after myocardial infarction.",
        ),
    ],
    "kidney": [
        RagItem(
            query="How is chronic kidney disease defined?",
            relevant_ids=["ckd_definition"],
            reference_answer="Chronic kidney disease is a reduced "
            "glomerular filtration rate or markers of kidney damage "
            "lasting more than three months.",
        ),
        RagItem(
            query="What raises the risk of chronic kidney disease?",
            relevant_ids=["ckd_risk"],
            reference_answer="Diabetes, hypertension, older age, and a "
            "family history of kidney disease are key risk factors for "
            "chronic kidney disease.",
        ),
        RagItem(
            query="How is chronic kidney disease managed?",
            relevant_ids=["ckd_management"],
            reference_answer="Management targets blood pressure and "
            "glycemic control, reduces proteinuria, and avoids "
            "nephrotoxic drugs.",
        ),
        RagItem(
            query="Which tests evaluate kidney function?",
            relevant_ids=["ckd_labs"],
            reference_answer="Evaluation uses serum creatinine, estimated "
            "GFR, urine albumin, and a complete urinalysis.",
        ),
        RagItem(
            query="Why does blood pressure control matter in kidney disease?",
            relevant_ids=["ckd_risk", "ckd_management"],
            reference_answer="Hypertension is both a risk factor for and "
            "a target of chronic kidney disease management, where blood "
            "pressure control slows disease progression.",
        ),
    ],
    "sepsis": [
        RagItem(
            query="How is sepsis defined?",
            relevant_ids=["sepsis_definition"],
            reference_answer="Sepsis is life-threatening organ "
            "dysfunction caused by a dysregulated host response to "
            "infection.",
        ),
        RagItem(
            query="How is sepsis recognized clinically?",
            relevant_ids=["sepsis_recognition"],
            reference_answer="Clinical recognition of sepsis combines "
            "suspected infection with organ dysfunction scored by SOFA or "
            "qSOFA.",
        ),
        RagItem(
            query="What is the treatment for sepsis?",
            relevant_ids=["sepsis_treatment"],
            reference_answer="Treatment of sepsis is broad-spectrum "
            "antibiotics within one hour of recognition, source control, "
            "and intravenous fluids.",
        ),
        RagItem(
            query="What does the first-hour sepsis bundle include?",
            relevant_ids=["sepsis_hour1"],
            reference_answer="The first-hour bundle delivers antibiotics, "
            "lactate measurement, blood cultures, and fluids promptly.",
        ),
        RagItem(
            query="Why are antibiotics given early in sepsis?",
            relevant_ids=["sepsis_definition", "sepsis_treatment", "sepsis_hour1"],
            reference_answer="Because sepsis is organ dysfunction from "
            "infection, broad-spectrum antibiotics are given within one "
            "hour of recognition as part of the first-hour bundle.",
        ),
    ],
}


@dataclass(frozen=True)
class DatasetStudy:
    """
    Collected metrics for one dataset across all five baselines.

    Parameters
    ----------
    preset : str
        Dataset preset name.
    n_train : int
        Number of training rows in the shared split.
    n_test : int
        Number of held-out rows in the shared split.
    n_features : int
        Number of engineered features.
    central : TrainResult
        Baseline 1 training result.
    central_metrics : ClassificationMetrics
        Baseline 1 held-out classification metrics.
    federated : TrainResult
        Baseline 2 training result.
    federated_metrics : FederatedMetrics
        Baseline 2 cost / convergence block.
    fed_classification : ClassificationMetrics
        Baseline 2 held-out classification metrics.
    parameter_bytes : int
        Bytes in a single federated weight exchange (``parameter_set_bytes``).
    rag_per_query : dict[str, RAGQualityMetrics]
        Baseline 3 RAG quality per query.
    rag_average : RAGQualityMetrics
        Baseline 3 averaged RAG quality.
    agents_without_rag : AgentMetrics
        Baseline 4 multi-agent block.
    agents_with_rag : AgentMetrics
        Baseline 5 multi-agent block (agents + RAG evidence).
    sample_predictions : list[str]
        Predicted classes for the sampled test rows (decision consistency).
    error : str | None
        Error message when the dataset could not be processed.
    """

    preset: str
    n_train: int
    n_test: int
    n_features: int
    central: TrainResult
    central_metrics: ClassificationMetrics
    federated: TrainResult
    federated_metrics: FederatedMetrics
    fed_classification: ClassificationMetrics
    parameter_bytes: int
    rag_per_query: dict[str, RAGQualityMetrics]
    rag_average: RAGQualityMetrics
    agents_without_rag: AgentMetrics
    agents_with_rag: AgentMetrics
    sample_predictions: list[str]
    error: str | None = None


def split_dataset(
    dataset_dir: Path, preset: str, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Reproduce the shared held-out split used by the training baselines.

    Reuses the exported ``prepare_tabular_data`` and the same
    ``train_test_split`` parameters as ``AnalysisService.train`` so the
    held-out rows are identical for every baseline.

    Parameters
    ----------
    dataset_dir : Path
        Directory containing the preset CSV.
    preset : str
        Preset name (must exist in ``api.services.PRESETS``).
    test_size : float
        Held-out fraction.
    seed : int
        Reproducibility seed.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        ``(train_x, test_x, train_y, test_y)``.

    Raises
    ------
    InvalidInputError
        If the CSV cannot be prepared.
    """

    file_name, target = PRESETS[preset]
    features, labels = prepare_tabular_data(dataset_dir / file_name, target, None)
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )


def run_centralized(
    service: AnalysisService, preset: str, test_size: float, seed: int
) -> TrainResult:
    """
    Baseline 1 — train a centralized model through the service.

    Parameters
    ----------
    service : AnalysisService
        Service bound to the dataset directory.
    preset : str
        Dataset preset name.
    test_size : float
        Held-out fraction.
    seed : int
        Reproducibility seed.

    Returns
    -------
    TrainResult
        Training result (fitted model replaces ``service.model``).
    """

    return service.train(preset=preset, federated=False, test_size=test_size, seed=seed)


def run_federated(
    service: AnalysisService,
    preset: str,
    clients: int,
    rounds: int,
    test_size: float,
    seed: int,
) -> tuple[TrainResult, FederatedMetrics]:
    """
    Baseline 2 — train a federated (FedAvg) model through the service.

    Parameters
    ----------
    service : AnalysisService
        Service bound to the dataset directory.
    preset : str
        Dataset preset name.
    clients : int
        Number of simulated hospital clients.
    rounds : int
        Number of federated rounds.
    test_size : float
        Held-out fraction.
    seed : int
        Reproducibility seed.

    Returns
    -------
    tuple[TrainResult, FederatedMetrics]
        Training result and the hydrated cost / convergence metrics.
    """

    result = service.train(
        preset=preset,
        federated=True,
        clients=clients,
        rounds=rounds,
        test_size=test_size,
        seed=seed,
    )
    raw = result.federated_metrics or {}
    fed = FederatedMetrics(
        **{
            key: value
            for key, value in raw.items()
            if key in FederatedMetrics.__dataclass_fields__
        }
    )
    return result, fed


def score_model(
    service: AnalysisService, test_x: pd.DataFrame, test_y: pd.Series
) -> ClassificationMetrics:
    """
    Score the service's fitted model on the shared held-out split.

    Parameters
    ----------
    service : AnalysisService
        Service whose ``model`` is fitted.
    test_x : pd.DataFrame
        Held-out features.
    test_y : pd.Series
        Held-out labels.

    Returns
    -------
    ClassificationMetrics
        Full accuracy / precision / recall / F1 / MCC / AUC block.
    """

    y_pred = np.asarray(service.model.predict(test_x))
    y_score = np.asarray(service.model.predict_proba(test_x), dtype=np.float64)
    return classification_metrics(
        test_y, y_pred, y_score, labels=service.model.classes_
    )


def build_dataset_pipeline(corpus: Sequence[tuple[str, str, str]]) -> RAGPipeline:
    """
    Build an ingested RAG pipeline over a literal per-dataset corpus.

    Parameters
    ----------
    corpus : Sequence[tuple[str, str, str]]
        ``(document id, source, text)`` triples.

    Returns
    -------
    RAGPipeline
        Ingested pipeline using the configured default embedder.
    """

    pipeline = RAGPipeline()
    pipeline.ingest_documents(
        [
            Document(id=doc_id, text=text, source=source)
            for doc_id, source, text in corpus
        ]
    )
    return pipeline


def evaluate_rag(
    pipeline: RAGPipeline,
    items: Sequence[RagItem],
    top_k: int = RAG_TOP_K,
) -> tuple[dict[str, RAGQualityMetrics], RAGQualityMetrics]:
    """
    Baseline 3 — compute RAG quality metrics over a fixed query set.

    Parameters
    ----------
    pipeline : RAGPipeline
        Ingested retrieval pipeline.
    items : Sequence[RagItem]
        Queries with ground-truth relevant ids and reference answers.
    top_k : int
        Retrieval depth.

    Returns
    -------
    tuple[dict[str, RAGQualityMetrics], RAGQualityMetrics]
        Per-query metrics and the arithmetic mean across queries.
    """

    per_query: dict[str, RAGQualityMetrics] = {}
    totals = np.zeros(4, dtype=float)
    for item in items:
        results = pipeline.retrieve(item.query, top_k=top_k)
        chunks = [(result.chunk.document_id, result.chunk.text) for result in results]
        quality = rag_quality_metrics(
            query=item.query,
            answer=item.reference_answer,
            retrieved_chunks=chunks,
            relevant_chunk_ids=item.relevant_ids,
            embedder=pipeline.retriever.embedder,
        )
        per_query[item.query] = quality
        totals += np.array(
            [
                quality.context_precision,
                quality.context_recall,
                quality.faithfulness,
                quality.answer_relevancy,
            ]
        )
    average = RAGQualityMetrics(*tuple(totals / len(items)))
    return per_query, average


def _report_sections(report: ClinicalReport) -> list[str]:
    """
    Extract the deterministic "task outputs" of a clinical report.

    Maps the five crew responsibilities onto the assembled report fields:
    coordinator summary, disease prediction, risk assessment, evidence
    retrieval, and treatment recommendations. Empty strings are kept so
    ``task_completion_rate`` reflects steps that produced no output.

    Parameters
    ----------
    report : ClinicalReport
        An assembled clinical report.

    Returns
    -------
    list[str]
        Five section outputs in execution order.
    """

    prediction = report.prediction.predicted_class if report.prediction else ""
    risk_parts: list[str] = []
    if report.risk is not None:
        risk_parts.append(report.risk.risk_level)
        risk_parts.extend(report.risk.risk_factors)
    return [
        report.patient_summary,
        prediction,
        " ".join(risk_parts),
        report.context,
        " ".join(report.recommendations),
    ]


def evaluate_agents(
    service: AnalysisService,
    preset: str,
    test_x: pd.DataFrame,
    n_patients: int,
    rag_pipeline: RAGPipeline | None = None,
) -> tuple[AgentMetrics, list[str]]:
    """
    Baselines 4/5 — run the deterministic crew over sample test rows.

    Parameters
    ----------
    service : AnalysisService
        Service with a fitted model (used for the prediction step).
    preset : str
        Dataset preset name (used for patient ids).
    test_x : pd.DataFrame
        Held-out features; the first ``n_patients`` rows are analyzed.
    n_patients : int
        Number of sample patients to analyze.
    rag_pipeline : RAGPipeline | None
        Evidence pipeline wired into the crew; ``None`` disables the
        evidence step (Baseline 4).

    Returns
    -------
    tuple[AgentMetrics, list[str]]
        Agent metrics over all crew outputs and the predicted classes.
    """

    service.rag_pipeline = rag_pipeline
    crew_results: list[str] = []
    predictions: list[str] = []
    for index in range(min(n_patients, len(test_x))):
        row = test_x.iloc[index].to_dict()
        age_value = row.get("age")
        age = (
            int(age_value) if age_value is not None and not pd.isna(age_value) else None
        )
        patient = PatientInfo(
            name=f"{preset} patient {index}",
            id=f"{preset}-{index}",
            age=age,
        )
        report = service.analyze(patient, row)
        crew_results.extend(_report_sections(report))
        if report.prediction is not None:
            predictions.append(report.prediction.predicted_class)
    return compute_agent_metrics(crew_results, predictions), predictions


def run_study(
    dataset_dir: Path,
    artifacts_dir: Path,
    clients: int,
    rounds: int,
    test_size: float,
    seed: int,
    n_patients: int,
    presets: Sequence[str] | None = None,
) -> list[DatasetStudy]:
    """
    Run all five baselines for the requested datasets.

    Parameters
    ----------
    dataset_dir : Path
        Directory containing the preset CSVs.
    artifacts_dir : Path
        Scratch directory for model artifacts written by training.
    clients : int
        Federated clients (Baselines 2-5).
    rounds : int
        Federated rounds (Baselines 2-5).
    test_size : float
        Shared held-out fraction.
    seed : int
        Reproducibility seed.
    n_patients : int
        Sample patients per dataset for the multi-agent baselines.
    presets : Sequence[str] | None
        Presets to run; defaults to all of ``PRESETS``.

    Returns
    -------
    list[DatasetStudy]
        One study record per processed dataset (in ``presets`` order).
    """

    targets = list(presets) if presets is not None else sorted(PRESETS)
    studies: list[DatasetStudy] = []
    for preset in targets:
        dataset_path = dataset_dir / PRESETS[preset][0]
        if not dataset_path.exists():
            logger.warning(
                "Skipping %s: %s not found under %s",
                preset,
                dataset_path.name,
                dataset_dir,
            )
            continue
        logger.info("Running baselines for %s (%s)", preset, dataset_path)
        service = AnalysisService(dataset_dir=dataset_dir, artifacts_dir=artifacts_dir)
        try:
            train_x, test_x, _train_y, test_y = split_dataset(
                dataset_dir, preset, test_size, seed
            )
            central = run_centralized(service, preset, test_size, seed)
            central_metrics = score_model(service, test_x, test_y)

            federated, fed_metrics = run_federated(
                service, preset, clients, rounds, test_size, seed
            )
            fed_classification = score_model(service, test_x, test_y)
            parameter_bytes = parameter_set_bytes(service.model.get_parameters())

            pipeline = build_dataset_pipeline(RAG_CORPORA[preset])
            rag_per_query, rag_average = evaluate_rag(pipeline, RAG_EVALUATION[preset])

            agents_without_rag, predictions = evaluate_agents(
                service, preset, test_x, n_patients, rag_pipeline=None
            )
            agents_with_rag, _ = evaluate_agents(
                service, preset, test_x, n_patients, rag_pipeline=pipeline
            )

            studies.append(
                DatasetStudy(
                    preset=preset,
                    n_train=len(train_x),
                    n_test=len(test_x),
                    n_features=test_x.shape[1],
                    central=central,
                    central_metrics=central_metrics,
                    federated=federated,
                    federated_metrics=fed_metrics,
                    fed_classification=fed_classification,
                    parameter_bytes=parameter_bytes,
                    rag_per_query=rag_per_query,
                    rag_average=rag_average,
                    agents_without_rag=agents_without_rag,
                    agents_with_rag=agents_with_rag,
                    sample_predictions=predictions,
                )
            )
            logger.info("Study complete for %s", preset)
        except Exception as error:  # noqa: BLE001 - record and continue
            logger.error("Study failed for %s: %s", preset, error)
            studies.append(
                DatasetStudy(
                    preset=preset,
                    n_train=0,
                    n_test=0,
                    n_features=0,
                    central=TrainResult("", "", "", 0.0),
                    central_metrics=ClassificationMetrics(
                        0.0, 0.0, 0.0, 0.0, None, None, None, None, 0
                    ),
                    federated=TrainResult("", "", "", 0.0),
                    federated_metrics=FederatedMetrics(0, 0, 0, 0, (), 0.0, (), None),
                    fed_classification=ClassificationMetrics(
                        0.0, 0.0, 0.0, 0.0, None, None, None, None, 0
                    ),
                    parameter_bytes=0,
                    rag_per_query={},
                    rag_average=RAGQualityMetrics(0.0, 0.0, 0.0, 0.0),
                    agents_without_rag=AgentMetrics(0.0, 0.0, 0.0),
                    agents_with_rag=AgentMetrics(0.0, 0.0, 0.0),
                    sample_predictions=[],
                    error=str(error),
                )
            )
    return studies


def _fmt(value: float | int | None, digits: int = 3) -> str:
    """Format a number, rendering ``None`` as ``n/a``."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _comm_cells(fed: FederatedMetrics) -> tuple[str, str]:
    """Communication cost and convergence-round cells for a federated row."""
    return _fmt(fed.total_bytes_exchanged), _fmt(fed.convergence_round)


def build_dataset_section(study: DatasetStudy, config: Mapping[str, Any]) -> str:
    """
    Render the markdown section (table + metric detail) for one dataset.

    Parameters
    ----------
    study : DatasetStudy
        Collected metrics for the dataset.
    config : Mapping[str, Any]
        Study configuration (test size, seed, clients, rounds, top-k,
        sample patients) for the method note.

    Returns
    -------
    str
        Markdown for the dataset.
    """

    name = DATASET_DISPLAY_NAMES.get(study.preset, study.preset)
    header = [
        f"## {name}",
        "",
        f"`{study.preset}` - {study.n_train} train / {study.n_test} test rows, "
        f"{study.n_features} features. Shared split: test_size={config['test_size']}, "
        f"seed={config['seed']}. Federated: {config['clients']} clients, "
        f"{config['rounds']} rounds. RAG top-k={config['rag_top_k']}. Agent sample "
        f"patients={config['n_patients']}.",
    ]
    if study.error:
        skipped = f"**Skipped** - {study.error}"
        return "\n".join([*header, "", skipped, ""])

    comm, conv = _comm_cells(study.federated_metrics)
    fed_acc = _fmt(study.fed_classification.accuracy)
    fed_f1 = _fmt(study.fed_classification.f1_macro)
    fed_auc = _fmt(study.fed_classification.roc_auc)
    cent_acc = _fmt(study.central_metrics.accuracy)
    cent_f1 = _fmt(study.central_metrics.f1_macro)
    cent_auc = _fmt(study.central_metrics.roc_auc)
    rag = study.rag_average
    rag_faith = _fmt(rag.faithfulness)
    rag_prec = _fmt(rag.context_precision)
    ag4 = study.agents_without_rag
    ag5 = study.agents_with_rag

    rows = [
        "| Baseline | Accuracy | F1 | ROC-AUC | Comm. cost (bytes) | "
        "Convergence round | RAG faithfulness | RAG context precision | "
        "Agent task completion | Agent collaboration |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| 1. Centralized | {cent_acc} | {cent_f1} | {cent_auc} | "
        "n/a | n/a | n/a | n/a | n/a | n/a |",
        f"| 2. Federated only | {fed_acc} | {fed_f1} | {fed_auc} | "
        f"{comm} | {conv} | n/a | n/a | n/a | n/a |",
        f"| 3. Federated + RAG | {fed_acc} | {fed_f1} | {fed_auc} | "
        f"{comm} | {conv} | {rag_faith} | {rag_prec} | n/a | n/a |",
        f"| 4. Federated + Multi-Agent | {fed_acc} | {fed_f1} | {fed_auc} | "
        f"{comm} | {conv} | n/a | n/a | "
        f"{_fmt(ag4.task_completion_rate)} | {_fmt(ag4.agent_collaboration_score)} |",
        f"| 5. Proposed (full) | {fed_acc} | {fed_f1} | {fed_auc} | "
        f"{comm} | {conv} | {rag_faith} | {rag_prec} | "
        f"{_fmt(ag5.task_completion_rate)} | {_fmt(ag5.agent_collaboration_score)} |",
    ]

    fed = study.federated_metrics
    ag4_comp = _fmt(ag4.task_completion_rate)
    ag5_comp = _fmt(ag5.task_completion_rate)
    ag4_collab = _fmt(ag4.agent_collaboration_score)
    ag5_collab = _fmt(ag5.agent_collaboration_score)
    detail: list[str] = [
        "",
        "### Metric detail",
        "",
        "- Comm. cost = total bytes exchanged over the whole federated run "
        f"({fed.n_rounds} rounds x {fed.n_clients} clients; "
        f"one weight set = {study.parameter_bytes} bytes).",
        f"- Convergence round: {_fmt(fed.convergence_round)} "
        f"(round-to-round accuracy deltas: "
        f"{[round(d, 4) for d in fed.accuracy_deltas]}).",
        "- Classification: centralized accuracy / F1 / ROC-AUC "
        f"({cent_acc} / {cent_f1} / {cent_auc}) vs federated "
        f"({fed_acc} / {fed_f1} / {fed_auc}).",
        "- RAG (averaged over 5 queries): context precision "
        f"{_fmt(rag.context_precision)}, context recall {_fmt(rag.context_recall)}, "
        f"faithfulness {_fmt(rag.faithfulness)}, answer relevancy "
        f"{_fmt(rag.answer_relevancy)}.",
        f"- Agent task completion: without RAG {ag4_comp} "
        f"(5 sections incl. empty evidence), with RAG {ag5_comp}; "
        f"agent collaboration without RAG {ag4_collab} / with RAG {ag5_collab}; "
        f"decision consistency {_fmt(ag5.decision_consistency)} "
        f"over {len(study.sample_predictions)} sampled patients "
        f"(predictions {study.sample_predictions}).",
        "- Baseline 4 runs the crew without the RAG evidence step; Baseline 5 "
        "wires the RAG pipeline into the same crew (evidence context fills "
        "the retrieval task, which is why task completion rises).",
    ]
    return "\n".join([*header, *rows, *detail])


def build_study_markdown(
    studies: Sequence[DatasetStudy], config: Mapping[str, Any]
) -> str:
    """
    Render the full study document (intro, tables, detail).

    Parameters
    ----------
    studies : Sequence[DatasetStudy]
        One record per processed dataset.
    config : Mapping[str, Any]
        Study configuration used for the method note.

    Returns
    -------
    str
        Markdown document (tables only; a hand-written Findings section is
        preserved by :func:`write_results`).
    """

    sections = [
        "# Baseline Comparison Study (paper §13)",
        "",
        "Generated by `backend/scripts/baseline_study.py` on "
        f"{date.today().isoformat()}. "
        "Every metric is produced by the existing modules "
        "(`evaluation/metrics.py`, `federated/metrics.py`, `rag/metrics.py`, "
        "`CrewAI/orchestrator/metrics.py`) - nothing here reimplements them.",
        "",
        "**Method (pilot scale).** Each dataset is split once with "
        f"`test_size={config['test_size']}`, `seed={config['seed']}`, and every "
        "baseline is scored on that same held-out split. Baselines 2-5 share the "
        f"federated model (`{config['clients']}` clients, "
        f"`{config['rounds']}` rounds); "
        "RAG and multi-agent layers do not retrain it, so their classification "
        "block is reported once and reused. The RAG layer is graded on 5 literal "
        "clinical queries per dataset against reference answers grounded in a "
        "literal per-dataset corpus. Agent metrics come from the deterministic, "
        "LLM-free crew over a handful of sample test rows. `n/a` means the metric "
        "does not apply to that configuration — no cells are fabricated.",
        "",
        "n8n is the orchestration layer (webhook → FastAPI → crew) already "
        "exercised live in `docs/CHANGELOG.md`'s verification entries; it adds no "
        "independent metric, so the Proposed row is the union of the classification, "
        "RAG, and agent metrics.",
        "",
    ]
    for study in studies:
        sections.append(build_dataset_section(study, config))
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


_FINDINGS_MARKER = "\n## Findings\n"


def write_results(out: Path, markdown: str) -> None:
    """
    Write the tables to disk, preserving a hand-written Findings section.

    The script regenerates the intro + tables on every run. If the output
    file already contains a ``## Findings`` section (added by a human after
    inspecting the numbers), it is re-appended unchanged so the narrative
    survives re-runs.

    Parameters
    ----------
    out : Path
        Destination markdown file.
    markdown : str
        Generated tables document.
    """

    if out.exists():
        existing = out.read_text(encoding="utf-8")
        if _FINDINGS_MARKER in existing:
            findings = existing[existing.index(_FINDINGS_MARKER) :]
            markdown = markdown.rstrip() + "\n\n" + findings.lstrip("\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """
    Run the baseline comparison study.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        ``0`` on success (datasets missing from ``DATASET_DIR`` are
        skipped with a warning, not an error).
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(os.environ.get("DATASET_DIR", ".")),
        help="Directory containing the preset CSVs (or DATASET_DIR env).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output markdown file.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/baseline_study"),
        help="Scratch directory for trained model artifacts.",
    )
    parser.add_argument("--clients", type=int, default=FEDERATED_CLIENTS)
    parser.add_argument("--rounds", type=int, default=FEDERATED_ROUNDS)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-patients", type=int, default=N_PATIENTS)
    parser.add_argument(
        "--only",
        choices=sorted(PRESETS),
        default=None,
        help="Run a single dataset (dev/testing).",
    )
    args = parser.parse_args(argv)

    config: Mapping[str, Any] = {
        "test_size": args.test_size,
        "seed": args.seed,
        "clients": args.clients,
        "rounds": args.rounds,
        "n_patients": args.n_patients,
        "rag_top_k": RAG_TOP_K,
    }
    presets = [args.only] if args.only is not None else None
    studies = run_study(
        dataset_dir=args.dataset_dir,
        artifacts_dir=args.artifacts_dir,
        clients=args.clients,
        rounds=args.rounds,
        test_size=args.test_size,
        seed=args.seed,
        n_patients=args.n_patients,
        presets=presets,
    )
    markdown = build_study_markdown(studies, config)
    write_results(args.out, markdown)
    sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
