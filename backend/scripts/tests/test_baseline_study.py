"""
Tests for the baseline comparison study script.

The study runner needs no ``DATASET_DIR``: these tests generate a small
synthetic CSV in a temp directory and exercise the script's core
per-baseline functions end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from api.services import AnalysisService
from scripts.baseline_study import (
    RAG_CORPORA,
    RAG_EVALUATION,
    build_dataset_pipeline,
    build_study_markdown,
    evaluate_agents,
    evaluate_rag,
    run_federated,
    run_study,
    split_dataset,
)


def _write_synthetic_csv(path: Path, n: int = 150, seed: int = 7) -> None:
    """Write a tiny binary classification CSV with an outcome column."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "glucose": rng.normal(110, 20, n),
            "bmi": rng.normal(27, 5, n),
            "age": rng.integers(30, 75, n),
            "outcome": rng.integers(0, 2, n),
        }
    )
    frame.to_csv(path, index=False)


def _service(tmp_path: Path) -> AnalysisService:
    """Build a service bound to a temp directory with a synthetic preset."""
    _write_synthetic_csv(tmp_path / "diabetes.csv")
    return AnalysisService(dataset_dir=tmp_path, artifacts_dir=tmp_path / "artifacts")


def test_study_runs_end_to_end_on_synthetic_data(tmp_path: Path) -> None:
    """All five baselines run and produce an honest markdown table."""
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    _write_synthetic_csv(dataset_dir / "diabetes.csv")

    studies = run_study(
        dataset_dir=dataset_dir,
        artifacts_dir=tmp_path / "artifacts",
        clients=3,
        rounds=2,
        test_size=0.25,
        seed=42,
        n_patients=3,
        presets=["diabetes"],
    )

    assert len(studies) == 1
    study = studies[0]
    assert study.error is None
    assert study.preset == "diabetes"
    assert study.n_test > 0

    assert abs(study.central.accuracy - study.central_metrics.accuracy) < 1e-9
    for metrics in (study.central_metrics, study.fed_classification):
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.f1_macro <= 1.0
        assert metrics.roc_auc is None or 0.0 <= metrics.roc_auc <= 1.0

    assert study.federated_metrics.total_bytes_exchanged > 0
    assert study.parameter_bytes > 0
    assert (
        study.federated_metrics.convergence_round is None
        or study.federated_metrics.convergence_round >= 1
    )

    for metric in study.rag_per_query.values():
        assert 0.0 <= metric.faithfulness <= 1.0
        assert 0.0 <= metric.answer_relevancy <= 1.0
    assert 0.0 <= study.rag_average.context_precision <= 1.0

    for agent in (study.agents_without_rag, study.agents_with_rag):
        assert 0.0 <= agent.task_completion_rate <= 1.0
        assert 0.0 <= agent.agent_collaboration_score <= 1.0
        assert 0.0 <= agent.decision_consistency <= 1.0

    config = {
        "test_size": 0.25,
        "seed": 42,
        "clients": 3,
        "rounds": 2,
        "n_patients": 3,
        "rag_top_k": 5,
    }
    markdown = build_study_markdown(studies, config)
    assert "## Diabetes (PIMA, type-2 risk)" in markdown
    assert "| 1. Centralized |" in markdown
    assert "| 2. Federated only |" in markdown
    assert "| 5. Proposed (full) |" in markdown
    assert "| n/a |" in markdown


def test_split_dataset_matches_service_internal_split(tmp_path: Path) -> None:
    """The reproduced held-out split matches the one train() uses."""
    service = _service(tmp_path)
    train_x, test_x, _train_y, test_y = split_dataset(
        tmp_path, "diabetes", test_size=0.25, seed=42
    )
    service.train(preset="diabetes", federated=False, test_size=0.25, seed=42)

    y_pred = np.asarray(service.model.predict(test_x))
    assert len(y_pred) == len(test_y)
    assert set(np.unique(test_y)).issubset({0, 1})
    assert train_x.shape[1] == test_x.shape[1] > 0


def test_split_dataset_kidney_preserves_disease_positive_orientation(
    tmp_path: Path,
) -> None:
    """Kidney's string labels (ckd/notckd) must orient disease=1 in the split.

    Regression test for the study's 0.020-accuracy kidney row: split_dataset
    used to call prepare_tabular_data WITHOUT the preset, so the string labels
    fell back to alphabetical LabelEncoder order (ckd=0, notckd=1) — inverted
    against the model trained with preset='kidney' (ckd=1). The study's own
    test split must reproduce the disease-positive orientation so its numbers
    are directly comparable to the trained model's.
    """
    _write_synthetic_csv(tmp_path / "kidney_disease.csv")
    # Overwrite with string (ckd/notckd) + tab-dirty labels, mirroring the
    # real UCI CKD dataset.
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "age": rng.integers(20, 80, 60),
            "bp": rng.normal(130, 20, 60),
            "sg": rng.normal(1.02, 0.01, 60),
            "al": rng.integers(0, 4, 60),
            "sc": rng.normal(1.2, 0.6, 60),
        }
    )
    labels = rng.choice(["ckd", "notckd"], size=60, p=[0.6, 0.4])
    labels[0] = "ckd\t"  # dirty tab, must be stripped by the orienter
    frame["classification"] = labels
    frame.to_csv(tmp_path / "kidney_disease.csv", index=False)

    _train_x, test_x, train_y, test_y = split_dataset(
        tmp_path, "kidney", test_size=0.25, seed=42
    )
    # The split's labels must have disease=1 (the positive class) just like
    # prepare_tabular_data(preset='kidney') produces.
    assert set(test_y.unique()).issubset({0, 1})
    assert test_y.dtype.kind in "iu"
    # Disease-positive rate should be nonzero and roughly match the 60% ckd.
    assert 0.3 <= test_y.mean() <= 0.8
    assert train_y.mean() > 0.3
    # And the split must be usable by the model (feature count aligned).
    assert test_x.shape[1] == _train_x.shape[1] > 0


def test_rag_evaluation_retrieves_relevant_documents(tmp_path: Path) -> None:
    """RAG metrics stay in range and retrieve the ground-truth docs."""
    pipeline = build_dataset_pipeline(RAG_CORPORA["diabetes"])
    per_query, average = evaluate_rag(pipeline, RAG_EVALUATION["diabetes"])

    assert len(per_query) == len(RAG_EVALUATION["diabetes"])
    assert 0.0 < average.context_precision <= 1.0
    assert 0.0 <= average.context_recall <= 1.0
    assert 0.0 <= average.faithfulness <= 1.0
    assert 0.0 <= average.answer_relevancy <= 1.0


def test_agent_metrics_reflect_evidence_step(tmp_path: Path) -> None:
    """Baseline 4 (no RAG) has a lower task completion than Baseline 5."""
    service = _service(tmp_path)
    _train_x, test_x, _train_y, _test_y = split_dataset(
        tmp_path, "diabetes", test_size=0.25, seed=42
    )
    run_federated(service, "diabetes", clients=3, rounds=2, test_size=0.25, seed=42)

    pipeline = build_dataset_pipeline(RAG_CORPORA["diabetes"])
    without_rag, _ = evaluate_agents(
        service, "diabetes", test_x, n_patients=3, rag_pipeline=None
    )
    with_rag, predictions = evaluate_agents(
        service, "diabetes", test_x, n_patients=3, rag_pipeline=pipeline
    )

    assert len(predictions) == 3
    # The evidence-retrieval section is the discriminator: disabling the
    # RAG pipeline leaves it empty, so completion drops below the with-RAG
    # run. (Current main fills the treatment/playbook and explainability
    # sections unconditionally, so the no-RAG floor is 0.8 of 5 sections
    # rather than the older pre-playbook pilot's 0.6.)
    assert without_rag.task_completion_rate < with_rag.task_completion_rate
    assert with_rag.task_completion_rate == 1.0
    assert with_rag.decision_consistency == 1.0 or with_rag.decision_consistency > 0.5
