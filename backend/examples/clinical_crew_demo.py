"""
End-to-end clinical crew demo: CSV -> preprocessing -> model -> crew.

Loads a hospital CSV, preprocesses it with the CSV pipeline, trains a
small tabular model, then runs the deterministic clinical crew
(prediction -> risk -> evidence retrieval -> clinical report) on one
patient row. The report is written as JSON.

The evidence step uses a small built-in medical corpus unless a corpus
directory of ``.txt``/``.md`` files is supplied with ``--corpus-dir``.

Usage (run from ``backend/``):

    python -m examples.clinical_crew_demo --preset diabetes

The dataset directory can be given with ``--dataset-dir`` or the
``DATASET_DIR`` environment variable.

This demo is fully offline: it never calls an LLM (ADR-008). To add
CrewAI agent narration instead, set ``CREW_LLM_API_KEY``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from CrewAI.orchestrator import ClinicalCrew
from CrewAI.orchestrator.schemas import PatientInfo
from models import TabularClassifier
from preprocessing.csv import CSVPipeline
from preprocessing.logger import get_logger
from rag import RAGPipeline

logger = get_logger(__name__)

PRESETS: dict[str, tuple[str, str]] = {
    "diabetes": ("diabetes.csv", "Outcome"),
    "heart": ("heart_disease_uci.csv", "num"),
    "kidney": ("kidney_disease.csv", "classification"),
    "sepsis": ("sepsis_icu_synthetic.csv", "sepsis_label"),
}

BUILTIN_CORPUS: list[str] = [
    "diabetes mellitus is managed with metformin, lifestyle changes, and "
    "regular glucose monitoring",
    "chronic hypertension management combines dietary sodium reduction, "
    "exercise, and blood pressure lowering medication",
    "sepsis is life-threatening organ dysfunction from infection and "
    "requires broad-spectrum antibiotics within one hour of recognition",
]


def prepare_data(
    dataset: Path, target: str, max_rows: int | None
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Preprocess a CSV and return the engineered features and target.

    Parameters
    ----------
    dataset : Path
        Path to the source CSV.
    target : str
        Target column name.
    max_rows : int | None
        Optional cap on the number of rows used.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        ``(features, labels, raw_rows)`` where ``raw_rows`` is the raw
        dataframe before preprocessing (used for clinical markers).
    """

    raw = pd.read_csv(dataset)
    if max_rows is not None:
        raw = raw.head(max_rows)

    seen: set[str] = set()
    cleaned: list[str] = []
    for column in raw.columns:
        name = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        if name in seen:
            name = f"{name}_{len(seen)}"
        seen.add(name)
        cleaned.append(name)
    raw.columns = cleaned
    target = str(target).strip().lower().replace(" ", "_").replace("-", "_")

    result = CSVPipeline().run(raw)
    features = result.dataframe.drop(columns=[target])
    labels = result.dataframe[target]
    with contextlib.suppress(TypeError, ValueError):
        labels = pd.to_numeric(labels).astype(int)
    return features, labels, raw


def build_pipeline(corpus_dir: Path | None) -> RAGPipeline:
    """
    Build and ingest a RAG pipeline from a corpus directory or a small
    built-in corpus.

    Parameters
    ----------
    corpus_dir : Path | None
        Directory of ``.txt``/``.md`` files, or None for the built-in
        corpus.

    Returns
    -------
    RAGPipeline
        Ingested retrieval pipeline.
    """

    pipeline = RAGPipeline()
    if corpus_dir is not None:
        from rag.documents import Document

        documents = [
            Document(id=path.name, text=path.read_text(encoding="utf-8"))
            for path in sorted(corpus_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}
        ]
        if not documents:
            raise ValueError(f"No .txt/.md documents under {corpus_dir}.")
        pipeline.ingest_documents(documents)
        logger.info("Ingested %d documents from %s", len(documents), corpus_dir)
    else:
        pipeline.ingest_texts(BUILTIN_CORPUS)
    return pipeline


def main(argv: list[str] | None = None) -> int:
    """Run the clinical crew demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Path to the source CSV.")
    parser.add_argument("--target", help="Target column name.")
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), help="Named dataset preset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(os.environ.get("DATASET_DIR", ".")),
        help="Directory for preset datasets (or DATASET_DIR env).",
    )
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("artifacts/crew"))
    args = parser.parse_args(argv)

    if args.preset is not None:
        file_name, preset_target = PRESETS[args.preset]
        dataset = args.dataset_dir / file_name
        target = args.target or preset_target
    else:
        if args.dataset is None or args.target is None:
            parser.error("Use --preset or provide both --dataset and --target.")
        dataset = args.dataset
        target = args.target

    features, labels, raw = prepare_data(dataset, target, args.max_rows)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=args.seed,
    )

    model = TabularClassifier(model_name="logistic").fit(train_x, train_y)
    patient_features = test_x.iloc[0].to_dict()
    patient_label = int(test_y.iloc[0])

    markers: dict[str, float] = {}
    for marker in ("glucose", "bmi", "age", "blood_pressure_systolic", "cholesterol"):
        if marker in raw.columns:
            value = raw[marker].dropna().iloc[0]
            try:
                markers[marker] = float(value)
            except (TypeError, ValueError):
                continue

    pipeline = build_pipeline(args.corpus_dir)
    crew = ClinicalCrew(
        patient=PatientInfo(name="Patient", id="demo-patient-1"),
        input_type="csv",
        model=model,
        features=patient_features,
        rag_pipeline=pipeline,
        markers=markers,
        recommendations=[
            "Review the report with a licensed physician before acting.",
        ],
    )

    report = crew.run_analysis()
    logger.info(
        "Prediction: %s (confidence %.2f), risk: %s",
        report.prediction.predicted_class if report.prediction else None,
        report.prediction.confidence if report.prediction else 0.0,
        report.risk.risk_level if report.risk else None,
    )

    payload = report.to_dict()
    payload["ground_truth"] = int(patient_label)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(payload, indent=2, default=float))
    logger.info("Artifacts written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
