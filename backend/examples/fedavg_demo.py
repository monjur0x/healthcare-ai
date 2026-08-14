"""
End-to-end federated learning demo: CSV -> preprocessing -> FedAvg.

Loads a hospital CSV, preprocesses it with the CSV pipeline, partitions
the training rows across simulated hospital clients, trains a tabular
MLP with the synchronous FedAvg server, and reports global metrics
against a central non-federated baseline.

Usage (run from ``backend/``):

    python -m examples.fedavg_demo --preset diabetes --clients 3 --rounds 3

Or with explicit files:

    python -m examples.fedavg_demo --dataset path/to/data.csv \\
        --target outcome --clients 3 --rounds 3

The dataset directory can be given with ``--dataset-dir`` or the
``DATASET_DIR`` environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from evaluation import evaluate_classifier
from federated import FedAvgServer, FederatedClient, make_global_evaluator
from models import TabularClassifier
from preprocessing.csv import CSVPipeline
from preprocessing.logger import get_logger

logger = get_logger(__name__)

# Convenience presets mapping a dataset name to (file name, target column).
PRESETS: dict[str, tuple[str, str]] = {
    "diabetes": ("diabetes.csv", "Outcome"),
    "heart": ("heart_disease_uci.csv", "num"),
    "kidney": ("kidney_disease.csv", "classification"),
    "sepsis": ("sepsis_icu_synthetic.csv", "sepsis_label"),
}


def _normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to lowercase snake_case (pipeline convention)."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for column in dataframe.columns:
        name = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        if name in seen:
            name = f"{name}_{len(seen)}"
        seen.add(name)
        cleaned.append(name)
    dataframe.columns = cleaned
    return dataframe


def prepare_data(
    dataset: Path, target: str, max_rows: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load, preprocess, and split a CSV into features and encoded labels.

    Parameters
    ----------
    dataset : Path
        Path to the source CSV.
    target : str
        Target column name.
    max_rows : int | None
        Optional row cap for large datasets.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Feature matrix and integer-encoded target labels.
    """

    dataframe = pd.read_csv(dataset)
    if max_rows is not None:
        dataframe = dataframe.sample(n=min(max_rows, len(dataframe)), random_state=42)
    dataframe = _normalize_columns(dataframe)
    target = target.strip().lower().replace(" ", "_").replace("-", "_")

    if target not in dataframe.columns:
        raise ValueError(f"Target column '{target}' not found in {dataset}.")

    y_raw = dataframe[target]
    feature_frame = dataframe.drop(columns=[target])
    for column in ("id", "subject_id"):
        if column in feature_frame.columns:
            feature_frame = feature_frame.drop(columns=[column])

    valid = y_raw.notna()
    feature_frame = feature_frame.loc[valid]
    y_raw = y_raw.loc[valid]

    if pd.api.types.is_string_dtype(y_raw):
        y_raw = y_raw.str.strip()
        for column in feature_frame.columns:
            if pd.api.types.is_string_dtype(feature_frame[column]):
                feature_frame[column] = feature_frame[column].str.strip()

    result = CSVPipeline(input_columns=tuple(feature_frame.columns)).run(feature_frame)
    features = result.dataframe.to_numpy(dtype=np.float64)
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("Pipeline produced no usable features.")

    y = LabelEncoder().fit_transform(y_raw.loc[result.dataframe.index])
    logger.info(
        "Prepared %d samples, %d features, %d classes",
        features.shape[0],
        features.shape[1],
        np.unique(y).size,
    )
    return features, y


def partition_clients(
    features: np.ndarray,
    labels: np.ndarray,
    n_clients: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition training data into class-balanced client shards."""
    counts = np.bincount(labels)
    if counts.min() < n_clients:
        raise ValueError(
            f"Rarest class has {counts.min()} samples, fewer than {n_clients} "
            "clients. Reduce --clients or drop the rare class."
        )
    splitter = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=seed)
    return [
        (features[index], labels[index])
        for index, _ in splitter.split(features, labels)
    ]


def build_report(
    server,
    baseline_accuracy: float,
    baseline_auc: float | None,
) -> dict:
    """Assemble the round-by-round demo report."""
    return {
        "baseline_accuracy": baseline_accuracy,
        "baseline_roc_auc": baseline_auc,
        "rounds": [round_result.to_dict() for round_result in server.history],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the federated learning demo."""
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
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--model", default="mlp", choices=("mlp", "logistic"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts/fedavg"))
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

    features, labels = prepare_data(dataset, target, args.max_rows)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        stratify=labels,
        random_state=args.seed,
    )

    shards = partition_clients(train_x, train_y, args.clients, args.seed)
    clients = [
        FederatedClient(
            lambda: TabularClassifier(model_name=args.model),
            shard_x,
            shard_y,
            test_x,
            test_y,
        )
        for shard_x, shard_y in shards
    ]
    evaluator = make_global_evaluator(
        lambda: TabularClassifier(model_name=args.model), test_x, test_y
    )

    server = FedAvgServer(
        clients=clients, num_rounds=args.rounds, evaluate_fn=evaluator
    ).run()

    baseline = evaluate_classifier(
        TabularClassifier(model_name=args.model).fit(train_x, train_y),
        test_x,
        test_y,
    )

    report = build_report(server, baseline.accuracy, baseline.roc_auc)
    logger.info("Baseline accuracy: %.4f", baseline.accuracy)
    for result in server.history:
        logger.info(
            "Round %d: accuracy=%.4f roc_auc=%s log_loss=%s",
            result.round_index,
            result.accuracy,
            result.roc_auc,
            result.log_loss,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    global_model = TabularClassifier(model_name=args.model)
    global_model.set_parameters(server.global_parameters)
    global_model.save(args.out / "global_model.joblib")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    logger.info("Artifacts written to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
