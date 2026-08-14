"""
End-to-end federated learning demo for the image path.

Federates the CNN classifier over a directory of class-labelled image
folders (e.g. the brain-tumor MRI dataset): image preprocessing ->
``ImageClassifier`` -> FedAvg rounds -> evaluation report. Mirrors
``fedavg_demo.py`` for tabular data so both model families share the
same report shape (baseline, federated metrics, per-round history).

Usage (run from ``backend/``):

    python -m examples.image_fedavg_demo --dataset-dir path/to/Training \\
        --clients 3 --rounds 3 --max-per-class 80
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from evaluation import evaluate_classifier
from federated import FedAvgServer, FederatedClient, make_global_evaluator
from models import ImageClassifier
from preprocessing.image import ImagePipeline
from preprocessing.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def discover_classes(dataset_dir: Path) -> dict[str, list[Path]]:
    """
    Map class labels to image paths from a folder-per-class layout.

    Parameters
    ----------
    dataset_dir : Path
        Directory whose immediate subfolders are class labels.

    Returns
    -------
    dict[str, list[Path]]
        Class label to sorted image file paths.

    Raises
    ------
    ValueError
        If no class folders are found.
    """

    classes: dict[str, list[Path]] = {}
    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir():
            continue
        images = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
        )
        if images:
            classes[folder.name] = images
    if not classes:
        raise ValueError(f"No class subfolders with images found under {dataset_dir}.")
    logger.info(
        "Discovered %d classes (%s)",
        len(classes),
        ", ".join(f"{name}={len(files)}" for name, files in classes.items()),
    )
    return classes


def subsample(
    classes: dict[str, list[Path]], max_per_class: int | None, seed: int
) -> list[tuple[Path, str]]:
    """
    Deterministically cap the per-class sample count.

    Parameters
    ----------
    classes : dict[str, list[Path]]
        Class label to image file paths.
    max_per_class : int | None
        Optional cap per class.
    seed : int
        Random seed for reproducible subsampling.

    Returns
    -------
    list[tuple[Path, str]]
        Flat (path, class label) list.
    """

    rng = np.random.default_rng(seed)
    sources: list[tuple[Path, str]] = []
    for label, files in classes.items():
        chosen = files
        if max_per_class is not None and len(files) > max_per_class:
            chosen = list(rng.choice(files, size=max_per_class, replace=False))
        sources.extend((Path(path), label) for path in chosen)
    return sources


def load_batch(
    sources: list[tuple[Path, str]],
    image_size: int,
    channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image sources into a normalized model-ready batch.

    Parameters
    ----------
    sources : list[tuple[Path, str]]
        (path, class label) pairs.
    image_size : int
        Square target size in pixels.
    channels : int
        Target channel count (1 or 3).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float32 ``(N, H, W, C)`` images and integer-encoded labels.
    """

    pipeline = ImagePipeline(
        size=(image_size, image_size),
        channels=channels,
        normalize=True,
        augment=False,
    )
    images = pipeline.transform_batch([path for path, _ in sources]).image
    labels = LabelEncoder().fit_transform([label for _, label in sources])
    logger.info("Preprocessed %d images to shape %s", images.shape[0], images.shape)
    return images, labels


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


def run_image_fedavg(
    dataset_dir: Path,
    n_clients: int,
    n_rounds: int,
    epochs: int,
    batch_size: int,
    image_size: int,
    channels: int,
    max_per_class: int | None,
    test_size: float,
    seed: int,
):
    """
    Run the image FedAvg demo end to end.

    Parameters
    ----------
    dataset_dir : Path
        Directory of class-labelled image folders.
    n_clients : int
        Number of simulated hospital clients.
    n_rounds : int
        Number of FedAvg rounds.
    epochs : int
        CNN training epochs (warm start and baseline).
    batch_size : int
        CNN training batch size.
    image_size : int
        Square target size in pixels.
    channels : int
        Target channel count (1 or 3).
    max_per_class : int | None
        Optional per-class sample cap.
    test_size : float
        Fraction of samples held out as the central test set.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[FedAvgServer, dict]
        Completed server and the JSON-friendly report.
    """

    classes = discover_classes(dataset_dir)
    sources = subsample(classes, max_per_class, seed)
    X, y = load_batch(sources, image_size, channels)

    train_x, test_x, train_y, test_y = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    def model_factory() -> ImageClassifier:
        return ImageClassifier(
            in_channels=channels,
            epochs=epochs,
            batch_size=batch_size,
        )

    shards = partition_clients(train_x, train_y, n_clients, seed)
    clients = [
        FederatedClient(model_factory, shard_x, shard_y, test_x, test_y)
        for shard_x, shard_y in shards
    ]

    evaluator_model = model_factory().fit(train_x, train_y)
    evaluator = make_global_evaluator(lambda: evaluator_model, test_x, test_y)

    server = FedAvgServer(
        clients=clients, num_rounds=n_rounds, evaluate_fn=evaluator
    ).run()

    global_model = model_factory().fit(train_x, train_y)
    global_model.set_parameters(server.global_parameters)

    baseline = evaluate_classifier(
        model_factory().fit(train_x, train_y), test_x, test_y
    )
    report = {
        "n_classes": len(np.unique(y)),
        "baseline_accuracy": baseline.accuracy,
        "baseline_roc_auc": baseline.roc_auc,
        "federated_metrics": server.metrics.to_dict(),
        "rounds": [result.to_dict() for result in server.history],
    }
    return server, report, global_model


def main(argv: list[str] | None = None) -> int:
    """Run the image federated learning demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3, choices=(1, 3))
    parser.add_argument("--max-per-class", type=int, default=80)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("artifacts/image_fedavg"))
    args = parser.parse_args(argv)

    server, report, global_model = run_image_fedavg(
        dataset_dir=args.dataset_dir,
        n_clients=args.clients,
        n_rounds=args.rounds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        channels=args.channels,
        max_per_class=args.max_per_class,
        test_size=args.test_size,
        seed=args.seed,
    )

    metrics = server.metrics
    logger.info("Baseline accuracy: %.4f", report["baseline_accuracy"])
    for result in server.history:
        logger.info(
            "Round %d: accuracy=%.4f roc_auc=%s log_loss=%s time=%.2fs bytes=%d",
            result.round_index,
            result.accuracy,
            result.roc_auc,
            result.log_loss,
            result.round_duration_s or 0.0,
            result.bytes_exchanged or 0,
        )
    logger.info(
        "Federated metrics: total_time=%.2fs exchanged=%d bytes per_round=%d "
        "convergence_round=%s",
        metrics.total_time_s,
        metrics.total_bytes_exchanged,
        metrics.bytes_exchanged_per_round,
        metrics.convergence_round,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    global_model.save(args.out / "global_model.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    logger.info("Artifacts written to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
