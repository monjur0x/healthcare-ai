"""
Train the brain-tumor MRI CNN classifier end to end.

Loads a class-labelled image dataset (e.g. the brain-tumor MRI dataset
with ``glioma`` / ``meningioma`` / ``notumor`` / ``pituitary`` folders),
preprocesses it with the image pipeline, fits an ``ImageClassifier`` on
the CPU, evaluates it on a hold-out split, and persists the artifact so
the API can serve image uploads via ``/api/v1/analyze/image``.

The dataset directory may be a plain folder-per-class layout, or the
root of the standard brain-tumor MRI archive (whose ``Training`` /
``Testing`` splits are merged before subsampling).

Usage (run from the repository root):

    python scripts/train_image_model.py --dataset /path/to/dataset \\
        --max-per-class 300 --epochs 5 --image-size 224

The artifact is written to ``backend/artifacts/brain/global_model.pt``
(the path the API reads via ``API_IMAGE_MODEL_PATH``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from evaluation import evaluate_classifier
from models import ImageClassifier
from preprocessing.image import ImagePipeline
from preprocessing.logger import get_logger
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

_SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def resolve_dataset_root(dataset_dir: Path) -> Path:
    """
    Return the directory of class-labelled image folders.

    Accepts either a plain folder-per-class layout or the standard
    brain-tumor MRI archive root (``Training`` / ``Testing`` splits).

    Parameters
    ----------
    dataset_dir : Path
        Candidate dataset root.

    Returns
    -------
    Path
        The root whose immediate subfolders are class labels.

    Raises
    ------
    ValueError
        If no class folders can be located.
    """

    candidate = dataset_dir
    for name in ("Training", "Testing"):
        split = dataset_dir / name
        if split.is_dir() and any(folder.is_dir() for folder in split.iterdir()):
            candidate = split
            break
    if not any(folder.is_dir() for folder in candidate.iterdir()):
        raise ValueError(f"No class subfolders found under {dataset_dir}.")
    return candidate


def discover_classes(dataset_root: Path) -> dict[str, list[Path]]:
    """
    Map class labels to image paths from a folder-per-class layout.

    Parameters
    ----------
    dataset_root : Path
        Directory whose immediate subfolders are class labels.

    Returns
    -------
    dict[str, list[Path]]
        Class label to sorted image file paths.
    """

    classes: dict[str, list[Path]] = {}
    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir():
            continue
        images = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
        )
        if images:
            classes[folder.name] = images
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
    logger.info("Using %d images (%d classes)", len(sources), len(classes))
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
        Float32 ``(N, H, W, C)`` images and string class labels.
    """

    pipeline = ImagePipeline(
        size=(image_size, image_size),
        channels=channels,
        normalize=True,
        augment=False,
    )
    images = pipeline.transform_batch([path for path, _ in sources]).image
    labels = np.asarray([label for _, label in sources])
    logger.info("Preprocessed %d images to shape %s", images.shape[0], images.shape)
    return images, labels


def train_image_model(
    dataset_dir: Path,
    image_size: int,
    channels: int,
    max_per_class: int | None,
    epochs: int,
    batch_size: int,
    test_size: float,
    seed: int,
) -> tuple[ImageClassifier, dict]:
    """
    Train the CNN on the dataset and return the model + metrics report.

    Parameters
    ----------
    dataset_dir : Path
        Dataset root (folder-per-class or archive layout).
    image_size : int
        Square target size in pixels.
    channels : int
        Target channel count (1 or 3).
    max_per_class : int | None
        Optional per-class sample cap.
    epochs : int
        CNN training epochs.
    batch_size : int
        CNN training batch size.
    test_size : float
        Fraction of samples held out for evaluation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[ImageClassifier, dict]
        Fitted model and the JSON-friendly metrics report.
    """

    root = resolve_dataset_root(dataset_dir)
    classes = discover_classes(root)
    sources = subsample(classes, max_per_class, seed)
    X, y = load_batch(sources, image_size, channels)

    train_x, test_x, train_y, test_y = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    model = ImageClassifier(
        in_channels=channels,
        epochs=epochs,
        batch_size=batch_size,
    ).fit(train_x, train_y)

    metrics = evaluate_classifier(model, test_x, test_y)
    logger.info(
        "Hold-out accuracy=%.4f roc_auc=%s f1=%.4f n=%d",
        metrics.accuracy,
        metrics.roc_auc,
        metrics.f1_macro,
        metrics.n_samples,
    )

    report = {
        "n_classes": len(np.unique(y)),
        "classes": [str(label) for label in np.unique(y)],
        "n_train": int(train_x.shape[0]),
        "n_test": int(test_x.shape[0]),
        "accuracy": float(metrics.accuracy),
        "roc_auc": float(metrics.roc_auc) if metrics.roc_auc is not None else None,
        "f1_macro": float(metrics.f1_macro),
        "image_size": image_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
    }
    return model, report


def main(argv: list[str] | None = None) -> int:
    """Train the brain-tumor CNN and persist the artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--channels", type=int, default=3, choices=(1, 3))
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("backend/artifacts/brain"))
    args = parser.parse_args(argv)

    model, report = train_image_model(
        dataset_dir=args.dataset,
        image_size=args.image_size,
        channels=args.channels,
        max_per_class=args.max_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "global_model.pt"
    model.save(model_path)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    logger.info("Model artifact written to %s", model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
