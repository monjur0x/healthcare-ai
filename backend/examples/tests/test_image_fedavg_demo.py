"""
Smoke tests for the image federated learning demo.

Runs the demo's core function on a tiny synthetic image tree so the
full CSV-free image path (preprocess -> shard -> FedAvg -> report) is
exercised without the real brain-tumor MRI dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from PIL import Image

from examples.image_fedavg_demo import (
    discover_classes,
    run_image_fedavg,
    subsample,
)


def _make_image(label: int, size: int = 16) -> Image.Image:
    """Generate a clearly separable two-class image."""
    array = np.zeros((size, size, 3), dtype=np.uint8)
    half = size // 2
    if label == 0:
        array[:, :half] = 255
    else:
        array[:, half:] = 255
    return Image.fromarray(array)


@pytest.fixture
def image_tree(tmp_path):
    for label in (0, 1):
        folder = tmp_path / f"class_{label}"
        folder.mkdir()
        for index in range(8):
            _make_image(label).save(folder / f"img_{index}.png")
    return tmp_path


def test_discover_classes(image_tree) -> None:
    classes = discover_classes(image_tree)
    assert set(classes) == {"class_0", "class_1"}
    assert len(classes["class_0"]) == 8
    assert len(classes["class_1"]) == 8


def test_subsample_caps_per_class(image_tree) -> None:
    classes = discover_classes(image_tree)
    sources = subsample(classes, max_per_class=4, seed=42)
    labels = [label for _, label in sources]
    assert len(sources) == 8
    assert labels.count("class_0") == 4
    assert labels.count("class_1") == 4


def test_run_image_fedavg_end_to_end(image_tree) -> None:
    server, report, global_model = run_image_fedavg(
        dataset_dir=image_tree,
        n_clients=2,
        n_rounds=2,
        epochs=1,
        batch_size=8,
        image_size=16,
        channels=3,
        max_per_class=8,
        test_size=0.25,
        seed=42,
    )

    assert report["n_classes"] == 2
    assert report["baseline_accuracy"] > 0.4
    assert report["federated_metrics"]["n_rounds"] == 2
    assert len(report["rounds"]) == 2
    assert server.history[-1].accuracy > 0.4
    assert global_model.is_fitted
    assert global_model.classes_.tolist() == [0, 1]
