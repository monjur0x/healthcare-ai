"""
Smoke tests for the clinical crew demo.
"""

from __future__ import annotations

import json
import subprocess
import sys

from pathlib import Path

import pandas as pd


def _write_synthetic_csv(path: Path) -> None:
    """Create a tiny binary classification CSV."""
    rng = __import__("numpy").random.default_rng(0)
    n = 120
    frame = pd.DataFrame(
        {
            "glucose": rng.normal(100, 20, n),
            "bmi": rng.normal(26, 4, n),
            "age": rng.integers(30, 70, n),
            "outcome": rng.integers(0, 2, n),
        }
    )
    frame.to_csv(path, index=False)


def test_crew_demo_writes_report(tmp_path: Path) -> None:
    dataset = tmp_path / "synthetic.csv"
    _write_synthetic_csv(dataset)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.clinical_crew_demo",
            "--dataset",
            str(dataset),
            "--target",
            "outcome",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["prediction"]["predicted_class"] in {"0", "1"}
    assert report["risk"]["risk_level"] in {"low", "medium", "high"}
    assert report["evidence"]
    assert report["doctor_notice"]


def test_crew_demo_accepts_corpus_dir(tmp_path: Path) -> None:
    dataset = tmp_path / "synthetic.csv"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.txt").write_text(
        "diabetes is managed with metformin and lifestyle changes",
        encoding="utf-8",
    )
    _write_synthetic_csv(dataset)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.clinical_crew_demo",
            "--dataset",
            str(dataset),
            "--target",
            "outcome",
            "--corpus-dir",
            str(corpus),
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["evidence"][0]["document_id"] == "one.txt"
