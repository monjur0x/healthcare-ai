"""
Smoke tests for the RAG demo.
"""

from __future__ import annotations

import json
import subprocess
import sys

from pathlib import Path

import pytest


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Create a tiny medical text corpus."""
    (tmp_path / "one.txt").write_text(
        "diabetes is managed with metformin and lifestyle changes",
        encoding="utf-8",
    )
    (tmp_path / "two.txt").write_text(
        "hypertension is treated with blood pressure lowering drugs",
        encoding="utf-8",
    )
    return tmp_path


def test_rag_demo_ingests_and_answers(corpus: Path, tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.rag_demo",
            "--corpus-dir",
            str(corpus),
            "--query",
            "metformin diabetes",
            "--top-k",
            "2",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["n_documents"] == 2
    assert report["queries"][0]["results"][0]["document_id"] == "one.txt"


def test_rag_demo_reports_metrics(corpus: Path, tmp_path: Path) -> None:
    ground_truth = tmp_path / "relevance.json"
    ground_truth.write_text(
        json.dumps({"metformin diabetes": ["one.txt"]}), encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.rag_demo",
            "--corpus-dir",
            str(corpus),
            "--ground-truth",
            str(ground_truth),
            "--top-k",
            "2",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    metrics = report["queries"][0]["metrics"]
    assert metrics["mrr"] == 1.0
    assert metrics["recall_at_k"] == 1.0
