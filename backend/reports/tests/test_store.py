"""Unit tests for the clinical report store."""

from __future__ import annotations

import pytest

from reports import ReportStore, ReportStoreError


@pytest.fixture()
def store(tmp_path):
    handle = ReportStore(tmp_path / "reports.db")
    yield handle
    handle.close()


def test_add_and_get_roundtrip(store):
    record = store.add(
        patient_id="p-1",
        preset="diabetes",
        report={"status": "success", "risk": {"risk_level": "high"}},
    )
    assert record.report_id == 1
    assert record.stored_at

    fetched = store.get(record.report_id)
    assert fetched is not None
    assert fetched.patient_id == "p-1"
    assert fetched.preset == "diabetes"
    assert fetched.report["risk"]["risk_level"] == "high"


def test_get_missing_returns_none(store):
    assert store.get(999) is None


def test_add_rejects_non_serializable(store):
    with pytest.raises(ReportStoreError):
        store.add(patient_id="p-1", preset="diabetes", report={"bad": {1, 2, 3}})
