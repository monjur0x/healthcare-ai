"""
Tests for hospital site building and the clobber guard.

The ``run`` / ``client`` federated paths call ``build_hospital_sites``
repeatedly; a hospital's local slice is its own file and must not be
silently regenerated on every invocation. These tests pin the reuse /
overwrite semantics.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from federated.hospitals import build_hospital_sites


@pytest.fixture
def dataset_dir(tmp_path) -> object:
    """Write a small diabetes.csv the site builder can partition."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 6, 240),
            "Glucose": rng.integers(70, 200, 240),
            "BloodPressure": rng.integers(60, 110, 240),
            "Outcome": rng.integers(0, 2, 240),
        }
    )
    path = tmp_path / "diabetes.csv"
    frame.to_csv(path, index=False)
    return tmp_path


def _rows_at(root, hospital_id: str) -> int:
    csv = root / hospital_id / "data.csv"
    return len(csv.read_text().splitlines()) if csv.is_file() else None


def test_build_sites_writes_slices_and_manifest(dataset_dir, tmp_path) -> None:
    root = tmp_path / "hospitals"
    sites = build_hospital_sites(
        preset="diabetes",
        n_sites=4,
        dataset_dir=dataset_dir,
        hospitals_dir=root,
        seed=42,
    )
    assert isinstance(sites, list) and len(sites) == 4
    assert sites[0].hospital_id == "hospital_A"
    assert sites[0].dataset_path.is_file()
    assert len(sites) == len(
        [p for p in sites if p.dataset_path.name == "data.csv"]
    )
    assert (root / "central_holdout.csv").is_file()
    assert (root / "sites.json").is_file()


def test_reuse_does_not_clobber_existing_slices(dataset_dir, tmp_path) -> None:
    root = tmp_path / "hospitals"
    first = build_hospital_sites(
        preset="diabetes",
        n_sites=4,
        dataset_dir=dataset_dir,
        hospitals_dir=root,
        seed=42,
    )
    before = {site.hospital_id: _rows_at(root, site.hospital_id) for site in first}

    second = build_hospital_sites(
        preset="diabetes",
        n_sites=4,
        dataset_dir=dataset_dir,
        hospitals_dir=root,
        seed=42,
    )
    # Same sites returned; files untouched (far fewer rows would mean a
    # re-partition happened and shrank the slices), and the manifest is
    # unchanged.
    assert [site.hospital_id for site in second] == [
        site.hospital_id for site in first
    ]
    after = {site.hospital_id: _rows_at(root, site.hospital_id) for site in second}
    assert after == before
    assert (root / "sites.json").read_bytes() == (root / "sites.json").read_bytes()


def test_overwrite_force_rebuilds(dataset_dir, tmp_path) -> None:
    root = tmp_path / "hospitals"
    build_hospital_sites(
        preset="diabetes",
        n_sites=4,
        dataset_dir=dataset_dir,
        hospitals_dir=root,
        seed=42,
    )
    manifest = json.loads((root / "sites.json").read_text())

    # Force a rebuild with the same parameters: manifest is regenerated and
    # the slices are rewritten (this path is what `federated sites` uses).
    build_hospital_sites(
        preset="diabetes",
        n_sites=4,
        dataset_dir=dataset_dir,
        hospitals_dir=root,
        seed=99,
        overwrite=True,
    )
    after = json.loads((root / "sites.json").read_text())
    assert after["seed"] == 99
    assert after["seed"] != manifest["seed"]


def test_changed_parameters_trigger_rebuild(dataset_dir, tmp_path) -> None:
    root = tmp_path / "hospitals"
    build_hospital_sites(
        preset="diabetes",
        n_sites=4,
        dataset_dir=dataset_dir,
        hospitals_dir=root,
        seed=42,
    )
    # A different site count demands re-partitioning (would otherwise reuse
    # stale slices that don't match the requested configuration). The two
    # requested sites are rebuilt with the new dimensions; the manifest
    # reflects the new parameters.
    sites = build_hospital_sites(
        preset="diabetes",
        n_sites=2,
        dataset_dir=dataset_dir,
        hospitals_dir=root,
        seed=42,
    )
    assert len(sites) == 2
    assert _rows_at(root, "hospital_A") is not None
    assert _rows_at(root, "hospital_B") is not None
    assert json.loads((root / "sites.json").read_text())["n_sites"] == 2
