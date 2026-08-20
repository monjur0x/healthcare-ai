"""
Hospital data layer for distributed federated learning.

Each hospital is a site that owns a local, non-overlapping slice of a
task's dataset. The hospital never ships raw rows anywhere; the Flower
client process loads its own CSV, preprocesses it locally with
:func:`preprocessing.loader.load_classification_frame`, and only model
weights travel over the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from preprocessing.loader import load_classification_frame, normalize_columns
from preprocessing.logger import get_logger

logger = get_logger(__name__)

#: Named dataset presets mapping a name to ``(file name, target column)``.
PRESETS: dict[str, tuple[str, str]] = {
    "diabetes": ("diabetes.csv", "Outcome"),
    "heart": ("heart_disease_uci.csv", "num"),
    "kidney": ("kidney_disease.csv", "classification"),
    "sepsis": ("sepsis_icu_synthetic.csv", "sepsis_label"),
}


@dataclass(frozen=True)
class HospitalConfig:
    """
    A single federated hospital site.

    Attributes
    ----------
    hospital_id : str
        Stable site identifier (e.g. ``"hospital_A"``).
    name : str
        Human-readable hospital name.
    dataset_path : Path
        Path to the hospital's own local CSV.
    target : str
        Target column name.
    """

    hospital_id: str
    name: str
    dataset_path: Path
    target: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the configuration to a JSON-friendly dictionary."""
        return {
            "hospital_id": self.hospital_id,
            "name": self.name,
            "dataset_path": str(self.dataset_path),
            "target": self.target,
        }


def build_hospital_sites(
    preset: str,
    n_sites: int,
    dataset_dir: str | Path,
    hospitals_dir: str | Path,
    seed: int = 42,
) -> list[HospitalConfig]:
    """
    Partition a preset dataset into per-hospital local CSV slices.

    The source CSV is read once and split into ``n_sites`` class-balanced
    shards. Each shard is written as a separate raw CSV under
    ``hospitals_dir/<hospital_id>/data.csv`` so every hospital preprocesses
    its own local copy. A single validation slice is additionally written
    to ``hospitals_dir/central_holdout.csv`` for server-side evaluation.

    Parameters
    ----------
    preset : str
        Dataset preset name (``"diabetes"``, ``"heart"``, ``"kidney"``,
        or ``"sepsis"``).
    n_sites : int
        Number of hospital sites to create.
    dataset_dir : str | Path
        Directory containing the source preset CSV.
    hospitals_dir : str | Path
        Root directory where per-hospital slices are written.
    seed : int
        Random seed for the stratified split.

    Returns
    -------
    list[HospitalConfig]
        One configuration per hospital site.

    Raises
    ------
    ValueError
        If the preset is unknown, the source CSV is missing, or the
        rarest class cannot support the requested number of sites.
    """

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from {sorted(PRESETS)}.")
    if n_sites < 1:
        raise ValueError("n_sites must be a positive integer.")

    file_name, target = PRESETS[preset]
    source = Path(dataset_dir) / file_name
    if not source.is_file():
        raise ValueError(f"Preset dataset not found: {source}")

    root = Path(hospitals_dir)
    root.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(source)
    raw = normalize_columns(raw)
    target = target.strip().lower().replace(" ", "_")
    if target not in raw.columns:
        raise ValueError(f"Target column '{target}' not found in {source}.")

    y_raw = raw[target]
    valid = y_raw.notna()
    raw = raw.loc[valid]
    y_raw = y_raw.loc[valid]

    if pd.api.types.is_string_dtype(y_raw):
        y_encoded = LabelEncoder().fit_transform(y_raw.str.strip())
    else:
        y_encoded = pd.to_numeric(y_raw).astype(int).to_numpy()

    counts = np.bincount(y_encoded)
    if counts.min() < n_sites + 1:
        raise ValueError(
            f"Rarest class has {counts.min()} samples, fewer than "
            f"{n_sites + 1} folds (sites + hold-out) require. Reduce "
            "n_sites or provide a larger dataset."
        )

    splitter = StratifiedKFold(n_splits=n_sites + 1, shuffle=True, random_state=seed)
    folds = list(splitter.split(raw, y_encoded))

    holdout_index = folds[0][1]
    holdout_path = root / "central_holdout.csv"
    raw.iloc[holdout_index].to_csv(holdout_path, index=False)
    logger.info(
        "Wrote central hold-out slice with %d rows to %s",
        len(holdout_index),
        holdout_path,
    )

    sites: list[HospitalConfig] = []
    for index, (_, test_index) in enumerate(folds[1:]):
        hospital_id = f"hospital_{chr(ord('A') + index)}"
        site_dir = root / hospital_id
        site_dir.mkdir(parents=True, exist_ok=True)
        slice_path = site_dir / "data.csv"
        raw.iloc[test_index].to_csv(slice_path, index=False)
        sites.append(
            HospitalConfig(
                hospital_id=hospital_id,
                name=f"Hospital {chr(ord('A') + index)}",
                dataset_path=slice_path,
                target=target,
            )
        )
        logger.info(
            "Wrote %s local slice with %d rows to %s",
            hospital_id,
            len(test_index),
            slice_path,
        )
    return sites


def load_hospital_dataset(
    hospital: HospitalConfig,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """
    Locally preprocess a hospital's own CSV slice.

    Parameters
    ----------
    hospital : HospitalConfig
        The hospital whose local data should be loaded.
    max_rows : int | None
        Optional cap on the number of rows used.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, dict[str, object]]
        Engineered feature frame, encoded labels, and scaler parameters
        (identical in shape to the API training loader output).
    """

    return load_classification_frame(hospital.dataset_path, hospital.target, max_rows)


__all__ = [
    "PRESETS",
    "HospitalConfig",
    "build_hospital_sites",
    "load_hospital_dataset",
]
