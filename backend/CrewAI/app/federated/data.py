"""Unified dataset construction and multi-hospital partition (Phase 1 & 3).

Four simulated hospitals each hold a *different* disease cohort. Raw data is
partitioned such that no hospital ever observes another hospital's rows; a
central orchestrator only ever sees model updates.

The pipeline works end-to-end offline by generating clinically-plausible
synthetic cohorts. If real CSVs (Pima / UCI Heart / UCI CKD) are dropped into
``settings.FL_DATA_DIR`` they are loaded and mapped to the unified schema
instead (identical column layout), so results are reproducible either way.

MIMIC-IV is *not* bundled: it requires credentialed PhysioNet access. The
``ehr`` hospital (Hospital D) uses a MIMIC-style synthetic cohort for the
same mortality/risk task.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import settings

# Unified feature schema shared by every hospital.
FEATURE_COLUMNS = [
    "age", "bmi", "sbp", "dbp", "glucose", "cholesterol", "hdl",
    "heart_rate", "creatinine", "hemoglobin", "spo2", "temp",
    "wbc", "platelet", "los_days",
]

CATEGORICAL_COLUMNS = ["hospital", "disease_group"]
LABEL_COLUMN = "label"  # 1 = high clinical risk, 0 = low


def _feature_datasets() -> dict[str, str]:
    return {
        "diabetes": "Pima Indians Diabetes",
        "heart": "UCI Heart Disease",
        "ckd": "UCI Chronic Kidney Disease",
        "ehr": "MIMIC-IV-style EHR (mortality/risk)",
    }


def _generate_diabetes(rng: np.random.Generator, n: int) -> pd.DataFrame:
    df = pd.DataFrame()
    df["age"] = _uniform(rng, n, 20, 85)
    df["bmi"] = _uniform(rng, n, 16, 46)
    df["sbp"] = _uniform(rng, n, 90, 190)
    df["dbp"] = _uniform(rng, n, 55, 120)
    df["glucose"] = _uniform(rng, n, 65, 220)
    df["cholesterol"] = _uniform(rng, n, 120, 300)
    df["hdl"] = _uniform(rng, n, 25, 80)
    df["heart_rate"] = _uniform(rng, n, 55, 110)
    df["creatinine"] = _uniform(rng, n, 0.5, 2.2)
    df["hemoglobin"] = _uniform(rng, n, 9.5, 17.5)
    df["spo2"] = _uniform(rng, n, 92, 100)
    df["temp"] = _uniform(rng, n, 35.8, 38.4)
    df["wbc"] = _uniform(rng, n, 3.5, 14.0)
    df["platelet"] = _uniform(rng, n, 120, 400)
    df["los_days"] = np.zeros(n)
    df["disease_group"] = "diabetes"
    score = (
        0.40 * ((df["glucose"] - 100) / 120).clip(0, 1)
        + 0.25 * ((df["bmi"] - 25) / 20).clip(0, 1)
        + 0.20 * ((df["sbp"] - 110) / 80).clip(0, 1)
        + 0.15 * ((df["age"] - 40) / 45).clip(0, 1)
        + 0.15 * _noise(rng, n)
    )
    df[LABEL_COLUMN] = (score > score.median()).astype(int)
    return df


def _generate_heart(rng: np.random.Generator, n: int) -> pd.DataFrame:
    df = pd.DataFrame()
    df["age"] = _uniform(rng, n, 29, 77)
    df["bmi"] = _uniform(rng, n, 18, 40)
    df["sbp"] = _uniform(rng, n, 94, 200)  # trestbps
    df["dbp"] = _uniform(rng, n, 60, 120)
    df["glucose"] = _uniform(rng, n, 70, 180)  # fbs proxy
    df["cholesterol"] = _uniform(rng, n, 126, 564)
    df["hdl"] = _uniform(rng, n, 22, 72)
    df["heart_rate"] = _uniform(rng, n, 71, 202)  # thalach
    df["creatinine"] = _uniform(rng, n, 0.6, 2.0)
    df["hemoglobin"] = _uniform(rng, n, 10, 17)
    df["spo2"] = _uniform(rng, n, 90, 100)
    df["temp"] = _uniform(rng, n, 35.9, 38.5)
    df["wbc"] = _uniform(rng, n, 4, 13)
    df["platelet"] = _uniform(rng, n, 130, 380)
    df["los_days"] = np.zeros(n)
    df["disease_group"] = "heart"
    score = (
        0.35 * ((df["cholesterol"] - 150) / 300).clip(0, 1)
        + 0.25 * ((df["sbp"] - 110) / 90).clip(0, 1)
        + 0.20 * ((df["age"] - 40) / 37).clip(0, 1)
        + 0.15 * ((df["heart_rate"] - 90) / 110).clip(0, 1)
        + 0.10 * ((df["glucose"] - 100) / 120).clip(0, 1)
        + 0.15 * _noise(rng, n)
    )
    df[LABEL_COLUMN] = (score > score.median()).astype(int)
    return df


def _generate_ckd(rng: np.random.Generator, n: int) -> pd.DataFrame:
    df = pd.DataFrame()
    df["age"] = _uniform(rng, n, 15, 90)
    df["bmi"] = _uniform(rng, n, 15, 40)
    df["sbp"] = _uniform(rng, n, 80, 200)  # bp
    df["dbp"] = _uniform(rng, n, 50, 120)
    df["glucose"] = _uniform(rng, n, 60, 200)  # bgr proxy
    df["cholesterol"] = _uniform(rng, n, 120, 300)
    df["hdl"] = _uniform(rng, n, 25, 80)
    df["heart_rate"] = _uniform(rng, n, 50, 110)
    df["creatinine"] = _uniform(rng, n, 0.4, 11.0)  # sc -> ckd driver
    df["hemoglobin"] = _uniform(rng, n, 6, 17)  # hemo
    df["spo2"] = _uniform(rng, n, 88, 100)
    df["temp"] = _uniform(rng, n, 35.7, 38.5)
    df["wbc"] = _uniform(rng, n, 2.5, 15.0)  # wbcc
    df["platelet"] = _uniform(rng, n, 100, 450)
    df["los_days"] = np.zeros(n)
    df["disease_group"] = "ckd"
    score = (
        0.40 * np.log1p(df["creatinine"]) / np.log1p(11.0)
        + 0.25 * ((140 - df["hemoglobin"]) / 80).clip(0, 1)
        + 0.15 * ((df["sbp"] - 110) / 90).clip(0, 1)
        + 0.10 * ((df["age"] - 40) / 50).clip(0, 1)
        + 0.10 * ((df["glucose"] - 100) / 120).clip(0, 1)
        + 0.15 * _noise(rng, n)
    )
    df[LABEL_COLUMN] = (score > score.median()).astype(int)
    return df


def _generate_ehr(rng: np.random.Generator, n: int) -> pd.DataFrame:
    df = pd.DataFrame()
    df["age"] = _uniform(rng, n, 18, 90)
    df["bmi"] = _uniform(rng, n, 16, 45)
    df["sbp"] = _uniform(rng, n, 80, 200)
    df["dbp"] = _uniform(rng, n, 50, 120)
    df["glucose"] = _uniform(rng, n, 60, 300)
    df["cholesterol"] = _uniform(rng, n, 110, 320)
    df["hdl"] = _uniform(rng, n, 20, 85)
    df["heart_rate"] = _uniform(rng, n, 45, 150)
    df["creatinine"] = _uniform(rng, n, 0.4, 10)
    df["hemoglobin"] = _uniform(rng, n, 7, 18)
    df["spo2"] = _uniform(rng, n, 78, 100)
    df["temp"] = _uniform(rng, n, 35.5, 40.5)
    df["wbc"] = _uniform(rng, n, 2, 25)
    df["platelet"] = _uniform(rng, n, 60, 500)
    df["los_days"] = _uniform(rng, n, 1, 25)
    df["disease_group"] = "ehr"
    acuity = (
        0.25 * ((100 - df["spo2"]) / 22).clip(0, 1)
        + 0.20 * ((df["heart_rate"] - 60) / 90).clip(0, 1)
        + 0.20 * ((df["glucose"] - 110) / 190).clip(0, 1)
        + 0.15 * df["los_days"] / 25
        + 0.15 * ((df["creatinine"] - 1) / 9).clip(0, 1)
        + 0.15 * ((df["temp"] - 37) / 3.5).abs().clip(0, 1)
        + 0.15 * _noise(rng, n)
    )
    df[LABEL_COLUMN] = (acuity > acuity.median()).astype(int)
    return df


def _uniform(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    return rng.uniform(lo, hi, n).round(2)


def _noise(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(-0.15, 0.15, n)


def _load_real(name: str, data_dir: Path) -> pd.DataFrame | None:
    """Load and map a real dataset CSV to the unified schema, if present."""
    candidates = [
        data_dir / f"{name}.csv",
        data_dir / f"healthcare-{name}.csv",
    ]
    for path in candidates:
        if path.exists():
            return _map_real(name, pd.read_csv(path))
    return None


def _map_real(name: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Best-effort mapping of a known dataset to FEATURE_COLUMNS."""
    df = pd.DataFrame()
    rng = np.random.default_rng(0)
    n = len(raw)
    raw.columns = [c.lower().strip().replace(" ", "_") for c in raw.columns]

    age = _col(raw, ["age"])
    df["age"] = age if age is not None else _uniform(rng, n, 30, 75)
    df["bmi"] = _col(raw, ["bmi", "mass_index"])
    if df["bmi"].isnull().all() and name == "pima":
        df["bmi"] = _uniform(rng, n, 16, 45)
    df["sbp"] = _col(raw, ["trestbps", "bp", "systolic", "blood_pressure_systolic"])
    df["dbp"] = _col(raw, ["dbp", "diastolic", "blood_pressure_diastolic"])
    if df["sbp"].isnull().all() and name == "pima":
        df["sbp"] = _uniform(rng, n, 95, 180)
        df["dbp"] = _col(raw, ["bp"])
    df["glucose"] = _col(raw, ["glucose", "bgr", "blood_glucose"])
    if df["glucose"].isnull().all() and name == "pima":
        df["glucose"] = _col(raw, ["glucose"])
    df["cholesterol"] = _col(raw, ["chol", "cholesterol", "tc"])
    if df["cholesterol"].isnull().all() and name == "heart":
        df["cholesterol"] = _col(raw, ["chol"])
    df["hdl"] = _col(raw, ["hdl"])
    if df["hdl"].isnull().all():
        df["hdl"] = _uniform(rng, n, 28, 70)
    df["heart_rate"] = _col(raw, ["thalach", "heart_rate", "pulse", "hr"])
    if df["heart_rate"].isnull().all() and name == "heart":
        df["heart_rate"] = _col(raw, ["thalach"])
    df["creatinine"] = _col(raw, ["sc", "creatinine", "serum_creatinine"])
    if df["creatinine"].isnull().all() and name == "ckd":
        df["creatinine"] = _uniform(rng, n, 0.5, 9)
    df["hemoglobin"] = _col(raw, ["hemo", "hemoglobin", "hgb", "hb"])
    if df["hemoglobin"].isnull().all() and name == "ckd":
        df["hemoglobin"] = _col(raw, ["hemo"])
    df["spo2"] = _col(raw, ["spo2", "oxygen_saturation"])
    if df["spo2"].isnull().all():
        df["spo2"] = _uniform(rng, n, 90, 100)
    df["temp"] = _col(raw, ["temp", "temperature"])
    if df["temp"].isnull().all():
        df["temp"] = _uniform(rng, n, 36, 38.5)
    df["wbc"] = _col(raw, ["wbcc", "wbc", "white_blood_cells"])
    if df["wbc"].isnull().all() and name == "ckd":
        df["wbc"] = _col(raw, ["wbcc"])
    df["platelet"] = _col(raw, ["platelet", "plt"])
    if df["platelet"].isnull().all():
        df["platelet"] = _uniform(rng, n, 120, 400)
    df["los_days"] = _col(raw, ["los", "length_of_stay"])
    if df["los_days"].isnull().all():
        df["los_days"] = np.zeros(n)

    for c in df.columns:
        if c == "disease_group":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        if df[c].isnull().all():
            df[c] = 0.0
        df[c] = df[c].fillna(df[c].median() if df[c].notna().sum() else 0.0)

    label_col = None
    for c in ["target", "diabetes", "ckd", "classification", "outcome", "label"]:
        if c in raw.columns:
            label_col = c
            break
    if label_col is not None:
        lab = pd.to_numeric(raw[label_col], errors="coerce")
        df[LABEL_COLUMN] = (lab > lab.median()).astype(int)
    else:
        score = (df["glucose"] / 200 + df["bmi"] / 45 + df["creatinine"] / 10) / 3
        df[LABEL_COLUMN] = (score > np.percentile(score, 50)).astype(int)

    df["disease_group"] = name
    return df[FEATURE_COLUMNS + [LABEL_COLUMN, "disease_group"]].reset_index(drop=True)


def _col(raw: pd.DataFrame, names: list[str]) -> "pd.Series[float]" | None:
    for name in names:
        if name in raw.columns:
            return pd.to_numeric(raw[name], errors="coerce").astype(float)
    return None


def build_hospital_datasets(
    n_per_hospital: int = 2000,
    seed: int | None = None,
    data_dir: Path | None = None,
) -> list[tuple[str, pd.DataFrame]]:
    """Build the four hospital cohorts.

    Returns a list of ``(hospital_name, dataframe)`` pairs, one per hospital.
    ``hospital`` column on each frame labels its owning site.
    """
    rng = np.random.default_rng(seed if seed is not None else settings.FL_SEED)
    data_dir = Path(data_dir) if data_dir else Path(settings.FL_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("hospital_a", "diabetes", "A", _generate_diabetes),
        ("hospital_b", "heart", "B", _generate_heart),
        ("hospital_c", "ckd", "C", _generate_ckd),
        ("hospital_d", "ehr", "D", _generate_ehr),
    ]

    datasets: list[tuple[str, pd.DataFrame]] = []
    for hname, dname, _tag, gen in specs:
        real = _load_real(dname, data_dir)
        df = real if real is not None else gen(rng, n_per_hospital)
        df = df.copy()
        df["hospital"] = hname
        df = df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS + [LABEL_COLUMN]].reset_index(drop=True)
        datasets.append((hname, df))

    return datasets


def full_dataset(n_per_hospital: int = 2000, seed: int | None = None) -> pd.DataFrame:
    """Concatenate all hospitals into a single (centralized-style) frame."""
    all_parts = [df.assign(_h=h) for h, df in build_hospital_datasets(n_per_hospital, seed)]
    return pd.concat(all_parts, ignore_index=True)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the numeric feature matrix from a unified frame."""
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)