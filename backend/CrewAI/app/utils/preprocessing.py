"""Data preprocessing utilities for healthcare data."""

import pandas as pd
import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


def preprocess_csv_data(csv_content: str | bytes) -> dict[str, Any]:
    """Preprocess CSV data for analysis.

    Args:
        csv_content: CSV content as string or bytes

    Returns:
        Dictionary with preprocessed data and metadata
    """
    try:
        # Parse CSV
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode('utf-8')

        df = pd.read_csv(pd.io.common.StringIO(csv_content))

        # Basic statistics
        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict()
        }

        # Clean data
        df_clean = clean_healthcare_data(df)

        # Extract key metrics
        key_metrics = extract_key_metrics(df_clean)

        # Identify potential conditions
        conditions = identify_conditions(key_metrics)

        return {
            "success": True,
            "dataframe": df_clean,
            "statistics": stats,
            "key_metrics": key_metrics,
            "potential_conditions": conditions,
            "quality_score": calculate_data_quality(df)
        }

    except Exception as e:
        logger.error(f"Error preprocessing CSV: {e}")
        return {
            "success": False,
            "error": str(e),
            "dataframe": pd.DataFrame(),
            "statistics": {},
            "key_metrics": {},
            "potential_conditions": [],
            "quality_score": 0.0
        }


def clean_healthcare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate healthcare data."""
    df_clean = df.copy()

    # Standardize column names
    df_clean.columns = [col.lower().strip().replace(' ', '_') for col in df_clean.columns]

    # Handle common column name variations
    column_mappings = {
        'bp_systolic': 'blood_pressure_systolic',
        'bp_diastolic': 'blood_pressure_diastolic',
        'systolic': 'blood_pressure_systolic',
        'diastolic': 'blood_pressure_diastolic',
        'hr': 'heart_rate',
        'bmi': 'bmi',
        'glucose': 'glucose',
        'hba1c': 'hba1c',
        'hb': 'hemoglobin',
        'hgb': 'hemoglobin'
    }

    df_clean = df_clean.rename(columns={
        k: v for k, v in column_mappings.items()
        if k in df_clean.columns
    })

    # Remove completely empty rows
    df_clean = df_clean.dropna(how='all')

    # Fill missing numeric values with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df_clean


def extract_key_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Extract key health metrics from preprocessed data."""
    metrics = {}

    # Common metric mappings
    metric_definitions = {
        "age": {"column": ["age"], "type": "direct"},
        "bmi": {"column": ["bmi"], "type": "direct"},
        "blood_pressure_systolic": {"column": ["blood_pressure_systolic", "systolic", "bp_systolic"], "type": "direct"},
        "blood_pressure_diastolic": {"column": ["blood_pressure_diastolic", "diastolic", "bp_diastolic"], "type": "direct"},
        "glucose": {"column": ["glucose", "blood_glucose", "fasting_glucose"], "type": "direct"},
        "heart_rate": {"column": ["heart_rate", "pulse", "hr"], "type": "direct"},
        "hemoglobin": {"column": ["hemoglobin", "hgb", "hb"], "type": "direct"},
        "creatinine": {"column": ["creatinine", "serum_creatinine"], "type": "direct"},
        "cholesterol": {"column": ["cholesterol", "total_cholesterol"], "type": "direct"},
        "hba1c": {"column": ["hba1c", "a1c", "hemoglobin_a1c"], "type": "direct"}
    }

    for metric_name, definition in metric_definitions.items():
        for col_name in definition["column"]:
            if col_name in df.columns:
                values = df[col_name].dropna()
                if len(values) > 0:
                    metrics[metric_name] = {
                        "value": float(values.mean()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "std": float(values.std()) if len(values) > 1 else 0,
                        "column": col_name
                    }
                break

    # Calculate derived metrics
    if "blood_pressure_systolic" in metrics and "blood_pressure_diastolic" in metrics:
        sys_bp = metrics["blood_pressure_systolic"]["value"]
        dia_bp = metrics["blood_pressure_diastolic"]["value"]
        metrics["pulse_pressure"] = {
            "value": sys_bp - dia_bp,
            "interpretation": "Normal" if 30 <= (sys_bp - dia_bp) <= 40 else "Abnormal"
        }

    return metrics


def identify_conditions(key_metrics: dict) -> list[dict[str, Any]]:
    """Identify potential health conditions based on metrics."""
    conditions = []

    # Diabetes indicators
    if "glucose" in key_metrics:
        glucose = key_metrics["glucose"]["value"]
        if glucose > 126:
            conditions.append({
                "condition": "Diabetes Mellitus",
                "indicator": f"Fasting glucose: {glucose:.1f} mg/dL",
                "threshold": "> 126 mg/dL",
                "confidence": min(0.9, (glucose - 126) / 74 + 0.6)
            })
        elif glucose > 100:
            conditions.append({
                "condition": "Pre-diabetes",
                "indicator": f"Fasting glucose: {glucose:.1f} mg/dL",
                "threshold": "100-126 mg/dL",
                "confidence": 0.7
            })

    # Hypertension indicators
    if "blood_pressure_systolic" in key_metrics:
        sys_bp = key_metrics["blood_pressure_systolic"]["value"]
        if sys_bp > 140:
            conditions.append({
                "condition": "Hypertension Stage 2",
                "indicator": f"Systolic BP: {sys_bp:.1f} mmHg",
                "threshold": "> 140 mmHg",
                "confidence": min(0.95, (sys_bp - 140) / 40 + 0.7)
            })
        elif sys_bp > 130:
            conditions.append({
                "condition": "Hypertension Stage 1",
                "indicator": f"Systolic BP: {sys_bp:.1f} mmHg",
                "threshold": "130-139 mmHg",
                "confidence": 0.75
            })

    # Obesity indicators
    if "bmi" in key_metrics:
        bmi = key_metrics["bmi"]["value"]
        if bmi > 30:
            conditions.append({
                "condition": "Obesity",
                "indicator": f"BMI: {bmi:.1f}",
                "threshold": "> 30 kg/m²",
                "confidence": 0.9
            })
        elif bmi > 25:
            conditions.append({
                "condition": "Overweight",
                "indicator": f"BMI: {bmi:.1f}",
                "threshold": "25-30 kg/m²",
                "confidence": 0.85
            })

    # Anemia indicators
    if "hemoglobin" in key_metrics:
        hb = key_metrics["hemoglobin"]["value"]
        if hb < 12:
            conditions.append({
                "condition": "Anemia",
                "indicator": f"Hemoglobin: {hb:.1f} g/dL",
                "threshold": "< 12 g/dL",
                "confidence": 0.85
            })

    # Kidney function indicators
    if "creatinine" in key_metrics:
        creat = key_metrics["creatinine"]["value"]
        if creat > 1.5:
            conditions.append({
                "condition": "Impaired Kidney Function",
                "indicator": f"Creatinine: {creat:.2f} mg/dL",
                "threshold": "> 1.5 mg/dL",
                "confidence": 0.8
            })

    return sorted(conditions, key=lambda x: x["confidence"], reverse=True)


def calculate_data_quality(df: pd.DataFrame) -> float:
    """Calculate data quality score (0-1)."""
    if df.empty:
        return 0.0

    # Completeness score
    completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))

    # Consistency score (check for outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    consistency_scores = []
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) > 2:
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            outliers = ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()
            consistency_scores.append(1 - (outliers / len(values)))

    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.5

    # Weighted quality score
    quality_score = (completeness * 0.6 + avg_consistency * 0.4)

    return round(quality_score, 3)
