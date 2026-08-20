"""
Pydantic schemas for the feedback / retrain loop.

These shapes describe clinician feedback and the accumulated store state.
The API-level retrain request/response models (which nest the training
response) live in ``api/schemas.py`` to avoid a feedback → api import
cycle.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DatasetPreset = Literal["diabetes", "heart", "kidney", "sepsis"]


class FeedbackRequest(BaseModel):
    """
    A clinician-confirmed label for a past analysis.

    Parameters
    ----------
    preset : DatasetPreset
        Dataset preset the feedback refers to.
    patient_id : str
        Patient study id the feedback is about.
    features : dict[str, float]
        Feature row that was analyzed (used to augment retraining).
    confirmed_label : int
        The clinician-confirmed outcome label (0/1).
    predicted_label : int | None
        The model's predicted label at analysis time, when available.
    confidence : float | None
        The model's confidence at analysis time, when available.
    """

    preset: DatasetPreset
    patient_id: str = Field(default="", min_length=1)
    features: dict[str, float] = Field(..., min_length=1)
    confirmed_label: int = Field(..., ge=0, le=1)
    predicted_label: int | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class FeedbackRecord(BaseModel):
    """
    A stored feedback sample.

    Parameters
    ----------
    id : int
        Store row id.
    preset : str
        Dataset preset.
    patient_id : str
        Patient study id.
    features : dict[str, float]
        Feature row that was analyzed (used to augment retraining).
    confirmed_label : int
        Clinician-confirmed outcome label.
    predicted_label : int | None
        Model prediction at analysis time (when recorded).
    confidence : float | None
        Model confidence at analysis time (when recorded).
    created_at : str
        ISO-8601 timestamp.
    """

    id: int
    preset: str
    patient_id: str
    features: dict[str, float] = Field(default_factory=dict)
    confirmed_label: int
    predicted_label: int | None = None
    confidence: float | None = None
    created_at: str


class FeedbackSummary(BaseModel):
    """
    Accumulated feedback state for one preset.

    Parameters
    ----------
    preset : str
        Dataset preset.
    dataset : str
        Source CSV file name.
    target : str
        Target column name.
    pending : int
        Number of stored feedback samples not yet consumed by a retrain.
    threshold : int
        Number of samples required before a retrain is allowed.
    ready : bool
        True when ``pending >= threshold``.
    recent : list[FeedbackRecord]
        Most recent stored samples (newest first).
    """

    preset: str
    dataset: str
    target: str
    pending: int = 0
    threshold: int = 0
    ready: bool = False
    recent: list[FeedbackRecord] = Field(default_factory=list)


class FeedbackStatus(BaseModel):
    """
    Feedback state across all presets.

    Parameters
    ----------
    retrain_enabled : bool
        Whether automated retraining is enabled.
    presets : list[FeedbackSummary]
        Per-preset feedback summaries ordered by name.
    """

    retrain_enabled: bool
    presets: list[FeedbackSummary] = Field(default_factory=list)


__all__ = [
    "DatasetPreset",
    "FeedbackRecord",
    "FeedbackRequest",
    "FeedbackStatus",
    "FeedbackSummary",
]
