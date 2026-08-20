"""
Feedback / retrain loop.

Records clinician-confirmed outcome labels for past analyses
(:mod:`feedback.store`) and exposes the state helpers used by the API
and n8n orchestration. No API or model logic lives here; retraining
itself is performed by ``AnalysisService`` using the stored features as
extra training rows. The API-level retrain request/response models live
in ``api.schemas`` to avoid a feedback → api import cycle.
"""

from .config import FeedbackSettings, settings
from .schemas import (
    DatasetPreset,
    FeedbackRecord,
    FeedbackRequest,
    FeedbackStatus,
    FeedbackSummary,
)
from .store import FeedbackStore, FeedbackStoreError

__all__ = [
    "DatasetPreset",
    "FeedbackRecord",
    "FeedbackRequest",
    "FeedbackSettings",
    "FeedbackStatus",
    "FeedbackStore",
    "FeedbackStoreError",
    "FeedbackSummary",
    "settings",
]
