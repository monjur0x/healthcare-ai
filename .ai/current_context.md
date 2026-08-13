# Current Context

## Current Milestone

Milestone 2 — Models (in progress)

## Current Module

backend/models

## Current Task

Shared model interface and first tabular model are complete (10 tests).
The next engineering work is an image model or a multimodal model.

## Completed

- Milestone 1: preprocessing (CSV + image + multimodal), 70 tests
- Shared model interface (`models/base.py`) — fit/predict/predict_proba/save/load
- Model exceptions (`models/exceptions.py`)
- `TabularClassifier` (`models/csv/tabular.py`) — sklearn gradient
  boosting / logistic / mlp, joblib persistence, deterministic seeds
- `backend/requirements.txt`
- Model unit tests (10 passing); total suite 80 passing

## Next Files (backend/models)

- image/ — visual models (EfficientNetV2 / DenseNet / Swin-T); requires torch
- multimodal/ — model consuming `FusionResult`
- integrate with `CSVPipeline` output directly (accepts DataFrame today)

## Design Notes

- Models consume preprocessing outputs; no preprocessing inside models.
- Reuse `preprocessing.config.settings.RANDOM_SEED` for reproducibility.
- Testing requires sklearn: use the CrewAI venv
  (`backend/CrewAI/.venv-opencode`) which has sklearn/joblib.

## Status

Models: shared + tabular complete. Image/multimodal and federation next.