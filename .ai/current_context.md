# Current Context

## Current Milestone

Milestone 1

## Current Module

backend/preprocessing

## Current Task

Preprocessing is implemented for CSV and image. The next engineering
work is the multimodal preprocessing module.

## Completed

- Preprocessing scaffolding (config, exceptions, logger)
- Preprocessing CSV pipeline (validator, cleaner, imputer, encoder, feature engineering, scaler, transformer, pipeline)
- CSV unit tests (27 passing)
- Preprocessing image pipeline (validator, loader, augmentation, normalization, pipeline, convenience API)
- Image unit tests (31 passing)
- Tooling config (backend/pyproject.toml, isort/ruff aligned)

## Next Files (backend/preprocessing/multimodal)

- fusion.py
- metadata.py

## Design Notes

Preprocessing pipeline must remain reusable by:

- Flower
- FastAPI
- CrewAI

No business logic.

No prediction.

Only preprocessing.

Image preprocessing is dependency-light (Pillow + NumPy). DICOM support
requires optional `pydicom`. Augmentation is deterministic via a seeded
RNG.

## Status

CSV and image complete. Multimodal in progress.
