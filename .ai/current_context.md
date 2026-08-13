# Current Context

## Current Milestone

Milestone 1 — complete

## Current Module

backend/preprocessing

## Current Task

Preprocessing is fully implemented (CSV, image, multimodal) with 70
passing unit tests. The next milestone is `models/`.

## Completed

- Preprocessing scaffolding (config, exceptions, logger)
- Preprocessing CSV pipeline (validator, cleaner, imputer, encoder, feature engineering, scaler, transformer, pipeline)
- CSV unit tests (27 passing)
- Preprocessing image pipeline (validator, loader, augmentation, normalization, pipeline, convenience API)
- Image unit tests (31 passing)
- Multimodal module (metadata + fusion, concatenate with summary/flatten image reduction)
- Multimodal unit tests (12 passing)
- Tooling config (backend/pyproject.toml, isort/ruff aligned)

## Next Files (backend/models)

Not yet scoped. Suggested start:

- models/csv/ — tabular models consuming CSVPipeline output
- models/image/ — vision models consuming ImagePipeline output
- models/architecture/ or multimodal — models consuming FusionResult

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

Milestone 1 complete. Models milestone next.