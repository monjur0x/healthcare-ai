# Changelog

## [Unreleased]

### Added

- `backend/preprocessing/csv/` module: `validator.py`, `cleaner.py`,
  `imputer.py`, `encoder.py`, `feature_engineering.py`, `scaler.py`,
  `transformer.py`, `pipeline.py`, and package `__init__.py`.
- Unit tests for the CSV preprocessing module (`tests/test_csv.py`,
  27 tests passing).
- `backend/pyproject.toml` with shared Black / Ruff / isort settings.

### Changed

- Replaced sklearn-based scaling with a dependency-light NumPy
  implementation in `scaler.py`.
- Standardized tooling on `backend/pyproject.toml`.

### Fixed

- Normalized column handling so all stages operate on lowercase
  snake_case column names.
- Graceful skip of unavailable feature-engineering features instead of
  hard failure (configurable via `strict=True`).