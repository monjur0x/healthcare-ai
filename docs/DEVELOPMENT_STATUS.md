# Development Status

## Legend

- [x] Implemented and tested.
- [ ] Not started.

---

## Milestone 1 — Preprocessing

### Package scaffolding

- [x] Package structure (`backend/preprocessing/__init__.py`)
- [x] Global configuration (`config.py`)
- [x] Centralized logging (`logger.py`)
- [x] Custom exceptions (`exceptions.py`)

### CSV preprocessing (`backend/preprocessing/csv`)

- [x] Validator (`validator.py`)
- [x] Cleaner / column normalization (`cleaner.py`)
- [x] Missing value imputation (`imputer.py`)
- [x] Categorical encoding (`encoder.py`)
- [x] Feature engineering (`feature_engineering.py`)
- [x] Scaling (`scaler.py`)
- [x] Transformer / pipeline orchestration (`transformer.py`)
- [x] High-level entry point (`pipeline.py`)
- [x] Unit tests (`tests/test_csv.py`) — 27 passing
### Image preprocessing (`backend/preprocessing/image`)

- [x] Validator (`validator.py`)
- [x] Loader (`loader.py`) — PNG/JPG via Pillow, DICOM via optional `pydicom`
- [x] Augmentation (`augmentation.py`) — deterministic, seeded
- [x] Normalization (`normalization.py`) — minmax / zero_mean / standard
- [x] Pipeline (`pipeline.py`) — load → validate → resize → augment → normalize
- [x] Convenience API (`preprocessing.py`) — single image, batch, directory
- [x] Unit tests (`tests/test_image.py`) — 31 passing

### Multimodal preprocessing (`backend/preprocessing/multimodal`)

- [ ] Fusion
- [ ] Metadata

---

## Tooling

- [x] `backend/pyproject.toml` with shared Black / Ruff / isort settings
- [ ] Lint/format/test commands documented before every session (see `AGENTS.md`)

---

## Not yet planned

Milestones for `models/`, `federated/`, `rag/`, `evaluation/`, `api/`
are defined at the repository level but not yet scoped in the backlog.