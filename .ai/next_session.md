# Next Session

## Objective

Implement the medical image preprocessing module: `backend/preprocessing/image`.

## Suggested Steps

1. Read `docs/SOFTWARE_ARCHITECTURE.md` and `.ai/current_context.md`.
2. Implement, in order:
   - `image/validator.py` — file type, dimension, channel validation.
   - `image/loader.py` — PNG/JPG/DICOM loading via Pillow (+ `pydicom` for `.dcm`).
   - `image/augmentation.py` — deterministic augmentation (fixed seed).
   - `image/normalization.py` — resize to `IMAGE_WIDTH/HEIGHT`, normalize to `IMAGE_CHANNELS`.
   - `image/pipeline.py` + `image/__init__.py`.
3. Add unit tests under `backend/preprocessing/tests/test_image.py`.
4. Run: `pytest`, black, ruff, isort (against `backend/pyproject.toml`).
5. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`, `docs/BACKLOG.md`,
   `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Normalize inputs before validation (match CSV module behavior).
- Dependency-light where reasonable; add `pydicom` to `backend/CrewAI/requirements.txt`
  only if needed for `.dcm` support.
- Log via `get_logger(__name__)`; raise custom exceptions from `exceptions.py`.