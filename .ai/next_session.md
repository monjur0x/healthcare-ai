# Next Session

## Objective

Scope and implement the prediction models milestone: `backend/models`.

## Suggested Steps

1. Read `docs/SOFTWARE_ARCHITECTURE.md` and `.ai/current_context.md`.
2. Read `docs/BACKLOG.md` and scope a Milestone-2 backlog for `models/`.
3. Consider implementing a minimal vertical slice first:
   - A simple CSV model (e.g., logistic regression on preprocessed
     `CSVResult` features).
   - A simple image model (e.g., small CNN on `ImageResult` batches).
   - One multimodal model consuming `FusionResult`.
4. Add unit tests under `backend/tests/` or `models/tests/`.
5. Run: `pytest`, black, ruff, isort (against `backend/pyproject.toml`).
6. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`,
   `docs/BACKLOG.md`, `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Models only perform ML/DL inference; preprocessing lives in
  `backend/preprocessing` and is never duplicated.
- Keep dependencies minimal; revisit choices (torch vs sklearn) in
  `docs/DECISIONS.md`.
- Log via `get_logger(__name__)`; raise custom exceptions.

## Open Questions

- Should `models/` use PyTorch, TensorFlow, or sklearn for the first
  iteration? Affects `requirements.txt` and whether image support
  already present in auxiliary environments can be reused.
- Whether to add a shared `models/base.py` abstract interface matching
  the preprocessing pipeline pattern.