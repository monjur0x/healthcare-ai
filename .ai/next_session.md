# Next Session

## Objective

Implement the multimodal preprocessing module: `backend/preprocessing/multimodal`.

## Suggested Steps

1. Read `docs/SOFTWARE_ARCHITECTURE.md` and `.ai/current_context.md`.
2. Implement, in order:
   - `multimodal/metadata.py` — shared metadata dataclasses/schema for
     images and tabular EHR records.
   - `multimodal/fusion.py` — align and combine preprocessed image and
     CSV outputs (reuse `ImagePipeline` and `CSVPipeline`; do not
     duplicate logic).
   - `multimodal/__init__.py`.
3. Add unit tests under `backend/preprocessing/tests/test_multimodal.py`.
4. Run: `pytest`, black, ruff, isort (against `backend/pyproject.toml`).
5. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`, `docs/BACKLOG.md`,
   `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Reuse existing preprocessing stages; never duplicate logic.
- Dependency-light where reasonable.
- Log via `get_logger(__name__)`; raise custom exceptions from `exceptions.py`.

## Open Questions

- Image normalization statistics are not persisted for inference-time
  consistency (stateless fallback for `standard` mode) — see
  `docs/BACKLOG.md`.
