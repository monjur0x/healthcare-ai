# Next Session

## Objective

Continue Milestone 2 — Models: build an image model or a multimodal model.

## Suggested Steps

1. Read `docs/SOFTWARE_ARCHITECTURE.md` and `.ai/current_context.md`.
2. Read `docs/BACKLOG.md` for the scoped Milestone 2 items.
3. Pick one of:
   - Image model: `models/image/` (e.g., small CNN on `ImageResult`
     batches, or a pretrained backbone). Requires adding torch to
     `backend/requirements.txt`; validate with the CrewAI venv.
   - Multimodal model: `models/multimodal/` consuming `FusionResult`
     (e.g., an MLP over the fused matrix) — dependency-light, no torch.
   - Federated tie-in: a `federated/` Flwr client wrapping
     `TabularClassifier`.
4. Add unit tests under `backend/models/tests/`.
5. Run: `pytest`, black, ruff, isort (against `backend/pyproject.toml`);
   use the CrewAI venv for sklearn/torch-dependent tests.
6. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`,
   `docs/BACKLOG.md`, `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Models perform inference only; reuse `preprocessing` outputs.
- Keep dependencies minimal; record framework choices in
  `docs/DECISIONS.md`.
- Log via `get_logger(__name__)`; raise model exceptions from
  `models/exceptions.py`.

## Open Questions

- torch vs sklearn-first for image models (image preprocessing is
  Pillow/NumPy-only today; the CrewAI venv already has torch).
- Whether `models/` should implement a multimodal fusion-predictor or
  defer to the existing `CrewAI/app/models/fusion.py`.