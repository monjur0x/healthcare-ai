# Next Session

## Objective

Continue Milestone 2 — Models: add the Flower federated learning tie-in
and the evaluation hooks.

## Suggested Steps

1. Read `docs/SOFTWARE_ARCHITECTURE.md` and `.ai/current_context.md`.
2. Read `docs/BACKLOG.md` for the scoped Milestone 2 items.
3. Pick one of:
   - Federated tie-in: `backend/federated/` with a Flower (flwr)
     server and a client that wraps a fitted `TabularClassifier`
     (state-free numpy weights via `get_parameters`/`set_parameters`).
   - Evaluation: `backend/evaluation/` metrics (ROC-AUC, PR-AUC, MCC)
     consumed from each model's `predict_proba`.
   - Wire a small end-to-end demo: `CSVPipeline` → `TabularClassifier`
     → metrics, to validate the contract.
4. Add unit tests under `backend/tests/` (avoid `test_image.py`
   basename collision: preprocessing already owns it).
5. Run: `pytest`, black, ruff, isort (against `backend/pyproject.toml`);
   use the CrewAI venv for sklearn/torch-dependent tests.
6. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`,
   `docs/BACKLOG.md`, `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Models perform inference only; reuse `preprocessing` outputs.
- `MODEL_*` env vars (seed, training hyperparameters) in
  `models/config.py`; keep weights exchange numpy-native for flwr.
- Log via `get_logger(__name__)`; raise model exceptions from
  `models/exceptions.py`.
- Record framework choices (flwr version) in `docs/DECISIONS.md`.

## Open Questions

- flwr version / whether a full server is needed or only the
  client-side parameter (de)serialization + a unit-tested `flwr` client.
- Whether evaluation uses a plain `sklearn.metrics` module or a custom
  `BackendMetrics` wrapper shared with CrewAI.