# Next Session

## Objective

Milestone 2 — Models: implement the Flower federated learning tie-in
(and optionally a small end-to-end demo pipeline).

## Suggested Steps

1. Read `docs/SOFTWARE_ARCHITECTURE.md` and `.ai/current_context.md`.
2. Read `docs/BACKLOG.md` for the scoped Milestone 2 items.
3. Implement `backend/federated/`:
   - Install `flwr` into the CrewAI venv
     (`backend/CrewAI/.venv-opencode/bin/pip install flwr`) and add it to
     `backend/requirements.txt`.
   - A client wrapping a fitted `TabularClassifier`: `get_parameters` /
     `set_parameters` over numpy weights, `fit` via
     `model.classifier.coefs_`/`intercepts_` (or the state dict for the
     CNN), plus `evaluate` using `evaluation.classification_metrics`.
   - A minimal FedAvg server strategy for a 2-3 client smoke test.
   - Prefer `flwr`'s in-process simulation or a local port for tests so
     the suite stays hermetic.
4. Add unit tests under `backend/federated/tests/`.
5. Run: `pytest`, black, ruff, isort (against `backend/pyproject.toml`);
   use the CrewAI venv for flwr/sklearn/torch-dependent tests.
6. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`,
   `docs/BACKLOG.md`, `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Models perform inference only; federation only orchestrates weights.
- Keep the flwr client numpy-native (no torch serialization for the
  tabular path); record the flwr version in `docs/DECISIONS.md`.
- Log via `get_logger(__name__)`; raise model exceptions from
  `models/exceptions.py`.

## Open Questions

- flwr version (0.13.x uses the `flwr.server` strategy API; 1.x moved to
  `flwr.simulation` / `Flower` config) — pin the version installed.
- Whether to federate all three model types now or only the tabular
  path first (weights extraction differs per framework).