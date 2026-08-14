# Next Session

## Objective

Milestone 2 — Models: finish the federated stack (server/simulation
driver) and optionally federate the CNN.

## Suggested Steps

1. Read `docs/SOFTWARE_ARCHITECTURE.md`, `docs/DECISIONS.md`, and
   `.ai/current_context.md`.
2. Read `docs/BACKLOG.md` for the scoped Milestone 2 items.
3. Implement `backend/federated/` server driver:
   - A `flwr.server.ServerApp` (or `start_server`) using
     `flwr.server.strategy.FedAvg` wired to `average_weights` semantics.
   - An in-process simulation driver (flwr `run_simulation`) so the test
     stays hermetic — 2-3 `FederatedClient`s over synthetic data.
   - Record round convergence by evaluating the global model after each
     round with `evaluation.classification_metrics`.
4. Optional: CNN federation — add `ImageClassifier.partial_fit` (a few
   gradient steps reusing the existing model weights) and a client test.
5. Add unit tests under `backend/federated/tests/`.
6. Run: `pytest`, black, ruff, isort (against `backend/pyproject.toml`);
   use the CrewAI venv (has flwr 1.33.0).
7. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`,
   `docs/BACKLOG.md`, `docs/DECISIONS.md` (if a new ADR),
   `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Federation only moves weights; training/inference stay in models.
- Keep the flwr client numpy-native; record API changes (e.g. any
  `NumPyClient` vs `ClientApp` divergence) in `docs/DECISIONS.md`.
- Log via `get_logger(__name__)`; raise model exceptions from
  `models/exceptions.py`.

## Open Questions

- Whether the driver should be a real `ServerApp` (networked) or the
  in-process `run_simulation` for now — the repo has no deployment
  target yet, so simulation is the hermetic default.
- Whether `partial_fit` for the CNN should accept an `epochs` arg or
  match the tabular one-pass contract.