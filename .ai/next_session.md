# Next Session

## Objective

Milestone 2 — Models: either exercise the federated driver with flwr's
real simulation API, or federate the CNN.

## Suggested Steps

1. Read `docs/SOFTWARE_ARCHITECTURE.md`, `docs/DECISIONS.md`, and
   `.ai/current_context.md`.
2. Read `docs/BACKLOG.md` for the scoped Milestone 2 items.
3. Pick one of:
   - Real flwr driver: wrap the synchronous `FedAvgServer` round logic
     in a `ServerApp`/`ClientApp` and run `flwr.simulation.run_simulation`
     (Ray backend) as an integration check; add `federated/metrics.py`
     for communication cost / convergence / training time.
   - CNN federation: add `ImageClassifier.partial_fit` (a few gradient
     steps reusing existing weights) and a client test so the image path
     joins rounds.
   - End-to-end demo: `CSVPipeline` → `TabularClassifier` → FedAvg
     rounds → `evaluation.classification_metrics` report.
4. Add unit tests under `backend/federated/tests/` or `backend/tests/`.
5. Run: `pytest`, black, ruff, isort (against `backend/pyproject.toml`);
   use the CrewAI venv (has flwr 1.33.0).
6. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`,
   `docs/BACKLOG.md`, `docs/DECISIONS.md` (if a new ADR),
   `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Federation only moves weights; training/inference stay in models.
- Keep the flwr client numpy-native; record API changes (e.g. any
  `NumPyClient` vs `ClientApp` divergence) in `docs/DECISIONS.md`.
- Log via `get_logger(__name__)`; raise model exceptions from
  `models/exceptions.py`.

## Open Questions

- Whether the real `run_simulation` integration check belongs in the
  unit suite (Ray spawn is slow/flaky) or as a separate smoke script.
- Whether `partial_fit` for the CNN should accept an `epochs` arg or
  match the tabular one-pass contract.