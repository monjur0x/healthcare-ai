# Next Session

## Objective

Milestone 2 — Models: federated metrics and/or real flwr simulation
deployment; or extend the demo to the image path.

## Suggested Steps

1. Read `docs/SOFTWARE_ARCHITECTURE.md`, `docs/DECISIONS.md`, and
   `.ai/current_context.md`.
2. Read `docs/BACKLOG.md` for the remaining scoped Milestone 2 items.
3. Pick one of:
   - Federated metrics (`federated/metrics.py`): communication cost
     (bytes exchanged per round), convergence (round-to-round deltas),
     training time (per-round + total wall time). Extend
     `FedAvgServer`/`RoundResult` to record per-round duration and
     wire the metrics into the demo report.
   - Real flwr driver: wrap the synchronous `FedAvgServer` round logic
     in a `ServerApp`/`ClientApp` and run `flwr.simulation.run_simulation`
     (Ray backend). BLOCKED until `ray` is installed (heavy dependency).
   - Image demo: extend `fedavg_demo.py` (or add `examples/` image
     script) to federate `ImageClassifier` over the brain-tumor MRI
     dataset; needs GPU for practical runtimes.
4. Add unit tests under `backend/federated/tests/` or `backend/tests/`.
5. Run: `pytest`, black, ruff, isort (against `backend/pyproject.toml`);
   use the CrewAI venv (has flwr 1.33.0).
6. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`,
   `docs/BACKLOG.md`, `docs/DECISIONS.md` (if a new ADR),
   `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Federation only moves weights; training/inference stay in models.
- `get_parameters` returns memory-shared NumPy views of the model
  state; snapshot with `.copy()` before the model trains further.
- Keep the flwr client numpy-native; record API changes (e.g. any
  `NumPyClient` vs `ClientApp` divergence) in `docs/DECISIONS.md`.
- Log via `get_logger(__name__)`; raise model exceptions from
  `models/exceptions.py`.

## Open Questions

- Whether the real `run_simulation` integration check belongs in the
  unit suite (Ray spawn is slow/flaky) or as a separate smoke script.
- Whether `partial_fit` for the CNN should accept an `epochs` arg or
  match the tabular one-pass contract (currently one-pass).