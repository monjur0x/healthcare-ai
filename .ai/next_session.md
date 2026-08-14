# Next Session

## Objective

Milestone 3+ — RAG module, CrewAI agents, or real flwr simulation
deployment (blocked on `ray`).

## Suggested Steps

1. Read `docs/SOFTWARE_ARCHITECTURE.md`, `docs/DECISIONS.md`, and
   `.ai/current_context.md`.
2. Read `docs/BACKLOG.md` for the scoped items.
3. Pick one of:
   - RAG module (`backend/rag/`): chunkers, embeddings, vector store
     (Qdrant), retriever; then a `rag_demo.py` example and tests.
   - CrewAI agents/tasks consuming preprocessing + model outputs.
   - Real flwr driver: wrap the synchronous `FedAvgServer` round logic
     in a `ServerApp`/`ClientApp` and run `flwr.simulation.run_simulation`
     (Ray backend). BLOCKED until `ray` is installed (heavy dependency).
4. Add unit tests under `backend/rag/tests/` or `backend/tests/`.
5. Run: `pytest`, black, ruff, isort (against `backend/pyproject.toml`);
   use the CrewAI venv (has flwr 1.33.0, torch, sklearn, PIL).
6. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`,
   `docs/BACKLOG.md`, `docs/DECISIONS.md` (if a new ADR),
   `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Preprocessing stays in `preprocessing/`, models in `models/`,
  retrieval in `rag/`, orchestration in `CrewAI/`, API in `api/`.
- Federation only moves weights; training/inference stay in models.
- `get_parameters` returns memory-shared NumPy views of the model
  state; snapshot with `.copy()` before the model trains further.
- Log via `get_logger(__name__)`; raise module exceptions from the
  module's `exceptions.py`.

## Open Questions

- Whether the real `run_simulation` integration check belongs in the
  unit suite (Ray spawn is slow/flaky) or as a separate smoke script.
- Whether the RAG retriever should use Qdrant or an in-memory vector
  store for hermetic tests (Qdrant needs a server / embedded mode).