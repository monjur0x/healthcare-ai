# Next Session

## Objective

Continue after Milestone 10 (evaluation-gap closure, committed + pushed).
Likely directions: real flwr deployment path, API hardening, metric
exposure, or the RAGAS/agent-metric baseline study. Read
`docs/DEVELOPMENT_STATUS.md` + `.ai/current_context.md` first (AGENTS.md
workflow).

## Suggested Steps

1. Read `docs/SYSTEM_SPECIFICATION.md`, `docs/SOFTWARE_ARCHITECTURE.md`,
   `docs/DEVELOPMENT_STATUS.md`, `docs/BACKLOG.md`, `docs/DECISIONS.md`,
   `.ai/current_context.md`.
2. Settle the open CrewAI question with the user: keep
   `gemini-3.7-flash` as the default (currently transient 503 "high
   demand") or switch to `gemini-3.6-flash`.
3. Pick one of the backlog directions:
   - **Expose metrics via the API**: add a `metrics`/`evaluate` endpoint
     that runs `rag_quality_metrics` and `compute_agent_metrics` on a
     report, or extend `POST /api/v1/analyze` responses with the
     `agent_metrics` block. Currently the Milestone 10 metrics are
     library functions + tests only.
   - **RAGAS-vs-heuristic calibration** (paper §12): compare the LLM-free
     `faithfulness` / `answer_relevancy` proxies against a judge-LLM
     baseline on a small labeled set; record findings in the docs.
   - **Baseline comparison study** (paper §13): run centralized vs
     federated vs federated+RAG vs federated+multi-agent on the shipped
     datasets and record accuracy / privacy / latency.
   - **Production DP pass**: re-run the federated DP path with Opacus
     `secure_mode=True` (experimentation UserWarning asks for a final
     secure retrain before release; see BACKLOG).
   - **Real flwr driver**: wrap the synchronous `FedAvgServer` round logic
     in a `ServerApp`/`ClientApp` and run `flwr.simulation.run_simulation`
     (Ray backend). BLOCKED until `ray` is installed (heavy dependency).
   - **API hardening**: file-upload endpoint for CSV / image inference,
     full OAuth (currently optional static `API_TOKEN`), deployment
     Dockerfile for `uvicorn api.main:app`.
   - **Image model lifecycle**: retraining / online / split reporting for
     the brain-tumor CNN (currently offline-only via
     `scripts/train_image_model.py`, ADR-012).
4. Add unit tests under the module's `tests/` directory; keep test-file
   basenames unique across `backend/` (a `test_metrics.py` collides
   between `evaluation/tests/` and `CrewAI/orchestrator/tests/`).
5. Run: `pytest preprocessing/tests models/tests evaluation/tests federated/tests rag/tests examples/tests CrewAI/orchestrator/tests api/tests`
   (from `backend/`) and `pytest dashboard/tests` (from `frontend/`;
   use the CrewAI venv — streamlit is installed there). Lint from
   `backend/` with the CrewAI venv binaries:
   `CrewAI/.venv-opencode/bin/{black,isort,ruff}` — e.g.
   `ruff check . ../frontend`, `black --check . ../frontend`,
   `isort --check-only . ../frontend --skip CrewAI/.venv-opencode`.
   Never run ruff from the repo root or with `--config
   backend/pyproject.toml` from root (both mis-handle the frontend /
   first-party detection).
6. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`,
   `docs/BACKLOG.md`, `docs/DECISIONS.md` (if a new ADR),
   `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Preprocessing stays in `preprocessing/`, models in `models/`,
  retrieval in `rag/`, orchestration in `CrewAI/orchestrator/`,
  API in `api/`, orchestration only in `n8n/` (n8n triggers, CrewAI
  reasons — `AGENTS.md`), view layer only in `frontend/` (ADR-010).
- RAG defaults stay TF-IDF + in-memory `VectorStore` (ADR-007); Chroma
  (`RAG_VECTOR_STORE=chroma`) and dense embeddings
  (`RAG_EMBEDDING_MODEL=sentence-transformer`) are opt-in behind the
  `Embedder` / `VectorStore` / `build_vector_store()` interfaces.
- API routes only validate + delegate to `api/services.AnalysisService`;
  domain exceptions are mapped to `APIError` subclasses (ADR-009);
  training is an API endpoint (`/api/v1/train`, ADR-011) — central fit
  default, federated only with `model_name='mlp'`.
- Image input follows ADR-012: `preprocessing.image` → `models.image` →
  `ClinicalCrew` image branch → same `ClinicalReport`; base64 image JSON
  decoded by the Pydantic validator in `AnalyzeImageRequest`.
- Federation only moves weights; training/inference stay in models.
- Agents orchestrate reasoning and consume pipeline outputs; they never
  implement ML algorithms (see `AGENTS.md`).
- CrewAI agents/tasks/crew construct hermetically (no LLM key); the LLM
  path (`ClinicalCrew.run_llm`) needs `CREW_LLM_API_KEY` and a provider
  extra installed.
- `get_parameters` returns memory-shared NumPy views of the model
  state; snapshot with `.copy()` before the model trains further.
- Privacy (ADR-013): DP + secure aggregation are opt-in per federated
  train request; DP requires a torch-backed model (`TorchMLPClassifier`);
  `SecureAggregator` masks only cancel under equal weights. Opacus
  `secure_mode=False` for experimentation — the production pass must
  re-run with `secure_mode=True`. `opacus>=1.5.0` is declared in
  `backend/requirements.txt`.
- Transport security (ADR-014): TLS/mTLS is a deployment-layer concern
  (reverse proxy), not in-process encryption; no application code
  change.
- Log via `get_logger(__name__)`; raise module exceptions from the
  module's `exceptions.py`.
- Background servers for live checks: start via
  `scripts/run_system.sh start` (`N8N_ENABLED=0` to skip n8n). Avoid
  `pkill -f "uvicorn api.main"` self-matches — use the `[u]vicorn`
  bracket trick. Kill stale uvicorn PIDs before starting if
  `/api/v1/model` 404s (an old process serves old code).

## Open Questions

- Whether the real `run_simulation` integration check belongs in the
  unit suite (Ray spawn is slow/flaky) or as a separate smoke script.
- Whether to keep `gemini-3.7-flash` as the CrewAI default (transient
  503 "high demand" on 2026-08-13) or fall back to `gemini-3.6-flash`,
  which verified working end-to-end.
- Whether RAGAS / agent metrics should be exposed through the API now
  (currently library functions + tests only) or saved for the paper's
  evaluation section.
- Whether a Next.js dashboard is still wanted on top of the Streamlit
  one (ADR-010 picks Streamlit for now).
- Whether image retraining should become an API endpoint despite
  ADR-012's offline-script choice, or stay offline behind a dashboard
  "rebuild model" trigger.
- `PRESETS` duplication: the registry lives in `api/services.py`, but
  `examples/fedavg_demo.py` and `examples/clinical_crew_demo.py` keep
  their own copies — consider a shared registry if another consumer
  appears.