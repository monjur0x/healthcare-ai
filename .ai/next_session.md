# Next Session

## Objective

Milestone 9+ — commit + push the staged Milestone 8.1 (image analysis +
friendly dashboard input), LLM/.env enabling, and Milestone 9
(privacy-preserving federated learning, paper §8) work once the user
gives the go-ahead, then continue with API hardening and the real flwr
deployment path.

## Suggested Steps

1. Read `docs/SOFTWARE_ARCHITECTURE.md`, `docs/DECISIONS.md`, and
   `.ai/current_context.md`.
2. Read `docs/BACKLOG.md` for the scoped items.
3. Settle the open CrewAI question with the user: keep
   `gemini-3.7-flash` as the default (currently transient 503 "high
   demand") or switch to `gemini-3.6-flash`.
4. Pick one of:
   - Commit + push (only with explicit user go-ahead): Milestone 8.1 +
     LLM/.env + Milestone 9 privacy work. Proposed focused commits:
     `feat(api)+feat(frontend) image analysis + friendly dashboard`,
     `feat(crew) enable Gemini LLM path + .env.example`,
     `feat(federated) DP + secure aggregation + privacy metrics (ADR-013)`.
   - Production DP pass: re-run the federated DP path with Opacus
     `secure_mode=True` (the experimentation UserWarning calls for a
     final secure retrain before release; see BACKLOG).
   - API hardening: file-upload endpoint for CSV / image inference (add
     an upload widget to `frontend/streamlit_app.py`), full OAuth
     (currently optional static `API_TOKEN`), deployment Dockerfile for
     `uvicorn api.main:app`, dashboard privacy-metrics panel from
     `federated_metrics.privacy`.
   - Image model lifecycle: expose retraining / online / split reporting
     for the brain-tumor CNN (currently offline-only via
     `scripts/train_image_model.py`, ADR-012), categorical-feature
     picklists in the dashboard form, per-class confidence histogram.
   - Real flwr driver: wrap the synchronous `FedAvgServer` round logic
     in a `ServerApp`/`ClientApp` and run `flwr.simulation.run_simulation`
     (Ray backend). BLOCKED until `ray` is installed (heavy dependency).
   - Downstream n8n branches: append report to a local file, notify via
     Slack/Discord (requires real credentials), docker-compose profile
     running n8n + FastAPI + Qdrant together.
5. Add unit tests under the module's `tests/` directory; keep test-file
   basenames unique across `backend/`.
6. Run: `pytest preprocessing/tests models/tests evaluation/tests federated/tests rag/tests examples/tests CrewAI/orchestrator/tests api/tests`
   (from `backend/`) and `pytest dashboard/tests` (from `frontend/`;
   use the CrewAI venv for frontend too — streamlit is installed there);
   black, ruff, isort; use the CrewAI venv
   (`backend/CrewAI/.venv-opencode`, has flwr 1.33.0, torch, sklearn,
   PIL, pydantic-settings, crewai 1.15.11, fastapi 0.138, streamlit
   1.61, opacus 1.6.0). Linters: `/tmp/opencode/lintenv/bin/{black,isort,ruff}`
   (recreate with `python -m venv /tmp/opencode/lintenv && pip install
   black isort ruff` if `/tmp` was cleared).
7. Update `docs/DEVELOPMENT_STATUS.md`, `docs/CHANGELOG.md`,
   `docs/BACKLOG.md`, `docs/DECISIONS.md` (if a new ADR),
   `.ai/current_context.md`, `.ai/next_session.md`.

## Conventions Reminder

- Preprocessing stays in `preprocessing/`, models in `models/`,
  retrieval in `rag/`, orchestration in `CrewAI/orchestrator/`,
  API in `api/`, orchestration only in `n8n/` (n8n triggers, CrewAI
  reasons — `AGENTS.md`), view layer only in `frontend/` (ADR-010).
- API routes only validate + delegate to `api/services.AnalysisService`;
  domain exceptions are mapped to `APIError` subclasses (ADR-009);
  training is an API endpoint (`/api/v1/train`, ADR-011) — central fit
  default, federated only with `model_name='mlp'`.
- Image input follows ADR-012: `preprocessing.image` → `models.image` →
  `ClinicalCrew` image branch → same `ClinicalReport`; base64 image JSON
  decoded by the Pydantic validator in `AnalyzeImageRequest` (a plain
  `bytes` field stores the base64 text — never rely on it to decode).
  Feature forms in the dashboard are driven by `GET /api/v1/model`;
  keep the raw-JSON fallback for when no model is configured.
- Federation only moves weights; training/inference stay in models.
- Agents orchestrate reasoning and consume pipeline outputs; they never
  implement ML algorithms (see `AGENTS.md`).
- CrewAI agents/tasks/crew construct hermetically (no LLM key); the LLM
  path (`ClinicalCrew.run_llm`) needs `CREW_LLM_API_KEY` and a provider
  extra installed.
- `get_parameters` returns memory-shared NumPy views of the model
  state; snapshot with `.copy()` before the model trains further.
- Privacy (ADR-013): DP + secure aggregation are opt-in per federated
  train request (`POST /api/v1/train` → `differential_privacy` /
  `noise_multiplier` / `max_grad_norm` / `privacy_delta` /
  `secure_aggregation`). DP requires a torch-backed model
  (`TorchMLPClassifier`); `SecureAggregator` masks only cancel under
  equal weights. Opacus `secure_mode=False` for experimentation — the
  production pass must re-run with `secure_mode=True`.
- Log via `get_logger(__name__)`; raise module exceptions from the
  module's `exceptions.py`.
- Background servers for live checks: start via
  `scripts/run_system.sh start` (`N8N_ENABLED=0` to skip n8n; n8n is
  `docker run -d --rm --name healthcare-n8n -p 5678:5678
  -v healthcare_n8n_data:/home/node/.n8n n8nio/n8n`). Avoid
  `pkill -f "uvicorn api.main"` self-matches — use the `[u]vicorn`
  bracket trick. `run_system.sh` trains the default model and serves
  the image model from `backend/artifacts/brain/global_model.pt`
  (`API_IMAGE_MODEL_PATH`); kill stale uvicorn PIDs before starting if
  `/api/v1/model` 404s (an old process serves old code).

## Open Questions

- Whether the real `run_simulation` integration check belongs in the
  unit suite (Ray spawn is slow/flaky) or as a separate smoke script.
- Whether the RAG production path should adopt sentence-transformers /
  Qdrant (ADR-007 defers this behind the `Embedder` / `VectorStore`
  interfaces).
- Which LLM provider to wire into `ClinicalCrew.run_llm`
  (needs an API key; never commit secrets).
- Whether to keep `gemini-3.7-flash` as the CrewAI default (it returned
  transient 503 "high demand" on 2026-08-13, two days after launch) or
  fall back to `gemini-3.6-flash`, which verified working end-to-end.
- Whether a Next.js dashboard is still wanted on top of the Streamlit
  one (ADR-010 picks Streamlit for now).
- Whether image retraining should become an API endpoint despite
  ADR-012's offline-script choice (long-running HTTP is poor UX), or
  stay offline behind a dashboard "rebuild model" trigger.
- `PRESETS` duplication: the registry lives in `api/services.py`, but
  `examples/fedavg_demo.py` and `examples/clinical_crew_demo.py` keep
  their own copies — consider a shared registry if another consumer
  appears.