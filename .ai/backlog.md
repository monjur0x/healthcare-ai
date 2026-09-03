# Backlog

Discovered-but-not-now items. Record here instead of implementing
mid-task (see AGENTS.md → Task Execution Rules).

## Deferred issues from the free-tier LLM tuning session

- [x] FIXED: RAG query-building — `build_disease_query` now appends
  elevated clinical markers (from `MARKER_THRESHOLDS`) to the
  disease-anchored query, so retrieval matches the patient's actual
  presentation, not just the condition name.
- [x] FIXED: Risk-score vs risk_factors inconsistency — `assess_risk`
  now takes `max(model P(disease), RISK_MARKER_WEIGHT * max normalized
  marker elevation)`; flagged markers can raise (never lower) the score,
  so the numeric score and the reported factors agree. New setting
  `CREW_RISK_MARKER_WEIGHT` (default 0.5; markers alone cap at medium).
- [x] FIXED: LLM narrative faithfulness — `create_tasks` now injects
  features, markers, and the assessed condition into the
  patient-analysis, disease-prediction, explanation, and risk-monitoring
  task descriptions via `_clinical_context_block`.
- [x] FIXED (same sweep): crew step-3 latent `NameError` when the RAG
  pipeline is absent; duplicate `@staticmethod` in `crew.py`;
  `crew_logging.py` infinite-recursion kickoff wrapper + F821 + dead
  duplicates; `scripts/run_rag_evaluation.py` never aggregated metrics
  (now persists mean P@k/R@k/MRR); `ingest_clinical_knowledge.py`
  duplicated blocks and used a file path as `output_dir`; silent
  `except: pass` in the treatment-planner / explainability agent routes
  now logged. Backend `ruff check` / `ruff format --check` fully clean.

## Privacy Layer (proposal §8 / flowchart AN node)

- Wire `federated/privacy.anonymize_frame` + `pseudonymize` into the
  hospital data-loading path so PII-like columns are dropped before any
  local training (currently implemented but unwired).

## Federation quality

- Shared-scaler study: clients currently train on unscaled canonical
  features (per-hospital scalers would desync FedAvg weight spaces).
  Evaluate round-0 federated standardization vs raw.
- Encrypted gRPC end-to-end demo across hosts with generated certs.

## Multi-Agent (proposal §6)

- Expose the six-agent CrewAI pipeline (A1 patient analysis … A6 risk
  monitoring) as distinct agents in `run_llm`, keeping the deterministic
  fallback path authoritative for prediction/risk/evidence.
- Wire `CrewAI/orchestrator/metrics.compute_agent_metrics` into the crew
  run and surface agent metrics in the report.

## RAG

- Expand bundled corpus (more conditions, WHO/CDC/NICE-style guideline
  summaries).
- [x] DONE: §12 RAG metrics wired via scripts/run_m3_evaluation.py;
  FAITHFULNESS_THRESHOLD now configurable (RAG_FAITHFULNESS_THRESHOLD) —
  set ~0.3 for TF-IDF (ceiling 0.66 verified) or use dense embedder at
  default 0.5.

## n8n

- [x] DONE (M3.4): `n8n/clinical-full.json` implements the 10-step flowchart
  workflow; verified rejection / low-risk / high-risk-notify paths live.
- [x] DONE: `n8n/risk-monitoring.json` polls `/api/v1/risk/alerts`
  every 15m and notifies clinicians per alert (activated locally).

## Dashboard

- [x] DONE: new "Risk Monitoring" tab — per-patient trend chart +
  direction metric, active escalation-alert list, clinician feedback
  form posting to `/api/v1/feedback` (client methods added).

## Housekeeping

- Image model training pipeline (currently inference-only; proposal
  future-extension §15 lists medical-image integration).

## Full-project review (2026-09-03)

7-module audit (preprocessing/models, federated, RAG, CrewAI, API,
risk/feedback/evaluation/scripts, frontend/n8n/config). Top
load-bearing claims verified by read (`crew.py:555`, `store_chroma.py:137`,
`client.py:103-106` + `privacy.py:145-230`, `transformer.py:156`,
`registry.py:39`). Not fixed — prioritized for future sessions.

### P0 — wrong results / broken paths / data leaks

- [x] FIXED: `backend/CrewAI/orchestrator/crew.py:555` — `_parse_report`
  missing `self` (bound-call `TypeError` on every `run_llm()` parse);
  signature fixed + `ClinicalReport | None` return type.
- [x] FIXED: `backend/preprocessing/csv/transformer.py:156` — encoder
  re-fit on every `transform()`; `CSVEncoder` gained `params()` /
  `from_params()` (mirroring `CSVScaler`), `CSVTransformer` /
  `CSVPipeline` accept `encoder_params`, `TabularClassifier` persists
  them in the joblib payload (old artifacts load with `None` → legacy
  behavior), train wires them via `prepare_tabular_data` + `train()`,
  `analyze_csv` reuses them; unseen label categories now raise
  `ValueError` instead of silent `NaN`, one-hot aligns to train columns.
- [x] FIXED: `backend/models/csv/torch_mlp.py:103,236` —
  `TorchMLPClassifier` gained `scaler_params` / `set_scaler_params` +
  `encoder_params` / `set_encoder_params` (parity with
  `TabularClassifier`), persisted in `save()` / restored in `load()`
  (`.get()` → old artifacts load with `None`); this also fixes the
  `AttributeError` at `api/services.py:1669` (`analyze_csv`) and the
  skipped-scaling path at `CrewAI/orchestrator/services.py:435` when a
  torch model is served, and engages the existing `hasattr` wiring in
  `train()` for federated torch models.
- [ ] `backend/preprocessing/image/normalization.py:189` — `_standard` on
  `uint8 0-255` with `~0.5` means, missing `/255` → values `~±500`.
- [x] FIXED: `backend/federated/client.py:103-106` — DP return
  discarded, original weights returned → DP-SGD had zero effect on
  shared weights. `_train_locally` now syncs the trained module back
  via `_apply_trained_weights` (direct `load_state_dict`, Opacus
  `_module.`-prefix strip on mismatch, loud `RuntimeError` on
  persistent mismatch instead of stale weights).
- [ ] `backend/federated/hospitals.py:211` + `distributed.py:595` —
  per-slice scalers never synced; heterogeneous canonical path unscaled vs
  partitioned scaled → FedAvg over incomparable spaces.
- [x] FIXED: `backend/federated/registry.py:39` (+ `risk/store.py`,
  `feedback/store.py`) — shared SQLite connection, no lock. All three
  stores gained an `RLock` (nested summary helpers are re-entrant),
  `check_same_thread=False` + `timeout=10`, and WAL journal mode for
  cross-process readers/writers; registry `register_model`
  count-then-insert is atomic under the lock, duplicate
  `record_round` rolls back then fails loud; `api/services.py`
  `_train_distributed` registry read wrapped in `try/finally`.
  Verified: 8-thread hammering on all three stores, unique versions,
  rollback-usable connection.
- [x] FIXED: `backend/rag/store_chroma.py:137` — trailing
  `_CHROMADB_AVAILABLE = False` overwrote the import probe, forcing
  `ImportError` even when installed; line removed so the probe governs.
  Verified: absent package → helpful `ImportError`, stubbed package →
  init/add/search/len functional.
- [ ] `backend/federated/distributed.py:619-624` — cert tuple passed as
  `root_certificates` → mTLS broken; `config.py:49` TLS off by default →
  plaintext hospital↔server gRPC.
- [ ] `backend/federated/__main__.py:197-203,270-276` — client/run paths
  re-call `build_hospital_sites`, clobbering `hospital_A-D/data.csv`.
- [ ] `backend/api/schemas.py:100-121` + `services.py:627-649` —
  `TrainRequest.dataset` accepts arbitrary paths, no confinement →
  local file read/traversal.

### P1 — invalid science / metrics / silent corruption

- [ ] FedAvg weighting: `federated/parameters.py:55`,
  `federated/server.py:250` unweighted; secure forces equal weights while
  non-secure is count-weighted (`distributed.py:290-293`).
- [ ] DP accounting: fresh `PrivacyEngine` per fit, flat client×round eps
  summed (`privacy.py:207-213`, `server.py:242-244`).
- [ ] OTP masks: fixed derivation, no seed/round nonce
  (`privacy.py:241,279-287`) → cross-round differencing leaks updates.
- [ ] Canonical: `canonical.py:170` maps kidney `glucose` from `bu`
  (blood urea); `canonical.py:36-48,139-152` schema omits proposal
  `Previous Diseases/Medication History`, heart maps 5/11.
- [ ] Agent metrics always zero: `crew.py:381-386` + `metrics.py:27-36`
  (`_output_of` misses `dict`/`list` shapes).
- [ ] LLM path tool-less: 5/7 agents get `tools=[]`
  (`agents.py:79-87` vs `crew.py:452-462`).
- [ ] API races: shared `AnalysisService` mutates model without lock
  (`api/services.py:364-396`); `_train_distributed` subprocess no timeout
  (`services.py:1196-1207`); routes leak business logic + swallow errors
  as 200 fallbacks (`routes.py:632-779,711-717,757-760`).
- [ ] Risk: `<MIN_TREND_POINTS` returns `latest_score=0.0/low`
  (`risk/store.py:243-254`); alerts stateless → n8n re-fires forever
  (`store.py:360-394`); `ALERTS_ENABLED` never enforced
  (`risk/config.py:40`).
- [ ] Feedback: `consumed` dropped from reads, list-then-consume
  non-atomic (`feedback/store.py:151-163`, `api/services.py:798-826`).
- [ ] Baselines: M3 B2/B4 and B3/B5 identical, latency double-counted
  (`scripts/run_m3_evaluation.py:439-455,200-214`); M2 single-class crash
  (`run_m2_experiment.py:84-94`); privacy MIA overlap + ignored CLI flags
  (`run_privacy_experiment.py:86-123`).
- [ ] RAG eval: 10 ground-truth IDs absent from `rag/corpus/`;
  script reimplements P/R/MRR with divergent MRR
  (`scripts/run_rag_evaluation.py:182-196`); TF-IDF default vs `0.5`
  faithfulness threshold (`rag/config.py:53-56`); BGE prefix on docs
  (`rag/embedder.py:240-245`); no vocab refit on incremental ingest
  (`rag/retriever.py:80-83`).

### P2 — gaps vs proposal / config drift

- [ ] Multimodal dirs empty (`preprocessing/multimodal/`,
  `models/multimodal/`) vs proposal imaging extension; explainability is
  magnitude-sort not SHAP/Grad-CAM (`crew.py:279-299`); treatment
  playbooks diabetes-only (`services.py:726-817`).
- [ ] Frontend: `frontend/requirements.txt` missing `pandas` (Risk tab
  crash at `streamlit_app.py:1441`); n8n `127.0.0.1:8000` breaks under
  Docker; hardcoded n8n credential IDs; `clinical-full-v2` undocumented.
- [ ] Config: `CREW_*` prefix bugs — `RISK_*`/`RAG_TOP_K` never bind
  (`orchestrator/config.py:63-87` vs `.env.example:151-171`);
  `LLM_BASE_URL/MAX_RETRIES`, `PREPROCESS_*`, `ACTIVE_PRESET`,
  `DOCTOR_NOTIFY_WEBHOOK`, `N8N_*` absent; personal absolute dataset
  paths in example; duplicate `RAG_TOP_K`.
- [ ] Docs: README endpoints table omits `/agents/*` + `/`
  (`README.md:318-341`); `risk-monitoring.json (to be added)` stale
  (`README.md:311`); `.gitignore` lacks generic `*.db/*.joblib` cover
  for relocated artifact paths.
