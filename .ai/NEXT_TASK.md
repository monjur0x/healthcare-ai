# Next Task (self-contained handoff — no conversation context needed)

> Authoritative "what to fix next" for a fresh AI session. Everything here is
> verified against current `main` (commit `4cef96e`, 2026-09-10). Repo is
> `backend/`. Venv python: `backend/CrewAI/.venv-opencode/bin/python`.

## 1. P0 — correctness / security (fix these first; all have known root causes)

1. **mTLS / encrypted gRPC is broken → plaintext hospital↔server traffic.**
   `backend/federated/distributed.py` passes a cert tuple as Flower's
   `root_certificates` (the API misuse the backlog audit flagged at
   `distributed.py:619-624`); TLS is off by default (`federated/config.py`).
   The proposal's "Encrypted Communication" privacy claim is therefore not
   operational. Fix the cert wiring OR state plaintext loudly in docs. Check
   `docs/PRIVACY_NOTES.md` §8 for the current claim.
2. **`build_hospital_sites` clobbers hospital data.**
   `backend/federated/__main__.py` re-runs site building on `client`/`run`
   paths (~lines 197, 270), overwriting `backend/data/hospitals/hospital_A–
   D/data.csv`. Contradicts the "local files never modified" canonical claim.
   Guard so the build runs once.
3. **Path traversal in `POST /api/v1/train`.**
   `TrainRequest.dataset` accepts arbitrary paths with no confinement
   (`backend/api/schemas.py:100-121`, `services.py:627-649`) → local file
   read. Confine/normalize.
4. **Image `_standard` normalization produces values ~±500.**
   `backend/preprocessing/image/normalization.py` `_standard` runs z-score on
   uint8 0-255 with mean ~0.5 but no `/255` → huge values. Add a `/255` (or
   correct mean/std) path for uint8 input.

## 2. P1 — research validity (the paper's honest-numbers issues)

1. [x] RESOLVED BY DISCLOSURE 2026-09-11: Baselines 2–5 share one
   federated model by design (RAG/MA layers never touch weights, so
   retraining per baseline would burn 16 extra FL runs for provably
   identical numbers). `BASELINE_STUDY_RESULTS.md` Method states this
   openly; RQ2/RQ3 are answered in retrieval/agent/ops terms.
2. [x] FIXED 2026-09-11: `run_m3_evaluation.agent_metrics_block` now
   samples a stratified 6-patient slice of the eval batch and reports
   separate without/with-RAG blocks (same 5-section shape as
   `baseline_study.evaluate_agents`); B4 takes without-RAG, B5 with-RAG.
   Stub-verified completion 0.8→1.0, collaboration 0.6→0.8 — the same
   signature as the baseline-study RQ2 numbers. 341 tests pass.
3. [x] FIXED 2026-09-11: `--seeds` (default 42–46); every cell is mean
   ± sample SD across repeated splits, Findings rewritten to the new
   numbers (diabetes Δ shrank +0.027 → +0.006 — the single-split luck
   this guards against). 342 tests pass. Deliberately no k-fold
   (folds × FL partitioning explodes the matrix for no extra honesty).

## 3. P2 — proposal gaps / improvements

1. [x] SPLIT 2026-09-11 (ADR-018): **readmission head DONE** —
   `train_outcome(preset="sepsis", outcome="readmission_30day")` + route
   branch + `report.readmission` attach + leakage exclusion (sepsis
   75→74 features, study regenerated); honest head numbers acc 0.946 /
   F1 0.486 / AUC 0.500 (5.4% positive — no ranking signal yet).
   **Mortality DATA-BLOCKED**: no shipped dataset has a mortality column
   (header-verified); needs credentialed MIMIC-IV extract (P2.6 track).
   Dashboard states the blocked reason instead of "Not estimated".
2. [x] DONE 2026-09-11 (ADR-019): **model-derived explanations** —
   tabular SHAP (`LinearExplainer` logistic / bounded `KernelExplainer`
   otherwise) via `CrewAI/orchestrator/explain.py`, Grad-CAM for the CNN,
   stratified ≤25-row background persisted per preset and preset-matched
   at serve time; Agent 5 tags `shap_linear|shap_kernel|grad_cam` vs
   labeled `magnitude_heuristic` fallback; route returns `shap_driven`.
   19 P2.2 tests; 371 backend tests pass, ruff clean.
3. **Treatment recommendation is a static playbook lookup.**
   `build_treatment_recommendations` returns hardcoded disease strings.
   Not evidence-grounded or model-graded.
4. **n8n RAG query builder is hardcoded to diabetes.**
   `n8n/clinical-full-v2.json` "Build RAG Query" node emits
   `'diabetes treatment'` / `'healthy lifestyle'` regardless of dataset.
5. **Multi-agent claim vs single-agent implementation.**
   README says "5 lean agents"; `crew.py` runs a single-agent LLM writer over
   a deterministic pipeline. If the paper's title claims "Multi-Agent,"
   either restore a genuine multi-agent LLM path or re-scope the proposal.
6. **MIMIC-IV never used.** Hospital D is a synthetic "MIMIC-IV-style" sepsis
   CSV; every model scores 1.000. The proposal lists MIMIC-IV as main dataset.
7. **Privacy budget is weak.** Cumulative ε ≈ 45 over 5 rounds
   (`docs/CHANGELOG.md`), far above target ε=4. Opacus `secure_mode=False`
   (P0 acknowledged). Retune, or report `ε=45` honestly as the headline.
8. **Dense embedder / bigger corpus for RAG.** TF-IDF context precision is
   0.30–0.35 because of the small per-dataset corpus; dense
   `SentenceTransformerEmbedder` already exists (`RAG_EMBEDDING_MODEL`).

## Docs to refresh when you make changes

- `docs/BASELINE_STUDY_RESULTS.md` — tables auto-regenerate, hand-written
  Findings section must be edited by hand to stay consistent.
- `docs/DECISIONS.md` — ADR-013/014/015 cover privacy/TLS/canonical; update if
  §1 changes.
- `docs/PRIVACY_NOTES.md` — keep the honest "what it does NOT provide" framing.
- `.ai/current_context.md` / this file — keep terse; append one line per
  completed fix so the next session doesn't re-derive.

## Golden rules

- Never commit `dataset/` (symlink) or `backend/.env`.
- Run `~/.local/bin/ruff check` on changed files; keep tests green
  (backend pytest from `backend/`, frontend pytest from `frontend/`).
- Backend + n8n run via `scripts/start_demo.py`; stop with `--stop`.
- The LLM/crewai path needs an API key — ask the user, never guess one.