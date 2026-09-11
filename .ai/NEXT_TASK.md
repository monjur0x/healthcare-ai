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

1. **Baselines 2–5 share one federated model** so their classification rows
   are identical by construction; only evidence/completeness/latency differ.
   RQ2/RQ3 can't be answered in prediction terms. To make them architecture
   baselines, each should retrain (central / FL / FL+RAG / FL+MA). See
   `scripts/baseline_study.py` `measure_baseline` / `run_study`.
2. **Agent metrics are structurally 1.0** (`scripts/run_m3_evaluation.py`
   `agent_metrics_block`: 3 identical calls, same patient id, deterministic
   pipeline). The restored `scripts/baseline_study.py` RQ2 at least shows a
   real RAG-vs-no-RAG discriminator (0.8→1.0); reconcile the two.
3. **No statistical rigor**: single seed 42, 5 RAG queries, 5 sample
   patients, no k-fold / CIs. The baseline doc already labels this pilot-scale.

## 3. P2 — proposal gaps / improvements

1. **Mortality & readmission prediction, binarized away.**
   Proposal expects mortality/readmission/CKD-stage outputs; everything was
   collapsed to binary `has_disease` (`services.py` `_preset_binary_labels`,
   `docs/DECISIONS.md` ADR-015). Frontend hardcodes
   `"Not estimated"` (`streamlit_app.py:490,496`; `clinical.py:958-959`).
   Add two small heads (mortality, readmission_30day) over the canonical
   tabular model.
2. **Explainability is magnitude-sort, not model-derived.**
   `crew.py:313-333` Explainability Expert sorts features by `abs(value)`.
   No SHAP/LIME/Grad-CAM (proposal §6/§7 promises them).
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