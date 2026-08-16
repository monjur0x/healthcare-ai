# Current Context

## Current Milestone

Milestone 12 — Baseline comparison study (paper §13) — complete, committed,
pushed (`feat(eval): add baseline comparison study (paper §13)`).

## Current Module

`backend/scripts/` (baseline_study.py · tests/) · `docs/BASELINE_STUDY_RESULTS.md`

## Current Task

None active — the study is done. The script runs all five proposal
configurations against the four shipped datasets, reusing the existing
metric modules (nothing reimplemented):

1. Centralized ML (`train(federated=False)`)
2. Federated-only (`train(federated=True, clients=3, rounds=5)`) +
   `FederatedMetrics`
3. Federated + RAG (`rag_quality_metrics` over 5 literal queries/dataset)
4. Federated + Multi-Agent (deterministic LLM-free crew + `compute_agent_metrics`)
5. Proposed full (union of classification + RAG + agent metrics; n8n noted
   as the qualitative orchestration layer, no fabricated metric)

Real results are in `docs/BASELINE_STUDY_RESULTS.md` (all four datasets) +
hand-written Findings for RQ1–RQ4 with pilot-scale caveats. The Findings
section survives script re-runs (preserved from the `## Findings` marker).

## Completed

- Milestones 1–11 — prior context (committed + pushed).
- Milestone 12 — baseline comparison study:
  - `backend/scripts/baseline_study.py` (+ `__init__.py`) — 5 baselines,
    shared split (test_size=0.25, seed=42), `n/a` for inapplicable metrics,
    stdout tables + repo-root output default
  - `backend/scripts/tests/test_baseline_study.py` — 4 passing, hermetic on
    a synthetic CSV (no `DATASET_DIR` in CI)
  - `docs/BASELINE_STUDY_RESULTS.md` — real numbers + Findings
  - `api/services.py::prepare_tabular_data` — strip string labels
    (`'ckd\t'` == `'ckd'`); unblocks the kidney preset's federated partition
  - Docs: BACKLOG item checked off, CHANGELOG `### Added` entry (headline
    numbers), DEVELOPMENT_STATUS Milestone 12 + Testing section, `.ai/*`
  - Headline: federated Δ accuracy +0.027 / +0.026 / +0.020 / 0.000 vs
    centralized; RAG recall 1.000 / precision 0.300–0.350; agent completion
    0.6→0.8 with RAG evidence
- Backend suite **335 passing** (+4), lint clean.

## Next Files (optional / backlog)

- Backlog candidates (pick one): patient persistence; mortality/readmission
  models; SHAP explainability; Qdrant store + dense-embedding RAG lever
  (the study showed RAG precision 0.3 at top-k=5); local/open-source LLM
  provider for the crew.

## Design Notes

- Study design decisions: classification block reused for baselines 2–5
  (RAG/agents do not retrain); RAG "answer" is a per-dataset reference
  answer grounded in a literal corpus (faithfulness is meaningful, not
  trivially 1.0); agent "task outputs" are the 5 report sections (summary /
  prediction / risk / evidence / recommendations), so completion is 0.6
  without RAG (evidence empty) and 0.8 with it (recommendations still empty
  in the deterministic path).
- `write_results` preserves everything from `## Findings` onward so a
  re-run does not clobber the hand-written narrative.
- Lint/tests: run from `backend/`; never ruff from the repo root.

## Status

Milestones 1–12 committed/pushed. Working tree clean (verify with
`git status`). Baseline study reproducible with
`DATASET_DIR=/path/to/datasets python scripts/baseline_study.py`.
