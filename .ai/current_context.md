# Current Context

## Project state (as of 2026-09-10)

Federated multi-agent healthcare CDS framework (proposal: `ai-automation-research.md`). Backend 309 tests + frontend 58 tests pass; `ruff` clean. Two recent commits on `main`/`origin`:

- `6b75585` — restored the deleted test suite + `scripts/baseline_study.py`; fixed `metrics._output_of` (empty-output over-count) and `crew._parse_report` (missing `@staticmethod`).
- `4cef96e` — fixed kidney label inversion in baseline study (`split_dataset` now passes `preset=preset`); added orientation regression test; refreshed `docs/BASELINE_STUDY_RESULTS.md`.

Working tree clean except untracked `dataset` symlink (keep out of commits).

## Entry points

- READ THIS FIRST for the fix list: `.ai/NEXT_TASK.md` (self-contained, prioritized).
- Detailed P0/P1/P2 audit (partly fixed, partly stale): `.ai/backlog.md`.
- Proposal: `ai-automation-research.md` · Architecture: `docs/SOFTWARE_ARCHITECTURE.md` · Decisions: `docs/DECISIONS.md` · Privacy honesty: `docs/PRIVACY_NOTES.md`.

## Environment

- Python venv: `backend/CrewAI/.venv-opencode/bin/python` (has torch/opacus/flwr/crewai/streamlit/chromadb/pytest).
- Ruff: `~/.local/bin/ruff`.
- Datasets live at `/home/monjur0x0/dataset` (symlinked as `dataset/`; real preset files: `diabetes.csv`, `heart_disease_uci.csv`, `kidney_disease.csv`, `sepsis_icu_synthetic.csv`). NOT committed.
- Backend start: `cd backend && DATASET_DIR=~/dataset ./CrewAI/.venv-opencode/bin/python -m uvicorn api.main:app --port 8000`. n8n via `scripts/start_demo.py` or Docker.
- Baseline study (run from `backend/`): `DATASET_DIR=~/dataset ./CrewAI/.venv-opencode/bin/python scripts/baseline_study.py` (regenerates `docs/BASELINE_STUDY_RESULTS.md` tables; preserves hand-written Findings section).
- Backend suite from `backend/`: `./CrewAI/.venv-opencode/bin/python -m pytest`. Frontend from `frontend/`: same python `-m pytest dashboard/tests/`.

## Verified live

- Kidney baseline bug fixed: study split now matches training orientation (kidney 0.980 acc / 1.000 ROC across all baselines).
- RQ1/RQ2/RQ3 answers in `docs/BASELINE_STUDY_RESULTS.md` Findings (regenerated tables match hand-written section).
- Frontend smoke tests updated for the 7-tab app (keyed widgets + session-state feature inputs).
- P2.1 readmission DONE 2026-09-11 (ADR-018): `train_outcome` + route `outcome` branch + `report.readmission` + leakage exclusion (sepsis 75→74 feats, study regen); mortality data-blocked. 352 tests pass.
- P2.2 explainability DONE 2026-09-11 (ADR-019): `explain.py` (SHAP tabular + Grad-CAM), per-preset background persist/match, Agent 5 method tags, `shap_driven` route flag, labeled heuristic fallback. 371 backend tests pass, ruff clean.
