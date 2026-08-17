# Current Context

## Current Milestone

Milestone 13 follow-up — CSV uploads through the n8n end-to-end workflow —
implementation complete and verified by tests; not yet committed (working
tree has the uncommitted changes).

## Current Module

`n8n/healthcare-endtoend.json` · `frontend/dashboard/client.py` ·
`frontend/streamlit_app.py` (CSV Upload mode)

## Current Task

Closed the backlog gap: CSV analysis previously went direct-to-FastAPI
only; the n8n end-to-end workflow carried structured features only. Now
the workflow orchestrates CSV analysis exactly like the feature path.

1. **Workflow** (`n8n/healthcare-endtoend.json`) — added **IF: CSV
   Input?** (true when the webhook `csv_b64` is a non-empty string) and
   **HTTP: Analyze CSV** (POST `csv`/`patient`/`markers`/
   `recommendations` → `/api/v1/analyze/csv`). Train-success and
   skip-train paths converge on the IF; CSV and structured branches merge
   back at the report-builder Code node; the CSV node's error output feeds
   the existing error formatter. JSON validated (targets exist) + embedded
   Code-node JS parses under node.
2. **Client** (`frontend/dashboard/client.py`) — new
   `analyze_csv_via_n8n()` (base64 `csv_b64` in the webhook payload,
   `preset`/`train` forwarding); n8n report extraction refactored into
   shared `_post_n8n_webhook()`.
3. **Dashboard** (`frontend/streamlit_app.py`) — CSV Upload submit now
   resolves the analysis route: n8n → `analyze_csv_via_n8n()`, direct →
   `analyze_csv()`; caption updated (n8n now forwards the file).
4. **Tests** — frontend suite **58 passing** (+4 client, +1 smoke).
   Lint clean (ruff/black).

## Completed

- Milestones 1–13 (incl. the Clinical Assessment rework) — committed +
  pushed (`feat(api)`, `feat(frontend)`, `docs(ai)`).
- CSV-through-n8n (this session, uncommitted):
  - `n8n/healthcare-endtoend.json` — CSV branch (IF + HTTP node) + README
    update (workflow steps, CSV uploads section, example payload note)
  - `frontend/dashboard/client.py` — `analyze_csv_via_n8n()`,
    `_post_n8n_webhook()` shared helper
  - `frontend/dashboard/tests/test_client.py` — 3 new tests (base64 +
    report parse, train/preset forwarding, workflow error)
  - `frontend/dashboard/tests/test_app_smoke.py` — 1 new smoke test
    (CSV upload via n8n route forwards bytes, no train when served
    preset matches)
  - `frontend/streamlit_app.py` — CSV submit routes via n8n when selected
  - Docs: n8n/README.md, docs/BACKLOG.md (item checked off),
    docs/CHANGELOG.md, docs/DEVELOPMENT_STATUS.md, `.ai/*`
- Frontend suite **58 passing**; lint clean.

## Next Files (optional / backlog)

- Commit the uncommitted changes (user's call).
- Backlog candidates (pick one): patient persistence + history;
  mortality/readmission models (Results page still "not estimated");
  SHAP explainability; multimodal fusion so the assessment summary no
  longer shows "image: Not provided"; scaler/encoder persistence for
  inference-time consistency; local/open-source LLM provider for the crew.

## Design Notes

- n8n stays orchestration-only: the CSV branch only forwards the base64
  bytes to the existing backend endpoint; parsing/preprocessing stay in
  `backend/preprocessing/csv` (AGENTS.md architecture rules intact).
- The CSV discriminator is `csv_b64` presence (string "is not empty"), not
  `input_type`, because the dashboard already sends `input_type: "csv"`
  for the structured-feature path too.
- The `Code: Analyze & Build Report` node reads train metadata inside a
  try/catch, so the CSV branch (which skips the train node) still builds
  the success summary — unchanged behaviour, verified by the code-node
  JS parse.
- Lint/tests: frontend from `frontend/`; never ruff from the repo root.

## Status

Milestones 1–13 committed/pushed. The CSV-through-n8n follow-up is
**uncommitted** (5 modified files). Frontend `pytest -q` from `frontend/`
= 58 passed.