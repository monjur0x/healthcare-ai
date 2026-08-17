# Next Session

## Objective

The CSV-through-n8n follow-up is implemented and verified (frontend 58,
lint clean) but **not yet committed** (5 modified files: n8n workflow,
client, streamlit_app, 2 test files). Commit it, then pick a backlog
direction.

## Done This Session (uncommitted)

- `n8n/healthcare-endtoend.json`: added **IF: CSV Input?** (non-empty
  `csv_b64`) + **HTTP: Analyze CSV** (→ `/api/v1/analyze/csv`); the
  train/skip-train paths converge on the IF and both branches merge at
  the report-builder Code node. JSON + embedded Code-node JS validated.
- `frontend/dashboard/client.py`: `analyze_csv_via_n8n()` (base64
  `csv_b64`, `preset`/`train` forwarding) + shared `_post_n8n_webhook()`.
- `frontend/streamlit_app.py`: CSV Upload mode resolves the route — n8n
  → `analyze_csv_via_n8n()`, direct → `analyze_csv()`.
- Tests: frontend 58 (+4 client, +1 smoke); n8n/README.md updated.

## Optional Next Steps

1. Commit the follow-up (await user instruction), e.g. `feat(n8n): route
   CSV uploads through the end-to-end workflow` (+ docs commit if split).
2. Optionally live-verify against a real n8n instance + backend (the
   committed workflow needs re-importing/activating; the dashboard's n8n
   route then forwards a real CSV upload).
3. Backlog candidates (pick a direction):
   - Patient persistence + history in the dashboard (each assessment is
     entered fresh)
   - Backend mortality/readmission risk models (Results page still shows
     "not estimated")
   - Model-derived / SHAP explainability for the decision report
   - Multimodal fusion so the assessment summary shows an image result
     instead of the static "Not provided"
   - Scaler/encoder persistence for inference-time consistency (uploaded
     CSV analysis re-fits the pipeline scaler on the uploaded rows)
   - Open-source / local LLM provider for the crew (`CrewSettings.
     LLM_PROVIDER` currently only has `google`)
4. Run tests + lint from `frontend/`; never ruff from the repo root.

## Open Questions

- Whether to commit now (user decides; nothing was committed this
  session).
- `gemini-3.7-flash` still returns transient 503 "high demand";
  `gemini-3.6-flash` verified stable — consider switching
  `CrewSettings.LLM_MODEL` default.