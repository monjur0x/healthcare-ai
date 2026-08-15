# n8n Workflows

Orchestration only. n8n triggers workflows and calls the FastAPI backend;
all AI reasoning stays in the CrewAI clinical crew (`backend/CrewAI/orchestrator`),
and all training / prediction / retrieval logic stays in the backend
modules (`models/`, `federated/`, `rag/`). n8n never implements or
duplicates ML logic.

## Architecture Fit

```
External caller (dashboard / automation / cron)
                    │
                    ▼
          n8n webhook trigger
                    │  (route, validate, orchestrate)
                    ▼
       FastAPI backend (backend/api)
                    │  /api/v1/train, /api/v1/analyze
                    ▼
       training → deterministic clinical crew pipeline
       (fit model → prediction → risk → RAG evidence → report)
                    │
                    ▼
        n8n stores the report + responds to the caller
```

## Workflows

| File                          | Trigger path                 | Purpose                                               |
| ----------------------------- | ---------------------------- | ----------------------------------------------------- |
| `healthcare-endtoend.json`    | `POST /webhook/healthcare-endtoend` | **Single end-to-end workflow**: optionally train a model, run a clinical analysis, write the report to disk, and return a structured result. |
| `clinical-analysis.json`      | `POST /webhook/healthcare-analyze`  | Minimal reference workflow: run one clinical analysis and return the full report + a compact summary. |

### `healthcare-endtoend.json` (recommended)

The single automation workflow. One webhook drives the full lifecycle so
nothing needs to be trained or configured manually first.

1. **Webhook** receives the request body.
2. **IF: Train First?** — when the caller sets `train: true` the workflow
   calls `POST /api/v1/train` first (see below), otherwise it skips
   straight to analysis (the API uses whatever model it already has).
3. **HTTP: Train Model** — trains on the requested `preset` (or
   `dataset` + `target`); central fit by default, or the federated FedAvg
   path with `federated: true` + `clients` / `rounds`. The backend saves
   the artifact and starts serving it immediately.
4. **HTTP: Analyze Patient** — posts the patient / features / markers to
   `POST /api/v1/analyze` (payload always read from the original webhook).
5. **Code: Analyze & Build Report** — validates the report, embeds the
   train metadata + full report in one JSON file, and attaches it as a
   binary payload.
6. **Write: Report to Disk** — writes the JSON to
   `/tmp/healthcare_reports/` (override with `output_dir` in the request).
7. **Respond to Webhook** — returns a structured `status: success`
   payload with the summary, the full clinical `report` (consumed
   directly by the Streamlit dashboard), and the `file_path`.
8. Errors from training, analysis, or validation are merged and returned
   as a single `status: error` payload with the failing `stage`.

Example payload:

```json
{
  "train": true,
  "preset": "diabetes",
  "federated": false,
  "clients": 3,
  "rounds": 3,
  "output_dir": "/tmp/healthcare_reports",
  "patient": { "id": "p-1001", "name": "Patient A", "age": 54 },
  "features": { "glucose": 148.0, "bmi": 27.3, "age": 54.0 },
  "markers": { "glucose": 148.0, "bmi": 27.3, "age": 54.0 }
}
```

Success response (summary + full report):

```json
{
  "status": "success",
  "request_id": "…",
  "dataset": "diabetes",
  "train_status": "trained",
  "model_accuracy": 0.75,
  "prediction": "1",
  "confidence": 0.72,
  "risk_level": "moderate",
  "evidence_count": 3,
  "file_path": "/tmp/healthcare_reports/report-<id>.json",
  "report": { "patient_summary": "…", "prediction": "…", "risk": "…", "evidence": [ "…" ], "recommendations": "…" }
}
```

The `report` field is the full `ClinicalReport`; the Streamlit dashboard
reads it from the response (no extra file round-trip).

### `clinical-analysis.json`

Minimal reference: webhook → `POST /api/v1/analyze` → validate →
respond. Kept as the simplest possible example; for a full lifecycle use
`healthcare-endtoend.json`.

## Configuration

1. **Import** — n8n → Workflows → ⋮ → Import from file, or drop the JSON
   into the n8n workflows directory. Use "Import from file".
2. **Base URL** — the HTTP nodes target `http://localhost:8000` by
   default. Change the `url` fields if the FastAPI backend is deployed
   elsewhere (edit each `HTTP: ...` node).
3. **Optional bearer token** — the backend accepts an optional static
   token (`API_TOKEN` env var, see `backend/api`). If it is set:
   - Create a credential of type **Header Auth** named `Healthcare API
     Token`, with header name `Authorization` and the token as value.
   - Associate it with the HTTP nodes (they already reference a
     placeholder credential of that name).
   - If `API_TOKEN` is empty the workflows run without the credential.
4. **Webhook URLs** — after activating a workflow, n8n prints the full
   webhook URL, e.g. `https://<n8n-host>/webhook/healthcare-endtoend` and
   `/webhook/healthcare-analyze`.

## Security Notes

- Never commit real bearer tokens or credentials; the committed JSON uses
  placeholder credential references only.
- The workflows pass patient data straight through to the backend; keep
  the n8n instance and the webhook endpoints behind your deployment's
  transport security (HTTPS / VPN).
- Report files are written to a local directory; the path is caller
  controlled (`output_dir`) — restrict the n8n instance to trusted
  callers and cap the output location if you expose it.

## Local Smoke Test

With the FastAPI backend running (`uvicorn api.main:app` from
`backend/`) and the workflow active:

```bash
curl -X POST http://localhost:5678/webhook/healthcare-endtoend \
  -H "Content-Type: application/json" \
  -d '{
    "train": true,
    "preset": "diabetes",
    "patient": {"id": "smoke-1"},
    "features": {"glucose": 148.0, "bmi": 27.3, "age": 54.0},
    "markers": {"glucose": 148.0, "bmi": 27.3, "age": 54.0}
  }'
```