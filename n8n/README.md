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
        n8n responds to the caller with the full report
```

## Workflows

| File                          | Trigger path                 | Purpose                                               |
| ----------------------------- | ---------------------------- | ----------------------------------------------------- |
| `healthcare-endtoend.json`    | `POST /webhook/healthcare-endtoend` | **Single end-to-end workflow**: optionally train a model, run a clinical analysis, and return the full clinical report in the webhook response. |
| `clinical-analysis.json`      | `POST /webhook/healthcare-analyze`  | Minimal reference workflow: run one clinical analysis and return the full report + a compact summary. |

### `healthcare-endtoend.json` (recommended)

The single automation workflow. One webhook drives the full lifecycle so
nothing needs to be trained or configured manually first.

1. **Webhook** receives the request body. The n8n webhook node nests the
   payload under `body` (`{headers, params, query, body}`), so every
   expression in the workflow reads fields via `$json.body.*`.
2. **IF: Train First?** — when the caller sets `train: true` the workflow
   calls `POST /api/v1/train` first (see below), otherwise it skips
   straight to analysis (the API uses whatever model it already has).
3. **HTTP: Train Model** — trains on the requested `preset` (or
   `dataset` + `target`); central fit by default, or the federated FedAvg
   path with `federated: true` + `clients` / `rounds`. The backend saves
   the artifact and starts serving it immediately.
4. **IF: CSV Input?** — when the caller supplies a base64-encoded CSV
   (`csv_b64`), the workflow posts it to `POST /api/v1/analyze/csv`;
   otherwise it continues to the structured-feature path.
5. **HTTP: Analyze CSV / HTTP: Analyze Patient** — posts the
   CSV / patient / features / markers to the matching endpoint (payload
   always read from the original webhook request body).
6. **Code: Analyze & Build Report** — validates the clinical report and
   emits a `status: success` summary alongside the full `report`.
7. **Respond to Webhook** — returns the summary + full clinical `report`
   as a single JSON object (consumed directly by the Streamlit
   dashboard).
8. Errors from training, analysis, or validation each respond as a
   single `status: error` payload with the failing `stage`.

### CSV uploads

The dashboard's CSV Upload mode works through this workflow too: the
caller base64-encodes the raw file bytes into `csv_b64` in the webhook
payload (alongside the usual `patient` / optional `markers` /
`recommendations`), and the **HTTP: Analyze CSV** node forwards them to
`POST /api/v1/analyze/csv`, where parsing and preprocessing happen in
`backend/preprocessing/csv`. `csv_b64` must be a non-empty string for
the CSV branch to be selected; structured-feature requests simply omit
it.

Example payload:

```json
{
  "train": true,
  "preset": "diabetes",
  "federated": false,
  "clients": 3,
  "rounds": 3,
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
   default. If n8n runs in a Docker container the backend must be reached
   through the Docker bridge gateway, e.g. `http://172.17.0.1:8000`
   (edit each `HTTP: ...` node) — `localhost` inside the container only
   reaches the container itself.
3. **Header Auth credential (required)** — the HTTP nodes use an
   `httpHeaderAuth` credential named **Healthcare API Token**. The
   credential object must exist on the instance and be associated with
   the nodes, otherwise every execution fails with *"Credential with ID
   … does not exist"* (the committed JSON carries a placeholder ID that
   must be replaced). When the backend `API_TOKEN` is unset the header
   value is ignored; when it is set, use header name `Authorization` and
   `Bearer <token>` as the value.
4. **Activate** — n8n 2.x uses draft/published workflow versions; a
   workflow only serves its webhook after it is **activated** (UI toggle
   or `POST /api/v1/workflows/{id}/activate` with an API key). Editing
   the workflow database alone is not enough.
5. **Webhook URLs** — after activating a workflow, n8n prints the full
   webhook URL, e.g. `https://<n8n-host>/webhook/healthcare-endtoend` and
   `/webhook/healthcare-analyze`.

## Security Notes

- Never commit real bearer tokens or credentials; the committed JSON uses
  placeholder credential references only.
- The workflows pass patient data straight through to the backend; keep
  the n8n instance and the webhook endpoints behind your deployment's
  transport security (HTTPS / VPN).
- n8n 2.34 restricts the Code node sandbox (`fs` is disallowed) and
  readWriteFile writes to `~/.n8n-files`; the end-to-end workflow
  therefore returns the report in the webhook response instead of writing
  it to disk. If you need disk archival, mount a volume under the n8n
  file sandbox base and use a `readWriteFile` node inside it.

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