# n8n Workflows

Orchestration only. n8n triggers workflows and calls the FastAPI backend;
all AI reasoning stays in the CrewAI clinical crew (`backend/CrewAI/orchestrator`),
and all prediction / retrieval logic stays in the backend modules
(`models/`, `rag/`). n8n never implements or duplicates ML logic.

## Architecture Fit

```
External caller (dashboard / automation / cron)
                    │
                    ▼
          n8n webhook trigger
                    │  (route, validate, orchestrate)
                    ▼
       FastAPI backend (backend/api)
                    │  /api/v1/analyze
                    ▼
    deterministic clinical crew pipeline
    (prediction → risk → RAG evidence → clinical report)
                    │
                    ▼
        n8n formats + responds to the caller
```

## Workflows

| File                                | Trigger path             | Purpose                                              |
| ----------------------------------- | ------------------------ | ---------------------------------------------------- |
| `clinical-analysis.json`            | `POST /webhook/healthcare-analyze` | Run one clinical analysis and return the full report + a compact summary. |
| `clinical-pipeline-modality.json`   | `POST /webhook/healthcare-pipeline` | Route an incoming request by `modality` (`image` vs CSV default) and run `/api/v1/analyze` with the matching `input_type`. |

Both workflows are webhook-triggered, validate the backend response,
format a structured `status: success|error` payload, and respond back to
the caller with `Respond to Webhook`.

### `clinical-analysis.json`

1. **Webhook** receives the request body.
2. **HTTP** posts the entire body to `POST http://localhost:8000/api/v1/analyze`.
3. **Validate & Summarize** checks the response is a clinical report and
   builds a compact summary (patient id, prediction, confidence, risk,
   evidence / recommendation counts).
4. **IF** routes valid reports to **Format Success** → respond, and
   malformed responses (or HTTP errors from the API) to the error branch
   → merge → respond with `status: "error"`.

Example payload:

```json
{
  "patient": { "id": "p-1001", "name": "Patient A", "age": 54, "notes": "" },
  "features": { "glucose": 148.0, "bmi": 27.3, "age": 54.0 },
  "markers": { "glucose": 148.0, "bmi": 27.3 },
  "recommendations": ["Review with a licensed physician."],
  "input_type": "csv"
}
```

### `clinical-pipeline-modality.json`

1. **Webhook** receives the request body.
2. **Normalize Input** extracts `patient` / `features` / `markers` /
   `recommendations` and a lowercased `modality` (defaults to `csv`).
3. **Switch** routes `modality == "image"` to the image branch and
   everything else to the CSV branch.
4. Each branch calls `/api/v1/analyze` with `input_type` set to `image`
   or `csv`.
5. Success results are merged and summarized; errors are merged and
   reported — the caller receives a single structured response either way.

Example payload:

```json
{
  "patient": { "id": "p-1002" },
  "features": { "glucose": 92.0, "bmi": 24.1, "age": 45.0 },
  "markers": { "glucose": 92.0 },
  "modality": "csv"
}
```

## Configuration

1. **Import** — n8n → Workflows → ⋮ → Import from file, or drop the JSON
   into the n8n workflows directory (`~/.n8n/nodes` is not used; use
   "Import from file").
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
   webhook URL: `https://<n8n-host>/webhook/healthcare-analyze` and
   `/webhook/healthcare-pipeline`.

## Security Notes

- Never commit real bearer tokens or credentials; the committed JSON uses
  placeholder credential references only.
- The workflows pass patient data straight through to the backend; keep
  the n8n instance and the webhook endpoints behind your deployment's
  transport security (HTTPS / VPN).

## Local Smoke Test

With the FastAPI backend running (`uvicorn api.main:app` from
`backend/`) and the workflow active:

```bash
curl -X POST http://localhost:5678/webhook/healthcare-analyze \
  -H "Content-Type: application/json" \
  -d '{
    "patient": {"id": "smoke-1"},
    "features": {"glucose": 148.0, "bmi": 27.3, "age": 54.0}
  }'
```
