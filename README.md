# Healthcare AI Framework

Federated multi-agent healthcare intelligence framework: preprocessing,
ML models, federated learning (Flower), CrewAI orchestration, RAG,
FastAPI, n8n automation, and a Streamlit dashboard.

**CPU-only friendly** — no GPU required. All models are small and run on
the CPU (Intel Iris Xe integrated graphics is fine).

## Components

| Component | Entry point | Purpose |
| --------- | ----------- | ------- |
| FastAPI backend | `backend/api/main.py` | Train / predict / retrieve / analyze |
| RAG (retrieval) | `backend/rag/` | TF-IDF (default) or dense embedding + in-memory / ChromaDB store, RAGAS-style quality metrics |
| Multi-agent crew | `backend/CrewAI/orchestrator/` | Deterministic tool pipeline + optional Gemini agents; agent-level metrics |
| Federated learning | `backend/federated/` | Flower FedAvg with opt-in DP (Opacus) + secure aggregation |
| Streamlit dashboard | `frontend/streamlit_app.py` | Clinical UI over the API |
| n8n automation | `n8n/healthcare-endtoend.json` | One workflow: train → analyze → store → respond |
| Datasets | `~/dataset/` | `diabetes.csv`, `heart_disease_uci.csv`, `kidney_disease.csv`, `sepsis_icu_synthetic.csv`, brain-tumor MRI |

## Quick Start

The one-command runner does everything (trains a default model, starts
the API, starts the dashboard, starts n8n in Docker):

```bash
cd /home/monjur0x0/Healthcare-AI
scripts/run_system.sh start        # N8N_ENABLED=0 to skip n8n
scripts/run_system.sh status       # what is running
scripts/run_system.sh stop         # stop everything
```

That gives you:

- Dashboard: http://localhost:8501
- API docs (Swagger): http://localhost:8000/docs
- n8n: http://localhost:5678

## Step-by-Step (manual)

Prerequisites (already present on the reference machine): a Python
3.12+ venv with the project dependencies and the datasets in
`~/dataset/`.

**1. Start the backend** (the model is loaded lazily, so the API starts
even before any model exists):

```bash
cd backend
DATASET_DIR=/home/monjur0x0/dataset \
  CrewAI/.venv-opencode/bin/python -m uvicorn api.main:app \
  --host 0.0.0.0 --port 8000
```

Check it: `curl localhost:8000/health`.

**2. Train a model** — now possible through the API itself (no manual
CLI step). Central fit:

```bash
curl -X POST localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{"preset": "diabetes", "model": "mlp"}'
# → {"model_path": ".../artifacts/diabetes/global_model.joblib", "accuracy": 0.81, ...}
```

Or federated (FedAvg over simulated hospital clients):

```bash
curl -X POST localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{"preset": "diabetes", "federated": true, "clients": 3, "rounds": 3}'
```

The backend starts serving the new model immediately — no restart.
Presets: `diabetes`, `heart`, `kidney`, `sepsis` (also usable via
`dataset` + `target` for arbitrary CSVs).

**3. Run a clinical analysis** (predict → risk → RAG evidence → report):

```bash
curl -X POST localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "patient": {"id": "p-1", "name": "Patient A", "age": 30},
    "features": {"pregnancies":5.0,"glucose":116.0,"bloodpressure":74.0,
                 "skinthickness":27.0,"insulin":102.5,"bmi":25.6,
                 "diabetespedigreefunction":0.201,"age":30.0},
    "markers": {"glucose":116.0,"bmi":25.6,"age":30.0}
  }'
```

Also available: `POST /api/v1/predict` (single row) and
`POST /api/v1/retrieve` (RAG evidence).

**3b. Image analysis (MRI upload)** — the backend ships with a trained
brain-tumor CNN (`glioma` / `meningioma` / `notumor` / `pituitary`)
loaded from `API_IMAGE_MODEL_PATH`. Send a base64-encoded image to
`POST /api/v1/analyze/image`, or use the dashboard's **Image (MRI
upload)** tab:

```bash
python - <<'EOF'
import base64, json, urllib.request
image = open("scan.png", "rb").read()
body = json.dumps({
    "patient": {"id": "p-img", "name": "Patient A", "age": 30},
    "image": base64.b64encode(image).decode(),
}).encode()
req = urllib.request.Request(
    "http://localhost:8000/api/v1/analyze/image", body,
    {"Content-Type": "application/json"})
print(urllib.request.urlopen(req).read().decode())
EOF
```

To retrain the CNN (the brain-tumor dataset must be extracted with
class folders like `glioma/`, `notumor/`):

```bash
cd /home/monjur0x0/Healthcare-AI
python scripts/train_image_model.py --dataset /path/to/dataset \
  --max-per-class 300 --epochs 6 --image-size 64
```

**4. Start the dashboard:**

```bash
cd frontend
../backend/CrewAI/.venv-opencode/bin/python -m streamlit run streamlit_app.py
```

**5. n8n (Docker)** — orchestrate the whole lifecycle in one workflow:

```bash
docker run -d --rm --name healthcare-n8n -p 5678:5678 \
  -v healthcare_n8n_data:/home/node/.n8n n8nio/n8n
```

Open http://localhost:5678, import `n8n/healthcare-endtoend.json`,
activate it, then drive everything with one request:

```bash
curl -X POST http://localhost:5678/webhook/healthcare-endtoend \
  -H "Content-Type: application/json" \
  -d '{
    "train": true,
    "preset": "diabetes",
    "patient": {"id": "smoke-1"},
    "features": {"pregnancies":5.0,"glucose":116.0,"bloodpressure":74.0,
                 "skinthickness":27.0,"insulin":102.5,"bmi":25.6,
                 "diabetespedigreefunction":0.201,"age":30.0},
    "markers": {"glucose":116.0,"bmi":25.6,"age":30.0}
  }'
```

The workflow trains the model (if requested), analyzes the patient,
writes the full report JSON to `/tmp/healthcare_reports/`, and returns a
structured success/error payload.

## API Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/health` | Liveness |
| GET | `/api/v1/model` | Model metadata (features / classes) |
| POST | `/api/v1/train` | Train a model (central or federated) and serve it |
| POST | `/api/v1/predict` | Classify one feature row |
| POST | `/api/v1/retrieve` | RAG evidence retrieval |
| POST | `/api/v1/analyze` | Full clinical report (prediction, risk, evidence, recommendations) |
| POST | `/api/v1/analyze/image` | Image-based clinical report (base64 image body) |

Optional bearer auth: set `API_TOKEN` and send
`Authorization: Bearer <token>` (all `/api/v1` routes).

## Configuration (environment variables)

- `API_MODEL_PATH` — path to a persisted tabular model (else empty; train via the API)
- `API_IMAGE_MODEL_PATH` — path to a persisted image CNN (else empty; train via `scripts/train_image_model.py`)
- `API_CORPUS_DIR` — RAG knowledge directory of `.txt`/`.md` (else built-in corpus)
- `API_DATASET_DIR` — base dir for preset datasets (else `DATASET_DIR`, else cwd)
- `API_ARTIFACTS_DIR` — where trained models are written (default `backend/artifacts`)
- `API_TOKEN` — optional bearer token
- `DATASET_DIR` — used by the demos and as the dataset-dir fallback

RAG settings (prefix `RAG_`, see `backend/rag/config.py`):

- `RAG_EMBEDDING_MODEL` — `tfidf` (default), `hashing`, or
  `sentence-transformer` (dense, opt-in, downloads a small model from
  Hugging Face on first use)
- `RAG_SENTENCE_TRANSFORMER_MODEL` — the sentence-transformer model name
  (default `BAAI/bge-small-en-v1.5`)
- `RAG_VECTOR_STORE` — `memory` (default, in-process NumPy) or `chroma`
  (persistent ChromaDB collection)
- `RAG_CHROMA_PERSIST_DIR` — directory for the ChromaDB collection when
  `RAG_VECTOR_STORE=chroma` (empty = ephemeral per process)
- `RAG_CHROMA_COLLECTION` — ChromaDB collection name (default
  `healthcare_rag`)

### Enabling the CrewAI LLM agents (Gemini)

By default the crew runs a fully offline, deterministic pipeline
(prediction → risk → evidence → report) with no model calls. To enable
the agentic path (`ClinicalCrew.run_llm`, CrewAI agents that enrich the
report with reasoning):

1. Get a Gemini API key from <https://aistudio.google.com/apikey>.
2. From `backend/`: `cp .env.example .env` and set `CREW_LLM_API_KEY=`.
   (Default provider/model: `google` / `gemini-3.7-flash`.)
3. Install the Google extra for CrewAI in the venv:
   `pip install crewai[google-genai]`.
4. Restart the backend. The crew now uses agents with Gemini; if the
   LLM call fails, it falls back to the deterministic report.

`backend/.env.example` documents every variable (API_ / CREW_ / MODEL_ /
RAG_). Never commit `.env` — it is gitignored.

## Privacy & Security

This framework is a research prototype; treat all outputs as
non-clinical. It applies defense-in-depth at three layers:

- **Data protection (in-process)** — PHI never leaves a client. When
  federated training runs with `differential_privacy=true`, local
  updates use Opacus DP-SGD; `secure_aggregation=true` adds a pairwise
  one-time-pad mask so per-client updates are hidden from the server.
  The training response reports epsilon, MIA-AUROC, and leakage rate
  (ADR-013). Raw records are never returned by the API.
- **Access control** — the API supports an optional static bearer token
  (`API_TOKEN`, off by default). Full OAuth is deferred to the backlog.
- **Transport security (encrypted communication)** — all inter-service
  traffic (dashboard ↔ API, n8n ↔ API) should be served over TLS at the
  deployment boundary, not encrypted inside the application (ADR-014).
  Run uvicorn behind a TLS-terminating reverse proxy such as nginx or
  Caddy; use mTLS where components must mutually authenticate. Localhost
  dev traffic can remain plain HTTP.

Secrets policy: never commit `.env`, API keys, tokens, passwords, private
datasets, or hospital records. `backend/.env.example` and `backend/.gitignore`
enforce this.

## Datasets

This machine ships the Pima Diabetes, UCI Heart Disease, UCI CKD, and a
synthetic sepsis ICU CSV (plus a brain-tumor MRI set for the image path).
**MIMIC-IV is not used here**: it requires PhysioNet credentialed access
and hundreds of GB of storage. The local CSVs are sufficient for the
full end-to-end system.

## Tests

```bash
# Backend (from backend/): 326 tests
CrewAI/.venv-opencode/bin/python -m pytest \
  preprocessing/tests models/tests evaluation/tests federated/tests \
  rag/tests examples/tests CrewAI/orchestrator/tests api/tests -q

# Frontend (from frontend/): 13 tests
../backend/CrewAI/.venv-opencode/bin/python -m pytest dashboard/tests -q
```