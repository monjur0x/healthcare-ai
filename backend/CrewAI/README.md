# Healthcare AI Backend - CrewAI

## Overview

Production-ready CrewAI backend for the research project:
**"Federated Multi-Agent Healthcare Intelligence Framework: Privacy-Preserving Clinical Decision Support using Federated Learning, RAG and Multi-Agent Systems."**

This backend provides a multi-agent AI system for clinical decision support, processing patient data (CSV and medical images) through a pipeline of specialized agents.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Endpoint                         │
│                   POST /api/run-healthcare-crew                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      HealthcareCrew                             │
│                   (Sequential Process)                          │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  Agent 1      │       │  Agent 2      │       │  Agent 3      │
│  Patient      │──────▶│  Disease      │──────▶│  Medical RAG  │
│  Analysis     │       │  Prediction   │       │  Evidence     │
└───────────────┘       └───────────────┘       └───────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  Agent 4      │       │  Agent 5      │       │  Agent 6      │
│  Treatment    │──────▶│  Explain-     │──────▶│  Risk         │
│  Planning     │       │  ability      │       │  Monitoring   │
└───────────────┘       └───────────────┘       └───────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      Agent 7          │
                    │  Clinical Report      │
                    │  Generation           │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   JSON Response       │
                    └───────────────────────┘
```

## Project Structure

```
backend/CrewAI/
├── app/
│   ├── __init__.py
│   ├── config.py              # Environment configuration
│   ├── main.py                # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # API endpoints
│   │   └── schemas.py         # Pydantic models
│   ├── crew/
│   │   ├── __init__.py
│   │   ├── agents.py          # CrewAI agents
│   │   ├── tasks.py           # CrewAI tasks
│   │   ├── crew.py            # Crew orchestration
│   │   └── tools.py           # Custom tools
│   ├── models/
│   │   ├── __init__.py
│   │   ├── csv_model.py       # CSV prediction model
│   │   ├── image_model.py     # Image analysis model
│   │   └── fusion.py          # Prediction fusion
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vector_store.py    # Qdrant integration
│   │   ├── retriever.py       # Medical retriever
│   │   └── embedder.py        # Sentence transformers
│   └── utils/
│       ├── __init__.py
│       ├── preprocessing.py   # Data preprocessing
│       ├── report.py          # Report generation
│       └── risk.py            # Risk calculation
├── tests/
│   └── test_healthcare.py
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

## Installation

### Prerequisites

- Python 3.12+
- Qdrant (optional, for vector storage)
- OpenAI API key or compatible LLM endpoint

### Setup

1. Clone the repository:
```bash
cd backend/CrewAI
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

### Running with Docker

```bash
docker-compose up -d
```

### Running Locally

```bash
python -m app.main
# or
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

### Endpoints

#### POST `/api/run-healthcare-crew`

Run the complete healthcare analysis pipeline.

**Request:**
- Content-Type: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| csv_file | File | No* | CSV file with patient health data |
| medical_image | File | No* | Medical image file |
| patient_name | String | Yes | Patient's full name |
| patient_id | String | Yes | Unique patient identifier |
| patient_age | Integer | Yes | Patient's age |
| notes | String | No | Additional clinical notes |

*At least one file (csv_file or medical_image) is required

**Response:**
```json
{
  "patient": {
    "name": "John Doe",
    "id": "P12345",
    "age": 45
  },
  "input_type": "csv",
  "patient_summary": "...",
  "prediction": {
    "primary_diagnosis": "Type 2 Diabetes",
    "secondary_diagnosis": "Hypertension",
    "confidence": 0.87,
    "severity": "moderate",
    "risk_level": "medium"
  },
  "clinical_findings": [],
  "image_findings": [],
  "evidence": [],
  "recommendations": [],
  "follow_up": [],
  "monitoring_plan": [],
  "explanation": "...",
  "limitations": "...",
  "doctor_notice": "This report is AI-assisted..."
}
```

### Example Usage

```bash
curl -X POST "http://localhost:8000/api/run-healthcare-crew" \
  -F "csv_file=@patient_data.csv" \
  -F "patient_name=John Doe" \
  -F "patient_id=P12345" \
  -F "patient_age=45"
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| LLM_MODEL | gpt-4o-mini | OpenAI model to use |
| LLM_API_KEY | - | Your API key |
| QDRANT_HOST | localhost | Qdrant host |
| QDRANT_PORT | 6333 | Qdrant port |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | Sentence transformer model |

## CrewAI Agents

1. **Patient Analysis Agent** - Validates and processes patient data
2. **Disease Prediction Agent** - Predicts disease risk
3. **Medical RAG Agent** - Retrieves clinical evidence
4. **Treatment Recommendation Agent** - Generates treatment plans
5. **Explainability Agent** - Explains AI predictions
6. **Risk Monitoring Agent** - Creates monitoring schedules
7. **Clinical Report Agent** - Generates final JSON report

## Integration with n8n

This backend is designed to work with n8n workflows:

1. n8n receives patient data
2. n8n calls `POST /api/run-healthcare-crew`
3. Backend processes and returns JSON response
4. n8n forwards results to downstream systems

## License

For research and educational use.
