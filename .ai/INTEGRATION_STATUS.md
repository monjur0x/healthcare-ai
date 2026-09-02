# Integration Status

## Disease-context propagation (fixed 2026-08-26)

```
Patient Input
  → Model Prediction            TabularClassifier (diabetes artifact)
  → Disease/Target Resolver     resolve_disease(active_preset) → DISEASE_REGISTRY
  → CrewAI Patient Analysis     Agent 1 (unchanged)
  → Disease Prediction Agent    Agent 2: enrich_prediction() attaches
                                {disease, predicted_label, p_pos, p_neg}
  → Disease-specific RAG query  build_disease_query():
                                  negative → "diabetes prevention risk factors screening guidelines"
                                  positive → "diabetes clinical guidelines diagnosis management treatment"
  → Disease-filtered retrieval  retrieve_evidence(topic="diabetes") —
                                corpus topics + 1.5× score boost
  → Treatment Recommendation    build_treatment_recommendations() playbook,
                                keyed by (disease, predicted_positive, level)
  → Explainability              label + P(condition), confidence labeled as such
  → Risk Monitoring             assess_risk(disease_context) → DISEASE_MONITORING
  → Report Writer               summary: "Predicted condition: Diabetes;
                                diabetes probability 99.9%. Overall risk: HIGH."
```

### Wiring points
- `API_ACTIVE_PRESET` env var declares the served model's disease.
  `/api/v1/train` overrides it in-process after fitting a preset model.
- `analyze_csv` inherits context via delegation to `analyze`.
- Image analysis intentionally has no disease context.
- LLM path receives `disease_context` in `crew.kickoff(inputs=...)`;
  the base report it merges over already carries the enriched fields.

### Known limitations
- Only the four preset diseases resolve; custom CSV uploads get empty
  disease context and generic playbooks/schedules (by design).
- Topic tags derive from filenames; new corpus files need matching
  keywords in `rag/corpus.py::TOPIC_KEYWORDS` to be filterable.
- The multi-disease canonical global model discriminates weakly per
  disease; single-disease assessment should serve the per-preset model.
