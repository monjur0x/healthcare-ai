# Progress

## 2026-08-26 — Disease-context pipeline fix (RAG integration)

**Status: COMPLETE & VERIFIED**

The clinical assessment pipeline now preserves disease context from the
model prediction through CrewAI to RAG retrieval, treatment
recommendations, and the final report.

### What was broken
- `ClinicalCrew` never received the dataset preset; `active_preset`
  stayed `None` for path-loaded models.
- The RAG query was built from raw class integers:
  `"clinical evidence and management for 0 at 99% confidence"`.
- RAG documents carried no topic metadata; retrieval returned any
  medically-adjacent chunk (coronary/heart-failure docs for a diabetes
  assessment).
- Agent 4 dumped raw evidence text as "treatment recommendations".
- Report summary and dashboard displayed `predicted_class` ("0") with
  "confidence" conflated with disease probability.
- Monitoring schedule was generic (annual physical / blood pressure)
  regardless of disease.

### What changed
| File | Change |
|---|---|
| `CrewAI/orchestrator/services.py` | `DISEASE_REGISTRY` (preset → disease, labels, positive class, rag topic); `resolve_disease`, `enrich_prediction`, `build_disease_query`, `build_rag_topic`, `build_treatment_recommendations`; `DISEASE_MONITORING` schedules; `assess_risk(disease_context=)`; `retrieve_evidence(topic=)` passes chunk topics into `EvidenceItem`; report summary uses human labels + P(condition) |
| `CrewAI/orchestrator/schemas.py` | `PredictionResult.disease/predicted_label/positive_probability/negative_probability`; `EvidenceItem.topics` |
| `CrewAI/orchestrator/crew.py` | Accepts `disease=` param; enriches prediction in Agent 2; Agent 3 uses disease query + topic boost; Agent 4 uses playbooks (+1 evidence source pointer); Agent 5 explanation distinguishes probability vs confidence; LLM kickoff receives `disease_context` |
| `rag/corpus.py` | `TOPIC_KEYWORDS` + `extract_topics()`; all corpus docs carry `metadata["topics"]` |
| `rag/documents.py` | `Chunk.metadata` inherited from Document |
| `rag/chunker.py` | Passes document metadata into chunks |
| `rag/retriever.py` | Topic-aware retrieval: 4× candidate pool, 1.5× score boost for matching topics, re-rank |
| `rag/pipeline.py` | `retrieve(..., topic=)` passthrough |
| `api/config.py` | New `API_ACTIVE_PRESET` — declares served model's condition |
| `api/services.py` | `from_config` reads `ACTIVE_PRESET`; `analyze()` passes `disease=self.active_preset` to crew |
| `frontend/streamlit_app.py` | "Predicted condition: No Diabetes", "Diabetes probability: X%", risk level; raw class/probabilities only in Technical details expander |

### Verification (3 diabetes patients, deterministic path)
Served model: `artifacts/diabetes/global_model.joblib` + `API_ACTIVE_PRESET=diabetes`.

| Input | p(condition) | Label | Risk | Evidence topics | Monitoring |
|---|---|---|---|---|---|
| glucose 85, BMI 19.5 | 0.000001 | No Diabetes | low (0.000001) | diabetes ×3 | HbA1c screening annual |
| glucose 130, BMI 28.5 | 0.031 | No Diabetes | low | diabetes ×3 | HbA1c screening annual |
| glucose 190, BMI 36.5 | 0.9995 | **Diabetes** | **high** | diabetes ×3 | HbA1c q3mo, home glucose daily, foot/eye/renal |

All 10 checks per patient passed:
disease preserved end-to-end · labels correct · risk == positive-class
probability · probabilities vary with input (1e-6 → 0.9995) · evidence
100% diabetes-topic · zero coronary/heart-failure contamination ·
recommendations disease-specific · monitoring disease-specific ·
summary/report use human labels.

Streamlit AppTest renders with 0 exceptions.

### Notes
- The multi-disease canonical model (`artifacts/multi_disease/…`) has
  weak per-disease discrimination (p_pos moved only 0.007→0.012 across
  extremes). For single-disease assessment serve the per-preset model;
  this is now the running configuration.
