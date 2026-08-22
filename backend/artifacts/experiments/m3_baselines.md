# M3 Evaluation — Proposal §12 Metrics + §13 Baselines

*Run:* `2026-08-22T19:41:30.668425+00:00`

## RAG metrics (average over 4 ground-truth queries)

| metric | value |
|---|---|
| precision_at_k | 0.5833 |
| recall_at_k | 0.8750 |
| mrr | 1.0000 |
| context_precision | 0.5833 |
| context_recall | 0.7917 |
| faithfulness | 0.0000 |
| answer_relevancy | 0.0085 |
| faithfulness_mean_cosine | 0.0415 |
| faithfulness_ceiling | 0.5791 |
| faithfulness_ceiling_mean_cosine | 0.3217 |

## Agent metrics

| metric | value |
|---|---|
| task_completion_rate | 1.000 |
| decision_consistency | 1.000 |
| agent_collaboration_score | 1.000 |

## §13 Baselines (same federated model)

| Baseline | Accuracy | Evidence | Completeness | Consistent | Latency (s) |
|---|---|---|---|---|---|
| B1 Centralized (M2) | 0.9119 | — | — | — | — |
| B2 FL only | 0.9250 | 0.0 | 0.857 | True | 0.0004 |
| B3 FL+RAG | 0.9250 | 3.0 | 1.0 | True | 0.001 |
| B4 FL+MA | 0.9250 | 0.0 | 0.857 | True | 0.0008 |
| B5 Proposed (+n8n) | 0.9250 | 3.0 | 1.0 | True | 0.0013 |