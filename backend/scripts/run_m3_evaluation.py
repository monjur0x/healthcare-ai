#!/usr/bin/env python3
"""
M3 evaluation: proposal §12 metrics + §13 baselines 2-5.

Wires the previously-unwired research metrics into real runs and stores
everything under ``artifacts/experiments/``:

* **RAG metrics** (``rag.metrics``) — precision@k / recall@k / MRR plus
  RAGAS-style context precision/recall, faithfulness, answer relevancy,
  measured against a ground-truth labeled subset of the bundled corpus.
* **Agent metrics** (``CrewAI.orchestrator.metrics``) — task completion
  rate, decision consistency, collaboration score over the deterministic
  pipeline stages.
* **Baselines** (proposal §13, same federated global model everywhere):
  - B2 Federated only            (no evidence retrieval)
  - B3 FL + RAG                  (evidence retrieved per analysis)
  - B4 FL + Multi-Agent          (deterministic stage pipeline + agent
                                  metrics + decision consistency)
  - B5 Proposed FL+MA+RAG+n8n    (B4 + B3 plus a live n8n trigger when
                                  reachable at ``N8N_BASE_URL``)

Prediction quality is identical across B2-B5 by construction (one global
model), so each baseline additionally reports evidence volume, report
completeness, decision consistency, and latency.

Usage (from backend/):
    PYTHONPATH=. RAG_VECTOR_STORE=memory python scripts/run_m3_evaluation.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..api.schemas import PatientInfo
from ..api.services import AnalysisService, build_rag_pipeline, load_predictive_model
from CrewAI.orchestrator.metrics import compute_agent_metrics
from ..federated.canonical import HOSPITAL_PRESETS, TARGET_COLUMN, load_canonical_frame
from preprocessing.logger import get_logger
from rag import TextChunker, TfidfEmbedder, load_bundled_corpus
from rag.metrics import (
    _cosine_similarity,
    _split_sentences,
    rag_quality_metrics,
    retrieval_metrics,
)

logger = get_logger(__name__)

BACKEND = Path(__file__).resolve().parent.parent
HOSPITALS_DIR = BACKEND / "data" / "hospitals"
EXPERIMENTS = BACKEND / "artifacts" / "experiments"
GLOBAL_MODEL = EXPERIMENTS.parent / "multi_disease" / "global_model.joblib"
SEED = 42

#: Ground-truth relevant bundled documents per disease query (§9 sources).
RAG_GROUND_TRUTH: dict[str, tuple[str, list[str]]] = {
    "diabetes": (
        "evidence-based management of type 2 diabetes including glucose "
        "targets and lifestyle measures",
        ["diabetes-mellitus", "obesity-metabolic-health"],
    ),
    "heart": (
        "coronary heart disease risk factors cholesterol and blood pressure control",
        ["coronary-heart-disease", "hypertension"],
    ),
    "kidney": (
        "chronic kidney disease staging creatinine hemoglobin albumin interpretation",
        ["chronic-kidney-disease", "clinical-laboratory-values"],
    ),
    "sepsis": (
        "sepsis recognition organ dysfunction lactate antibiotics within one hour",
        ["sepsis", "clinical-laboratory-values"],
    ),
}

COMPLETENESS_FIELDS = (
    "patient_summary",
    "prediction",
    "risk",
    "monitoring_schedule",
    "evidence",
    "limitations",
    "doctor_notice",
)


def load_eval_batch(per_class: int = 20) -> pd.DataFrame:
    """Stratified labeled sample pooled from the four canonical frames."""
    frames = [
        load_canonical_frame(
            str(HOSPITALS_DIR / hospital / "data.csv"), HOSPITAL_PRESETS[hospital]
        )
        for hospital in sorted(HOSPITAL_PRESETS)
    ]
    pooled_x = pd.concat([x for x, _ in frames], ignore_index=True)
    pooled_y = pd.concat([y for _, y in frames], ignore_index=True)

    idx = (
        pooled_y.index.to_series()
        .groupby(pooled_y)
        .sample(n=per_class, random_state=SEED)
        .sort_index()
    )
    return pooled_x.loc[idx].assign(**{TARGET_COLUMN: pooled_y.loc[idx]})


def chunk_index(corpus_docs) -> dict[str, list[str]]:
    """Map document id -> its chunk ids using the production chunker."""
    chunker = TextChunker()
    mapping: dict[str, list[str]] = {}
    for doc in corpus_docs:
        mapping[doc.id] = [c.id for c in chunker.chunk(doc)]
    return mapping


def prediction_block(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def completeness(report) -> float:
    """Fraction of key clinical-report fields present/non-empty."""
    checks = [
        bool(report.patient_summary),
        report.prediction is not None,
        report.risk is not None,
        bool(report.risk.monitoring_schedule),
        len(report.evidence) > 0,
        bool(report.limitations),
        bool(report.doctor_notice),
    ]
    return sum(checks) / len(checks)


def measure_baseline(
    svc: AnalysisService, batch: pd.DataFrame, use_rag: bool, max_analyze: int = 6
) -> dict:
    """Run one §13 baseline configuration over the labeled batch.

    ``use_rag`` controls whether the analyze path performs evidence
    retrieval (B2 federated-only runs with retrieval disabled).
    """

    model = svc.model
    original_pipeline = svc.rag_pipeline
    svc.rag_pipeline = original_pipeline if use_rag else None
    feature_names = list(model.feature_names or batch.columns[:-1])

    t0 = time.perf_counter()
    probs = model.predict_proba(batch[feature_names].to_numpy(dtype="float64"))
    preds = model.predict(batch[feature_names].to_numpy(dtype="float64"))
    predict_time = time.perf_counter() - t0

    block = prediction_block(
        batch[TARGET_COLUMN].to_numpy(),
        preds,
        probs[:, 1],
    )

    latencies, evidence_counts, completeness_scores, risk_levels = [], [], [], []
    last_report = None
    repeat_stable: bool | None = None
    for _, row in batch.head(max_analyze).iterrows():
        features = {name: float(row[name]) for name in feature_names}
        markers = {
            k: features[k]
            for k in ("glucose", "bmi", "blood_pressure", "creatinine")
            if k in features
        }
        start = time.perf_counter()
        report = svc.analyze(
            patient=PatientInfo(name="M3Eval", id=f"M3-{int(row.name)}"),
            features=features,
            markers=markers,
            input_type="csv",
        )
        latencies.append(time.perf_counter() - start)
        evidence_counts.append(len(report.evidence))
        completeness_scores.append(completeness(report))
        risk_levels.append(report.risk.risk_level)
        last_report = report
        if repeat_stable is None:
            # Re-run the SAME patient once; stable risk level => consistent.
            rerun = svc.analyze(
                patient=PatientInfo(name="M3Eval", id=f"M3R-{int(row.name)}"),
                features=features,
                markers=markers,
                input_type="csv",
            )
            repeat_stable = rerun.risk.risk_level == report.risk.risk_level
            latencies.append(time.perf_counter() - start)

    svc.rag_pipeline = original_pipeline

    return {
        "prediction": block,
        "predict_only_time_s": round(predict_time, 3),
        "analyze_latency_avg_s": round(sum(latencies) / len(latencies), 4),
        "avg_evidence_items": round(sum(evidence_counts) / len(evidence_counts), 2),
        "report_completeness": round(
            sum(completeness_scores) / len(completeness_scores), 3
        ),
        "decision_consistent": bool(repeat_stable),
        "use_rag": use_rag,
    }, last_report


def rag_metrics_block(
    svc: AnalysisService, embedder, doc_chunks: dict[str, list[str]]
) -> dict:
    """§12 RAG metrics against ground-truth bundled-document relevance."""
    out: dict[str, dict[str, float]] = {}
    for disease, (query, relevant) in RAG_GROUND_TRUTH.items():
        results = svc.rag_pipeline.retrieve(query, top_k=3)
        doc_ids = [r.chunk.document_id for r in results]
        chunks = [(r.chunk.id, r.chunk.text) for r in results]

        report = svc.analyze(
            patient=PatientInfo(name="RAGProbe", id=f"RAG-{disease}"),
            features={
                "age": 50,
                "gender": 1,
                "bmi": 30,
                "blood_pressure": 85,
                "heart_rate": 80,
                "spo2": 96,
                "glucose": 140,
                "creatinine": 1.2,
                "cholesterol": 220,
                "hemoglobin": 13,
                "albumin": 4,
            },
            markers={"glucose": 140},
            input_type="csv",
        )
        answer = report.patient_summary or ""

        # Expand document-level ground truth to the chunk level.
        relevant_chunk_ids = {
            cid for doc in relevant for cid in doc_chunks.get(doc, [])
        }

        ret = retrieval_metrics(relevant_ids=relevant, retrieved_ids=doc_ids, k=3)
        qual = rag_quality_metrics(
            query=query,
            answer=answer,
            retrieved_chunks=chunks,
            relevant_chunk_ids=relevant_chunk_ids,
            embedder=embedder,
        )
        # Sanity ceiling: a corpus-derived answer must score ~1.0,
        # proving the metric discriminates (template summaries score low).
        ceiling = rag_quality_metrics(
            query=query,
            answer=chunks[0][-1],
            retrieved_chunks=chunks,
            relevant_chunk_ids=relevant_chunk_ids,
            embedder=embedder,
        )

        def _mean_best_cosine(answer_text: str, ctx: list) -> float:
            sentences = _split_sentences(answer_text)
            texts = [c[-1] for c in ctx]
            if not sentences or not texts:
                return 0.0
            s_vec = embedder.embed(sentences)
            c_vec = embedder.embed(texts)
            return sum(
                max(_cosine_similarity(s, c) for c in c_vec) for s in s_vec
            ) / len(sentences)

        out[disease] = {
            **ret.to_dict(),
            **qual.to_dict(),
            # Thresholded metrics are calibrated for dense embedders; the
            # raw TF-IDF cosines below make sparse-embedder runs readable.
            "faithfulness_mean_cosine": round(_mean_best_cosine(answer, chunks), 4),
            "faithfulness_ceiling": ceiling.faithfulness,
            "faithfulness_ceiling_mean_cosine": round(
                _mean_best_cosine(chunks[0][-1], chunks), 4
            ),
        }
        m = out[disease]
        logger.info(
            "RAG %s: P@3=%.2f R@3=%.2f MRR=%.2f",
            disease,
            m["precision_at_k"],
            m["recall_at_k"],
            m["mrr"],
        )
    return out


def agent_metrics_block(svc: AnalysisService) -> dict:
    """§12 Agent metrics over the deterministic pipeline stages."""
    features = {
        "age": 55,
        "gender": 1,
        "bmi": 36,
        "blood_pressure": 92,
        "heart_rate": 88,
        "spo2": 95,
        "glucose": 185,
        "creatinine": 1.8,
        "cholesterol": 240,
        "hemoglobin": 12,
        "albumin": 3.5,
    }
    predictions, stages = [], []
    for _ in range(3):
        report = svc.analyze(
            patient=PatientInfo(name="AgentProbe", id="AG-1"),
            features=features,
            markers={"glucose": 185, "bmi": 36},
            input_type="csv",
        )
        import json

        predictions.append(str(report.prediction.predicted_class))
        stages = [
            json.dumps(report.prediction.model_dump()),
            json.dumps(report.risk.model_dump()),
            json.dumps([e.model_dump() for e in report.evidence]),
            report.patient_summary,
        ]
    return compute_agent_metrics(stages, predictions).to_dict()


def n8n_probe(base_url: str | None) -> dict:
    """Trigger the live end-to-end workflow when n8n is reachable."""
    if not base_url:
        return {"reachable": False, "triggered": False}
    try:
        urllib.request.urlopen(f"{base_url}/healthz", timeout=2)
    except (urllib.error.URLError, TimeoutError):
        return {"reachable": False, "triggered": False}

    payload = json.dumps(
        {
            "preset": "diabetes",
            "patient": {"name": "M3 Baseline", "id": "M3-N8N"},
            "features": {
                "pregnancies": 6,
                "glucose": 190,
                "bloodpressure": 92,
                "skinthickness": 35,
                "insulin": 180,
                "bmi": 42,
                "diabetespedigreefunction": 1.2,
                "age": 55,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/webhook/healthcare-endtoend",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.load(resp)
        return {
            "reachable": True,
            "triggered": True,
            "status": body.get("status"),
            "risk_level": body.get("risk_level"),
            "latency_s": round(time.perf_counter() - start, 2),
        }
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as error:
        logger.warning("n8n trigger failed: %s", error)
        return {"reachable": True, "triggered": False, "error": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 proposal-metrics evaluation")
    parser.add_argument("--model", default=str(GLOBAL_MODEL))
    parser.add_argument("--n8n-url", default=os.environ.get("N8N_BASE_URL", ""))
    args = parser.parse_args()

    EXPERIMENTS.mkdir(parents=True, exist_ok=True)

    svc = AnalysisService(
        model=load_predictive_model(args.model),
        artifacts_dir=EXPERIMENTS.parent,
        dataset_dir=Path(
            os.environ.get("DATASET_DIR", str(BACKEND / ".." / "dataset"))
        ),
        rag_pipeline=build_rag_pipeline(None),
    )
    corpus = load_bundled_corpus()
    embedder = TfidfEmbedder().fit([doc.text for doc in corpus])

    print("Loading stratified eval batch…")
    batch = load_eval_batch(per_class=20)
    print(f"  rows={len(batch)} positives={int(batch[TARGET_COLUMN].sum())}")

    # ---- §12 RAG metrics -------------------------------------------------
    print("\n=== RAG METRICS (ground-truth corpus relevance) ===")
    corpus_docs = load_bundled_corpus()
    doc_chunks = chunk_index(corpus_docs)
    rag_block = rag_metrics_block(svc, embedder, doc_chunks)
    avg = {
        key: round(sum(v[key] for v in rag_block.values()) / len(rag_block), 4)
        for key in next(iter(rag_block.values()))
    }
    print(f"  averages: {avg}")

    # ---- §12 Agent metrics ----------------------------------------------
    print("\n=== AGENT METRICS (deterministic pipeline stages) ===")
    agents_block = agent_metrics_block(svc)
    print(f"  {agents_block}")

    # ---- §13 Baselines ---------------------------------------------------
    print("\n=== BASELINES (same federated global model) ===")
    b2, _ = measure_baseline(svc, batch, use_rag=False)
    print(
        f"B2 federated-only : acc={b2['prediction']['accuracy']:.4f} "
        f"evidence={b2['avg_evidence_items']} complete={b2['report_completeness']}"
    )
    b3, _ = measure_baseline(svc, batch, use_rag=True)
    print(
        f"B3 FL+RAG         : acc={b3['prediction']['accuracy']:.4f} "
        f"evidence={b3['avg_evidence_items']} complete={b3['report_completeness']}"
    )
    b4, _ = measure_baseline(svc, batch, use_rag=False)
    b4["agent_metrics"] = agents_block
    print(
        f"B4 FL+Multi-Agent : consistent={b4['decision_consistent']} "
        f"complete={b4['report_completeness']}"
    )
    b5, _ = measure_baseline(svc, batch, use_rag=True)
    b5["agent_metrics"] = agents_block
    b5["n8n"] = n8n_probe(args.n8n_url or None)
    print(
        f"B5 proposed       : complete={b5['report_completeness']} "
        f"n8n={b5['n8n'].get('triggered', False)}"
    )

    # Reuse stored centralized/federated numbers for the full table.
    m2_path = EXPERIMENTS / "m2_results.json"
    m2 = json.loads(m2_path.read_text()) if m2_path.exists() else {}

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": str(Path(args.model).resolve()),
        "rag_metrics_per_query": rag_block,
        "rag_metrics_average": avg,
        "agent_metrics": agents_block,
        "baselines": {
            "B2_federated_only": b2,
            "B3_fl_rag": b3,
            "B4_fl_multi_agent": b4,
            "B5_proposed_full": b5,
        },
        "prior_results": {
            "B1_centralized": m2.get("centralized", {}).get("metrics"),
            "B2_federated_m2": m2.get("federated", {}).get("final_metrics"),
        },
    }
    (EXPERIMENTS / "m3_metrics.json").write_text(json.dumps(report, indent=2))

    md = [
        "# M3 Evaluation — Proposal §12 Metrics + §13 Baselines",
        "",
        f"*Run:* `{report['timestamp']}`",
        "",
        "## RAG metrics (average over 4 ground-truth queries)",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    md += [f"| {k} | {v:.4f} |" for k, v in avg.items()]
    md += ["", "## Agent metrics", "", "| metric | value |", "|---|---|"]
    md += [f"| {k} | {v:.3f} |" for k, v in agents_block.items()]

    def _row(label: str, b: dict) -> str:
        return (
            f"| {label} | {b['prediction']['accuracy']:.4f} "
            f"| {b['avg_evidence_items']} | {b['report_completeness']} "
            f"| {b['decision_consistent']} | {b['analyze_latency_avg_s']} |"
        )

    cent_acc = f"{m2['centralized']['metrics']['accuracy']:.4f}" if m2 else "n/a"
    md += [
        "",
        "## §13 Baselines (same federated model)",
        "",
        "| Baseline | Accuracy | Evidence | Completeness | Consistent | Latency (s) |",
        "|---|---|---|---|---|---|",
        f"| B1 Centralized (M2) | {cent_acc} | — | — | — | — |",
        _row("B2 FL only", b2),
        _row("B3 FL+RAG", b3),
        _row("B4 FL+MA", b4),
        _row("B5 Proposed (+n8n)", b5),
    ]

    (EXPERIMENTS / "m3_baselines.md").write_text("\n".join(md))

    print("\n✅ stored:")
    print(f"   {EXPERIMENTS / 'm3_metrics.json'}")
    print(f"   {EXPERIMENTS / 'm3_baselines.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
