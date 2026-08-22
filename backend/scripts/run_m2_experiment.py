#!/usr/bin/env python3
"""
M2 experiment: Centralized vs Federated on four specialty hospitals.

Per the research proposal, each hospital owns a different disease dataset:

    Hospital A - Pima Diabetes        (data/hospitals/hospital_A/data.csv)
    Hospital B - UCI Heart Disease    (data/hospitals/hospital_B/data.csv)
    Hospital C - Chronic Kidney       (data/hospitals/hospital_C/data.csv)
    Hospital D - MIMIC-IV style sepsis(data/hospitals/hospital_D/data.csv)

Every hospital maps its local columns onto the shared canonical schema
(federated.canonical) and a binary ``has_disease`` target so FedAvg can
average weights. Raw rows never leave the hospital.

The experiment:
1. Builds a stratified train/test split over the union of mapped frames.
2. Trains the CENTRALIZED baseline on the pooled training partition.
3. Runs the FLOWER federation (4 clients, N rounds); each client trains
   only on its own local rows; the global model is evaluated on the same
   held-out test partition.
4. Records accuracy / precision / recall / F1 / ROC-AUC / PR-AUC / MCC,
   training time, rounds, and communication cost into
   artifacts/experiments/m2_results.json + m2_results.md.

Usage (from backend/):
    PYTHONPATH=. python scripts/run_m2_experiment.py --rounds 10
"""

from __future__ import annotations

import argparse
import json
import time

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from federated.canonical import HOSPITAL_PRESETS, load_canonical_frame
from federated.client import FederatedClient
from federated.server import FedAvgServer, make_global_evaluator
from models import TabularClassifier
from preprocessing.logger import get_logger

logger = get_logger(__name__)

REPO_BACKEND = Path(__file__).resolve().parent.parent
HOSPITALS_DIR = REPO_BACKEND / "data" / "hospitals"
ARTIFACTS_DIR = REPO_BACKEND / "artifacts" / "experiments"

SEED = 42


def load_all_hospitals() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Load every hospital's local CSV through the canonical adapters."""
    hospitals: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for hospital_id in sorted(HOSPITAL_PRESETS):
        path = HOSPITALS_DIR / hospital_id / "data.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Place each hospital's specialty CSV first."
            )
        features, labels = load_canonical_frame(
            str(path), HOSPITAL_PRESETS[hospital_id]
        )
        hospitals[hospital_id] = (features, labels)
    return hospitals


def classification_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    """Full prediction metric block required by the proposal."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def run_centralized(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> tuple[dict, TabularClassifier]:
    """Train the pooled centralized baseline and evaluate it."""
    model = TabularClassifier(model_name="mlp", random_state=SEED)
    start = time.perf_counter()
    model.fit(train_x.values.astype(np.float64), train_y.to_numpy())
    elapsed = time.perf_counter() - start

    pred = model.predict(test_x)
    prob = model.predict_proba(test_x)[:, 1]
    return (
        {
            "metrics": classification_metrics(test_y, pred, prob),
            "train_time_s": elapsed,
        },
        model,
    )


def run_federated(
    hospitals: dict[str, tuple[pd.DataFrame, pd.Series]],
    test_x: np.ndarray,
    test_y: np.ndarray,
    rounds: int,
) -> tuple[dict, FedAvgServer]:
    """
    Federate the four hospitals and evaluate every round's global model.

    Each client receives ONLY its own hospital frame — mirroring the
    proposal where raw data never leaves the site.
    """

    def make_client_model() -> TabularClassifier:
        return TabularClassifier(model_name="mlp", random_state=SEED)

    clients = [
        FederatedClient(
            make_client_model,
            features.to_numpy(dtype=np.float64),
            labels.to_numpy(),
            test_x,
            test_y,
        )
        for features, labels in hospitals.values()
    ]
    evaluator = make_global_evaluator(make_client_model, test_x, test_y)

    server = FedAvgServer(clients=clients, num_rounds=rounds, evaluate_fn=evaluator)
    start = time.perf_counter()
    server.run()
    elapsed = time.perf_counter() - start

    metrics = server.metrics.to_dict()
    history = [result.to_dict() for result in server.history]

    # Final-model prediction metrics on the shared test partition.
    global_model = make_client_model()
    global_model.set_parameters(server.global_parameters)
    pred = global_model.predict(test_x)
    prob = global_model.predict_proba(test_x)[:, 1]

    per_round_accuracy = [entry["accuracy"] for entry in history]
    result = {
        "final_metrics": classification_metrics(test_y, pred, prob),
        "rounds": rounds,
        "per_round": history,
        "accuracy_per_round": per_round_accuracy,
        "log_loss_first": history[0]["log_loss"] if history else None,
        "log_loss_last": history[-1]["log_loss"] if history else None,
        "total_train_time_s": elapsed,
        "bytes_exchanged_total": metrics.get("total_bytes_exchanged"),
        "bytes_per_round": metrics.get("bytes_exchanged_per_round"),
        "convergence_round": metrics.get("convergence_round"),
    }
    return result, server


def main() -> int:
    parser = argparse.ArgumentParser(description="M2 centralized vs federated")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading four specialty hospitals through canonical schema…")
    hospitals = load_all_hospitals()
    for hospital_id, (features, labels) in hospitals.items():
        print(
            f"  {hospital_id} ({HOSPITAL_PRESETS[hospital_id]:>8}): "
            f"{features.shape[0]:>5} rows, {int(labels.sum())} positives"
        )

    # Pooled split — identical test partition for both paradigms so the
    # comparison is apples-to-apples.
    all_x = pd.concat([f for f, _ in hospitals.values()], ignore_index=True)
    all_y = pd.concat([labels for _, labels in hospitals.values()], ignore_index=True)
    train_pool_x, test_x, train_pool_y, test_y = train_test_split(
        all_x, all_y, test_size=args.test_size, stratify=all_y, random_state=SEED
    )
    test_x_np = test_x.to_numpy(dtype=np.float64)
    test_y_np = test_y.to_numpy()
    print(
        f"\nPooled: {len(all_x)} rows | train={len(train_pool_x)} "
        f"test={len(test_x)} | positive rate={all_y.mean():.2%}\n"
    )

    print("=== CENTRALIZED BASELINE ===")
    central_result, central_model = run_centralized(
        train_pool_x, train_pool_y, test_x_np, test_y_np
    )
    for key, value in central_result["metrics"].items():
        print(f"  {key:>9}: {value:.4f}")
    print(f"  training: {central_result['train_time_s']:.2f}s")

    print(f"\n=== FEDERATED ({len(hospitals)} hospitals, {args.rounds} rounds) ===")
    fed_result, fed_server = run_federated(hospitals, test_x_np, test_y_np, args.rounds)
    for key, value in fed_result["final_metrics"].items():
        print(f"  {key:>9}: {value:.4f}")
    print(
        f"  training: {fed_result['total_train_time_s']:.2f}s "
        f"across {fed_result['rounds']} rounds"
    )
    print(f"  bytes exchanged: {fed_result['bytes_exchanged_total']:,}")
    accs = fed_result["accuracy_per_round"]
    if len(accs) >= 2:
        print(f"  convergence: acc {accs[0]:.4f} → {accs[-1]:.4f}")

    # Persist the federated global model next to the report.
    global_model = TabularClassifier(model_name="mlp", random_state=SEED)
    global_model.set_parameters(fed_server.global_parameters)
    model_path = ARTIFACTS_DIR / "global_model_m2.joblib"
    global_model.save(model_path)
    central_model_path = ARTIFACTS_DIR / "central_model_m2.joblib"
    central_model.save(central_model_path)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "hospitals": {
            hospital_id: {
                "preset": HOSPITAL_PRESETS[hospital_id],
                "n_rows": int(features.shape[0]),
                "n_positive": int(labels.sum()),
            }
            for hospital_id, (features, labels) in hospitals.items()
        },
        "canonical_features": list(all_x.columns),
        "dataset": {"rows": len(all_x), "test_rows": len(test_x)},
        "rounds": args.rounds,
        "centralized": central_result,
        "federated": fed_result,
        "delta_fed_minus_central": {
            key: fed_result["final_metrics"][key] - central_result["metrics"][key]
            for key in central_result["metrics"]
        },
        "artifacts": {
            "federated_model": str(model_path),
            "centralized_model": str(central_model_path),
        },
    }

    json_path = ARTIFACTS_DIR / "m2_results.json"
    json_path.write_text(json.dumps(report, indent=2))

    md = [
        "# M2 Experiment — Centralized vs Federated (4 specialty hospitals)",
        "",
        f"*Run:* `{report['timestamp']}` · seed {SEED} · {args.rounds} rounds",
        "",
        "| Metric | Centralized | Federated | Delta (Fed-Cent) |",
        "|--------|-------------|-----------|----------------|",
    ]
    for key in central_result["metrics"]:
        cent = central_result["metrics"][key]
        fed = fed_result["final_metrics"][key]
        md.append(f"| {key} | {cent:.4f} | {fed:.4f} | {fed - cent:+.4f} |")
    md += [
        f"| train time (s) | {central_result['train_time_s']:.2f} | "
        f"{fed_result['total_train_time_s']:.2f} | — |",
        f"| communication | — | {fed_result['bytes_exchanged_total']:,} bytes / "
        f"{fed_result['rounds']} rounds | — |",
        "",
        "## Accuracy per federated round",
        "",
        "| Round | Global accuracy | Log loss |",
        "|-------|-----------------|----------|",
    ]
    for entry in fed_result["per_round"]:
        row = (
            f"| {entry['round_index']} | {entry['accuracy']:.4f} "
            f"| {entry['log_loss']:.4f} |"
        )
        md.append(row)
    md.append("")
    (ARTIFACTS_DIR / "m2_results.md").write_text("\n".join(md))

    print("\n✅ Results stored:")
    print(f"   {json_path}")
    print(f"   {ARTIFACTS_DIR / 'm2_results.md'}")
    print(f"   {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
