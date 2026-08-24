#!/usr/bin/env python3
"""
Privacy Experiment Runner (reproducible)

Runs a federated learning experiment with DP + Secure Aggregation and
persists all privacy metrics as reproducible JSON.

Every run records:
- random seed, dataset/preset, federation config
- DP config (epsilon target, delta, noise multiplier)
- secure aggregation configuration
- model type
- all measured privacy metrics (epsilon per-round + cumulative,
  MIA AUROC, leakage rate, attack resistance score)
- timestamp / run ID

Usage:
    PYTHONPATH=. python scripts/run_privacy_experiment.py \
        --preset diabetes --clients 4 --rounds 5 \
        --dp --noise-multiplier 1.1 --secure-aggregation
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split

from federated.client import FederatedClient
from federated.privacy import PrivacyConfig, inspect_federation_payloads
from federated.server import FedAvgServer, make_global_evaluator
from models import TabularClassifier, TorchMLPClassifier
from preprocessing.logger import get_logger

logger = get_logger(__name__)

BACKEND = Path(__file__).resolve().parent.parent
SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproducible privacy experiment")
    parser.add_argument(
        "--preset",
        default="diabetes",
        choices=["diabetes", "heart", "kidney", "sepsis"],
    )
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dp", action="store_true")
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--privacy-delta", type=float, default=1e-5)
    parser.add_argument("--epsilon-target", type=float, default=4.0)
    parser.add_argument("--secure-aggregation", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/experiments")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / f"privacy_{args.preset}_{args.rounds}r"
    out_dir.mkdir(parents=True, exist_ok=True)

    from federated.canonical import HOSPITAL_PRESETS, load_canonical_frame

    # ── Load hospital data through canonical schema ────────────────────
    logger.info("Loading hospitals via canonical schema…")
    hospitals: dict[str, tuple] = {}
    for hid in sorted(HOSPITAL_PRESETS):
        path = BACKEND / "data" / "hospitals" / hid / "data.csv"
        features, labels = load_canonical_frame(str(path), HOSPITAL_PRESETS[hid])
        hospitals[hid] = (features, labels)

    all_x = pd.concat([f for f, _ in hospitals.values()], ignore_index=True)
    all_y = pd.concat([labels for _, labels in hospitals.values()], ignore_index=True)

    _train_pool_x, test_x, _train_pool_y, test_y = train_test_split(
        all_x, all_y, test_size=0.25, stratify=all_y, random_state=args.seed
    )
    test_n = test_x.to_numpy(dtype="float64")
    test_labels_n = test_y.to_numpy()

    feature_names = list(all_x.columns)
    n_features = len(feature_names)
    n_classes = int(all_y.nunique())

    # ── Build clients with DP ────────────────────────────────────────
    privacy_config = (
        PrivacyConfig(
            enabled=args.dp,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            delta=args.privacy_delta,
            epsilon_target=args.epsilon_target,
        )
        if args.dp
        else None
    )

    def make_client_model():
        if args.dp:
            return TorchMLPClassifier(
                n_features=n_features, n_classes=n_classes, seed=args.seed
            )
        return TabularClassifier(model_name="mlp", random_state=args.seed)

    shards = []
    for features, labels in hospitals.values():
        xn = features.to_numpy(dtype="float64")
        yn = labels.to_numpy()
        mid = int(len(yn) * 0.8)
        shards.append((xn[:mid], yn[:mid]))

    member_x = __import__("numpy").concatenate([sx for sx, _ in shards], axis=0)

    clients = [
        FederatedClient(
            make_client_model, sx, sy, test_n, test_labels_n, privacy=privacy_config
        )
        for sx, sy in shards
    ]
    evaluator = make_global_evaluator(make_client_model, test_n, test_labels_n)

    # ── Run federation ────────────────────────────────────────────────
    server = FedAvgServer(
        clients=clients,
        num_rounds=args.rounds,
        evaluate_fn=evaluator,
        secure_aggregation=args.secure_aggregation,
    )
    start = time.perf_counter()
    server.run()
    elapsed = time.perf_counter() - start

    metrics = server.metrics.to_dict()

    # ── Measure MIA ───────────────────────────────────────────────────
    global_model = make_client_model()
    global_model.set_parameters(server.global_parameters)

    proba_member = global_model.predict_proba(member_x)[:, 1]
    proba_holdout = global_model.predict_proba(test_n)[:, 1]

    from federated.privacy import membership_inference_auroc

    mia_auroc = membership_inference_auroc(proba_member, proba_holdout)
    mia_counts = {"train_members": len(member_x), "holdout_nonmembers": len(test_n)}

    epsilon = server.max_epsilon or 0.0
    cumulative_epsilon = server.cumulative_epsilon_upper_bound or epsilon
    per_round_eps = server.per_round_epsilons

    # ── Payload inspection ────────────────────────────────────────────
    payload_inspection = server.payload_inspection
    if not payload_inspection:
        raw_updates = [c.get_parameters({}) for c in clients]
        payload_inspection = inspect_federation_payloads(raw_updates, feature_names)

    leakage_rate = payload_inspection["leakage_rate"]

    from federated.privacy import privacy_metrics_summary

    privacy_block = privacy_metrics_summary(
        epsilon=cumulative_epsilon,
        delta=args.privacy_delta if args.dp else 0.0,
        mia_auroc=mia_auroc,
        leakage_rate=leakage_rate,
        num_samples=len(member_x),
        epsilon_target=args.epsilon_target,
        secure_aggregation=args.secure_aggregation,
        per_round_epsilons=per_round_eps or None,
        epsilon_composition_method=(
            "naive_sum_upper_bound" if len(per_round_eps) > 1 else "single_round"
        ),
        mia_sample_counts=mia_counts,
        payload_inspection=payload_inspection,
    )

    # ── Persist reproducible JSON ─────────────────────────────────────
    run_id = f"privacy_{args.preset}_{args.rounds}r_{int(time.time())}"
    report = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "dataset_preset": args.preset,
        "federation": {
            "n_clients": args.clients,
            "n_rounds": args.rounds,
            "distributed": False,
            "model_type": "TorchMLPClassifier" if args.dp else "TabularClassifier(mlp)",
        },
        "dp_config": {
            "enabled": args.dp,
            "noise_multiplier": args.noise_multiplier,
            "max_grad_norm": args.max_grad_norm,
            "delta": args.privacy_delta,
            "epsilon_target": args.epsilon_target,
        }
        if args.dp
        else {"enabled": False},
        "secure_aggregation": args.secure_aggregation,
        "canonical_features": feature_names,
        "dataset": {
            "total_rows": len(all_x),
            "train_rows": len(member_x),
            "test_rows": len(test_n),
        },
        "measured_privacy_metrics": privacy_block,
        "federated_metrics": metrics,
        "total_wall_time_s": round(elapsed, 3),
    }

    json_path = out_dir / "privacy_results.json"
    json_path.write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 60}")
    print("PRIVACY EXPERIMENT RESULTS")
    print(f"{'=' * 60}")
    print(f"  DP enabled         : {args.dp}")
    print(f"  Secure Aggregation : {args.secure_aggregation}")
    print(f"  Rounds             : {args.rounds} x {args.clients} clients")
    if args.dp:
        print(f"  Per-round ε        : {[round(e, 4) for e in per_round_eps]}")
        print(
            f"  Cumulative ε       : {cumulative_epsilon:.4f} (naive_sum_upper_bound)"
        )
        print(f"  δ                  : {args.privacy_delta}")
        print(f"  Budget used        : {privacy_block['privacy_budget_used_pct']}%")
    print(
        f"  MIA AUROC          : {mia_auroc:.4f} (simplified confidence-based baseline)"
    )
    print(f"  Attack Resistance  : {privacy_block['attack_resistance_score']}")
    print(f"  Data Leakage Rate  : {leakage_rate:.4f} (measured from payloads)")
    acc_last = (
        metrics.get('accuracy_deltas', [None])[-1]
        if metrics.get('accuracy_deltas')
        else 'N/A'
    )
    print(f"  Accuracy delta     : {acc_last}")

    print(f"\n✅ Saved to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
