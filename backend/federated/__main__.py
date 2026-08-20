"""
Launcher for the distributed federated deployment.

Entry points to run hospitals as separate processes connected over gRPC:

* ``python -m federated run`` — build hospital sites, start the server
  and one client process per hospital in this orchestrator, wait, and
  report the registered global model.
* ``python -m federated server ...`` — start only the Flower gRPC server
  (for truly separate hosts).
* ``python -m federated client ...`` — connect one hospital to a running
  server.
* ``python -m federated sites ...`` — build the per-hospital local data
  slices only.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from pathlib import Path
from typing import Any

import numpy as np

from federated.config import settings
from federated.distributed import ModelSpec
from federated.hospitals import PRESETS, build_hospital_sites
from federated.registry import ModelRegistry
from preprocessing.logger import get_logger

logger = get_logger(__name__)


def _dataset_dir() -> Path:
    """Resolve the dataset directory from settings or environment."""
    base = settings.DATASET_DIR or os.environ.get("DATASET_DIR", "")
    return Path(base) if base else Path.cwd()


def _hospital_root() -> Path:
    """Resolve the hospitals data root directory."""
    return Path(settings.HOSPITALS_DIR)


def _spec_from_preset(preset: str, differential_privacy: bool, seed: int) -> ModelSpec:
    """Determine the shared model architecture from the full source CSV."""
    from preprocessing.loader import load_classification_frame

    features, labels, _ = load_classification_frame(
        _dataset_dir() / PRESETS[preset][0], PRESETS[preset][1]
    )
    return ModelSpec(
        n_features=int(features.shape[1]),
        n_classes=int(labels.nunique()),
        feature_names=tuple(features.columns),
        differential_privacy=differential_privacy,
        seed=seed,
    )


def _spec_from_args(
    n_features: int, n_classes: int, differential_privacy: bool, seed: int
) -> ModelSpec:
    """Construct a :class:`ModelSpec` from explicit CLI values."""
    return ModelSpec(
        n_features=n_features,
        n_classes=n_classes,
        differential_privacy=differential_privacy,
        seed=seed,
    )


def _with_feature_names(spec: ModelSpec, feature_names: str | None) -> ModelSpec:
    """Attach a comma-separated feature schema to a :class:`ModelSpec`."""
    if not feature_names:
        return spec
    names = tuple(name for name in feature_names.split(",") if name)
    return ModelSpec(
        n_features=spec.n_features,
        n_classes=spec.n_classes,
        feature_names=names,
        differential_privacy=spec.differential_privacy,
        seed=spec.seed,
    )


def _preset_target(preset: str) -> str:
    """Resolve the target column for a preset name."""
    return {
        "diabetes": "outcome",
        "heart": "num",
        "kidney": "classification",
        "sepsis": "sepsis_label",
    }[preset]


def cmd_sites(args: argparse.Namespace) -> int:
    """Build the per-hospital local data slices."""
    sites = build_hospital_sites(
        preset=args.preset,
        n_sites=args.hospitals,
        dataset_dir=_dataset_dir(),
        hospitals_dir=_hospital_root(),
        seed=args.seed,
    )
    for site in sites:
        print(f"{site.hospital_id}\t{site.name}\t{site.dataset_path}\t{site.target}")
    return 0


def cmd_server(args: argparse.Namespace) -> int:
    """Run only the Flower gRPC server process."""
    from federated.distributed import run_distributed_server
    from preprocessing.loader import load_classification_frame

    spec = _spec_from_args(
        args.n_features, args.n_classes, args.differential_privacy, args.seed
    )
    spec = _with_feature_names(spec, args.feature_names)
    registry = ModelRegistry(settings.REGISTRY_PATH)

    holdout: tuple[Any, np.ndarray] | None = None
    holdout_csv = _hospital_root() / "central_holdout.csv"
    if holdout_csv.is_file():
        features, labels, _ = load_classification_frame(
            holdout_csv, _preset_target(args.preset)
        )
        features = spec.align_features(features)
        holdout = (features, labels.to_numpy())

    run_id, model_path, version = run_distributed_server(
        address=args.address,
        num_rounds=args.rounds,
        num_clients=args.hospitals,
        model_spec=spec,
        registry=registry,
        preset=args.preset,
        secure_aggregation=args.secure_aggregation,
        holdout=holdout,
        artifacts_dir=settings.ARTIFACTS_DIR,
        min_available=args.hospitals,
    )
    registry.close()
    print(f"run_id={run_id} model_path={model_path} version={version}")
    return 0


def cmd_client(args: argparse.Namespace) -> int:
    """Connect one hospital client to a running server."""
    from federated.distributed import run_hospital_client
    from federated.privacy import PrivacyConfig

    sites = build_hospital_sites(
        preset=args.preset,
        n_sites=args.hospitals,
        dataset_dir=_dataset_dir(),
        hospitals_dir=_hospital_root(),
        seed=args.seed,
    )
    hospital = next((site for site in sites if site.hospital_id == args.hospital), None)
    if hospital is None:
        logger.error("Unknown hospital id %r", args.hospital)
        return 1

    spec = _spec_from_args(
        args.n_features, args.n_classes, args.differential_privacy, args.seed
    )
    spec = _with_feature_names(spec, args.feature_names)
    privacy = (
        PrivacyConfig(
            enabled=args.differential_privacy,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            delta=args.privacy_delta,
        )
        if args.differential_privacy
        else None
    )
    run_hospital_client(
        address=args.address,
        hospital=hospital,
        model_spec=spec,
        privacy=privacy,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Orchestrate a full distributed run on this machine."""
    registry_path = Path(settings.REGISTRY_PATH)
    hospitals_dir = _hospital_root()

    sites = build_hospital_sites(
        preset=args.preset,
        n_sites=args.hospitals,
        dataset_dir=_dataset_dir(),
        hospitals_dir=hospitals_dir,
        seed=args.seed,
    )

    spec = _spec_from_preset(args.preset, args.differential_privacy, args.seed)
    n_features = spec.n_features
    n_classes = spec.n_classes
    feature_names = ",".join(spec.feature_names)

    server_cmd = [
        sys.executable,
        "-m",
        "federated",
        "server",
        "--address",
        args.address,
        "--preset",
        args.preset,
        "--hospitals",
        str(args.hospitals),
        "--rounds",
        str(args.rounds),
        "--n-features",
        str(n_features),
        "--n-classes",
        str(n_classes),
        "--feature-names",
        feature_names,
        "--seed",
        str(args.seed),
    ]
    if args.secure_aggregation:
        server_cmd.append("--secure-aggregation")
    if args.differential_privacy:
        server_cmd.append("--differential-privacy")

    client_cmd = [
        sys.executable,
        "-m",
        "federated",
        "client",
        "--address",
        args.address,
        "--preset",
        args.preset,
        "--hospitals",
        str(args.hospitals),
        "--n-features",
        str(n_features),
        "--n-classes",
        str(n_classes),
        "--feature-names",
        feature_names,
        "--seed",
        str(args.seed),
        "--hospital",
        "{hospital}",
    ]
    if args.differential_privacy:
        client_cmd += [
            "--differential-privacy",
            "--noise-multiplier",
            str(args.noise_multiplier),
            "--max-grad-norm",
            str(args.max_grad_norm),
            "--privacy-delta",
            str(args.privacy_delta),
        ]

    logger.info("Starting distributed server process: %s", " ".join(server_cmd))
    server_proc = subprocess.Popen(server_cmd)
    time.sleep(2.0)

    client_procs = []
    for site in sites:
        hospital_cmd = [part.format(hospital=site.hospital_id) for part in client_cmd]
        logger.info("Starting hospital client: %s", site.hospital_id)
        client_procs.append(subprocess.Popen(hospital_cmd))

    server_exit = server_proc.wait()
    for proc in client_procs:
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()

    registry = ModelRegistry(registry_path)
    latest = registry.latest_model(args.preset)
    registry.close()

    if server_exit != 0:
        logger.error("Distributed server exited with code %d", server_exit)
        return server_exit
    if latest is None:
        logger.error("No model was registered for preset %r", args.preset)
        return 1

    print(f"run_id={latest['run_id']}")
    print(f"model_path={latest['model_path']}")
    print(f"version={latest['version']}")
    print(f"accuracy={latest['accuracy']}")
    print(f"roc_auc={latest['roc_auc']}")
    print(f"epsilon={latest['epsilon']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="federated",
        description="Distributed federated healthcare learning.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--preset",
        required=True,
        choices=["diabetes", "heart", "kidney", "sepsis"],
    )
    common.add_argument("--seed", type=int, default=42)

    sites = subparsers.add_parser(
        "sites", parents=[common], help="Build hospital data slices."
    )
    sites.add_argument("--hospitals", type=int, default=4)
    sites.set_defaults(func=cmd_sites)

    server = subparsers.add_parser(
        "server", parents=[common], help="Run the Flower server."
    )
    server.add_argument("--address", default=settings.SERVER_ADDRESS)
    server.add_argument("--hospitals", type=int, default=4)
    server.add_argument("--rounds", type=int, default=3)
    server.add_argument("--n-features", type=int, required=True)
    server.add_argument("--n-classes", type=int, required=True)
    server.add_argument("--feature-names", default="")
    server.add_argument("--secure-aggregation", action="store_true")
    server.add_argument("--differential-privacy", action="store_true")
    server.set_defaults(func=cmd_server)

    client = subparsers.add_parser(
        "client", parents=[common], help="Connect a hospital client."
    )
    client.add_argument("--address", default="127.0.0.1:8080")
    client.add_argument("--hospitals", type=int, default=4)
    client.add_argument("--hospital", required=True)
    client.add_argument("--n-features", type=int, required=True)
    client.add_argument("--n-classes", type=int, required=True)
    client.add_argument("--feature-names", default="")
    client.add_argument("--differential-privacy", action="store_true")
    client.add_argument("--noise-multiplier", type=float, default=1.1)
    client.add_argument("--max-grad-norm", type=float, default=1.0)
    client.add_argument("--privacy-delta", type=float, default=1e-5)
    client.set_defaults(func=cmd_client)

    run = subparsers.add_parser(
        "run", parents=[common], help="Run server + hospitals here."
    )
    run.add_argument("--address", default=settings.SERVER_ADDRESS)
    run.add_argument("--hospitals", type=int, default=4)
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--secure-aggregation", action="store_true")
    run.add_argument("--differential-privacy", action="store_true")
    run.add_argument("--noise-multiplier", type=float, default=1.1)
    run.add_argument("--max-grad-norm", type=float, default=1.0)
    run.add_argument("--privacy-delta", type=float, default=1e-5)
    run.set_defaults(func=cmd_run)

    return parser


def main() -> int:
    """Parse arguments and dispatch to the requested subcommand."""
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
