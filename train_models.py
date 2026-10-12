"""Train the model set for a selected local dataset and save a reproducible score report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from clinicflow.data import PROJECT_ROOT, load_dataset
from clinicflow.modeling import evaluate_all, simulate_capacity


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(key: str) -> None:
    data = load_dataset(key)
    print(f"Dataset: {data.label}; rows after validation: {len(data.y):,}")

    def progress(step: int, total: int, name: str, seconds: float) -> None:
        print(f"[{step}/{total}] {name}: fit {seconds:.1f}s", flush=True)

    result = evaluate_all(data.X, data.y, data.groups, progress=progress)
    best_model = str(result.metrics.iloc[0]["model"])
    best_probability = result.probabilities[best_model]
    holdout_size = len(result.y_test)
    capacity_scenarios = {}
    for fraction in (0.05, 0.10, 0.20):
        capacity = max(1, math.ceil(holdout_size * fraction))
        capacity_scenarios[f"top_{int(fraction * 100)}_percent"] = simulate_capacity(
            result.y_test,
            best_probability,
            mode="Top-K capacity",
            capacity=capacity,
        )
    threshold_scenarios = {
        f"{threshold:.2f}": simulate_capacity(
            result.y_test,
            best_probability,
            mode="Risk threshold",
            threshold=threshold,
        )
        for threshold in (0.25, 0.35, 0.50)
    }
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(results_dir / f"{key}_metrics.csv", index=False)
    raw_path = (
        PROJECT_ROOT / "data" / "simulated" / "appointments_demo.csv"
        if key == "simulated"
        else PROJECT_ROOT / "data" / "public" / "KaggleV2-May-2016.csv"
    )
    report = {
        "dataset_key": key,
        "dataset_label": data.label,
        "source": data.source,
        "dataset_sha256": file_sha256(raw_path),
        "audit": data.audit,
        "split": result.metadata,
        "model_count_trained": len(result.models),
        "model_failures": result.failures,
        "simulator_model": best_model,
        "simulator_top_k": capacity_scenarios,
        "simulator_thresholds": threshold_scenarios,
        "metrics": result.metrics.to_dict(orient="records"),
    }
    (results_dir / f"{key}_evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"\n{len(result.models)} models trained; report saved in results/.")
    print(result.metrics.to_string(index=False))
    if result.failures:
        print("Model failures:", result.failures, file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["simulated", "public_brazil_2016", "both"], default="both"
    )
    args = parser.parse_args()
    keys = ["simulated", "public_brazil_2016"] if args.dataset == "both" else [args.dataset]
    for key in keys:
        run(key)


if __name__ == "__main__":
    main()

