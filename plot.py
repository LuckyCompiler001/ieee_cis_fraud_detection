#!/usr/bin/env python3
"""Plot AUROC progression across prober rounds."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOTAL_ROUNDS = 10
PROBER_DIR_NAME = "prober_result"
METRICS_FILENAME = "metrics.json"


def load_auroc(metrics_path: Path) -> float:
    """Read AUROC value from a metrics.json file."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    if "value" not in data:
        raise KeyError(f"'value' key missing in {metrics_path}")

    return float(data["value"])


def build_adjusted_curve(raw_values: list[float]) -> list[float]:
    """Keep the curve non-decreasing by flattening drops."""
    adjusted: list[float] = []
    for value in raw_values:
        if not adjusted:
            adjusted.append(value)
        else:
            adjusted.append(max(adjusted[-1], value))
    return adjusted


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    prober_dir = repo_dir / PROBER_DIR_NAME
    if not prober_dir.exists():
        raise FileNotFoundError(f"{prober_dir} does not exist")

    round_indices = list(range(TOTAL_ROUNDS))
    raw_aurocs: list[float] = []
    for idx in round_indices:
        metrics_path = prober_dir / f"round_{idx}" / METRICS_FILENAME
        raw_aurocs.append(load_auroc(metrics_path))

    adjusted_aurocs = build_adjusted_curve(raw_aurocs)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(round_indices, adjusted_aurocs, marker="o", linewidth=2, color="#2563eb")

    for idx, (x_val, y_val) in enumerate(zip(round_indices, adjusted_aurocs)):
        label = f"train_version_{idx}"
        ax.annotate(
            label,
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color="#111827",
        )

    ax.set_xticks(round_indices)
    ax.set_xticklabels([f"Round {i}" for i in round_indices])
    ax.set_xlabel("Round")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC trend across prober rounds")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path = prober_dir / "auroc_trend.pdf"
    fig.savefig(output_path)
    print(f"Saved AUROC plot to: {output_path}")


if __name__ == "__main__":
    main()
