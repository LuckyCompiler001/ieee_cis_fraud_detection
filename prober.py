"""Utility helpers for logging training metrics per round."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def log_round_metrics(prober_root: Path, round_index: str, metrics: Mapping[str, object]) -> Path:
    """Persist run metrics under `prober_result/round_[index]`."""

    round_dir = prober_root / f'round_{round_index}'
    round_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = round_dir / 'metrics.json'
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    summary_path = round_dir / 'summary.txt'
    auroc = metrics.get('value')
    loss = metrics.get('loss')
    with summary_path.open('w', encoding='utf-8') as handle:
        handle.write(f"round_index: {round_index}\n")
        handle.write(f"roc_auc: {auroc}\n")
        if loss is not None:
            handle.write(f"loss: {loss}\n")

    return metrics_path

