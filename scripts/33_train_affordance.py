#!/usr/bin/env python3
"""Train and calibrate the MiniGrid skill-success affordance model."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from affordance.data import read_dataset
from affordance.features import AffordanceFeatureEncoder
from affordance.model import LearnedAffordanceModel


def group_is_validation(row, folds=5):
    context = row.get("context", {})
    group = f"{context.get('env_id')}:{context.get('seed')}"
    digest = hashlib.sha256(group.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % folds == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="outputs/affordance/skill_outcomes.jsonl")
    parser.add_argument("--output", default="models/minigrid_affordance_v1.json")
    parser.add_argument("--report", default="outputs/affordance/training_report.json")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    encoder = AffordanceFeatureEncoder()
    rows = [row for row in read_dataset(args.dataset, encoder)
            if float(row.get("hard_feasibility", 0.0)) > 0]
    if not rows:
        parser.error("Dataset has no hard-feasible skill outcomes.")
    train_rows = [row for row in rows if not group_is_validation(row)]
    validation_rows = [row for row in rows if group_is_validation(row)]
    if not validation_rows or len({row["label"] for row in train_rows}) < 2:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(rows))
        cut = max(1, int(len(rows) * 0.8))
        train_rows = [rows[index] for index in order[:cut]]
        validation_rows = [rows[index] for index in order[cut:]]
    x_train = np.asarray([row["features"] for row in train_rows])
    y_train = np.asarray([row["label"] for row in train_rows])
    x_validation = np.asarray([row["features"] for row in validation_rows])
    y_validation = np.asarray([row["label"] for row in validation_rows])
    model = LearnedAffordanceModel(
        encoder=encoder, hidden_size=args.hidden_size, seed=args.seed
    )
    metrics = model.fit(
        x_train, y_train, x_validation, y_validation,
        epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed,
    )
    model.training_metadata.update({
        "dataset": str(args.dataset),
        "dataset_cases": len(rows),
        "train_skill_counts": dict(Counter(row["skill"] for row in train_rows)),
        "validation_skill_counts": dict(Counter(row["skill"] for row in validation_rows)),
    })
    model_path = model.save(args.output)
    report = {
        "model": str(model_path), "model_type": model.model_type,
        "feature_schema": encoder.metadata(), "temperature": model.temperature,
        "train_cases": len(train_rows), "validation_cases": len(validation_rows),
        "metrics": metrics, "training_metadata": model.training_metadata,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
