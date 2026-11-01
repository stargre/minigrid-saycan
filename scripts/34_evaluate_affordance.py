#!/usr/bin/env python3
"""Evaluate learned affordance calibration on a held-out JSONL dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from affordance.data import read_dataset
from affordance.model import LearnedAffordanceModel


def is_validation_group(row, folds=5):
    context = row.get("context", {})
    group = f"{context.get('env_id')}:{context.get('seed')}"
    digest = hashlib.sha256(group.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % folds == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/minigrid_affordance_v1.json")
    parser.add_argument("--dataset", default="outputs/affordance/skill_outcomes.jsonl")
    parser.add_argument("--output", default="outputs/affordance/evaluation_report.json")
    parser.add_argument(
        "--all-samples", action="store_true",
        help="Evaluate all rows instead of the held-out env/seed groups.",
    )
    args = parser.parse_args()
    model = LearnedAffordanceModel.load(args.model)
    rows = [row for row in read_dataset(args.dataset, model.encoder)
            if float(row.get("hard_feasibility", 0.0)) > 0]
    if not args.all_samples:
        rows = [row for row in rows if is_validation_group(row)]
    if not rows:
        parser.error("No matching evaluation rows were found.")
    x = np.asarray([row["features"] for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    probabilities = model.predict_array(x)
    by_skill = defaultdict(lambda: ([], []))
    for row, label, probability in zip(rows, labels, probabilities):
        by_skill[row["skill"]][0].append(label)
        by_skill[row["skill"]][1].append(probability)
    report = {
        "model": args.model, "dataset": args.dataset,
        "split": "all" if args.all_samples else "held_out_env_seed_groups",
        "overall": model.metrics(labels, probabilities),
        "by_skill": {
            skill: model.metrics(values[0], values[1])
            for skill, values in sorted(by_skill.items())
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
