"""JSONL skill-outcome collection for learned affordance training."""
from __future__ import annotations

import json
from pathlib import Path

from affordance.features import AffordanceFeatureEncoder


SUCCESS_STATUSES = {
    "skill_complete", "target_found", "target_reached", "done_issued",
}


class AffordanceDatasetWriter:
    def __init__(self, path, context=None, encoder=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context = dict(context or {})
        self.encoder = encoder or AffordanceFeatureEncoder()
        self.rows_written = 0

    def start(self, task, belief, call, assessment):
        return {
            "dataset_version": "minigrid_skill_outcomes_v1",
            "feature_schema_version": self.encoder.schema_version,
            "feature_schema_hash": self.encoder.schema_hash,
            "features": self.encoder.encode(belief, call),
            "task": str(task),
            "skill": call.get("name"),
            "call": call,
            "hard_feasibility": float(
                assessment.get("hard_feasibility", assessment.get("feasibility", 0.0))
            ),
            "planner_affordance": float(assessment.get("feasibility", 0.0)),
            "context": dict(self.context),
        }

    def finish(self, sample, result):
        if sample is None:
            return
        status = str(result.get("status", "unknown"))
        success = bool(result.get("success") and status in SUCCESS_STATUSES)
        row = {
            **sample,
            "label": int(success),
            "success": success,
            "status": status,
            "low_level_actions": int(result.get("low_level_actions", 0) or 0),
            "controller_steps": int(result.get("controller_steps", 0) or 0),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.rows_written += 1


def read_dataset(path, encoder=None):
    encoder = encoder or AffordanceFeatureEncoder()
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("feature_schema_hash") != encoder.schema_hash:
                raise ValueError(f"Feature schema mismatch at line {line_number}.")
            if len(row.get("features", [])) != encoder.dimension:
                raise ValueError(f"Feature dimension mismatch at line {line_number}.")
            rows.append(row)
    return rows
