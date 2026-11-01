"""Tests for feature encoding, learned Can scores, persistence and collection."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from affordance.data import AffordanceDatasetWriter, read_dataset
from affordance.evaluator import LearnedAffordanceEvaluator
from affordance.features import AffordanceFeatureEncoder
from affordance.model import LearnedAffordanceModel


class Belief:
    agent_pos = (0, 0)
    agent_dir = 0
    inventory_status = "empty"
    held_object = None
    completed_subgoals = []
    known_map = {
        (0, 0): {"object": "empty", "color": "unknown", "state": "unknown"},
        (1, 0): {"object": "key", "color": "yellow", "state": "unknown"},
        (0, 1): {"object": "wall", "color": "grey", "state": "unknown"},
    }

    def cell_at(self, pos):
        return self.known_map.get(tuple(pos))


class Runtime:
    belief = Belief()
    terminated = False
    truncated = False


CALL = {
    "id": "pickup:yellow:key", "name": "PickUp",
    "args": {"target": {"object": "key", "color": "yellow"}},
    "description": "Pick up yellow key",
}


class LearnedAffordanceTests(unittest.TestCase):
    def test_encoder_is_fixed_length_and_parameter_sensitive(self):
        encoder = AffordanceFeatureEncoder()
        first = encoder.encode(Belief(), CALL)
        other = encoder.encode(Belief(), {
            **CALL,
            "name": "OpenDoor",
            "args": {"target": {"object": "door", "color": "blue"}},
        })
        self.assertEqual(len(first), encoder.dimension)
        self.assertNotEqual(first, other)
        self.assertEqual(len(encoder.schema_hash), 16)

    def test_model_trains_calibrates_and_round_trips(self):
        encoder = AffordanceFeatureEncoder()
        base = np.asarray(encoder.encode(Belief(), CALL), dtype=float)
        x = np.stack([base.copy() for _ in range(40)])
        y = np.asarray([0] * 20 + [1] * 20)
        x[:20, 0] = 0.0
        x[20:, 0] = 1.0
        model = LearnedAffordanceModel(encoder, hidden_size=8, seed=3)
        model.fit(x[:32], y[:32], x[32:], y[32:], epochs=80, seed=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = LearnedAffordanceModel.load(path)
            np.testing.assert_allclose(
                model.predict_array(x), loaded.predict_array(x), atol=1e-12
            )

    def test_hard_mask_cannot_be_overridden_by_learned_model(self):
        class AlwaysHigh:
            model_type = "test_model"

            @staticmethod
            def predict(_belief, _call):
                return 0.99

        evaluator = LearnedAffordanceEvaluator(Runtime(), AlwaysHigh())
        impossible = {
            "name": "UnlockDoor",
            "args": {"target": {"object": "door", "color": "blue"}},
        }
        result = evaluator.evaluate(impossible)
        self.assertEqual(result["hard_feasibility"], 0.0)
        self.assertEqual(result["feasibility"], 0.0)
        self.assertEqual(result["affordance_source"], "hard_mask")

    def test_unseen_skill_falls_back_instead_of_extrapolating(self):
        class FindOnly:
            model_type = "test_model"

            @staticmethod
            def supports(call):
                return call.get("name") == "Find"

            @staticmethod
            def predict(_belief, _call):
                return 0.01

        evaluator = LearnedAffordanceEvaluator(Runtime(), FindOnly())
        explore = evaluator.evaluate({"name": "Explore", "args": {}})
        self.assertEqual(explore["feasibility"], 1.0)
        self.assertIsNone(explore["learned_affordance"])
        self.assertEqual(explore["affordance_source"], "unseen_skill_fallback")

    def test_dataset_writer_records_real_skill_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            writer = AffordanceDatasetWriter(
                path, context={"env_id": "test", "seed": 4}
            )
            sample = writer.start(
                "pick up key", Belief(), CALL,
                {"feasibility": 1.0, "reason": "ready"},
            )
            writer.finish(sample, {
                "success": True, "status": "skill_complete",
                "low_level_actions": 2, "controller_steps": 2,
            })
            rows = read_dataset(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label"], 1)
            self.assertEqual(rows[0]["context"]["seed"], 4)
            self.assertEqual(len(rows[0]["features"]), AffordanceFeatureEncoder().dimension)


if __name__ == "__main__":
    unittest.main()
