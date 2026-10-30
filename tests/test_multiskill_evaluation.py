"""Contract tests for the T1-T5 compositional evaluation protocol."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from minigrid.core.world_object import Ball


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "09_batch_evaluation.py"
SPEC = importlib.util.spec_from_file_location("multiskill_batch_eval", SCRIPT_PATH)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


class FakeGrid:
    def __init__(self):
        self.objects = {
            (1, 1): Ball("red"),
            (2, 2): SimpleNamespace(is_open=False),
        }

    def get(self, x, y):
        return self.objects.get((x, y))


class FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.target_pos = (1, 1)
        self.door_pos = (2, 2)
        self.ball_side = "right"
        self.carrying = None
        self.grid = FakeGrid()


class MultiSkillEvaluationTests(unittest.TestCase):
    def test_conditional_door_is_still_required_by_fixed_wall_layout(self):
        env = FakeEnv()
        task = (
            "Grab the red ball and put it above the green goal; "
            "open the yellow door if needed."
        )

        oracle = EVAL.ground_truth_success(env, "T5", task)

        self.assertTrue(oracle["door_required"])
        self.assertFalse(oracle["door_requirement_satisfied"])
        self.assertFalse(oracle["success"])

        env.grid.objects[env.door_pos].is_open = True
        oracle = EVAL.ground_truth_success(env, "T5", task)
        self.assertTrue(oracle["success"])

    def test_t2_and_t4_enforce_explicit_open_before_pickup_order(self):
        correct = ["Find", "NavigateTo", "OpenDoor", "PickUp", "PlaceNear"]
        incorrect = ["Find", "NavigateTo", "PickUp", "OpenDoor", "PlaceNear"]

        self.assertTrue(EVAL.order_constraints("T2", correct, True)["passed"])
        self.assertFalse(EVAL.order_constraints("T2", incorrect, True)["passed"])
        self.assertTrue(EVAL.order_constraints("T4", correct, True)["passed"])
        self.assertFalse(EVAL.order_constraints("T4", incorrect, True)["passed"])

    def test_t5_conditional_wording_still_requires_open_before_placement_in_this_layout(self):
        with_open = ["Find", "OpenDoor", "PickUp", "PlaceNear"]
        without_open = ["Find", "PickUp", "PlaceNear"]

        self.assertTrue(EVAL.order_constraints("T5", with_open, True)["passed"])
        self.assertFalse(EVAL.order_constraints("T5", without_open, True)["passed"])

    def test_t1_t3_require_pickup_and_open_before_placement_without_ordering_each_other(self):
        pickup_first = ["PickUp", "OpenDoor", "PlaceNear"]
        door_first = ["OpenDoor", "PickUp", "PlaceNear"]

        self.assertTrue(EVAL.order_constraints("T1", pickup_first, True)["passed"])
        self.assertTrue(EVAL.order_constraints("T1", door_first, True)["passed"])
        self.assertTrue(EVAL.order_constraints("T3", pickup_first, True)["passed"])
        self.assertTrue(EVAL.order_constraints("T3", door_first, True)["passed"])


if __name__ == "__main__":
    unittest.main()
