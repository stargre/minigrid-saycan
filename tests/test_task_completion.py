"""Regression tests for observation-derived task completion evidence."""
import unittest

from planners.unified_saycan_planner import UnifiedSayCanPlanner


class FakeBelief:
    # The agent has crossed the goal/door cells and picked up the key, so those
    # objects are no longer all represented as objects in the current map.
    known_map = {}
    inventory_status = "yellow_key"
    held_object = {"object": "key", "color": "yellow", "state": "held"}
    completed_subgoals = ["pick", "pickup", "unlock_door", "open_door", "reach"]


class FakeRuntime:
    belief = FakeBelief()
    terminated = True
    truncated = False
    last_reward = 0.96


class TaskCompletionTests(unittest.TestCase):
    def test_found_objects_remain_evidence_after_pickup_and_traversal(self):
        planner = UnifiedSayCanPlanner.__new__(UnifiedSayCanPlanner)
        planner.runtime = FakeRuntime()
        planner.history = [
            {
                "call": {
                    "name": "Find",
                    "args": {"target": {"object": "key", "color": "yellow"}},
                },
                "result": {
                    "success": True,
                    "target": {"object": "key", "color": "yellow", "label": "yellow key"},
                },
            },
            {
                "call": {
                    "name": "Find",
                    "args": {"target": {"object": "door", "color": "yellow"}},
                },
                "result": {
                    "success": True,
                    "target": {"object": "door", "color": "yellow", "label": "yellow door"},
                },
            },
            {
                "call": {
                    "name": "Find",
                    "args": {"target": {"object": "goal", "color": "green"}},
                },
                "result": {
                    "success": True,
                    "target": {"object": "goal", "color": "green", "label": "green goal"},
                },
            },
        ]
        task = (
            "Find the yellow key, unlock the yellow door, open the yellow door, "
            "and reach the green goal."
        )
        self.assertTrue(planner._task_success(task))

    def test_find_task_still_fails_without_any_target_evidence(self):
        planner = UnifiedSayCanPlanner.__new__(UnifiedSayCanPlanner)
        planner.runtime = FakeRuntime()
        planner.runtime.belief = FakeBelief()
        planner.runtime.belief.known_map = {}
        planner.runtime.belief.held_object = None
        planner.history = [
            {
                "call": {"name": "Explore", "args": {}},
                "result": {"success": True},
            }
        ]
        # There is no historical Find result for the blue ball.
        self.assertFalse(planner._task_success("Find the blue ball"))


if __name__ == "__main__":
    unittest.main()
