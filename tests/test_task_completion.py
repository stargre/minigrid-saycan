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

    def test_get_object_task_does_not_finish_when_find_only_succeeded(self):
        class FetchBelief:
            known_map = {
                (2, 1): {"object": "key", "color": "grey", "state": "unknown"}
            }
            inventory_status = "empty"
            held_object = None
            completed_subgoals = []

        class FetchRuntime:
            belief = FetchBelief()
            terminated = False
            truncated = False
            last_reward = 0.0

        planner = UnifiedSayCanPlanner.__new__(UnifiedSayCanPlanner)
        planner.runtime = FetchRuntime()
        planner.history = [
            {
                "call": {
                    "name": "Find",
                    "args": {"target": {"object": "key", "color": "grey"}},
                },
                "result": {
                    "success": True,
                    "status": "target_found",
                    "target": {"object": "key", "color": "grey"},
                },
            }
        ]

        self.assertFalse(planner._task_success("go get a grey key"))

    def test_get_object_task_requires_the_requested_object_to_be_held(self):
        class FetchBelief:
            known_map = {}
            inventory_status = "blue_ball"
            held_object = {"object": "ball", "color": "blue", "state": "held"}
            completed_subgoals = ["pick", "pickup"]

        class FetchRuntime:
            belief = FetchBelief()
            terminated = False
            truncated = False
            last_reward = 0.0

        planner = UnifiedSayCanPlanner.__new__(UnifiedSayCanPlanner)
        planner.runtime = FetchRuntime()
        planner.history = [
            {
                "call": {
                    "name": "Find",
                    "args": {"target": {"object": "key", "color": "grey"}},
                },
                "result": {
                    "success": True,
                    "status": "target_found",
                    "target": {"object": "key", "color": "grey"},
                },
            },
            {
                "call": {
                    "name": "PickUp",
                    "args": {"target": {"object": "ball", "color": "blue"}},
                },
                "result": {"success": True, "status": "skill_complete"},
            },
        ]

        self.assertFalse(planner._task_success("go get a grey key"))
        self.assertTrue(planner._task_success("go get a blue ball"))

    def test_pickup_and_place_task_completes_after_object_is_released(self):
        class PlacementBelief:
            known_map = {}
            inventory_status = "empty"
            held_object = None
            completed_subgoals = ["pick", "pickup", "place"]

        class PlacementRuntime:
            belief = PlacementBelief()
            terminated = False
            truncated = False
            last_reward = 0.0

        planner = UnifiedSayCanPlanner.__new__(UnifiedSayCanPlanner)
        planner.runtime = PlacementRuntime()
        planner.history = [
            {
                "call": {
                    "name": "PickUp",
                    "args": {"target": {"object": "ball", "color": "red"}},
                },
                "result": {"success": True, "status": "skill_complete"},
            },
            {
                "call": {
                    "name": "PlaceNear",
                    "args": {
                        "object": {"object": "ball", "color": "red"},
                        "target": {"object": "box", "color": "green"},
                    },
                },
                "result": {"success": True, "status": "skill_complete"},
            },
        ]

        self.assertTrue(
            planner._task_success("pick up the red ball and place it near the green box")
        )

    def test_get_to_door_is_not_misclassified_as_object_pickup(self):
        planner = UnifiedSayCanPlanner.__new__(UnifiedSayCanPlanner)
        planner.runtime = FakeRuntime()
        planner.runtime.terminated = False
        planner.runtime.truncated = False
        planner.runtime.last_reward = 0.0
        planner.history = [
            {
                "call": {
                    "name": "NavigateTo",
                    "args": {"target": {"object": "door", "color": "green"}},
                },
                "result": {"success": True, "status": "skill_complete"},
            }
        ]
        self.assertFalse(planner._task_success("get to the green door"))

        planner.runtime.terminated = True
        planner.runtime.last_reward = 0.95
        self.assertTrue(planner._task_success("get to the green door"))

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

    def test_goto_door_does_not_finish_after_navigation_alone(self):
        planner = UnifiedSayCanPlanner.__new__(UnifiedSayCanPlanner)
        planner.runtime = FakeRuntime()
        planner.runtime.terminated = False
        planner.runtime.truncated = False
        planner.runtime.last_reward = 0.0
        planner.history = [
            {
                "call": {
                    "name": "NavigateTo",
                    "args": {"target": {"object": "door", "color": "green"}},
                },
                "result": {"success": True, "status": "skill_complete"},
            }
        ]
        self.assertFalse(planner._task_success("go to the green door"))

        # Even adjacency alone is insufficient; the environment must confirm
        # successful termination after the required Done interaction.
        planner.runtime.terminated = True
        planner.runtime.last_reward = 0.0
        self.assertFalse(planner._task_success("go to the green door"))

        planner.runtime.last_reward = 0.95
        self.assertTrue(planner._task_success("go to the green door"))

    def test_gotoobject_requires_positive_terminal_done_not_find_or_navigation(self):
        class ObjectBelief:
            inventory_status = "empty"
            held_object = None
            completed_subgoals = []

        class ObjectRuntime:
            belief = ObjectBelief()
            terminated = False
            truncated = False
            last_reward = 0.0

        planner = UnifiedSayCanPlanner.__new__(UnifiedSayCanPlanner)
        planner.runtime = ObjectRuntime()
        planner.history = [
            {
                "call": {
                    "name": "Find",
                    "args": {"target": {"object": "key", "color": "red"}},
                },
                "result": {"success": True, "status": "target_found"},
            }
        ]
        task = "go to the red key"
        self.assertFalse(planner._task_success(task))

        planner.history = [
            {
                "call": {
                    "name": "NavigateTo",
                    "args": {"target": {"object": "key", "color": "red"}},
                },
                "result": {"success": True, "status": "skill_complete"},
            }
        ]
        self.assertFalse(planner._task_success(task))

        # Done at the wrong location terminates the official environment with
        # zero reward; that must not be accepted as task completion.
        planner.history.append({
            "call": {
                "name": "Done",
                "args": {"target": {"object": "key", "color": "red"}},
            },
            "result": {"success": True, "status": "done_issued"},
        })
        planner.runtime.terminated = True
        planner.runtime.last_reward = 0.0
        self.assertFalse(planner._task_success(task))

        planner.runtime.last_reward = 0.95
        self.assertTrue(planner._task_success(task))

        planner.runtime.truncated = True
        self.assertFalse(planner._task_success(task))

    def test_find_obligation_only_applies_to_explicit_search_target(self):
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
            }
        ]
        # The key is explicitly requested by Find. Door and goal are mentioned
        # only as later action targets and must not require their own Find calls.
        task = (
            "Find the yellow key, unlock the yellow door, open the yellow door, "
            "and reach the green goal."
        )
        self.assertTrue(planner._task_success(task))


if __name__ == "__main__":
    unittest.main()
