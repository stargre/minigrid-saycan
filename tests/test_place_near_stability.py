"""Regression tests for stable PlaceNear execution targets."""
import unittest
from unittest.mock import patch

from skills.manipulation_skills import PlaceNearSkill
from skills.navigation import ObservationNavigator


def cell(object_name, color="unknown"):
    return {"object": object_name, "color": color, "state": "unknown"}


class FakeBelief:
    def __init__(self):
        self.agent_pos = (0, 0)
        self.agent_dir = 0
        self.known_map = {
            (5, 5): cell("box", "purple"),
            (1, 0): cell("empty"),
            (0, 1): cell("empty"),
        }
        self.held_object = {"object": "key", "color": "blue", "state": "held"}
        self.inventory_status = "blue_key"
        self.completed_subgoals = []

    def cell_at(self, pos):
        return self.known_map.get(tuple(pos))

    def _mark_completed(self, name):
        if name not in self.completed_subgoals:
            self.completed_subgoals.append(name)


class FakeRuntime:
    def __init__(self):
        self.belief = FakeBelief()


class PlaceNearStabilityTests(unittest.TestCase):
    def test_selected_placement_cell_stays_fixed_across_controller_ticks(self):
        runtime = FakeRuntime()
        skill = PlaceNearSkill(runtime)
        navigated_targets = []
        skill.navigator.step_to_adjacent = lambda pos: (
            navigated_targets.append(tuple(pos))
            or {"success": True, "status": "action_executed", "actions_executed": 1}
        )

        with patch(
            "skills.manipulation_skills.select_placement",
            side_effect=[
                ((5, 5), (1, 0), None),
                ((5, 5), (0, 1), None),
            ],
        ) as selector:
            skill.step(
                object={"object": "key", "color": "blue"},
                target={"object": "box", "color": "purple"},
                relation="near",
            )
            # The agent moves; a fresh distance ranking would prefer another
            # nearby cell, but the active skill must keep its commitment.
            runtime.belief.agent_pos = (0, 1)
            skill.step(
                object={"object": "key", "color": "blue"},
                target={"object": "box", "color": "purple"},
                relation="near",
            )

        self.assertEqual(navigated_targets, [(1, 0), (1, 0)])
        selector.assert_called_once()

    def test_occupied_committed_cell_triggers_reselection(self):
        runtime = FakeRuntime()
        skill = PlaceNearSkill(runtime)
        navigated_targets = []
        skill.navigator.step_to_adjacent = lambda pos: (
            navigated_targets.append(tuple(pos))
            or {"success": True, "status": "action_executed", "actions_executed": 1}
        )

        with patch(
            "skills.manipulation_skills.select_placement",
            side_effect=[
                ((5, 5), (1, 0), None),
                ((5, 5), (0, 1), None),
            ],
        ) as selector:
            skill.step(
                object={"object": "key", "color": "blue"},
                target={"object": "box", "color": "purple"},
                relation="near",
            )
            runtime.belief.known_map[(1, 0)] = cell("wall")
            skill.step(
                object={"object": "key", "color": "blue"},
                target={"object": "box", "color": "purple"},
                relation="near",
            )

        self.assertEqual(navigated_targets, [(1, 0), (0, 1)])
        self.assertEqual(selector.call_count, 2)

    def test_reached_approach_cell_is_not_abandoned_due_to_stale_map_cell(self):
        runtime = FakeRuntime()
        runtime.belief.agent_pos = (1, 0)
        runtime.belief.agent_dir = 0
        runtime.belief.known_map[(1, 0)] = cell("wall")

        navigator = ObservationNavigator(runtime)
        navigator._active_target = (2, 0)
        navigator._approach_pos = (1, 0)

        result = navigator.step_to_adjacent((2, 0))

        self.assertEqual(result["status"], "target_adjacent")
        self.assertEqual(navigator._approach_pos, (1, 0))


if __name__ == "__main__":
    unittest.main()
