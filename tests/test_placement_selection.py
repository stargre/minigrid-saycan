"""Regression tests for shared placement-cell selection."""
import unittest

from skills.placement import select_placement


class FakeBelief:
    def __init__(self, known_map, agent_pos=(0, 0)):
        self.known_map = known_map
        self.agent_pos = agent_pos

    def cell_at(self, position):
        return self.known_map.get(tuple(position))


class FakeNavigator:
    def __init__(self, reachable_approaches):
        self.reachable_approaches = {tuple(pos) for pos in reachable_approaches}

    @staticmethod
    def _safe_walkable(cell):
        return bool(cell and cell.get("object") in {"empty", "floor"})

    def _find_path(self, start, goal):
        return [tuple(start), tuple(goal)] if tuple(goal) in self.reachable_approaches else None


def cell(object_name, color="unknown"):
    return {"object": object_name, "color": color, "state": "unknown"}


class PlacementSelectionTests(unittest.TestCase):
    def test_near_skips_nearest_but_unreachable_cell(self):
        belief = FakeBelief({
            (2, 2): cell("box", "green"),
            (3, 2): cell("empty"),
            (2, 3): cell("empty"),
            (1, 3): cell("empty"),
        })
        navigator = FakeNavigator(reachable_approaches={(1, 3)})

        destination, placement, error = select_placement(
            belief, navigator, [(2, 2)], "near"
        )

        self.assertEqual(destination, (2, 2))
        self.assertEqual(placement, (2, 3))
        self.assertIsNone(error)

    def test_known_but_unreachable_empty_cell_is_not_feasible(self):
        belief = FakeBelief({
            (2, 2): cell("box", "green"),
            (2, 1): cell("empty"),
        })
        navigator = FakeNavigator(reachable_approaches=set())

        destination, placement, error = select_placement(
            belief, navigator, [(2, 2)], "above"
        )

        self.assertIsNone(destination)
        self.assertIsNone(placement)
        self.assertEqual(error, "no_known_path")

    def test_occupied_fixed_relation_cell_is_reported(self):
        belief = FakeBelief({
            (2, 2): cell("box", "green"),
            (2, 1): cell("wall"),
        })
        navigator = FakeNavigator(reachable_approaches={(1, 1)})

        destination, placement, error = select_placement(
            belief, navigator, [(2, 2)], "above"
        )

        self.assertIsNone(destination)
        self.assertIsNone(placement)
        self.assertEqual(error, "placement_cell_occupied")

    def test_unknown_fixed_relation_cell_is_reported(self):
        belief = FakeBelief({(2, 2): cell("box", "green")})
        navigator = FakeNavigator(reachable_approaches={(1, 1)})

        destination, placement, error = select_placement(
            belief, navigator, [(2, 2)], "above"
        )

        self.assertIsNone(destination)
        self.assertIsNone(placement)
        self.assertEqual(error, "placement_cell_unknown")

    def test_unsupported_relation_is_rejected(self):
        belief = FakeBelief({(2, 2): cell("box", "green")})
        navigator = FakeNavigator(reachable_approaches={(1, 1)})

        destination, placement, error = select_placement(
            belief, navigator, [(2, 2)], "diagonal"
        )

        self.assertIsNone(destination)
        self.assertIsNone(placement)
        self.assertEqual(error, "unsupported_relation")

    def test_second_destination_is_used_when_nearest_has_no_reachable_placement(self):
        belief = FakeBelief({
            (1, 1): cell("box", "green"),
            (5, 5): cell("box", "green"),
            (1, 0): cell("empty"),
            (5, 4): cell("empty"),
            (4, 4): cell("empty"),
        })
        navigator = FakeNavigator(reachable_approaches={(4, 4)})

        destination, placement, error = select_placement(
            belief, navigator, [(1, 1), (5, 5)], "above"
        )

        self.assertEqual(destination, (5, 5))
        self.assertEqual(placement, (5, 4))
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
