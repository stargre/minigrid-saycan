"""Regression tests for official MiniGrid Fetch task grounding."""
import unittest

from skills.dependency import TaskDependencyResolver
from skills.target import parse_target_specs


class FakeBelief:
    # Distractors are intentionally visible alongside the requested object.
    known_map = {
        (1, 1): {"object": "key", "color": "red", "state": "unknown"},
        (2, 2): {"object": "ball", "color": "blue", "state": "unknown"},
        (3, 3): {"object": "key", "color": "yellow", "state": "unknown"},
        (4, 4): {"object": "ball", "color": "green", "state": "unknown"},
    }

    @staticmethod
    def find_objects(*args, **kwargs):
        return []


class FetchTaskContractTests(unittest.TestCase):
    def test_official_fetch_mission_variants_parse_one_target(self):
        missions = (
            "get a red key",
            "go get a blue ball",
            "fetch a yellow key",
            "go fetch a grey ball",
            "you must fetch a green key",
        )
        expected = (
            ("key", "red"),
            ("ball", "blue"),
            ("key", "yellow"),
            ("ball", "grey"),
            ("key", "green"),
        )

        for mission, target in zip(missions, expected):
            with self.subTest(mission=mission):
                specs = parse_target_specs(mission)
                self.assertEqual(
                    [(spec.object_name, spec.color) for spec in specs],
                    [target],
                )

    def test_fetch_candidates_exclude_all_distractor_objects(self):
        calls = TaskDependencyResolver().candidate_calls(
            "fetch a red key",
            belief=FakeBelief(),
        )

        target_calls = [
            call for call in calls
            if call["name"] in {"Find", "PickUp"}
        ]
        observed_targets = {
            (
                call["name"],
                call["args"]["target"]["object"],
                call["args"]["target"].get("color"),
            )
            for call in target_calls
        }

        self.assertEqual(
            observed_targets,
            {
                ("Find", "key", "red"),
                ("PickUp", "key", "red"),
            },
        )

    def test_get_a_target_uses_the_same_fetch_candidate_constraint(self):
        calls = TaskDependencyResolver().candidate_calls(
            "go get a yellow ball",
            belief=FakeBelief(),
        )
        pickup_targets = [
            call["args"]["target"]
            for call in calls
            if call["name"] == "PickUp"
        ]

        self.assertEqual(len(pickup_targets), 1)
        self.assertEqual(pickup_targets[0]["object"], "ball")
        self.assertEqual(pickup_targets[0]["color"], "yellow")


if __name__ == "__main__":
    unittest.main()
