import unittest
from skills.target import TargetSpec, parse_target_specs, relation_offset
from skills.dependency import TaskDependencyResolver, make_call
from llm.skill_relevance import KeywordRelevanceScorer


class TargetTests(unittest.TestCase):
    def test_target_matching_is_parameterized(self):
        red_ball = TargetSpec.from_value({"object": "ball", "color": "red"})
        self.assertTrue(red_ball.matches({"object": "ball", "color": "red", "state": "unknown"}))
        self.assertFalse(red_ball.matches({"object": "ball", "color": "blue", "state": "unknown"}))

    def test_target_parser(self):
        specs = parse_target_specs("Find the red ball and unlock the yellow door with a key")
        keys = {item.key for item in specs}
        self.assertIn("red:ball", keys)
        self.assertIn("yellow:door", keys)
        self.assertIn("key", keys)

    def test_relations(self):
        self.assertEqual(relation_offset("above"), (0, -1))
        self.assertEqual(relation_offset("left_of"), (-1, 0))
        self.assertIsNone(relation_offset("diagonal"))


class DependencyTests(unittest.TestCase):
    def test_unlock_candidate_has_parameterized_target(self):
        calls = TaskDependencyResolver().candidate_calls("Find a key and unlock the yellow door")
        self.assertTrue(any(c["name"] == "UnlockDoor" for c in calls))
        self.assertTrue(any(c["name"] == "PickUp" and c["args"]["target"]["object"] == "key" for c in calls))

    def test_place_near_candidate(self):
        calls = TaskDependencyResolver().candidate_calls("Pick up the red ball and place it above the green goal")
        place = [c for c in calls if c["name"] == "PlaceNear"]
        self.assertTrue(place)
        self.assertEqual(place[0]["args"]["relation"], "above")

    def test_goal_entry_candidate_uses_reach(self):
        calls = TaskDependencyResolver().candidate_calls("Unlock the door and reach the green goal")
        self.assertTrue(any(
            c["name"] == "Reach" and c["args"]["target"]["object"] == "goal"
            for c in calls
        ))
        self.assertFalse(any(
            c["name"] == "NavigateTo" and c["args"]["target"]["object"] == "goal"
            for c in calls
        ))


class RelevanceTests(unittest.TestCase):
    def test_offline_scores_are_bounded(self):
        calls = [make_call("Find", target={"object": "key"}), make_call("UnlockDoor", target={"object": "door", "color": "yellow"})]
        scores = KeywordRelevanceScorer().score("Find a key and unlock the yellow door", calls)
        self.assertEqual(set(scores), {c["id"] for c in calls})
        self.assertTrue(all(0 <= x["relevance"] <= 1 for x in scores.values()))


if __name__ == "__main__":
    unittest.main()
