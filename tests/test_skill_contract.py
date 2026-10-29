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

    def test_gotodoor_uses_done_without_opening_target(self):
        calls = TaskDependencyResolver().candidate_calls("go to the red door")
        self.assertTrue(any(
            c["name"] == "Done" and c["args"]["target"]["object"] == "door"
            for c in calls
        ))
        self.assertFalse(any(c["name"] in {"OpenDoor", "UnlockDoor"} for c in calls))

    def test_explicit_door_color_excludes_other_observed_door_colors(self):
        class Belief:
            known_map = {
                (1, 1): {"object": "door", "color": "blue", "state": "closed"},
                (3, 3): {"object": "door", "color": "green", "state": "closed"},
            }

            @staticmethod
            def find_objects(*args, **kwargs):
                return []

        calls = TaskDependencyResolver().candidate_calls("go to the green door", Belief())
        target_colors = {
            call.get("args", {}).get("target", {}).get("color")
            for call in calls
            if call.get("args", {}).get("target", {}).get("object") == "door"
        }
        self.assertIn("green", target_colors)
        self.assertNotIn("blue", target_colors)

    def test_open_door_task_exposes_door_interaction(self):
        calls = TaskDependencyResolver().candidate_calls("open the yellow door")
        self.assertTrue(any(c["name"] == "OpenDoor" for c in calls))

    def test_place_near_candidate(self):
        calls = TaskDependencyResolver().candidate_calls("Pick up the red ball and place it above the green goal")
        place = [c for c in calls if c["name"] == "PlaceNear"]
        self.assertTrue(place)
        self.assertEqual(place[0]["args"]["relation"], "above")

    def test_putnear_candidate_exposes_ordered_pickup_and_destination(self):
        task = "put the red ball near the green box"
        calls = TaskDependencyResolver().candidate_calls(task)

        pickup_calls = [call for call in calls if call["name"] == "PickUp"]
        self.assertEqual(len(pickup_calls), 1, pickup_calls)
        self.assertEqual(pickup_calls[0]["args"]["target"]["object"], "ball")
        self.assertEqual(pickup_calls[0]["args"]["target"]["color"], "red")

        place_calls = [call for call in calls if call["name"] == "PlaceNear"]
        self.assertEqual(len(place_calls), 1, place_calls)
        self.assertEqual(place_calls[0]["args"]["object"]["object"], "ball")
        self.assertEqual(place_calls[0]["args"]["object"]["color"], "red")
        self.assertEqual(place_calls[0]["args"]["target"]["object"], "box")
        self.assertEqual(place_calls[0]["args"]["target"]["color"], "green")
        self.assertEqual(place_calls[0]["args"]["relation"], "near")

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

    def test_reach_scores_high_for_goal_entry(self):
        call = make_call("Reach", target={"object": "goal", "color": "green"})
        scores = KeywordRelevanceScorer().score(
            "Reach the green goal", [call]
        )
        self.assertGreaterEqual(scores[call["id"]]["relevance"], 0.9)

    def test_goto_door_prefers_done_for_the_requested_color(self):
        calls = TaskDependencyResolver().candidate_calls("go to the green door")
        wrong_color = make_call(
            "NavigateTo", target={"object": "door", "color": "blue"}
        )
        calls.append(wrong_color)
        scores = KeywordRelevanceScorer().score("go to the green door", calls)

        done_green = next(
            call for call in calls
            if call["name"] == "Done"
            and call["args"]["target"].get("color") == "green"
        )
        self.assertGreaterEqual(scores[done_green["id"]]["relevance"], 0.95)
        self.assertLessEqual(scores[wrong_color["id"]]["relevance"], 0.01)


if __name__ == "__main__":
    unittest.main()
