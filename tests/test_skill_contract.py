import unittest

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX
from skills.target import TargetSpec, parse_target_specs, relation_offset
from skills.dependency import TaskDependencyResolver, make_call
from llm.skill_relevance import KeywordRelevanceScorer
from skills.feasibility import SkillFeasibilityEvaluator
from perception.belief_state import BeliefState


class TargetTests(unittest.TestCase):
    def test_target_matching_is_parameterized(self):
        red_ball = TargetSpec.from_value({"object": "ball", "color": "red"})
        self.assertTrue(red_ball.matches({"object": "ball", "color": "red", "state": "unknown"}))
        self.assertFalse(red_ball.matches({"object": "ball", "color": "blue", "state": "unknown"}))

    def test_target_parser_handles_singular_box(self):
        singular = parse_target_specs("put the blue key near the purple box")
        plural = parse_target_specs("put the blue key near the purple boxes")

        self.assertEqual(
            [(item.object_name, item.color) for item in singular],
            [("key", "blue"), ("box", "purple")],
        )
        self.assertEqual(
            [(item.object_name, item.color) for item in plural],
            [("key", "blue"), ("box", "purple")],
        )

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


class BeliefStateTests(unittest.TestCase):
    def test_agent_occlusion_preserves_underlying_open_door(self):
        image = np.zeros((7, 7, 3), dtype=np.uint8)
        image[:, :, 0] = OBJECT_TO_IDX["unseen"]
        image[3, 6] = [OBJECT_TO_IDX["agent"], 0, 0]

        belief = BeliefState()
        belief.agent_pos = (2, 3)
        belief.known_map[belief.agent_pos] = {
            "object": "door",
            "color": "yellow",
            "state": "open",
            "last_seen": 1,
        }

        belief._integrate_observation({
            "image": image,
            "direction": 0,
        })

        self.assertEqual(
            belief.known_map[belief.agent_pos]["object"], "door"
        )
        self.assertEqual(
            belief.known_map[belief.agent_pos]["color"], "yellow"
        )
        self.assertEqual(
            belief.known_map[belief.agent_pos]["state"], "open"
        )


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

    def test_gotoobject_uses_done_for_only_the_explicit_object(self):
        class Belief:
            known_map = {
                (1, 1): {"object": "key", "color": "red", "state": "unknown"},
                (2, 1): {"object": "ball", "color": "blue", "state": "unknown"},
                (3, 1): {"object": "box", "color": "green", "state": "unknown"},
                (4, 1): {"object": "key", "color": "blue", "state": "unknown"},
            }

            @staticmethod
            def find_objects(*args, **kwargs):
                return []

        calls = TaskDependencyResolver().candidate_calls(
            "go to the red key", Belief()
        )
        target_specs = {
            (call.get("args", {}).get("target", {}).get("object"),
             call.get("args", {}).get("target", {}).get("color"))
            for call in calls if call.get("args", {}).get("target")
        }
        self.assertEqual(target_specs, {("key", "red")})
        self.assertTrue(any(
            call["name"] == "Done"
            and call["args"]["target"]["object"] == "key"
            and call["args"]["target"]["color"] == "red"
            for call in calls
        ))
        self.assertFalse(any(call["name"] == "PickUp" for call in calls))

    def test_get_to_pickable_object_is_navigation_not_pickup(self):
        calls = TaskDependencyResolver().candidate_calls("get to the red key")
        self.assertTrue(any(
            call["name"] == "Done"
            and call["args"]["target"]["object"] == "key"
            and call["args"]["target"]["color"] == "red"
            for call in calls
        ))
        self.assertFalse(any(call["name"] == "PickUp" for call in calls))

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

    def test_multiskill_place_near_uses_the_explicit_goal_across_t1_t5(self):
        tasks = [
            "Pick up the red ball, open the yellow door, and place it above the green goal.",
            "Open the yellow door first, then pick up the red ball, and finally place it above the green goal.",
            "Take the red ball, open the yellow door, and drop it above the green goal.",
            "Open the yellow door, then place the red ball above the green goal after picking it up.",
            "Grab the red ball and put it above the green goal; open the yellow door if needed.",
        ]

        resolver = TaskDependencyResolver()
        for task in tasks:
            with self.subTest(task=task):
                calls = resolver.candidate_calls(task)
                place_calls = [
                    call for call in calls if call["name"] == "PlaceNear"
                ]

                self.assertEqual(len(place_calls), 1)
                args = place_calls[0]["args"]
                self.assertEqual(args["object"]["object"], "ball")
                self.assertEqual(args["object"]["color"], "red")
                self.assertEqual(args["target"]["object"], "goal")
                self.assertEqual(args["target"]["color"], "green")
                self.assertEqual(args["relation"], "above")

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


class FeasibilityTests(unittest.TestCase):
    def test_explore_prerequisite_uses_canonical_skill_id(self):
        class Belief:
            inventory_status = "empty"
            held_object = None
            agent_pos = (0, 0)
            agent_dir = 0
            known_map = {
                (0, 0): {
                    "object": "empty",
                    "color": "unknown",
                    "state": "unknown",
                }
            }

            @staticmethod
            def cell_at(_position):
                return None

        class Runtime:
            belief = Belief()
            terminated = False
            truncated = False

        evaluator = SkillFeasibilityEvaluator(Runtime())
        assessment = evaluator.evaluate({
            "id": "unknown",
            "name": "UnknownSkill",
            "args": {},
        })

        self.assertEqual(
            assessment["prerequisites"][0]["id"],
            make_call("Explore")["id"],
        )
        self.assertEqual(assessment["prerequisites"][0]["id"], "explore")

    def test_find_is_infeasible_when_target_is_held(self):
        class Belief:
            inventory_status = "yellow_ball"
            held_object = {"object": "ball", "color": "yellow", "state": "held"}
            known_map = {}
            agent_pos = (0, 0)
            agent_dir = 0

            @staticmethod
            def cell_at(_position):
                return None

        class Runtime:
            belief = Belief()
            terminated = False
            truncated = False

        evaluator = SkillFeasibilityEvaluator(Runtime())
        call = make_call("Find", target={"object": "ball", "color": "yellow"})
        assessment = evaluator.evaluate(call)

        self.assertEqual(assessment["feasibility"], 0.0)
        self.assertIn("already held", assessment["reason"])


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
