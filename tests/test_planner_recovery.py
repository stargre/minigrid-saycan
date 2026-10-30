"""Regression tests for recoverable planner failures and scoring boundaries."""
import unittest
from unittest.mock import Mock

from planners.unified_saycan_planner import UnifiedSayCanPlanner


class FakeBelief:
    observation_count = 0
    completed_subgoals = []
    known_map = {}
    inventory_status = "empty"
    held_object = None

    def summary(self):
        return {"observation_count": self.observation_count}


class FakeRuntime:
    def __init__(self):
        self.action_count = 0
        self.terminated = False
        self.truncated = False
        self.belief = FakeBelief()


class PlannerRecoveryTests(unittest.TestCase):
    def planner(self):
        planner = UnifiedSayCanPlanner.__new__(UnifiedSayCanPlanner)
        planner.runtime = FakeRuntime()
        planner.history = []
        planner.completed_ids = set()
        planner.blocked_ids = {}
        planner._queued = {}
        planner._run_start_actions = 0
        planner.max_actions = 100
        planner.max_skill_actions = 10
        planner.relevance = Mock()
        planner.library = Mock()
        planner.feasibility = Mock()
        planner.dependencies = Mock()
        planner._task_success = Mock(return_value=False)
        return planner

    def test_exploration_exhausted_is_not_a_hard_stop(self):
        planner = self.planner()
        statuses = iter([
            {"success": False, "status": "exploration_exhausted"},
            {"success": True, "status": "task_progress"},
        ])
        planner.step = Mock(side_effect=lambda task: next(statuses))
        result = planner.run("find a red ball", max_high_level_steps=2)
        self.assertEqual(planner.step.call_count, 2)
        self.assertEqual(result["status"], "task_progress")

    def test_zero_relevance_has_deterministic_fallback(self):
        planner = self.planner()
        call = {"id": "Find:ball", "name": "Find", "args": {}, "description": "Find ball"}
        planner._candidate_pool = Mock(return_value=[
            (call, {"feasibility": 1.0, "reason": "reachable"})
        ])
        planner.relevance.score.return_value = {
            "Find:ball": {"relevance": 0.0, "rationale": "low"}
        }
        planner._execute_skill_to_terminal = Mock(return_value={
            "success": False, "status": "exploration_exhausted", "low_level_actions": 0
        })
        result = planner.step("find a ball")
        self.assertEqual(result["status"], "exploration_exhausted")
        self.assertTrue(result["record"]["zero_score_fallback"])
        planner.relevance.score.assert_called_once()

    def test_feasible_done_has_priority_over_higher_qwen_explore_score(self):
        planner = self.planner()
        explore = {
            "id": "Explore", "name": "Explore", "args": {},
            "description": "Explore the environment",
        }
        done = {
            "id": "Done:key:red", "name": "Done",
            "args": {"target": {"object": "key", "color": "red"}},
            "description": "Finish at the red key",
        }
        planner._candidate_pool = Mock(return_value=[
            (explore, {"feasibility": 1.0, "reason": "frontier remains"}),
            (done, {"feasibility": 1.0, "reason": "target is adjacent"}),
        ])
        planner.relevance.score.return_value = {
            explore["id"]: {"relevance": 1.0, "rationale": "Explore the map"},
            done["id"]: {"relevance": 0.0, "rationale": "Low semantic score"},
        }
        planner._execute_skill_to_terminal = Mock(return_value={
            "success": True, "status": "done_issued", "low_level_actions": 1,
        })

        result = planner.step("go to the red key")

        planner._execute_skill_to_terminal.assert_called_once_with(done)
        self.assertEqual(result["record"]["call"]["name"], "Done")
        self.assertTrue(result["record"]["terminal_priority"])
        self.assertFalse(result["record"]["zero_score_fallback"])
        self.assertEqual(result["record"]["selected_candidate_id"], done["id"])
        self.assertEqual(result["record"]["selection_policy"], "terminal_priority")
        self.assertEqual(
            result["record"]["candidate_rankings"][0]["id"], explore["id"]
        )
        self.assertIn(
            "priority: feasible terminal Done action",
            result["record"]["relevance_rationale"],
        )

    def test_infeasible_done_does_not_override_feasible_explore(self):
        planner = self.planner()
        explore = {
            "id": "Explore", "name": "Explore", "args": {},
            "description": "Explore the environment",
        }
        done = {
            "id": "Done:key:red", "name": "Done",
            "args": {"target": {"object": "key", "color": "red"}},
            "description": "Finish at the red key",
        }
        planner._candidate_pool = Mock(return_value=[
            (explore, {"feasibility": 1.0, "reason": "frontier remains"}),
            (done, {"feasibility": 0.0, "reason": "target is not adjacent"}),
        ])
        planner.relevance.score.return_value = {
            explore["id"]: {"relevance": 0.8, "rationale": "Explore next"},
        }
        planner._execute_skill_to_terminal = Mock(return_value={
            "success": True, "status": "action_executed", "low_level_actions": 1,
        })

        result = planner.step("go to the red key")

        planner._execute_skill_to_terminal.assert_called_once_with(explore)
        self.assertEqual(result["record"]["call"]["name"], "Explore")
        self.assertFalse(result["record"]["terminal_priority"])

    def test_compound_task_does_not_force_intermediate_done_priority(self):
        planner = self.planner()
        explore = {
            "id": "Explore", "name": "Explore", "args": {},
            "description": "Explore the environment",
        }
        done = {
            "id": "Done:door:yellow", "name": "Done",
            "args": {"target": {"object": "door", "color": "yellow"}},
            "description": "Finish at the yellow door",
        }
        planner._candidate_pool = Mock(return_value=[
            (explore, {"feasibility": 1.0, "reason": "frontier remains"}),
            (done, {"feasibility": 1.0, "reason": "door is adjacent"}),
        ])
        planner.relevance.score.return_value = {
            explore["id"]: {"relevance": 1.0, "rationale": "Continue the task"},
            done["id"]: {"relevance": 0.0, "rationale": "Intermediate interaction"},
        }
        planner._execute_skill_to_terminal = Mock(return_value={
            "success": True, "status": "action_executed", "low_level_actions": 1,
        })

        result = planner.step("unlock the yellow door and reach the green goal")

        planner._execute_skill_to_terminal.assert_called_once_with(explore)
        self.assertEqual(result["record"]["call"]["name"], "Explore")
        self.assertFalse(result["record"]["terminal_priority"])


    def test_conditional_door_open_is_not_an_unconditional_task_subgoal(self):
        planner = self.planner()
        planner._task_success = UnifiedSayCanPlanner._task_success.__get__(
            planner, UnifiedSayCanPlanner
        )
        planner.runtime.belief.completed_subgoals = ["pick", "pickup", "place"]
        planner.history = [{"call": {"name": "PickUp"}}]

        conditional_task = (
            "Grab the red ball and put it above the green goal; "
            "open the yellow door if needed."
        )
        self.assertTrue(planner._task_success(conditional_task))

        mandatory_task = (
            "Grab the red ball, open the yellow door, "
            "and put it above the green goal."
        )
        self.assertFalse(planner._task_success(mandatory_task))

    def test_candidate_ranking_trace_contains_relevance_and_feasibility(self):
        planner = self.planner()
        explore = {
            "id": "Explore", "name": "Explore", "args": {},
            "description": "Explore the environment",
        }
        pickup = {
            "id": "PickUp:ball:red", "name": "PickUp",
            "args": {"target": {"object": "ball", "color": "red"}},
            "description": "Pick up the red ball",
        }
        planner._candidate_pool = Mock(return_value=[
            (explore, {"feasibility": 1.0, "reason": "frontier remains"}),
            (pickup, {"feasibility": 0.8, "reason": "target is reachable"}),
        ])
        planner.relevance.score.return_value = {
            explore["id"]: {"relevance": 0.2, "rationale": "Explore is secondary"},
            pickup["id"]: {"relevance": 0.9, "rationale": "Acquire requested object"},
        }
        planner._execute_skill_to_terminal = Mock(return_value={
            "success": True, "status": "skill_complete", "low_level_actions": 1,
        })

        result = planner.step("pick up the red ball")

        ranking = result["record"]["candidate_rankings"]
        self.assertEqual(
            [item["id"] for item in ranking],
            [pickup["id"], explore["id"]],
        )
        self.assertEqual(ranking[0]["score"], 0.72)
        self.assertEqual(ranking[0]["feasibility"], 0.8)
        self.assertEqual(
            result["record"]["selection_policy"],
            "max_relevance_times_feasibility",
        )

    def test_infeasible_candidates_are_not_sent_to_relevance_scorer(self):
        planner = self.planner()
        feasible = {"id": "OpenDoor:yellow:door", "name": "OpenDoor", "args": {}, "description": "Open door"}
        infeasible = {"id": "PlaceNear:ball:goal:above", "name": "PlaceNear", "args": {}, "description": "Place ball"}
        planner._candidate_pool = Mock(return_value=[
            (feasible, {"feasibility": 1.0, "reason": "ready"}),
            (infeasible, {"feasibility": 0.0, "reason": "not holding"}),
        ])
        planner.relevance.score.return_value = {
            feasible["id"]: {"relevance": 0.9, "rationale": "next step"}
        }
        planner._execute_skill_to_terminal = Mock(return_value={
            "success": True, "status": "task_progress", "low_level_actions": 1
        })
        planner.runtime.action_count = 1
        result = planner.step("open door and place ball")
        sent_calls = planner.relevance.score.call_args.args[1]
        self.assertEqual([c["id"] for c in sent_calls], [feasible["id"]])
        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
