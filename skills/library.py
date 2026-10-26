"""Compatibility facade for the observation-grounded SkillLibraryV1."""
from skills.observation_runtime import ObservationRuntime
from skills.unified_library import SkillLibraryV1
from skills.feasibility import SkillFeasibilityEvaluator
from skills.dependency import TaskDependencyResolver
from llm.skill_relevance import KeywordRelevanceScorer


class SkillLibrary:
    """Legacy facade; new code should instantiate SkillLibraryV1(runtime)."""
    def __init__(self, env, movement=None, picker=None, opener=None, placer=None):
        if isinstance(env, ObservationRuntime):
            self.runtime = env
        else:
            self.runtime = ObservationRuntime(env)
            obs = env.unwrapped.gen_obs()
            self.runtime.reset(obs, initial_inventory="empty")
        self.library = SkillLibraryV1(self.runtime)
        self.feasibility = SkillFeasibilityEvaluator(self.runtime)
        self.dependencies = TaskDependencyResolver()
        self.relevance_scorer = KeywordRelevanceScorer()
        self.completed_subgoals = []
        self.inventory_status = "empty"

    @staticmethod
    def _legacy_call(name):
        aliases = {
            "pick": {"name": "PickUp", "args": {"target": {"object": "ball", "color": "red"}}},
            "open_door": {"name": "OpenDoor", "args": {"target": {"object": "door", "color": "yellow"}}},
            "unlock_door": {"name": "UnlockDoor", "args": {"target": {"object": "door", "color": "yellow"}}},
            "place": {"name": "PlaceNear", "args": {"object": {"object": "ball", "color": "red"}, "target": {"object": "goal", "color": "green"}, "relation": "above"}},
            "explore": {"name": "Explore", "args": {}},
        }
        return aliases.get(name)

    def score_candidates(self, task):
        calls = self.dependencies.candidate_calls(task, self.runtime.belief)
        relevance = self.relevance_scorer.score(task, calls, self.runtime.belief.summary())
        output = []
        for call in calls:
            feasibility = self.feasibility.evaluate(call)
            rel = relevance.get(call["id"], {}).get("relevance", 0.0)
            output.append({
                "id": call["id"], "name": call["name"], "call": call,
                "relevance": rel, "feasibility": feasibility["feasibility"],
                "feasibility_reason": feasibility["reason"],
                "prerequisites": feasibility["prerequisites"],
                "score": rel * feasibility["feasibility"],
            })
        return output

    def execute(self, skill_name):
        call = skill_name if isinstance(skill_name, dict) else self._legacy_call(skill_name)
        if call is None:
            return {"success": False, "status": "unknown_skill", "reason": f"Unknown skill: {skill_name}", "actions_executed": 0}
        result = self.library.execute(call)
        if result.get("success"):
            name = call["name"]
            if name not in self.completed_subgoals:
                self.completed_subgoals.append(name)
            self.inventory_status = self.runtime.belief.inventory_status
        return result
