"""SayCan-style planner over the frozen observation-grounded skill library."""
from skills.dependency import TaskDependencyResolver
from skills.feasibility import SkillFeasibilityEvaluator
from skills.unified_library import SkillLibraryV1
from llm.skill_relevance import KeywordRelevanceScorer


class UnifiedSayCanPlanner:
    def __init__(self, runtime, relevance_scorer=None, skill_library=None, max_actions=450):
        self.runtime = runtime
        self.library = skill_library or SkillLibraryV1(runtime)
        self.dependencies = TaskDependencyResolver()
        self.feasibility = SkillFeasibilityEvaluator(runtime, self.dependencies)
        self.relevance = relevance_scorer or KeywordRelevanceScorer()
        self.max_actions = int(max_actions)
        self.history = []
        self.completed_ids = set()
        self._queued = {}

    @staticmethod
    def _terminal(result):
        return result.get("status") in {"target_found", "skill_complete", "done_issued", "door_already_open"}

    def _candidate_pool(self, task):
        for call in self.dependencies.candidate_calls(task, self.runtime.belief):
            self._queued.setdefault(call["id"], call)
        enriched = []
        for call in list(self._queued.values()):
            if call["id"] in self.completed_ids:
                continue
            assessment = self.feasibility.evaluate(call)
            if assessment["feasibility"] <= 0:
                for prerequisite in assessment["prerequisites"]:
                    self._queued.setdefault(prerequisite["id"], prerequisite)
            enriched.append((call, assessment))
        for call in list(self._queued.values()):
            if call["id"] in self.completed_ids or any(c["id"] == call["id"] for c, _ in enriched):
                continue
            enriched.append((call, self.feasibility.evaluate(call)))
        return enriched

    def step(self, task):
        if self.runtime.action_count >= self.max_actions:
            return {"success": False, "status": "max_actions_reached", "history": self.history}
        pool = self._candidate_pool(task)
        calls = [call for call, _ in pool]
        relevance = self.relevance.score(task, calls, self.runtime.belief.summary())
        scored = []
        for call, assessment in pool:
            item = relevance.get(call["id"], {"relevance": 0.0, "rationale": "No relevance score."})
            rel, feasible = float(item["relevance"]), float(assessment["feasibility"])
            scored.append({
                "call": call, "relevance": rel, "feasibility": feasible,
                "score": rel * feasible, "rationale": item.get("rationale", ""),
                "feasibility_reason": assessment["reason"],
            })
        scored.sort(key=lambda x: (-x["score"], x["call"]["id"]))
        selected = next((x for x in scored if x["score"] > 0), None)
        if selected is None:
            return {"success": False, "status": "no_feasible_candidate", "history": self.history}
        call = selected["call"]
        result = self.library.execute(call)
        record = {
            "round": len(self.history) + 1, "call": call,
            "relevance": selected["relevance"], "feasibility": selected["feasibility"],
            "score": selected["score"], "relevance_rationale": selected["rationale"],
            "feasibility_reason": selected["feasibility_reason"],
            "result": result, "belief": self.runtime.belief.summary(),
        }
        self.history.append(record)
        if self._terminal(result):
            self.completed_ids.add(call["id"])
        return {"success": bool(result.get("success")), "status": result.get("status"), "record": record}

    def run(self, task, max_high_level_steps=500):
        start_actions = self.runtime.action_count
        status = "planner_stopped"
        for _ in range(int(max_high_level_steps)):
            if self.runtime.terminated or self.runtime.truncated:
                status = "episode_ended"
                break
            if self.runtime.action_count - start_actions >= self.max_actions:
                status = "max_actions_reached"
                break
            result = self.step(task)
            status = result.get("status", "unknown")
            if status in {"no_feasible_candidate", "target_not_found", "exploration_exhausted", "exploration_stalled", "episode_ended", "max_actions_reached"}:
                break
        success = self._task_success(task)
        return {
            "success": success, "status": "task_complete" if success else status,
            "high_level_steps": len(self.history), "low_level_actions": self.runtime.action_count-start_actions,
            "completed_calls": sorted(self.completed_ids), "history": self.history,
        }

    def _task_success(self, task):
        text = str(task).lower()
        done = set(self.runtime.belief.completed_subgoals)
        if any(w in text for w in ("place", "put", "drop", "set down")):
            if self.runtime.belief.inventory_status != "empty" or not ({"place", "drop"} & done):
                return False
        if "unlock" in text and not ({"unlock_door", "open_door"} & done):
            return False
        if "open" in text and "door" in text and not ({"open_door", "unlock_door"} & done):
            return False
        if any(w in text for w in ("pick", "take", "grab", "fetch")) and not ({"pick", "pickup"} & done):
            return False
        return bool(self.history)
