"""SayCan-style planner with persistent closed-loop skill execution.

The LLM (or offline scorer) selects a semantic skill. The selected deterministic
controller is then called repeatedly until it succeeds, fails terminally, stalls,
or reaches a bounded action budget. Primitive actions never go to the LLM.
"""
from skills.dependency import TaskDependencyResolver
from skills.feasibility import SkillFeasibilityEvaluator
from skills.unified_library import SkillLibraryV1
from skills.target import parse_target_specs
from llm.skill_relevance import KeywordRelevanceScorer


class UnifiedSayCanPlanner:
    # These statuses mean the selected controller still has work to do.
    PROGRESS_STATUSES = {"action_executed", "searching", "phase_complete"}
    TERMINAL_FAILURE_STATUSES = {
        "target_not_found",
        "exploration_exhausted",
        "exploration_stalled",
        "skill_timeout",
        "skill_stalled",
        "invalid_arguments",
        "unsupported_relation",
        "not_pickable",
        "no_known_path",
        "requires_key",
        "gripper_not_empty",
        "not_holding_object",
        "placement_cell_unknown",
        "placement_cell_occupied",
        "pickup_not_confirmed",
        "placement_not_confirmed",
        "open_not_confirmed",
        "unlock_not_confirmed",
        "episode_ended",
    }

    def __init__(
        self,
        runtime,
        relevance_scorer=None,
        skill_library=None,
        max_actions=450,
        max_skill_actions=300,
    ):
        self.runtime = runtime
        self.library = skill_library or SkillLibraryV1(runtime)
        self.dependencies = TaskDependencyResolver()
        self.feasibility = SkillFeasibilityEvaluator(runtime, self.dependencies)
        self.relevance = relevance_scorer or KeywordRelevanceScorer()
        self.max_actions = int(max_actions)
        self.max_skill_actions = int(max_skill_actions)
        self.history = []
        self.completed_ids = set()
        # call id -> observation_count at which the call failed. A later new
        # observation allows a retry; otherwise a failed call cannot spin forever.
        self.blocked_ids = {}
        self._queued = {}
        self._run_start_actions = runtime.action_count

    def _refresh_blocked(self):
        current_observation = self.runtime.belief.observation_count
        self.blocked_ids = {
            call_id: observed_at
            for call_id, observed_at in self.blocked_ids.items()
            if observed_at == current_observation
        }

    def _candidate_pool(self, task):
        self._refresh_blocked()

        for call in self.dependencies.candidate_calls(task, self.runtime.belief):
            self._queued.setdefault(call["id"], call)

        enriched = []
        for call in list(self._queued.values()):
            call_id = call["id"]
            if call_id in self.completed_ids or call_id in self.blocked_ids:
                continue

            assessment = self.feasibility.evaluate(call)
            if assessment["feasibility"] <= 0:
                for prerequisite in assessment["prerequisites"]:
                    self._queued.setdefault(prerequisite["id"], prerequisite)
            enriched.append((call, assessment))

        # Include newly discovered prerequisites in the same selection round.
        for call in list(self._queued.values()):
            call_id = call["id"]
            if (
                call_id in self.completed_ids
                or call_id in self.blocked_ids
                or any(existing["id"] == call_id for existing, _ in enriched)
            ):
                continue
            enriched.append((call, self.feasibility.evaluate(call)))
        return enriched

    def _execute_skill_to_terminal(self, call):
        """Run one selected skill controller until a terminal result."""
        start_actions = self.runtime.action_count
        trace = []
        result = None

        for _ in range(self.max_skill_actions):
            if self.runtime.terminated or self.runtime.truncated:
                result = {
                    "success": False,
                    "status": "episode_ended",
                    "reason": "The episode ended while the skill was executing.",
                    "skill": call["name"],
                    "actions_executed": 0,
                }
                break

            if self.runtime.action_count - self._run_start_actions >= self.max_actions:
                result = {
                    "success": False,
                    "status": "max_actions_reached",
                    "reason": "The global low-level action budget was exhausted.",
                    "skill": call["name"],
                    "actions_executed": 0,
                }
                break

            before_actions = self.runtime.action_count
            result = self.library.execute(call)
            after_actions = self.runtime.action_count
            status = result.get("status", "unknown")

            trace.append({
                "controller_step": len(trace) + 1,
                "status": status,
                "action": result.get("action"),
                "reason": result.get("reason", ""),
                "actions_executed": after_actions - before_actions,
                "estimated_position": self.runtime.belief.summary()["estimated_position"],
            })

            if status not in self.PROGRESS_STATUSES:
                break

            # A phase transition such as pickup -> placement can legitimately
            # change skill state without consuming a primitive action.
            if after_actions == before_actions and status != "phase_complete":
                result = {
                    **result,
                    "success": False,
                    "status": "skill_stalled",
                    "reason": "The controller reported progress but executed no primitive action.",
                }
                break
        else:
            result = {
                "success": False,
                "status": "skill_timeout",
                "reason": f"Skill exceeded {self.max_skill_actions} controller steps.",
                "skill": call["name"],
                "actions_executed": 0,
            }

        result = dict(result or {})
        result["controller_steps"] = len(trace)
        result["low_level_actions"] = self.runtime.action_count - start_actions
        result["execution_trace"] = trace
        return result

    def step(self, task):
        if self.runtime.action_count - self._run_start_actions >= self.max_actions:
            return {"success": False, "status": "max_actions_reached", "history": self.history}

        pool = self._candidate_pool(task)
        if not pool:
            return {"success": False, "status": "no_feasible_candidate", "history": self.history}

        calls = [call for call, _ in pool]
        # Exactly one relevance-scoring call per high-level skill selection.
        relevance = self.relevance.score(task, calls, self.runtime.belief.summary())
        scored = []

        for call, assessment in pool:
            item = relevance.get(
                call["id"],
                {"relevance": 0.0, "rationale": "No relevance score."},
            )
            rel = float(item["relevance"])
            feasible = float(assessment["feasibility"])
            scored.append({
                "call": call,
                "relevance": rel,
                "feasibility": feasible,
                "score": rel * feasible,
                "rationale": item.get("rationale", ""),
                "feasibility_reason": assessment["reason"],
            })

        scored.sort(key=lambda item: (-item["score"], item["call"]["id"]))
        selected = next((item for item in scored if item["score"] > 0), None)
        if selected is None:
            return {
                "success": False,
                "status": "no_feasible_candidate",
                "history": self.history,
                "candidates": [
                    {
                        "id": item["call"]["id"],
                        "name": item["call"]["name"],
                        "relevance": item["relevance"],
                        "feasibility": item["feasibility"],
                        "reason": item["feasibility_reason"],
                    }
                    for item in scored
                ],
            }

        call = selected["call"]
        result = self._execute_skill_to_terminal(call)
        record = {
            "round": len(self.history) + 1,
            "call": call,
            "relevance": selected["relevance"],
            "feasibility": selected["feasibility"],
            "score": selected["score"],
            "relevance_rationale": selected["rationale"],
            "feasibility_reason": selected["feasibility_reason"],
            "result": result,
            "belief": self.runtime.belief.summary(),
        }
        self.history.append(record)

        status = result.get("status", "unknown")
        if result.get("success") and status not in self.PROGRESS_STATUSES:
            self.completed_ids.add(call["id"])
        elif status in self.TERMINAL_FAILURE_STATUSES or status == "max_actions_reached":
            self.blocked_ids[call["id"]] = self.runtime.belief.observation_count

        return {
            "success": bool(result.get("success")),
            "status": status,
            "record": record,
        }

    def run(self, task, max_high_level_steps=100):
        self._run_start_actions = self.runtime.action_count
        status = "planner_stopped"

        for _ in range(int(max_high_level_steps)):
            if self.runtime.terminated or self.runtime.truncated:
                status = "episode_ended"
                break
            if self.runtime.action_count - self._run_start_actions >= self.max_actions:
                status = "max_actions_reached"
                break

            result = self.step(task)
            status = result.get("status", "unknown")
            # A skill-level search failure is not necessarily a task failure.
            # For example, Find(ball) may exhaust the currently reachable room
            # because a closed door separates the ball from the agent. Re-score
            # candidates so the planner can navigate to/open the door and retry.
            if status in {
                "no_feasible_candidate",
                "exploration_exhausted",
                "exploration_stalled",
                "episode_ended",
                "max_actions_reached",
                "skill_timeout",
                "skill_stalled",
            }:
                break

        success = self._task_success(task)
        return {
            "success": success,
            "status": "task_complete" if success else status,
            "high_level_steps": len(self.history),
            "low_level_actions": self.runtime.action_count - self._run_start_actions,
            "completed_calls": sorted(self.completed_ids),
            "blocked_calls": sorted(self.blocked_ids),
            "history": self.history,
        }

    def _task_success(self, task):
        """Belief-side completion check; evaluation scripts must also check GT."""
        text = str(task).lower()
        done = set(self.runtime.belief.completed_subgoals)

        if any(word in text for word in ("place", "put", "drop", "set down")):
            if self.runtime.belief.inventory_status != "empty" or not ({"place", "drop"} & done):
                return False

        if "unlock" in text and not ({"unlock_door", "open_door"} & done):
            return False
        if "open" in text and "door" in text and not ({"open_door", "unlock_door"} & done):
            return False
        if any(word in text for word in ("pick", "take", "grab", "fetch")):
            if not ({"pick", "pickup"} & done):
                return False

        goal_entry_intent = any(
            phrase in text
            for phrase in ("reach", "enter", "arrive at", "get to", "go to", "navigate to", "cross to")
        )
        if "goal" in text and goal_entry_intent:
            if "reach" not in done and not (
                self.runtime.terminated and self.runtime.last_reward > 0
            ):
                return False

        # Finding is an evidence requirement, not a requirement that every
        # object remain visible forever. Picked-up objects disappear from their
        # map cells, and the agent may overwrite a door/goal cell while crossing
        # it. Accept current map evidence, a matching held object, or a
        # successful Find result recorded earlier in this run.
        if any(word in text for word in ("find", "search", "locate")):
            requested = parse_target_specs(task)
            belief = self.runtime.belief
            held = getattr(belief, "held_object", None)
            for spec in requested:
                known_now = any(
                    spec.matches(cell) for cell in belief.known_map.values()
                )
                held_now = bool(held and spec.matches(held))
                found_earlier = any(
                    record.get("call", {}).get("name") == "Find"
                    and record.get("result", {}).get("success")
                    and spec.matches(record.get("result", {}).get("target", {}))
                    for record in self.history
                )
                if not (known_now or held_now or found_earlier):
                    return False

        return bool(self.history)
