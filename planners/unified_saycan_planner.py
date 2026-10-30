"""SayCan-style planner with persistent closed-loop skill execution.

The LLM (or offline scorer) selects a semantic skill. Deterministic controllers
execute primitive actions through one runtime gateway. Recoverable skill
failures trigger replanning; hard episode/budget failures terminate the run.
"""
import os
import re

from skills.dependency import TaskDependencyResolver
from skills.feasibility import SkillFeasibilityEvaluator
from skills.unified_library import SkillLibraryV1
from skills.target import PICKABLE_OBJECTS, parse_target_specs
from llm.skill_relevance import KeywordRelevanceScorer


_FIND_CLAUSE = re.compile(
    r"\b(?:find|search(?:\s+for)?|locate)\s+(.*?)"
    r"(?=(?:\s*,?\s*(?:(?:and|then)\s+)?"
    r"(?:find|search|locate|unlock|open|reach|enter|go(?:\s+to)?|"
    r"navigate(?:\s+to)?|pick(?:\s+up)?|take|grab|place|put|drop|"
    r"move|cross|finish|complete)\b)|[.;]|$)",
    re.IGNORECASE,
)


def _explicit_find_targets(task):
    targets, seen = [], set()
    for match in _FIND_CLAUSE.finditer(str(task)):
        for spec in parse_target_specs(match.group(1)):
            if spec.key not in seen:
                targets.append(spec)
                seen.add(spec.key)
    return targets


class UnifiedSayCanPlanner:
    PROGRESS_STATUSES = {"action_executed", "searching", "phase_complete"}
    TERMINAL_FAILURE_STATUSES = {
        "target_not_found", "exploration_exhausted", "exploration_stalled",
        "skill_timeout", "skill_stalled", "invalid_arguments",
        "unsupported_relation", "not_pickable", "no_known_path",
        "requires_key", "gripper_not_empty", "not_holding_object",
        "placement_cell_unknown", "placement_cell_occupied",
        "pickup_not_confirmed", "placement_not_confirmed",
        "open_not_confirmed", "unlock_not_confirmed", "episode_ended",
    }
    # These failures end the current skill, not the whole task. The planner
    # blocks that exact call until new observations arrive and then re-scores.
    RECOVERABLE_FAILURE_STATUSES = {
        "target_not_found", "exploration_exhausted", "exploration_stalled",
        "no_known_path", "requires_key", "placement_cell_unknown",
        "placement_cell_occupied", "no_feasible_candidate",
    }
    HARD_STOP_STATUSES = {
        "episode_ended", "max_actions_reached", "skill_timeout", "skill_stalled",
    }

    def __init__(self, runtime, relevance_scorer=None, skill_library=None,
                 max_actions=450, max_skill_actions=300):
        self.runtime = runtime
        self.library = skill_library or SkillLibraryV1(runtime)
        self.dependencies = TaskDependencyResolver()
        self.feasibility = SkillFeasibilityEvaluator(runtime, self.dependencies)
        self.relevance = relevance_scorer or KeywordRelevanceScorer()
        self.max_actions = int(max_actions)
        self.max_skill_actions = int(max_skill_actions)
        self.history = []
        self.completed_ids = set()
        self.blocked_ids = {}
        self._queued = {}
        self._run_start_actions = runtime.action_count

    def _refresh_blocked(self):
        current_observation = self.runtime.belief.observation_count
        self.blocked_ids = {
            call_id: observed_at for call_id, observed_at in self.blocked_ids.items()
            if observed_at == current_observation
        }

    def _candidate_pool(self, task):
        self._refresh_blocked()
        for call in self.dependencies.candidate_calls(task, self.runtime.belief):
            self._queued.setdefault(call["id"], call)

        enriched = []
        for call in list(self._queued.values()):
            if call["id"] in self.completed_ids or call["id"] in self.blocked_ids:
                continue
            assessment = self.feasibility.evaluate(call)
            if assessment["feasibility"] <= 0:
                for prerequisite in assessment["prerequisites"]:
                    self._queued.setdefault(prerequisite["id"], prerequisite)
            enriched.append((call, assessment))

        for call in list(self._queued.values()):
            if (call["id"] in self.completed_ids or call["id"] in self.blocked_ids
                    or any(existing["id"] == call["id"] for existing, _ in enriched)):
                continue
            enriched.append((call, self.feasibility.evaluate(call)))
        return enriched

    def _execute_skill_to_terminal(self, call):
        start_actions = self.runtime.action_count
        trace, result = [], None
        for _ in range(self.max_skill_actions):
            if self.runtime.terminated or self.runtime.truncated:
                result = {"success": False, "status": "episode_ended",
                          "reason": "The episode ended while the skill was executing.",
                          "skill": call["name"], "actions_executed": 0}
                break
            if self.runtime.action_count - self._run_start_actions >= self.max_actions:
                result = {"success": False, "status": "max_actions_reached",
                          "reason": "The global low-level action budget was exhausted.",
                          "skill": call["name"], "actions_executed": 0}
                break

            before = self.runtime.action_count
            result = self.library.execute(call)
            after = self.runtime.action_count
            status = result.get("status", "unknown")
            trace.append({
                "controller_step": len(trace) + 1, "status": status,
                "action": result.get("action"), "reason": result.get("reason", ""),
                "actions_executed": after - before,
                "estimated_position": self.runtime.belief.summary()["estimated_position"],
            })
            if status not in self.PROGRESS_STATUSES:
                break
            if after == before and status != "phase_complete":
                result = {**result, "success": False, "status": "skill_stalled",
                          "reason": "Controller reported progress without a primitive action."}
                break
        else:
            result = {"success": False, "status": "skill_timeout",
                      "reason": f"Skill exceeded {self.max_skill_actions} controller steps.",
                      "skill": call["name"], "actions_executed": 0}

        result = dict(result or {})
        result["controller_steps"] = len(trace)
        result["low_level_actions"] = self.runtime.action_count - start_actions
        result["execution_trace"] = trace
        return result

    @staticmethod
    def _is_navigation_only_done_task(task):
        """Limit forced Done priority to navigation-only benchmark tasks."""
        text = str(task).lower()
        movement_intent = any(
            phrase in text
            for phrase in (
                "reach", "enter", "arrive at", "get to", "go to",
                "navigate to", "cross to",
            )
        )
        if not movement_intent:
            return False

        # Composite tasks must retain planner flexibility: an intermediate
        # Done candidate must not pre-empt opening/unlocking, acquisition,
        # placement, or an explicitly requested search.
        if any(
            word in text
            for word in (
                "open", "unlock", "pick", "take", "grab", "carry", "fetch",
                "place", "put", "drop", "set down", "find", "search", "locate",
            )
        ) or re.search(r"\bget\s+(?:a|an|the|it|them)\b", text):
            return False

        return any(
            spec.object_name == "door" or spec.object_name in PICKABLE_OBJECTS
            for spec in parse_target_specs(task)
        )

    def step(self, task):
        if self.runtime.action_count - self._run_start_actions >= self.max_actions:
            return {"success": False, "status": "max_actions_reached", "history": self.history}

        pool = self._candidate_pool(task)
        if not pool:
            return {"success": False, "status": "no_feasible_candidate", "history": self.history}

        # Ask the LLM only about skills that the deterministic feasibility
        # evaluator says can execute now. Infeasible calls remain available as
        # prerequisite sources, but cannot waste tokens or dominate relevance.
        feasible_pool = [(call, assessment) for call, assessment in pool
                         if float(assessment["feasibility"]) > 0]
        if not feasible_pool:
            return {
                "success": False, "status": "no_feasible_candidate",
                "history": self.history,
                "candidates": [{
                    "id": call["id"], "name": call["name"], "relevance": 0.0,
                    "feasibility": float(assessment["feasibility"]),
                    "reason": assessment["reason"],
                } for call, assessment in pool],
            }

        calls = [call for call, _ in feasible_pool]
        relevance_source = type(self.relevance).__name__
        relevance_error = None
        try:
            relevance = self.relevance.score(task, calls, self.runtime.belief.summary())
        except Exception as exc:
            # Keep the task running when a remote scorer times out or returns
            # malformed data. The fallback is recorded explicitly in history.
            relevance = KeywordRelevanceScorer().score(
                task, calls, self.runtime.belief.summary()
            )
            error_text = str(exc)
            api_key = os.getenv("DASHSCOPE_API_KEY", "")
            if api_key:
                error_text = error_text.replace(api_key, "[REDACTED]")
            relevance_error = f"{type(exc).__name__}: {error_text[:240]}"
            relevance_source = f"KeywordRelevanceScorer (API error: {type(exc).__name__})"
        scored = []
        for call, assessment in feasible_pool:
            item = relevance.get(call["id"], {"relevance": 0.0, "rationale": "Missing relevance score."})
            rel = float(item["relevance"])
            feasible = float(assessment["feasibility"])
            scored.append({
                "call": call, "relevance": rel, "feasibility": feasible,
                "score": rel * feasible, "rationale": item.get("rationale", ""),
                "feasibility_reason": assessment["reason"],
            })

        scored.sort(key=lambda item: (-item["score"], item["call"]["id"]))

        # A feasible Done call is the task's terminal interaction. Once its
        # preconditions are met, do not let semantic relevance select Explore
        # or another skill that could move the agent away from the target.
        # DoneSkill still performs the final orientation check, and task
        # success is verified separately from the environment's terminal reward.
        terminal = (
            next(
                (item for item in scored if item["call"].get("name") == "Done"),
                None,
            )
            if self._is_navigation_only_done_task(task)
            else None
        )
        terminal_priority = terminal is not None
        if terminal_priority:
            selected = terminal
            selected["rationale"] = (
                str(selected.get("rationale", ""))
                + " [priority: feasible terminal Done action]"
            ).strip()
            zero_score_fallback = False
        else:
            # A zero relevance response must not dead-end an otherwise feasible
            # planner. Use the strongest feasible candidate as a deterministic
            # fallback, and make the fallback explicit in the history.
            selected = next((item for item in scored if item["score"] > 0), None)
            zero_score_fallback = selected is None
            if selected is None:
                selected = max(
                    scored,
                    key=lambda item: (item["feasibility"], item["call"]["id"]),
                )
                selected["rationale"] = (
                    str(selected.get("rationale", ""))
                    + " [fallback: all feasible candidates had zero relevance]"
                ).strip()

        call = selected["call"]
        result = self._execute_skill_to_terminal(call)
        candidate_rankings = [
            {
                "id": item["call"]["id"],
                "name": item["call"]["name"],
                "args": item["call"].get("args", {}),
                "relevance": item["relevance"],
                "feasibility": item["feasibility"],
                "score": item["score"],
                "rationale": item["rationale"],
                "feasibility_reason": item["feasibility_reason"],
            }
            for item in scored
        ]
        selection_policy = (
            "terminal_priority" if terminal_priority
            else "zero_score_fallback" if zero_score_fallback
            else "max_relevance_times_feasibility"
        )
        record = {
            "round": len(self.history) + 1, "call": call,
            "selected_candidate_id": call["id"],
            "selection_policy": selection_policy,
            "candidate_rankings": candidate_rankings,
            "relevance": selected["relevance"], "feasibility": selected["feasibility"],
            "score": selected["score"], "zero_score_fallback": zero_score_fallback,
            "terminal_priority": terminal_priority,
            "relevance_rationale": selected["rationale"],
            "relevance_source": relevance_source,
            "relevance_error": relevance_error,
            "feasibility_reason": selected["feasibility_reason"],
            "result": result, "belief": self.runtime.belief.summary(),
        }
        self.history.append(record)
        status = result.get("status", "unknown")
        if result.get("success") and status not in self.PROGRESS_STATUSES:
            self.completed_ids.add(call["id"])
        elif status in self.TERMINAL_FAILURE_STATUSES or status == "max_actions_reached":
            self.blocked_ids[call["id"]] = self.runtime.belief.observation_count

        return {"success": bool(result.get("success")), "status": status, "record": record}

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
            if self._task_success(task):
                status = "task_complete"
                break
            if status in self.HARD_STOP_STATUSES:
                break
            # Skill-level search failures are recoverable. step() blocks the
            # failed call for the current observation; the next loop iteration
            # tries other feasible candidates. A no-candidate result is only
            # terminal if replanning cannot identify another feasible action.
            if status == "no_feasible_candidate":
                status = self._try_recovery_candidate(task)
                if status == "recovery_exhausted":
                    break

        success = self._task_success(task)
        return {
            "success": success, "status": "task_complete" if success else status,
            "high_level_steps": len(self.history),
            "low_level_actions": self.runtime.action_count - self._run_start_actions,
            "completed_calls": sorted(self.completed_ids),
            "blocked_calls": sorted(self.blocked_ids), "history": self.history,
        }

    def _try_recovery_candidate(self, task):
        """Try a bounded, observation-grounded recovery action if one exists."""
        pool = self._candidate_pool(task)
        feasible = [(call, assessment) for call, assessment in pool
                    if float(assessment["feasibility"]) > 0]
        if not feasible:
            return "recovery_exhausted"
        # Prefer Explore when it can produce new observations; otherwise choose
        # the feasible candidate with the highest feasibility and stable ID.
        explore = [(c, a) for c, a in feasible if c.get("name") == "Explore"]
        call, assessment = (explore or feasible)[0]
        result = self._execute_skill_to_terminal(call)
        record = {
            "round": len(self.history) + 1, "call": call,
            "selected_candidate_id": call["id"],
            "selection_policy": "deterministic_recovery",
            "candidate_rankings": [
                {
                    "id": candidate["id"],
                    "name": candidate["name"],
                    "args": candidate.get("args", {}),
                    "relevance": None,
                    "feasibility": float(candidate_assessment["feasibility"]),
                    "score": None,
                    "rationale": "Not scored by Qwen; selected by deterministic recovery.",
                    "feasibility_reason": candidate_assessment["reason"],
                }
                for candidate, candidate_assessment in feasible
            ],
            "relevance": 0.0, "feasibility": float(assessment["feasibility"]),
            "score": 0.0, "zero_score_fallback": True,
            "relevance_rationale": "Deterministic recovery: no feasible candidate received a positive relevance score.",
            "feasibility_reason": assessment["reason"], "result": result,
            "belief": self.runtime.belief.summary(),
        }
        self.history.append(record)
        status = result.get("status", "unknown")
        if result.get("success") and status not in self.PROGRESS_STATUSES:
            self.completed_ids.add(call["id"])
        elif status in self.TERMINAL_FAILURE_STATUSES or status == "max_actions_reached":
            self.blocked_ids[call["id"]] = self.runtime.belief.observation_count
        return status

    def _task_success(self, task):
        text = str(task).lower()
        done = set(self.runtime.belief.completed_subgoals)
        if any(word in text for word in ("place", "put", "drop", "set down")):
            if self.runtime.belief.inventory_status != "empty" or not ({"place", "drop"} & done):
                return False
        # Conditional instructions such as "open the door if needed" do not
        # impose an unconditional door-open subgoal. If opening is actually
        # necessary to complete placement/navigation, the environment-grounded
        # skill chain will still have to open it before the task can succeed.
        door_action_optional = bool(
            re.search(
                r"\b(?:if|when)\s+(?:it\s+is\s+)?(?:needed|necessary)\b",
                text,
            )
        )
        if "unlock" in text and not door_action_optional and not ({"unlock_door", "open_door"} & done):
            return False
        if "open" in text and "door" in text and not door_action_optional and not ({"open_door", "unlock_door"} & done):
            return False
        target_specs = parse_target_specs(task)
        pickup_targets = [
            spec for spec in target_specs
            if spec.object_name in PICKABLE_OBJECTS
        ]
        get_pickable_intent = bool(
            re.search(r"\bget\s+(?:a|an|the)\s+", text)
            and pickup_targets
        )
        release_intent = any(
            word in text for word in ("place", "put", "drop", "set down")
        )
        acquisition_intent = (
            any(word in text for word in ("pick", "take", "grab", "fetch"))
            or get_pickable_intent
        )
        if acquisition_intent:
            if not ({"pick", "pickup"} & done):
                return False
            held = getattr(self.runtime.belief, "held_object", None)
            if (
                pickup_targets
                and not release_intent
                and not any(spec.matches(held) for spec in pickup_targets)
            ):
                return False
        goal_entry_intent = any(
            phrase in text for phrase in
            ("reach", "enter", "arrive at", "get to", "go to", "navigate to", "cross to")
        )

        # Official MiniGrid GoToDoor missions require a terminal environment
        # success, not merely a successful Find/NavigateTo skill. The Done
        # action is the environment's task-completion interaction.
        door_navigation_only = (
            "door" in text
            and goal_entry_intent
            and not any(
                word in text
                for word in (
                    "open", "unlock", "pick", "take", "grab", "place", "put",
                    "drop", "find", "search", "locate",
                )
            )
        )
        if door_navigation_only:
            return bool(
                self.runtime.terminated
                and not self.runtime.truncated
                and self.runtime.last_reward > 0
            )

        # Official GoToObject missions use the same terminal Done interaction
        # as GoToDoor, but their target is a pickable object (key, ball, or box).
        # Merely finding or navigating next to the object is not task success;
        # the environment must confirm a positive-reward terminal transition.
        object_navigation_only = (
            goal_entry_intent
            and any(spec.object_name in PICKABLE_OBJECTS for spec in target_specs)
            and not acquisition_intent
            and not release_intent
            and not any(word in text for word in ("find", "search", "locate"))
        )
        if object_navigation_only:
            done_issued = any(
                record.get("call", {}).get("name") == "Done"
                for record in self.history
            )
            return bool(
                done_issued
                and self.runtime.terminated
                and not self.runtime.truncated
                and self.runtime.last_reward > 0
            )

        if "goal" in text and goal_entry_intent:
            if "reach" not in done and not (self.runtime.terminated and self.runtime.last_reward > 0):
                return False

        if any(word in text for word in ("find", "search", "locate")):
            belief = self.runtime.belief
            held = getattr(belief, "held_object", None)
            for spec in _explicit_find_targets(task):
                known_now = any(spec.matches(cell) for cell in belief.known_map.values())
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
