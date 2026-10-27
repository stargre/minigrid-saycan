"""Hidden-state leakage audit for skill_library_v1.

The audit has two layers:
1. Static checks on decision-making modules for direct environment-state reads
   or primitive-action execution outside ObservationRuntime.
2. A guarded environment proxy that exposes only step(), rejects every other
   attribute access, and verifies every action originates from
   ObservationRuntime.execute_action.

Ground-truth inspection in this script is evaluation-only and happens after the
planner finishes; it is never passed into the planner or skill library.
"""
import inspect
import re
from pathlib import Path

from minigrid.core.world_object import Ball

from environments.saycan_grid import SayCanGridEnv
from llm.skill_relevance import KeywordRelevanceScorer
from planners.unified_saycan_planner import UnifiedSayCanPlanner
from skills.observation_runtime import ObservationRuntime


ROOT = Path(__file__).resolve().parents[1]
DECISION_FILES = [
    "skills/unified_library.py",
    "skills/observation_explorer.py",
    "skills/navigation.py",
    "skills/feasibility.py",
    "skills/dependency.py",
    "skills/target.py",
    "perception/belief_state.py",
    "planners/unified_saycan_planner.py",
    "llm/skill_relevance.py",
]
FORBIDDEN_PATTERNS = {
    "direct hidden-state attribute access": re.compile(
        r"\b(?:self\.)?env\.(?:grid|agent_pos|agent_dir|carrying|"
        r"door_pos|ball_pos|goal_pos|target_pos|unwrapped)\b"
        r"|\b(?:runtime|self\.runtime)\.env\."
        r"(?:grid|agent_pos|agent_dir|carrying|door_pos|ball_pos|"
        r"goal_pos|target_pos|unwrapped)\b"
    ),
    "primitive action bypasses ObservationRuntime": re.compile(
        r"\b(?:self\.)?env\.step\s*\("
        r"|\b(?:self\.)?runtime\.env\.step\s*\("
    ),
    "wrapper/environment oracle access": re.compile(
        r"\.get_wrapper_attr\s*\(|\b(?:env|runtime\.env)\.unwrapped\b"
    ),
    "label/ground-truth info used in decisions": re.compile(
        r"\b(?:info|_info)\s*\[\s*['\"]"
        r"(?:is_success|ball_pos|door_pos|target_pos|goal_pos|ball_side)['\"]"
    ),
}


def static_audit():
    failures = []
    for relative in DECISION_FILES:
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"Missing decision module: {relative}")
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for label, pattern in FORBIDDEN_PATTERNS.items():
                if pattern.search(line):
                    failures.append(
                        f"{relative}:{line_number}: {label}: {stripped.strip()}"
                    )
    if failures:
        raise AssertionError(
            "Static hidden-state audit failed:\n  - " + "\n  - ".join(failures)
        )
    print(f"PASS static audit: {len(DECISION_FILES)} decision modules scanned")


class StepOnlyEnv:
    """Proxy that exposes only step() and rejects privileged reads."""

    def __init__(self, inner):
        object.__setattr__(self, "_StepOnlyEnv__inner", inner)
        object.__setattr__(self, "step_calls", 0)

    def step(self, action):
        caller = inspect.currentframe().f_back
        if (
            caller is None
            or caller.f_code.co_name != "execute_action"
            or caller.f_globals.get("__name__") != "skills.observation_runtime"
        ):
            raise AssertionError(
                "Primitive action bypassed ObservationRuntime.execute_action."
            )
        object.__setattr__(self, "step_calls", self.step_calls + 1)
        inner = object.__getattribute__(self, "_StepOnlyEnv__inner")
        return inner.step(action)

    def __getattr__(self, name):
        raise AssertionError(
            f"Hidden-state access attempted through runtime.env.{name}; "
            "only env.step(action) is permitted."
        )


def ground_truth_success_for_evaluation_only(env):
    """Oracle used only after planning, never exposed to the decision process."""
    placed = env.grid.get(*env.target_pos)
    door = env.grid.get(*env.door_pos)
    return bool(
        isinstance(placed, Ball)
        and placed.color == "red"
        and getattr(door, "is_open", False)
        and getattr(env, "carrying", None) is None
    )


def guarded_end_to_end(seed):
    env = SayCanGridEnv(render_mode=None)
    try:
        obs, _reset_info = env.reset(seed=seed)
        guarded_env = StepOnlyEnv(env)
        runtime = ObservationRuntime(guarded_env)
        runtime.reset(obs, initial_inventory="empty")
        planner = UnifiedSayCanPlanner(
            runtime,
            relevance_scorer=KeywordRelevanceScorer(),
            max_actions=450,
            max_skill_actions=300,
        )
        task = (
            "Pick up the red ball, open the yellow door, "
            "and place it above the green goal."
        )
        result = planner.run(task, max_high_level_steps=100)
        ground_truth_success = ground_truth_success_for_evaluation_only(env)

        assert result.get("success"), (
            f"seed={seed}: planner failed under guarded env: "
            f"{result.get('status')}"
        )
        assert ground_truth_success, (
            f"seed={seed}: independent post-run environment verification failed"
        )
        assert guarded_env.step_calls == runtime.action_count, (
            f"seed={seed}: {guarded_env.step_calls} environment steps but "
            f"{runtime.action_count} runtime-counted actions"
        )
        print(
            f"PASS guarded end-to-end seed={seed}: "
            f"actions={runtime.action_count}, "
            f"high_level_steps={result.get('high_level_steps')}"
        )
    finally:
        env.close()


def main():
    static_audit()
    for seed in (0, 7, 42, 45, 46):
        guarded_end_to_end(seed)
    print("Hidden-state audit passed: no detected leakage in guarded runs.")


if __name__ == "__main__":
    main()
