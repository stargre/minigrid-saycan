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
import ast
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
HIDDEN_ATTRIBUTES = {
    "grid", "agent_pos", "agent_dir", "carrying",
    "door_pos", "ball_pos", "goal_pos", "target_pos", "unwrapped",
}
GROUND_TRUTH_INFO_KEYS = {
    "is_success", "ball_pos", "door_pos", "target_pos",
    "goal_pos", "ball_side",
}


def _attribute_chain(node):
    """Return a dotted AST attribute chain without evaluating source code."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attribute_chain(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _is_environment_receiver(chain):
    """Recognize env references used by the decision code."""
    return bool(chain) and (
        chain == "env"
        or chain.endswith(".env")
    )


def static_audit():
    """Inspect executable AST nodes, never comments/docstrings/string literals."""
    failures = []
    for relative in DECISION_FILES:
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"Missing decision module: {relative}")
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError as exc:
            failures.append(f"{relative}:{exc.lineno}: Python syntax error: {exc.msg}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                chain = _attribute_chain(node)
                receiver, _, attr = chain.rpartition(".")
                if attr in HIDDEN_ATTRIBUTES and _is_environment_receiver(receiver):
                    failures.append(
                        f"{relative}:{node.lineno}: direct hidden-state access: {chain}"
                    )
                if attr == "get_wrapper_attr":
                    failures.append(
                        f"{relative}:{node.lineno}: wrapper attribute oracle access: {chain}"
                    )

            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    receiver = _attribute_chain(func.value)
                    if func.attr == "step" and _is_environment_receiver(receiver):
                        failures.append(
                            f"{relative}:{node.lineno}: primitive action bypass: {receiver}.step(...)"
                        )
                    if func.attr == "get_wrapper_attr":
                        failures.append(
                            f"{relative}:{node.lineno}: wrapper attribute oracle access"
                        )

                # Catch simple dynamic access such as getattr(env, "grid").
                if (
                    isinstance(func, ast.Name)
                    and func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    receiver = _attribute_chain(node.args[0])
                    attr = node.args[1].value
                    if _is_environment_receiver(receiver) and attr in HIDDEN_ATTRIBUTES:
                        failures.append(
                            f"{relative}:{node.lineno}: dynamic hidden-state access: "
                            f"getattr({receiver}, {attr!r})"
                        )

            elif isinstance(node, ast.Subscript):
                value_chain = _attribute_chain(node.value)
                key_node = node.slice
                if (
                    value_chain in {"info", "_info"}
                    and isinstance(key_node, ast.Constant)
                    and key_node.value in GROUND_TRUTH_INFO_KEYS
                ):
                    failures.append(
                        f"{relative}:{node.lineno}: ground-truth info used in decisions: "
                        f"{value_chain}[{key_node.value!r}]"
                    )

    if failures:
        raise AssertionError(
            "Static hidden-state audit failed:\n  - " + "\n  - ".join(failures)
        )
    print(
        f"PASS static audit: {len(DECISION_FILES)} decision modules scanned "
        "(AST-only; comments and docstrings ignored)"
    )

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
