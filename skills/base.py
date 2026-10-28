"""Shared base class for observation-grounded high-level skills."""
from skills.navigation import ObservationNavigator
from skills.target import TargetSpec


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class UnifiedSkill:
    name = "Skill"

    def __init__(self, runtime):
        self.runtime = runtime
        self.belief = runtime.belief
        self.navigator = ObservationNavigator(runtime)
        self.max_internal_steps = 1

    def _result(self, success, status, reason, **kwargs):
        return {
            "success": bool(success),
            "status": status,
            "reason": reason,
            "skill": self.name,
            "actions_executed": int(kwargs.pop("actions_executed", 0)),
            **kwargs,
        }

    def _positions(self, target):
        spec = TargetSpec.from_value(target)
        return [
            (tuple(pos), cell)
            for pos, cell in self.belief.known_map.items()
            if spec.matches(cell)
        ]

    def _nearest(self, target):
        items = self._positions(target)
        if not items:
            return None, None
        return min(items, key=lambda item: _dist(item[0], self.belief.agent_pos))
