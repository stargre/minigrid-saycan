"""Search-oriented skills: Explore and Find."""
from skills.base import UnifiedSkill
from skills.observation_explorer import ObservationExplorer
from skills.target import TargetSpec


class ExploreSkill(UnifiedSkill):
    name = "Explore"

    def __init__(self, runtime):
        super().__init__(runtime)
        self.explorer = ObservationExplorer(runtime)

    def step(self, **kwargs):
        result = self.explorer.step()
        result["skill"] = self.name
        return result


class FindSkill(UnifiedSkill):
    name = "Find"

    def step(self, target, **kwargs):
        spec = TargetSpec.from_value(target)
        pos, cell = self._nearest(spec.to_dict())
        if pos is not None:
            return self._result(
                True, "target_found",
                "Target is supported by observation evidence in the belief map.",
                target=spec.to_dict(), evidence={"position": list(pos), "cell": dict(cell)},
            )
        result = self.explorer.step()
        result["skill"] = self.name
        result["target"] = spec.to_dict()
        if result.get("status") == "exploration_exhausted":
            result["success"] = False
            result["status"] = "target_not_found"
            result["reason"] = "No reachable unexplored frontier remains; target not found."
        elif result.get("status") == "action_executed":
            result["success"] = False
            result["status"] = "searching"
        return result

    def __init__(self, runtime):
        super().__init__(runtime)
        self.explorer = ObservationExplorer(runtime)
