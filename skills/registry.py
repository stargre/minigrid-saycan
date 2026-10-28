"""Public skill registry and stable execution contract."""
from skills.doors import OpenDoorSkill, UnlockDoorSkill
from skills.interaction_skill import DoneSkill
from skills.manipulation_skills import MoveObstacleSkill, PickUpSkill, PlaceNearSkill
from skills.navigation_skills import NavigateToSkill, ReachSkill
from skills.search import ExploreSkill, FindSkill


class SkillLibraryV1:
    """Frozen skill registry and uniform execution contract.

    Contract: execute(call) -> {success, status, reason, skill,
    actions_executed, ...}. LLMs never issue primitive MiniGrid actions.
    """

    VERSION = "skill_library_v1"

    def __init__(self, runtime):
        self.runtime = runtime
        self.skills = {
            "Explore": ExploreSkill(runtime),
            "Find": FindSkill(runtime),
            "NavigateTo": NavigateToSkill(runtime),
            "Reach": ReachSkill(runtime),
            "PickUp": PickUpSkill(runtime),
            "OpenDoor": OpenDoorSkill(runtime),
            "UnlockDoor": UnlockDoorSkill(runtime),
            "PlaceNear": PlaceNearSkill(runtime),
            "MoveObstacle": MoveObstacleSkill(runtime),
            "Done": DoneSkill(runtime),
        }

    @property
    def names(self):
        return tuple(self.skills.keys())

    def execute(self, call: dict) -> dict:
        if not isinstance(call, dict) or "name" not in call:
            raise TypeError("call must be a mapping containing 'name'")
        name = str(call["name"])
        if name not in self.skills:
            return {
                "success": False, "status": "unknown_skill",
                "reason": f"Unknown skill: {name}", "skill": name,
                "actions_executed": 0,
            }
        args = dict(call.get("args", {}))
        try:
            result = self.skills[name].step(**args)
        except (TypeError, ValueError) as exc:
            return {
                "success": False, "status": "invalid_arguments",
                "reason": str(exc), "skill": name, "actions_executed": 0,
            }
        result.setdefault("skill", name)
        result.setdefault("actions_executed", 0)
        result.setdefault("success", False)
        result.setdefault("status", "unknown")
        result.setdefault("reason", "")
        return result
