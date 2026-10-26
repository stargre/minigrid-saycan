from minigrid.core.actions import Actions

from skills.navigation import ObservationNavigator


class PickSkill:
    """
    观测驱动的红球拾取技能。

    1. 从 BeliefState 中寻找已观测到的红球。
    2. 规划到红球相邻位置。
    3. 面向红球执行 pickup。
    4. 根据动作后的观测确认拾取成功。
    """

    def __init__(self, runtime, color="red"):
        self.runtime = runtime
        self.belief = runtime.belief
        self.color = color

        self.navigator = ObservationNavigator(runtime)
        self.target_pos = None

    def _find_target(self):
        if self.target_pos is not None:
            cell = self.belief.cell_at(self.target_pos)

            if (
                cell is not None
                and cell["object"] == "ball"
                and cell["color"] == self.color
            ):
                return self.target_pos

            self.target_pos = None
            self.navigator.reset_target()

        balls = self.belief.find_objects(
            "ball",
            color=self.color,
        )

        if not balls:
            return None

        start = tuple(self.belief.agent_pos)

        target_pos, _ = min(
            balls,
            key=lambda item: (
                abs(item[0][0] - start[0])
                + abs(item[0][1] - start[1])
            ),
        )

        self.target_pos = tuple(target_pos)
        return self.target_pos

    def step(self):
        if self.belief.inventory_status == "red_ball":
            return {
                "success": True,
                "status": "skill_complete",
                "reason": "The red ball is already held.",
                "actions_executed": 0,
            }

        if self.belief.inventory_status != "empty":
            return {
                "success": False,
                "status": "inventory_unknown",
                "reason": (
                    "The current inventory state is not "
                    "confirmed to be empty."
                ),
                "actions_executed": 0,
            }

        target = self._find_target()

        if target is None:
            return {
                "success": False,
                "status": "target_not_found",
                "reason": (
                    "The red ball has not been observed. "
                    "Explore the environment first."
                ),
                "actions_executed": 0,
            }

        result = self.navigator.step_to_adjacent(target)

        if result["status"] != "target_adjacent":
            result["skill"] = "pick"
            result["target_position"] = list(target)
            return result

        # 此时已经通过观测确认目标在正前方相邻格。
        action_result = self.runtime.execute_action(
            Actions.pickup
        )

        success = (
            self.belief.inventory_status == "red_ball"
            and "pick" in self.belief.completed_subgoals
        )

        if success:
            self.target_pos = None
            self.navigator.reset_target()

        return {
            "success": success,
            "status": (
                "skill_complete"
                if success
                else "pickup_not_confirmed"
            ),
            "reason": (
                "Pickup confirmed by the next observation."
                if success
                else (
                    "Pickup was attempted, but the updated "
                    "belief does not confirm that the ball "
                    "is being held."
                )
            ),
            "action": "pickup",
            "actions_executed": action_result[
                "actions_executed"
            ],
            "inventory_status": (
                self.belief.inventory_status
            ),
            "target_position": list(target),
            "terminated": action_result["terminated"],
        }


class PlaceSkill:
    """
    观测驱动的放置技能。

    目标由已观测到的绿色目标推导：
        target_pos = (goal_x, goal_y - 1)

    不读取环境内部的 target_pos。
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.belief = runtime.belief

        self.navigator = ObservationNavigator(runtime)
        self.goal_pos = None
        self.target_pos = None

    def _find_goal(self):
        goals = self.belief.find_objects(
            "goal",
            color="green",
        )

        if not goals:
            return self.goal_pos

        start = tuple(self.belief.agent_pos)

        goal_pos, _ = min(
            goals,
            key=lambda item: (
                abs(item[0][0] - start[0])
                + abs(item[0][1] - start[1])
            ),
        )

        self.goal_pos = tuple(goal_pos)
        self.target_pos = (
            self.goal_pos[0],
            self.goal_pos[1] - 1,
        )

        return self.goal_pos

    def step(self):
        if (
            "place" in self.belief.completed_subgoals
            and self.belief.inventory_status == "empty"
        ):
            return {
                "success": True,
                "status": "skill_complete",
                "reason": (
                    "Placement was previously confirmed "
                    "by observations."
                ),
                "actions_executed": 0,
            }

        if self.belief.inventory_status != "red_ball":
            return {
                "success": False,
                "status": "not_holding_object",
                "reason": (
                    "The red ball must be held before "
                    "the placement skill can execute."
                ),
                "actions_executed": 0,
            }

        if self._find_goal() is None:
            return {
                "success": False,
                "status": "goal_not_found",
                "reason": (
                    "The green goal has not been observed. "
                    "Explore the environment first."
                ),
                "actions_executed": 0,
            }

        target = self.target_pos
        cell = self.belief.cell_at(target)

        if cell is None:
            return {
                "success": False,
                "status": "target_not_known",
                "reason": (
                    "The placement cell above the goal "
                    "has not yet been observed."
                ),
                "target_position": list(target),
                "actions_executed": 0,
            }

        # 只允许将红球放到已知的空地。
        if cell["object"] not in {"empty", "floor"}:
            return {
                "success": False,
                "status": "target_occupied",
                "reason": (
                    "The placement cell is not known to "
                    "be empty."
                ),
                "target_position": list(target),
                "observed_cell": cell["object"],
                "actions_executed": 0,
            }

        result = self.navigator.step_to_adjacent(target)

        if result["status"] != "target_adjacent":
            result["skill"] = "place"
            result["target_position"] = list(target)
            return result

        # 导航器已确认目标格位于正前方。
        action_result = self.runtime.execute_action(
            Actions.drop
        )

        success = (
            self.belief.inventory_status == "empty"
            and "place" in self.belief.completed_subgoals
        )

        if success:
            self.navigator.reset_target()

        return {
            "success": success,
            "status": (
                "skill_complete"
                if success
                else "drop_not_confirmed"
            ),
            "reason": (
                "Placement above the green goal was "
                "confirmed by the updated belief."
                if success
                else (
                    "Drop was attempted, but placement "
                    "at the required target has not been "
                    "confirmed."
                )
            ),
            "action": "drop",
            "actions_executed": action_result[
                "actions_executed"
            ],
            "inventory_status": (
                self.belief.inventory_status
            ),
            "target_position": list(target),
            "terminated": action_result["terminated"],
        }
