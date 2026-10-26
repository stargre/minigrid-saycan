from skills.approach_door import ApproachDoorSkill
from skills.interaction import OpenDoorSkill
from skills.manipulation import PickSkill, PlaceSkill
from skills.observation_explorer import ObservationExplorer


class ObservationTaskController:
    """
    观测驱动的完整任务控制器。

    不读取环境的全局坐标、完整地图或 info。
    每次决策至多执行一个底层动作。
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.belief = runtime.belief

        self.explorer = ObservationExplorer(runtime)
        self.approacher = ApproachDoorSkill(
            runtime,
            color="yellow",
        )
        self.opener = OpenDoorSkill(
            runtime,
            color="yellow",
        )
        self.picker = PickSkill(
            runtime,
            color="red",
        )
        self.placer = PlaceSkill(runtime)

    def is_task_complete(self):
        return (
            "pick" in self.belief.completed_subgoals
            and "open_door"
            in self.belief.completed_subgoals
            and "place" in self.belief.completed_subgoals
            and self.belief.inventory_status == "empty"
        )

    @staticmethod
    def _tag(result, skill_name):
        result = dict(result)
        result["skill"] = skill_name
        return result

    def _closed_door_exists(self):
        doors = self.belief.find_objects(
            "door",
            color="yellow",
            state="closed",
        )
        return bool(doors)

    def _handle_closed_door(self):
        """
        接近观测到的关闭门。
        到达门前后，才调用开门技能。
        """
        result = self.approacher.step()

        if result.get("status") == "target_adjacent":
            result = self.opener.step()

            return self._tag(
                result,
                "open_door",
            )

        if result.get("status") == "action_executed":
            return self._tag(
                result,
                "approach_door",
            )

        # 没有已知路径时，先探索以补充地图，
        # 不使用隐藏坐标替代导航。
        result = self.explorer.step()

        return self._tag(
            result,
            "explore",
        )

    def step(self):
        if self.is_task_complete():
            return {
                "success": True,
                "status": "task_complete",
                "skill": "controller",
                "actions_executed": 0,
            }

        if self.runtime.terminated or self.runtime.truncated:
            return {
                "success": False,
                "status": "episode_ended",
                "skill": "controller",
                "reason": (
                    "The episode ended before the task "
                    "was confirmed complete."
                ),
                "actions_executed": 0,
            }

        inventory = self.belief.inventory_status
        closed_door = self._closed_door_exists()

        # --------------------------------------------------
        # Case A: 已经持有红球，需要找到并到达放置位置。
        # --------------------------------------------------
        if inventory == "red_ball":
            goals = self.belief.find_objects(
                "goal",
                color="green",
            )

            if goals:
                result = self.placer.step()

                if result.get("status") in {
                    "action_executed",
                    "skill_complete",
                }:
                    return self._tag(
                        result,
                        "place",
                    )

                if result.get("status") == "target_occupied":
                    return self._tag(
                        result,
                        "place",
                    )

            # 目标尚未观测到，或已知路径不足。
            if closed_door:
                return self._handle_closed_door()

            return self._tag(
                self.explorer.step(),
                "explore",
            )

        # --------------------------------------------------
        # Case B: 双手为空，优先尝试拾取已观测到的红球。
        # --------------------------------------------------
        if inventory == "empty":
            balls = self.belief.find_objects(
                "ball",
                color="red",
            )

            if balls:
                result = self.picker.step()

                if result.get("status") in {
                    "action_executed",
                    "skill_complete",
                }:
                    return self._tag(
                        result,
                        "pick",
                    )

            # 红球不可达时，如果已知关闭门，则先打开门。
            if closed_door:
                return self._handle_closed_door()

            # 目标未知或地图尚不完整时继续探索。
            return self._tag(
                self.explorer.step(),
                "explore",
            )

        # 未知持有状态时，不贸然执行拾取或放置。
        if closed_door:
            return self._handle_closed_door()

        return self._tag(
            self.explorer.step(),
            "explore",
        )

    def run(
        self,
        max_low_level_actions=450,
        log_every=10,
    ):
        """
        运行完整任务。

        max_low_level_actions 默认 450，
        低于环境配置的 500 步上限。
        """
        start_count = self.runtime.action_count
        decisions = 0

        log_every = max(1, int(log_every))
        next_log = log_every
        max_decisions = max_low_level_actions * 4

        for _ in range(max_decisions):
            if self.is_task_complete():
                break

            actions_used = (
                self.runtime.action_count - start_count
            )

            if actions_used >= max_low_level_actions:
                break

            if self.runtime.terminated or self.runtime.truncated:
                break

            result = self.step()
            decisions += 1

            actions_used = (
                self.runtime.action_count - start_count
            )

            if actions_used >= next_log:
                print(
                    f"[Progress] actions={actions_used}, "
                    f"skill={result.get('skill')}, "
                    f"status={result.get('status')}, "
                    f"position={list(self.belief.agent_pos)}, "
                    f"known_cells={len(self.belief.known_map)}, "
                    f"inventory={self.belief.inventory_status}, "
                    f"completed={self.belief.completed_subgoals}"
                )

                next_log = (
                    actions_used // log_every + 1
                ) * log_every

            if result.get("status") in {
                "exploration_exhausted",
                "exploration_stalled",
                "requires_key",
                "target_occupied",
                "not_holding_object",
                "target_not_traversable",
            }:
                break

        success = self.is_task_complete()

        if success:
            status = "task_complete"
        elif (
            self.runtime.terminated
            or self.runtime.truncated
        ):
            status = "episode_ended_before_confirmation"
        elif (
            self.runtime.action_count - start_count
            >= max_low_level_actions
        ):
            status = "max_actions_reached"
        else:
            status = "planner_stopped"

        return {
            "success": success,
            "status": status,
            "decisions": decisions,
            "low_level_actions": (
                self.runtime.action_count - start_count
            ),
            "inventory_status": (
                self.belief.inventory_status
            ),
            "completed_subgoals": list(
                self.belief.completed_subgoals
            ),
            "known_cells": len(
                self.belief.known_map
            ),
        }
