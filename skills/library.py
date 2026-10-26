from minigrid.core.world_object import Ball, Door
from llm.qwen_relevance import QwenRelevanceScorer
from perception.observation_summary import observation_to_belief


class SkillLibrary:
    """统一管理技能的任务相关性、可行性和执行。"""

    def __init__(self, env, movement, picker, opener, placer):
        self.env = env
        self.base_env = env.unwrapped
        self.movement = movement
        self.picker = picker
        self.opener = opener
        self.placer = placer

        # LLM 只负责任务相关性评分
        self.relevance_scorer = QwenRelevanceScorer()

        # 仅根据成功的技能执行反馈更新，不读取隐藏状态
        self.completed_subgoals = []
        self.inventory_status = "empty"
        self.last_llm_scores = {}

    def _has_reachable_adjacent(self, target_pos, empty_only=False):
        x, y = target_pos
        start = tuple(
            int(v) for v in self.base_env.agent_pos
        )

        candidates = [
            (x, y - 1),
            (x - 1, y),
            (x + 1, y),
            (x, y + 1),
        ]

        for pos in candidates:
            px, py = pos

            if not (
                0 <= px < self.base_env.width
                and 0 <= py < self.base_env.height
            ):
                continue

            if empty_only and self.base_env.grid.get(px, py) is not None:
                continue

            if self.movement.find_path(start, pos) is not None:
                return True

        return False

    @staticmethod
    def _first_match_position(task, keywords):
        task = task.lower()
        positions = [
            task.find(keyword.lower())
            for keyword in keywords
            if task.find(keyword.lower()) >= 0
        ]
        return min(positions) if positions else None

    def score_candidates(self, task):
        """
        使用局部观测计算技能可行性；
        使用千问计算任务相关性。

        注意：这里只修正评分阶段。
        execute() 中的旧技能执行接口仍需后续改造。
        """
        obs = self.base_env.gen_obs()

        belief = observation_to_belief(
            obs,
            inventory_status=self.inventory_status,
            completed_subgoals=self.completed_subgoals,
        )

        visible = belief["visible_objects"]
        completed = set(self.completed_subgoals)
        inventory = self.inventory_status

        def is_adjacent(item):
            offset = item.get("relative_offset", {})
            forward = int(offset.get("forward", 999))
            lateral = int(offset.get("lateral", 999))
            return abs(forward) + abs(lateral) == 1

        red_balls = [
            item for item in visible
            if item["object"] == "ball"
            and item["color"] == "red"
        ]

        closed_yellow_doors = [
            item for item in visible
            if item["object"] == "door"
            and item["color"] == "yellow"
            and item.get("state") == "closed"
        ]

        # 只能确认目标已出现在局部观测中时，才考虑立即操作。
        pick_feasible = float(
            "pick" not in completed
            and inventory == "empty"
            and any(is_adjacent(item) for item in red_balls)
        )

        open_feasible = float(
            "open_door" not in completed
            and any(
                is_adjacent(item)
                for item in closed_yellow_doors
            )
        )

        # 尚未建立目标格的局部占用判断，因此保守地禁止放置。
        place_feasible = 0.0

        # 探索执行技能尚未实现，不能把它伪装成可执行动作。
        explore_feasible = 0.0

        candidates = [
            {"name": "pick", "feasibility": pick_feasible},
            {"name": "open_door", "feasibility": open_feasible},
            {"name": "place", "feasibility": place_feasible},
            {"name": "explore", "feasibility": explore_feasible},
        ]

        llm_scores = self.relevance_scorer.score(
            task=task,
            belief_summary=belief,
        )
        self.last_llm_scores = llm_scores

        for candidate in candidates:
            item = llm_scores[candidate["name"]]
            candidate["relevance"] = item["relevance"]
            candidate["rationale"] = item["rationale"]
            candidate["score"] = (
                candidate["relevance"]
                * candidate["feasibility"]
            )

        return candidates

    def _record_skill_result(self, skill_name, result):
        """仅根据技能执行结果更新记忆。"""
        if not result.get("success", False):
            return

        if skill_name not in self.completed_subgoals:
            self.completed_subgoals.append(skill_name)

        if skill_name == "pick":
            self.inventory_status = "red_ball"
        elif skill_name == "place":
            self.inventory_status = "empty"

    def execute(self, skill_name):
        """
        兼容现有技能接口。

        当前执行器仍使用环境内部坐标；
        这部分将在下一阶段改造成观测驱动执行。
        """
        base = self.base_env

        if skill_name == "pick":
            result = self.picker.pick(base.ball_pos)
        elif skill_name == "open_door":
            result = self.opener.open_door(base.door_pos)
        elif skill_name == "place":
            result = self.placer.place(base.target_pos)
        else:
            raise ValueError(f"Unknown skill: {skill_name}")

        self._record_skill_result(skill_name, result)
        return result
