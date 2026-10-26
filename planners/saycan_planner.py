from minigrid.core.world_object import Ball, Door

from skills.library import SkillLibrary


class SayCanPlanner:
    """SayCan 风格的高层技能选择器。"""

    def __init__(self, env, movement, picker, opener, placer):
        self.env = env
        self.base_env = env.unwrapped

        self.library = SkillLibrary(
            env, movement, picker, opener, placer
        )

    def is_task_complete(self):
        grid = self.base_env.grid

        ball = grid.get(*self.base_env.target_pos)
        door = grid.get(*self.base_env.door_pos)

        return (
            isinstance(ball, Ball)
            and ball.color == "red"
            and isinstance(door, Door)
            and door.is_open
            and self.base_env.carrying is None
        )

    def run(self, task, max_high_level_steps=10):
        """反复评分、选择、执行，直到任务完成或无法继续。"""
        history = []

        for step in range(1, max_high_level_steps + 1):
            if self.is_task_complete():
                print("Task completed.")
                return True, history

            candidates = self.library.score_candidates(task)

            print(f"\n{'=' * 55}")
            print(f"Planning round {step}")
            print(f"Task: {task}")
            print("-" * 55)
            print(
                f"{'Skill':<12}"
                f"{'Relevance':>12}"
                f"{'Feasibility':>14}"
                f"{'Score':>10}"
            )

            for candidate in candidates:
                print(
                    f"{candidate['name']:<12}"
                    f"{candidate['relevance']:>12.2f}"
                    f"{candidate['feasibility']:>14.2f}"
                    f"{candidate['score']:>10.2f}"
                )

            # 候选技能按照固定顺序排列；
            # 同分时选择排在前面的技能。
            selected = max(
                candidates,
                key=lambda candidate: candidate["score"],
            )

            if selected["score"] <= 0:
                print("No relevant and feasible skill remains.")
                return False, history

            skill_name = selected["name"]

            print(f"\nSelected skill: {skill_name}")

            # 记录执行前的状态
            state_before = {
                "position": tuple(
                    int(v) for v in self.base_env.agent_pos
                ),
                "carrying": (
                    self.base_env.carrying is not None
                ),
            }

            result = self.library.execute(skill_name)

            record = {
                "round": step,
                "skill": skill_name,
                "relevance": selected["relevance"],
                "feasibility": selected["feasibility"],
                "score": selected["score"],
                "state_before": state_before,
                "result": result,
            }

            history.append(record)

            print("Execution result:", result)

            if not result["success"]:
                print("Skill execution failed. Stopping planner.")
                return False, history

            # 下一轮重新读取环境状态并重新评分
            if self.is_task_complete():
                print("\nTask completed successfully.")
                return True, history

        print("Maximum high-level planning steps reached.")
        return self.is_task_complete(), history
