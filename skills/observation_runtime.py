from perception.belief_state import BeliefState


class ObservationRuntime:
    """
    观测驱动的底层动作执行器。

    只向上层提供动作结果和新观测，不暴露环境 info。
    env 可以是原始 MiniGrid 环境或 FrameRecorder 包装器。
    """

    def __init__(self, env, belief=None):
        self.env = env
        self.belief = belief or BeliefState()

        self.action_count = 0
        self.last_obs = None
        self.last_reward = 0.0
        self.terminated = False
        self.truncated = False

    def reset(self, obs, initial_inventory="empty"):
        self.belief.reset(
            obs,
            initial_inventory=initial_inventory,
        )

        self.last_obs = obs
        self.action_count = 0
        self.last_reward = 0.0
        self.terminated = False
        self.truncated = False

    def execute_action(self, action):
        if self.last_obs is None:
            raise RuntimeError(
                "Call reset(obs) before executing actions."
            )

        if self.terminated or self.truncated:
            raise RuntimeError(
                "The episode has ended; reset before acting again."
            )

        previous_obs = self.last_obs

        # info 被刻意丢弃，不进入决策链。
        obs, reward, terminated, truncated, _info = (
            self.env.step(int(action))
        )

        self.action_count += 1
        self.last_reward = float(reward)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)

        self.belief.update(
            previous_obs=previous_obs,
            action=action,
            next_obs=obs,
        )

        self.last_obs = obs

        return {
            "obs": obs,
            "reward": float(reward),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "action": int(action),
            "actions_executed": 1,
        }
