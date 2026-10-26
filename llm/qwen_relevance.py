import json
import math
import os

from openai import OpenAI


SKILL_NAMES = ("pick", "open_door", "place", "explore")

SYSTEM_PROMPT = """
你是 SayCan 风格机器人规划器中的任务相关性评分模块。

根据用户任务、当前已知状态和候选技能，分别评估每个技能
对完成任务的相关程度。

返回合法 JSON，格式如下：
{
  "pick": {"relevance": 0.9, "rationale": "..."},
  "open_door": {"relevance": 0.8, "rationale": "..."},
  "place": {"relevance": 0.7, "rationale": "..."},
  "explore": {"relevance": 0.6, "rationale": "..."}
}

要求：
1. relevance 必须是 0 到 1 之间的数字。
2. 只评估任务相关性，不评估技能物理可行性。
3. 不得猜测未观察到的物体位置或门的状态。
4. 未观察到的物体仍可能与最终任务相关。
5. 不要把相关性分数解释为技能成功概率。
6. 所有技能都必须返回分数和简短理由。
7. 只输出 JSON，不要输出 Markdown。
"""


class QwenRelevanceScorer:
    """调用阿里云百炼 API 评估候选技能的任务相关性。"""

    def __init__(self):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置")

        self.model = os.getenv("QWEN_MODEL", "qwen-plus")

        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            timeout=30.0,
            max_retries=2,
        )

    def score(self, task, belief_summary):
        payload = {
            "task": task,
            "observation_derived_state": belief_summary,
            "candidate_skills": {
                "pick": "接近并拾取红球",
                "open_door": "接近并打开关闭的门",
                "place": "将手中的物体放置到目标位置",
                "explore": "探索未知区域以获得新的观测",
            },
        }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload, ensure_ascii=False
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=500,
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("千问返回了空响应")

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"千问返回的内容不是合法 JSON：{content}"
            ) from exc

        result = {}
        for name in SKILL_NAMES:
            if name not in data:
                raise ValueError(f"千问响应缺少技能：{name}")

            item = data[name]
            score = float(item["relevance"])

            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"{name} 的相关性分数无效：{score}"
                )

            result[name] = {
                "relevance": score,
                "rationale": str(item["rationale"]),
            }

        return result
