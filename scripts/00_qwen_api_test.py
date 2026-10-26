import json
import os

from openai import OpenAI


api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("请先配置 DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url=os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    timeout=30.0,
    max_retries=2,
)

model = os.getenv("QWEN_MODEL", "qwen-turbo")

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": (
                "你是机器人规划助手。"
                "请仅返回合法的JSON对象，不要添加Markdown代码块。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请给出拾取红球和打开门这两个技能的任务相关性分数。"
                "返回JSON，包含pick和open_door两个字段，"
                "每个字段包含relevance和rationale。"
            ),
        },
    ],
    response_format={"type": "json_object"},
    temperature=0.1,
    max_tokens=300,
)

content = response.choices[0].message.content
if not content:
    raise RuntimeError("模型返回了空内容")

result = json.loads(content)

print("Model:", model)
print("Response:")
print(json.dumps(result, ensure_ascii=False, indent=2))

if response.usage:
    print("\nToken usage:")
    print("Input:", response.usage.prompt_tokens)
    print("Output:", response.usage.completion_tokens)

print("\nPASS: Qwen API is working.")
