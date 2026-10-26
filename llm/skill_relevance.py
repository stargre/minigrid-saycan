"""Task-relevance scoring for arbitrary parameterized skills."""
import json
import math
import os


class KeywordRelevanceScorer:
    """Offline fallback for smoke tests; use an LLM adapter for experiments."""
    def score(self, task, calls, belief_summary=None):
        text = str(task).lower()
        result = {}
        for call in calls:
            name = call["name"]
            if name == "Explore":
                score = 0.45 if any(w in text for w in ("find", "search", "locate", "explore")) else 0.15
            elif name == "Find":
                score = 0.75 if any(w in text for w in ("find", "search", "locate", "key", "door", "ball", "object", "fetch")) else 0.35
            elif name == "NavigateTo":
                score = 0.75 if any(w in text for w in ("go to", "navigate", "reach", "approach", "door")) else 0.25
            elif name == "PickUp":
                score = 0.9 if any(w in text for w in ("pick", "take", "grab", "fetch", "key", "carry")) else 0.2
            elif name in {"OpenDoor", "UnlockDoor"}:
                score = 0.9 if any(w in text for w in ("door", "open", "unlock", "key")) else 0.1
            elif name == "PlaceNear":
                score = 0.9 if any(w in text for w in ("place", "put", "drop", "above", "below", "left", "right", "near")) else 0.1
            elif name == "MoveObstacle":
                score = 0.85 if any(w in text for w in ("move", "push", "clear", "blocked", "obstacle")) else 0.1
            elif name == "Done":
                score = 0.8 if any(w in text for w in ("done", "finish", "complete", "go to the door")) else 0.1
            else:
                score = 0.1
            result[call["id"]] = {"relevance": min(1.0, max(0.0, score)), "rationale": "Offline keyword fallback; not an LLM judgement."}
        return result


class QwenSkillRelevanceScorer:
    """Qwen-compatible OpenAI client; requires DASHSCOPE_API_KEY."""
    def __init__(self):
        from openai import OpenAI
        key = os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured.")
        self.model = os.getenv("QWEN_MODEL", "qwen-plus")
        self.client = OpenAI(
            api_key=key,
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            timeout=30.0, max_retries=2,
        )

    def score(self, task, calls, belief_summary=None):
        payload = {
            "task": task,
            "observation_derived_state": belief_summary or {},
            "candidate_skills": [{"id": c["id"], "name": c["name"], "description": c.get("description", "")} for c in calls],
        }
        prompt = (
            "Score only semantic task relevance for every candidate. Do not score physical feasibility. "
            "Do not infer unobserved positions or hidden states. Return JSON mapping each candidate id "
            "to {relevance: number in [0,1], rationale: short string}. Every id must be present."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":prompt},{"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
            response_format={"type":"json_object"}, temperature=0.1, max_tokens=1200,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Qwen returned an empty response.")
        data = json.loads(content)
        result = {}
        for call in calls:
            item = data.get(call["id"])
            if not isinstance(item, dict):
                raise ValueError(f"Qwen response is missing candidate {call['id']}")
            score = float(item.get("relevance"))
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid relevance score for {call['id']}: {score}")
            result[call["id"]] = {"relevance": score, "rationale": str(item.get("rationale", ""))}
        return result
