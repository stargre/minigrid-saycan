"""Task-relevance scoring adapters for SayCan-style skill selection."""
import json
import math
import os
import time
from collections import OrderedDict


class KeywordRelevanceScorer:
    """Deterministic offline baseline; not an LLM judgement."""
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
            elif name == "Reach":
                target = call.get("args", {}).get("target", {})
                target_name = target.get("object", target.get("object_name", "")) if isinstance(target, dict) else str(target)
                move_words = ("reach", "enter", "arrive at", "get to", "go to", "navigate to", "cross to")
                score = 0.95 if str(target_name).lower() == "goal" and "goal" in text and any(w in text for w in move_words) else 0.15
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
            result[call["id"]] = {"relevance": min(1.0, max(0.0, score)), "rationale": "Offline keyword fallback."}
        return result


class QwenSkillRelevanceScorer:
    """Low-latency Qwen-compatible adapter; requires DASHSCOPE_API_KEY."""
    def __init__(self):
        from openai import OpenAI
        key = os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured.")
        self.model = os.getenv("QWEN_MODEL", "qwen-plus")
        self.timeout = float(os.getenv("QWEN_TIMEOUT_SECONDS", "12"))
        self.max_retries = int(os.getenv("QWEN_MAX_RETRIES", "0"))
        self.client = OpenAI(
            api_key=key,
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self.last_latency_s = 0.0
        self.request_count = 0
        self._cache = OrderedDict()
        self.cache_size = int(os.getenv("QWEN_SCORE_CACHE_SIZE", "64"))

    @staticmethod
    def _target_progress(call, task, state):
        name = call.get("name", "")
        done = set(state.get("completed_subgoals", []))
        if name in {"PickUp", "PlaceNear", "OpenDoor", "UnlockDoor", "Reach", "Done"}:
            if name == "PickUp" and ({"pick", "pickup"} & done):
                return "already completed or possibly redundant"
            if name == "OpenDoor" and ({"open_door", "unlock_door"} & done):
                return "already completed or possibly redundant"
            if name == "PlaceNear" and ({"place", "drop"} & done):
                return "already completed or possibly redundant"
        held = state.get("held_object")
        if name == "PlaceNear" and not held:
            return "requires holding the requested object; feasibility is checked separately"
        return "consider whether this advances an unfinished task stage"

    def score(self, task, calls, belief_summary=None):
        state = belief_summary or {}
        compact_calls = [{
            "id": c["id"], "skill": c["name"],
            "arguments": c.get("args", {}),
            "description": c.get("description", ""),
            "progress_hint": self._target_progress(c, task, state),
        } for c in calls]
        payload = {
            "task": str(task),
            "progress": {
                "completed_subgoals": state.get("completed_subgoals", []),
                "held_object": state.get("held_object"),
                "inventory_status": state.get("inventory_status"),
                "estimated_position": state.get("estimated_position"),
                "known_cell_count": state.get("known_cell_count"),
            },
            "candidates": compact_calls,
        }
        cache_key = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return dict(self._cache[cache_key])

        system_prompt = (
            "You are the semantic-relevance component of a SayCan-style planner. "
            "Choose relevance for the NEXT skill, not general relevance to the whole task. "
            "Score only semantic fit and progress toward the earliest unfinished task stage; "
            "physical feasibility is computed separately. Prefer actions that advance unfinished "
            "stages, avoid repeating completed actions, and use candidate arguments to distinguish "
            "targets. Use only the supplied task/state; never infer hidden positions. "
            "Return compact JSON: {candidate_id: {relevance: 0..1, rationale: <=8 words}}. "
            "Include every candidate ID exactly once."
        )
        start = time.monotonic()
        self.request_count += 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=min(600, max(120, 28 * len(calls))),
        )
        self.last_latency_s = time.monotonic() - start
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Qwen returned an empty response.")
        data = json.loads(content)
        result = {}
        for call in calls:
            item = data.get(call["id"])
            if not isinstance(item, dict):
                raise ValueError(f"Qwen response is missing candidate {call['id']}")
            value = float(item.get("relevance"))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid relevance score for {call['id']}: {value}")
            result[call["id"]] = {"relevance": value, "rationale": str(item.get("rationale", ""))[:160]}

        self._cache[cache_key] = result
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return dict(result)
