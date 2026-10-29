"""Task-relevance scoring adapters for SayCan-style skill selection."""
import json
import math
import os
import time
from collections import OrderedDict

from skills.target import TargetSpec, parse_target_specs


class KeywordRelevanceScorer:
    """Deterministic offline baseline; not an LLM judgement.

    In addition to verb matching, this scorer penalizes candidate arguments
    that contradict an explicitly requested object color. This matters for
    generated MiniGrid missions such as "go to the green door", where other
    colored doors may already be visible in the observation-derived map.
    """

    def score(self, task, calls, belief_summary=None):
        text = str(task).lower()
        requested_by_object = {}
        for spec in parse_target_specs(text):
            requested_by_object.setdefault(spec.object_name, []).append(spec)

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
                movement_intent = any(
                    phrase in text
                    for phrase in ("go to", "navigate to", "reach", "approach", "arrive at", "get to")
                )
                score = 0.98 if ("door" in text and movement_intent) else (
                    0.8 if any(w in text for w in ("done", "finish", "complete")) else 0.1
                )
            else:
                score = 0.1

            # Do not let a semantically relevant verb outweigh an explicitly
            # mismatched target argument (e.g. NavigateTo(blue door) for a
            # mission that specifically asks for the green door).
            target = call.get("args", {}).get("target")
            if target is not None:
                try:
                    target_spec = TargetSpec.from_value(target)
                except (TypeError, ValueError):
                    target_spec = None
                if target_spec is not None:
                    requested = requested_by_object.get(target_spec.object_name, [])
                    requested_colors = {spec.color for spec in requested if spec.color}
                    if (
                        target_spec.color
                        and requested_colors
                        and target_spec.color not in requested_colors
                    ):
                        score = min(score, 0.01)

            result[call["id"]] = {
                "relevance": min(1.0, max(0.0, score)),
                "rationale": "Offline keyword and target-argument relevance.",
            }
        return result

class QwenSkillRelevanceScorer:
    """Qwen-compatible semantic scorer; requires DASHSCOPE_API_KEY."""

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
        self.api_latencies_s = []
        self.parse_retry_count = 0
        self._cache = OrderedDict()
        self.cache_size = int(os.getenv("QWEN_SCORE_CACHE_SIZE", "64"))

    @staticmethod
    def _target_progress(call, task, state):
        name = call.get("name", "")
        done = set(state.get("completed_subgoals", []))
        if name == "PickUp" and ({"pick", "pickup"} & done):
            return "already completed or possibly redundant"
        if name == "OpenDoor" and ({"open_door", "unlock_door"} & done):
            return "already completed or possibly redundant"
        if name == "PlaceNear" and ({"place", "drop"} & done):
            return "already completed or possibly redundant"
        if name == "PlaceNear" and not state.get("held_object"):
            return "requires holding the requested object; feasibility is checked separately"
        return "consider whether this advances an unfinished task stage"

    @staticmethod
    def _parse_score_response(content, calls):
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Qwen returned an empty response.")
        text = content.strip()
        fence = chr(96) * 3
        if text.startswith(fence):
            lines = text.splitlines()
            if lines and lines[0].lstrip().startswith(fence):
                lines = lines[1:]
            if lines and lines[-1].strip() == fence:
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Qwen scoring response must be a JSON object.")
        expected = {call["id"] for call in calls}
        actual = set(data)
        if expected != actual:
            raise ValueError(f"Qwen candidate IDs mismatch; missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}")
        result = {}
        for call in calls:
            cid = call["id"]
            item = data[cid]
            raw = item.get("relevance") if isinstance(item, dict) else item
            rationale = str(item.get("rationale", "Qwen semantic relevance."))[:160] if isinstance(item, dict) else "Qwen semantic relevance."
            if isinstance(raw, bool):
                raise ValueError(f"Invalid relevance score for {cid}: boolean")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid relevance score for {cid}: {raw!r}") from exc
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid relevance score for {cid}: {value}")
            result[cid] = {"relevance": value, "rationale": rationale}
        return result

    def _request_content(self, messages, max_tokens):
        started = time.perf_counter()
        self.request_count += 1
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages,
                response_format={"type": "json_object"},
                temperature=0, max_tokens=max_tokens,
            )
            choices = getattr(response, "choices", None)
            if not choices:
                return None
            return getattr(getattr(choices[0], "message", None), "content", None)
        finally:
            elapsed = time.perf_counter() - started
            self.last_latency_s = elapsed
            self.api_latencies_s.append(elapsed)

    def score(self, task, calls, belief_summary=None):
        state = belief_summary or {}
        compact_calls = [{
            "id": c["id"], "skill": c["name"], "arguments": c.get("args", {}),
            "description": c.get("description", ""),
            "progress_hint": self._target_progress(c, task, state),
        } for c in calls]
        payload = {
            "task": str(task),
            "progress": {k: state.get(k) for k in (
                "completed_subgoals", "held_object", "inventory_status",
                "estimated_position", "known_cell_count")},
            "candidates": compact_calls,
        }
        cache_key = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return dict(self._cache[cache_key])

        system_prompt = (
            "You are the semantic-relevance component of a SayCan-style planner. "
            "Score each candidate for the NEXT skill, not general relevance to the whole task. "
            "Score semantic fit and progress toward the earliest unfinished task stage; "
            "physical feasibility is computed separately. Prefer advancing unfinished stages, "
            "avoid repeating completed actions, and use arguments to distinguish targets. "
            "Use only supplied task/state; never infer hidden positions. Return ONLY a compact "
            "JSON object mapping every candidate ID to a numeric relevance score from 0 to 1. "
            "Include every ID exactly once. No rationale, markdown, or surrounding text."
        )
        user_message = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        try:
            result = self._parse_score_response(
                self._request_content(messages, min(1200, max(256, 64 * len(calls)))), calls
            )
        except (json.JSONDecodeError, TypeError, ValueError) as first_error:
            self.parse_retry_count += 1
            retry_messages = [
                {"role": "system", "content": system_prompt + " Previous output was invalid/incomplete. Return a complete valid JSON object now, numeric scores only."},
                {"role": "user", "content": user_message},
            ]
            try:
                result = self._parse_score_response(
                    self._request_content(retry_messages, min(1600, max(512, 96 * len(calls)))), calls
                )
            except (json.JSONDecodeError, TypeError, ValueError) as retry_error:
                raise ValueError(
                    "Qwen returned invalid scoring JSON after one format retry. "
                    f"First: {type(first_error).__name__}: {str(first_error)[:120]}; "
                    f"retry: {type(retry_error).__name__}: {str(retry_error)[:160]}"
                ) from retry_error

        self._cache[cache_key] = result
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return dict(result)
