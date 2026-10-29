"""Target descriptions shared by all parameterized robot skills.

Targets are semantic descriptions, not environment coordinates. Coordinates are
returned only as evidence from the agent's own belief map.
"""
from dataclasses import dataclass
import re
from typing import Any, Optional


OBJECT_ALIASES = {
    "balls": "ball",
    "sphere": "ball",
    "spheres": "ball",
    "keys": "key",
    "doors": "door",
    "boxes": "box",
    "crate": "box",
    "crates": "box",
    "goals": "goal",
    "walls": "wall",
    "objects": "object",
}
COLORS = {"red", "blue", "green", "purple", "yellow", "grey", "gray"}
PICKABLE_OBJECTS = {"ball", "key", "box"}
RELATION_OFFSETS = {
    "above": (0, -1),
    "on_top_of": (0, -1),
    "below": (0, 1),
    "left_of": (-1, 0),
    "right_of": (1, 0),
}


def canonical_object(value: str) -> str:
    value = str(value).strip().lower().replace("-", "_")
    return OBJECT_ALIASES.get(value, value)


@dataclass(frozen=True)
class TargetSpec:
    """A reusable object description independent of object identity/location."""

    object_name: str
    color: Optional[str] = None
    state: Optional[str] = None
    label: Optional[str] = None

    @classmethod
    def from_value(cls, value: Any) -> "TargetSpec":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            parsed = parse_target_specs(value)
            if parsed:
                # Prefer a target that describes the whole phrase.
                return parsed[0]
            noun = canonical_object(value)
            return cls(object_name=noun, label=value.strip())
        if isinstance(value, dict):
            object_name = value.get(
                "object_name", value.get("object", value.get("type", "object"))
            )
            color = value.get("color")
            state = value.get("state")
            label = value.get("label", value.get("description"))
            return cls(
                object_name=canonical_object(object_name),
                color=str(color).lower() if color else None,
                state=str(state).lower() if state else None,
                label=str(label) if label else None,
            )
        raise TypeError(
            "target must be a TargetSpec, a natural-language string, or a mapping"
        )

    @property
    def key(self) -> str:
        parts = [self.color, self.object_name, self.state]
        return ":".join(str(part) for part in parts if part)

    @property
    def display_name(self) -> str:
        if self.label:
            return self.label
        parts = [self.color, self.state, self.object_name]
        return " ".join(str(part) for part in parts if part)

    def to_dict(self) -> dict:
        result = {"object": self.object_name}
        if self.color:
            result["color"] = self.color
        if self.state:
            result["state"] = self.state
        result["label"] = self.display_name
        return result

    def matches(self, cell: Optional[dict]) -> bool:
        if not cell:
            return False
        actual_object = canonical_object(cell.get("object", "unknown"))
        if self.object_name == "object":
            if actual_object not in PICKABLE_OBJECTS:
                return False
        elif actual_object != self.object_name:
            return False
        if self.color and str(cell.get("color", "")).lower() != self.color:
            return False
        if self.state and str(cell.get("state", "")).lower() != self.state:
            return False
        return True


_TARGET_PATTERN = re.compile(
    r"\b(?:(red|blue|green|purple|yellow|grey|gray)\s+)?"
    r"(?:(locked|closed|open)\s+)?"
    r"(balls?|spheres?|keys?|doors?|box(?:es)?|crates?|goals?|lava|walls?|objects?)\b",
    re.IGNORECASE,
)


def parse_target_specs(text: str) -> list[TargetSpec]:
    """Extract simple color/object phrases; this is not an LLM parser."""
    specs = []
    seen = set()
    for match in _TARGET_PATTERN.finditer(str(text)):
        color, state, noun = match.groups()
        object_name = canonical_object(noun)
        spec = TargetSpec(
            object_name=object_name,
            color=color.lower() if color else None,
            state=state.lower() if state else None,
            label=" ".join(
                part for part in (color, state, object_name) if part
            ),
        )
        if spec.key not in seen:
            specs.append(spec)
            seen.add(spec.key)
    return specs


def relation_offset(relation: str) -> Optional[tuple[int, int]]:
    relation = str(relation or "near").strip().lower().replace("-", "_")
    aliases = {
        "over": "above",
        "on": "above",
        "top": "above",
        "under": "below",
        "beneath": "below",
        "left": "left_of",
        "right": "right_of",
        "nearby": "near",
        "adjacent": "near",
        "beside": "near",
        "next_to": "near",
    }
    relation = aliases.get(relation, relation)
    return RELATION_OFFSETS.get(relation)
