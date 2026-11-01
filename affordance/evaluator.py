"""Combine learned success probability with deterministic safety constraints."""
from __future__ import annotations

import os

from affordance.model import LearnedAffordanceModel
from skills.feasibility import SkillFeasibilityEvaluator


class LearnedAffordanceEvaluator:
    """Can(s, skill) = hard deterministic mask * learned success probability."""

    def __init__(self, runtime, model, dependency_resolver=None,
                 probability_floor=0.02):
        self.runtime = runtime
        self.model = (
            LearnedAffordanceModel.load(model)
            if isinstance(model, (str, bytes, os.PathLike)) else model
        )
        if self.model is None:
            raise ValueError("A trained affordance model is required.")
        self.hard = SkillFeasibilityEvaluator(runtime, dependency_resolver)
        self.probability_floor = float(probability_floor)

    def evaluate(self, call):
        deterministic = self.hard.evaluate(call)
        hard_score = float(deterministic["feasibility"])
        if hard_score <= 0:
            return {
                **deterministic,
                "hard_feasibility": 0.0,
                "learned_affordance": 0.0,
                "affordance_source": "hard_mask",
            }
        supports = getattr(self.model, "supports", None)
        if callable(supports) and not supports(call):
            return {
                **deterministic,
                "hard_feasibility": hard_score,
                "learned_affordance": None,
                "affordance_source": "unseen_skill_fallback",
                "reason": (
                    f"{deterministic['reason']} The trained model has no "
                    "examples for this skill; deterministic Can is retained."
                ),
            }
        probability = float(self.model.predict(self.runtime.belief, call))
        probability = min(1.0, max(self.probability_floor, probability))
        return {
            **deterministic,
            "feasibility": hard_score * probability,
            "hard_feasibility": hard_score,
            "learned_affordance": probability,
            "affordance_source": self.model.model_type,
            "reason": (
                f"{deterministic['reason']} Learned skill-success probability="
                f"{probability:.4f}."
            ),
        }
