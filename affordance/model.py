"""Dependency-light calibrated MLP for learned SayCan affordances."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from affordance.features import AffordanceFeatureEncoder


def _sigmoid(value):
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


class LearnedAffordanceModel:
    """Predict P(skill succeeds | observation belief, parameterized skill)."""

    model_type = "calibrated_numpy_mlp_v1"

    def __init__(self, encoder=None, hidden_size=32, seed=0):
        self.encoder = encoder or AffordanceFeatureEncoder()
        self.hidden_size = int(hidden_size)
        rng = np.random.default_rng(seed)
        scale = math.sqrt(2.0 / max(1, self.encoder.dimension))
        self.w1 = rng.normal(0.0, scale, (self.encoder.dimension, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size, dtype=np.float64)
        self.w2 = rng.normal(0.0, 0.1, (self.hidden_size, 1))
        self.b2 = np.zeros(1, dtype=np.float64)
        self.mean = np.zeros(self.encoder.dimension, dtype=np.float64)
        self.std = np.ones(self.encoder.dimension, dtype=np.float64)
        self.temperature = 1.0
        self.training_metadata = {}

    def _logits(self, x):
        normalized = (np.asarray(x, dtype=np.float64) - self.mean) / self.std
        hidden = np.maximum(0.0, normalized @ self.w1 + self.b1)
        return (hidden @ self.w2 + self.b2).reshape(-1)

    def predict_array(self, x):
        return _sigmoid(self._logits(x) / max(0.05, float(self.temperature)))

    def predict(self, belief, call):
        vector = self.encoder.encode(belief, call)
        return float(self.predict_array(np.asarray([vector]))[0])

    def supports(self, call):
        """Return whether training metadata contains this skill family."""
        skill = call.get("name") if isinstance(call, dict) else str(call)
        counts = self.training_metadata.get("train_skill_counts", {})
        return int(counts.get(skill, 0)) > 0

    @staticmethod
    def metrics(labels, probabilities):
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        p = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 1e-7, 1 - 1e-7)
        if len(y) == 0:
            return {"cases": 0}
        accuracy = float(np.mean((p >= 0.5) == (y >= 0.5)))
        brier = float(np.mean((p - y) ** 2))
        log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        ece = 0.0
        for low in np.linspace(0.0, 0.9, 10):
            mask = (p >= low) & (p < low + 0.1)
            if np.any(mask):
                ece += float(np.mean(mask)) * abs(float(np.mean(p[mask]) - np.mean(y[mask])))
        positives = np.where(y >= 0.5)[0]
        negatives = np.where(y < 0.5)[0]
        auc = None
        if len(positives) and len(negatives):
            wins = sum(
                float(p[i] > p[j]) + 0.5 * float(p[i] == p[j])
                for i in positives for j in negatives
            )
            auc = float(wins / (len(positives) * len(negatives)))
        return {
            "cases": int(len(y)), "positive_rate": float(np.mean(y)),
            "accuracy": accuracy, "brier": brier, "log_loss": log_loss,
            "ece_10_bin": float(ece), "roc_auc": auc,
        }

    def fit(self, train_x, train_y, validation_x=None, validation_y=None,
            epochs=250, learning_rate=0.01, batch_size=64, l2=1e-4, seed=0):
        x = np.asarray(train_x, dtype=np.float64)
        y = np.asarray(train_y, dtype=np.float64).reshape(-1, 1)
        if x.ndim != 2 or x.shape[1] != self.encoder.dimension:
            raise ValueError("Training features do not match the encoder schema.")
        if len(x) < 2 or len(np.unique(y)) < 2:
            raise ValueError("Training requires at least two samples and both outcome classes.")
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        xn = (x - self.mean) / self.std
        rng = np.random.default_rng(seed)
        params = [self.w1, self.b1, self.w2, self.b2]
        m = [np.zeros_like(item) for item in params]
        v = [np.zeros_like(item) for item in params]
        step = 0
        positive_weight = float((len(y) - y.sum()) / max(1.0, y.sum()))
        positive_weight = min(8.0, max(0.25, positive_weight))
        for _ in range(int(epochs)):
            order = rng.permutation(len(xn))
            for start in range(0, len(order), int(batch_size)):
                idx = order[start:start + int(batch_size)]
                xb, yb = xn[idx], y[idx]
                z1 = xb @ self.w1 + self.b1
                h = np.maximum(0.0, z1)
                logits = h @ self.w2 + self.b2
                probs = _sigmoid(logits)
                weights = np.where(yb > 0.5, positive_weight, 1.0)
                dz2 = weights * (probs - yb) / len(idx)
                dw2 = h.T @ dz2 + l2 * self.w2
                db2 = dz2.sum(axis=0)
                dh = dz2 @ self.w2.T
                dz1 = dh * (z1 > 0)
                dw1 = xb.T @ dz1 + l2 * self.w1
                db1 = dz1.sum(axis=0)
                grads = [dw1, db1, dw2, db2]
                step += 1
                for number, (param, grad) in enumerate(zip(params, grads)):
                    m[number] = 0.9 * m[number] + 0.1 * grad
                    v[number] = 0.999 * v[number] + 0.001 * (grad * grad)
                    mhat = m[number] / (1 - 0.9 ** step)
                    vhat = v[number] / (1 - 0.999 ** step)
                    param -= learning_rate * mhat / (np.sqrt(vhat) + 1e-8)

        self.temperature = 1.0
        if validation_x is not None and validation_y is not None and len(validation_x):
            vx = np.asarray(validation_x, dtype=np.float64)
            vy = np.asarray(validation_y, dtype=np.float64).reshape(-1)
            logits = self._logits(vx)
            candidates = np.exp(np.linspace(math.log(0.2), math.log(5.0), 121))
            losses = []
            for temperature in candidates:
                p = np.clip(_sigmoid(logits / temperature), 1e-7, 1 - 1e-7)
                losses.append(float(-np.mean(vy * np.log(p) + (1 - vy) * np.log(1 - p))))
            self.temperature = float(candidates[int(np.argmin(losses))])
        train_metrics = self.metrics(train_y, self.predict_array(x))
        validation_metrics = None
        if validation_x is not None and validation_y is not None and len(validation_x):
            validation_metrics = self.metrics(
                validation_y, self.predict_array(np.asarray(validation_x, dtype=np.float64))
            )
        self.training_metadata = {
            "train": train_metrics, "validation": validation_metrics,
            "epochs": int(epochs), "learning_rate": float(learning_rate),
            "batch_size": int(batch_size), "l2": float(l2), "seed": int(seed),
        }
        return dict(self.training_metadata)

    def to_dict(self):
        return {
            "model_type": self.model_type,
            "feature_schema": self.encoder.metadata(),
            "hidden_size": self.hidden_size,
            "temperature": self.temperature,
            "mean": self.mean.tolist(), "std": self.std.tolist(),
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "training_metadata": self.training_metadata,
        }

    def save(self, path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        encoder = AffordanceFeatureEncoder()
        schema = data.get("feature_schema", {})
        if schema.get("schema_hash") != encoder.schema_hash:
            raise ValueError(
                "Affordance model feature schema is incompatible with this code: "
                f"{schema.get('schema_hash')} != {encoder.schema_hash}"
            )
        model = cls(encoder=encoder, hidden_size=int(data["hidden_size"]))
        for name in ("mean", "std", "w1", "b1", "w2", "b2"):
            setattr(model, name, np.asarray(data[name], dtype=np.float64))
        model.temperature = float(data.get("temperature", 1.0))
        model.training_metadata = dict(data.get("training_metadata", {}))
        return model
