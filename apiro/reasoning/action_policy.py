"""Feature-weighted, externally trainable investigation action policy."""

from __future__ import annotations

import json
import math
from pathlib import Path


DEFAULT_WEIGHTS = {
    "bias": -0.4,
    "uncertainty": 1.2,
    "small_margin": 1.1,
    "evidence_gap": 1.0,
    "contradiction": 1.5,
    "novelty": 0.5,
    "cost": -0.8,
}


class ActionValuePolicy:
    """Score expected information gain using loadable linear weights."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({key: float(value) for key, value in weights.items()})

    def score(self, features: dict[str, float]) -> float:
        value = self.weights["bias"]
        for name, feature in features.items():
            value += self.weights.get(name, 0.0) * min(1.0, max(0.0, float(feature)))
        return round(1.0 / (1.0 + math.exp(-value)), 6)

    def fit(
        self,
        examples: list[tuple[dict[str, float], float]],
        *,
        epochs: int = 200,
        learning_rate: float = 0.1,
    ) -> None:
        """Fit logistic weights from action features and observed utility labels."""
        if not examples:
            raise ValueError("at least one training example is required")
        feature_names = sorted({name for row, _target in examples for name in row})
        for name in feature_names:
            self.weights.setdefault(name, 0.0)
        for _ in range(max(1, int(epochs))):
            gradients = {"bias": 0.0, **{name: 0.0 for name in feature_names}}
            for features, target in examples:
                prediction = self.score(features)
                error = prediction - min(1.0, max(0.0, float(target)))
                gradients["bias"] += error
                for name in feature_names:
                    gradients[name] += error * float(features.get(name, 0.0))
            scale = float(len(examples))
            for name, gradient in gradients.items():
                self.weights[name] -= learning_rate * gradient / scale

    @classmethod
    def load(cls, path: Path) -> "ActionValuePolicy":
        path = Path(path)
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload.get("weights", payload))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"weights": self.weights}, indent=2) + "\n",
            encoding="utf-8",
        )


__all__ = ["ActionValuePolicy", "DEFAULT_WEIGHTS"]
