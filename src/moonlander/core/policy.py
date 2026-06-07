"""Trained-policy forward pass (CONTRACT §2 ``step_policy``, schema §11).

Pure stdlib — imports nothing beyond ``math`` and ``json`` (enforced by
tests/test_policy.py). The policy is a tiny MLP: obs(14) → tanh(hidden) →
4 logits → argmax → the same Discrete(4) classic-controls mapping the
Gymnasium env uses. Weights come from ``python -m moonlander.train_cem``
as a JSON string — the very artifact the browser fetches, so JS never
sees the network, only the frames it flies.
"""

import json
import math

FORMAT = "mlp-tanh-argmax/v1"

# argmax index -> classic controls — EXACTLY env.py's Discrete(4) mapping:
# 0 noop, 1 rotate left (+1 = CCW), 2 rotate right (-1), 3 thrust.
ACTIONS = ((0, False), (1, False), (-1, False), (0, True))


def _check_vector(name, v, n):
    if not isinstance(v, list) or len(v) != n:
        raise ValueError(f"{name} must be a list of {n} numbers")
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)) \
                or not math.isfinite(x):
            raise ValueError(
                f"{name} contains a non-finite or non-number entry: {x!r}"
            )


def _check_matrix(name, m, rows, cols):
    if not isinstance(m, list) or len(m) != rows:
        raise ValueError(f"{name} must be a list of {rows} rows")
    for j, row in enumerate(m):
        _check_vector(f"{name}[{j}]", row, cols)


class Policy:
    """Deterministic MLP policy: ``act(obs) -> (rotate, thrust)``."""

    def __init__(self, w1, b1, w2, b2):
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2

    @classmethod
    def from_json(cls, policy_json):
        """Parse + validate a §11 policy JSON string. ValueError on any problem."""
        data = json.loads(policy_json)
        if not isinstance(data, dict):
            raise ValueError(
                f"policy JSON must be an object, got {type(data).__name__}"
            )
        if data.get("format") != FORMAT:
            raise ValueError(
                f"policy format must be {FORMAT!r}, got {data.get('format')!r}"
            )
        sizes = data.get("sizes")
        if (not isinstance(sizes, list) or len(sizes) != 3 or sizes[0] != 14
                or sizes[2] != 4 or not isinstance(sizes[1], int)
                or isinstance(sizes[1], bool) or sizes[1] < 1):
            raise ValueError(f"sizes must be [14, hidden >= 1, 4], got {sizes!r}")
        hidden = sizes[1]
        _check_matrix("w1", data.get("w1"), hidden, 14)
        _check_vector("b1", data.get("b1"), hidden)
        _check_matrix("w2", data.get("w2"), 4, hidden)
        _check_vector("b2", data.get("b2"), 4)
        return cls(data["w1"], data["b1"], data["w2"], data["b2"])

    def act(self, obs):
        """obs (14 floats) -> (rotate, thrust). Ties pick the lowest index."""
        h = [math.tanh(sum(w * x for w, x in zip(row, obs)) + b)
             for row, b in zip(self.w1, self.b1)]
        logits = [sum(w * v for w, v in zip(row, h)) + b
                  for row, b in zip(self.w2, self.b2)]
        return ACTIONS[max(range(4), key=logits.__getitem__)]
