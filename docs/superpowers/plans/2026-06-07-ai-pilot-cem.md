# AI Pilot (CEM-trained MLP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The simplest learned pilot — a 308-parameter tanh MLP trained by the cross-entropy method — flying lander 0 in the browser behind a `P`-key / `AI`-button AI PILOT mode, with `web/ml.html` explaining every line of the machinery.

**Architecture:** Pure-stdlib forward pass in `core/policy.py` (ships in the wheel, runs in Pyodide), `Game.set_policy/step_policy` mirroring the autopilot pattern, numpy CEM trainer at `moonlander/train_cem.py` writing `web/assets/policy.json` (committed artifact), JS only toggles mode and renders frames. CONTRACT.md is updated in the same commits as the code it specifies. Version bumps 0.4.0 → 0.5.0 in the four coupled spots.

**Tech Stack:** Python stdlib (core), numpy + gymnasium (trainer only), pytest, vanilla JS/canvas (web). No new dependencies.

Spec: `docs/superpowers/specs/2026-06-07-ml-pilot-design.md`.

Conventions used below: repo root `/Users/bijanmehr/MultiLander`, venv python `.venv/bin/python`, run tests with `.venv/bin/python -m pytest`. Commit after each task. Reference docs: CONTRACT sections cited as §N.

---

### Task 1: `core/policy.py` — pure-stdlib MLP forward pass

**Files:**
- Create: `src/moonlander/core/policy.py`
- Test: `tests/test_policy.py`

- [ ] **Step 1: Write the failing tests** (forward pass, action mapping, validation, purity)

Create `tests/test_policy.py`:

```python
"""Trained-policy forward pass + Game.set_policy/step_policy (CONTRACT §2/§11)."""

import json
import math
import subprocess
import sys
import textwrap

import pytest

from moonlander.core import policy as policy_module
from moonlander.core.game import Game
from moonlander.core.policy import ACTIONS, FORMAT, Policy


def policy_json(hidden=2, w1=None, b1=None, w2=None, b2=None):
    """A valid §11 policy JSON string; weights default to zeros."""
    return json.dumps({
        "format": FORMAT,
        "sizes": [14, hidden, 4],
        "w1": w1 if w1 is not None else [[0.0] * 14 for _ in range(hidden)],
        "b1": b1 if b1 is not None else [0.0] * hidden,
        "w2": w2 if w2 is not None else [[0.0] * hidden for _ in range(4)],
        "b2": b2 if b2 is not None else [0.0] * 4,
    })


# --------------------------------------------------------------- forward pass

def test_zero_weights_tie_break_picks_action_0_noop():
    p = Policy.from_json(policy_json())
    assert p.act([0.5] * 14) == (0, False)  # all logits equal -> lowest index


def test_bias_selects_each_action_with_envs_discrete4_mapping():
    # §11: argmax index -> classic controls exactly as env.py's Discrete(4):
    # 0 noop, 1 rotate left (+1 = CCW), 2 rotate right (-1), 3 thrust.
    expected = {0: (0, False), 1: (1, False), 2: (-1, False), 3: (0, True)}
    for idx, controls in expected.items():
        b2 = [0.0] * 4
        b2[idx] = 1.0
        p = Policy.from_json(policy_json(b2=b2))
        assert p.act([0.0] * 14) == controls


def test_forward_pass_matches_hand_computed_reference():
    # hidden=1: h = tanh(w1·obs + b1); logits = w2*h + b2 — recomputed here
    # with raw math and compared against the module's decision.
    obs = [0.1 * i for i in range(14)]
    w1 = [[(-1.0) ** i * 0.2 for i in range(14)]]
    b1 = [0.3]
    w2 = [[1.0], [-2.0], [0.5], [0.0]]
    b2 = [0.0, 0.1, -0.2, 0.05]

    h = math.tanh(sum(w * x for w, x in zip(w1[0], obs)) + b1[0])
    logits = [w2[k][0] * h + b2[k] for k in range(4)]
    want = ACTIONS[max(range(4), key=logits.__getitem__)]

    p = Policy.from_json(policy_json(hidden=1, w1=w1, b1=b1, w2=w2, b2=b2))
    assert p.act(obs) == want


def test_hidden_sign_flips_the_decision():
    # Pins tanh + both matmuls: h > 0 makes logit[1] = 2h the max (rotate
    # left); h < 0 makes the zero logits win, lowest index 2 (rotate right).
    def mk(scale):
        w1 = [[0.0] * 14]
        w1[0][0] = scale
        return Policy.from_json(policy_json(
            hidden=1, w1=w1, w2=[[1.0], [2.0], [0.0], [0.0]]))

    obs = [0.0] * 14
    obs[0] = 1.0
    assert mk(1.0).act(obs) == (1, False)
    assert mk(-1.0).act(obs) == (-1, False)


# ---------------------------------------------------------------- validation

@pytest.mark.parametrize("mutate, match", [
    (lambda d: d.update(format="mlp/v0"), "format"),
    (lambda d: d.update(sizes=[14, 0, 4]), "sizes"),
    (lambda d: d.update(sizes=[13, 2, 4]), "sizes"),
    (lambda d: d.update(sizes=[14, 2]), "sizes"),
    (lambda d: d.update(b1=[0.0]), "b1"),                       # wrong length
    (lambda d: d["w1"][0].pop(), r"w1\[0\]"),                   # ragged row
    (lambda d: d["w2"][1].__setitem__(0, float("nan")), r"w2\[1\]"),
    (lambda d: d["b2"].__setitem__(2, "x"), "b2"),
])
def test_from_json_rejects_bad_payloads(mutate, match):
    d = json.loads(policy_json())
    mutate(d)
    with pytest.raises(ValueError, match=match):
        Policy.from_json(json.dumps(d))


def test_from_json_rejects_non_object():
    with pytest.raises(ValueError, match="object"):
        Policy.from_json("[1, 2, 3]")


# -------------------------------------------------------------------- purity

def test_policy_module_imports_only_math_and_json():
    # Fresh interpreter with json (and its transitive deps) PRELOADED into
    # `before`, so the only new module the policy file may pull in is math;
    # the only module references it may hold are math + json (the
    # Pyodide-core rule, same style as the autopilot purity test).
    code = textwrap.dedent(
        """
        import json  # preload json + its deps into `before`
        import importlib.util, sys, types

        path = sys.argv[1]
        before = set(sys.modules)
        spec = importlib.util.spec_from_file_location("pp_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        loaded = set(sys.modules) - before - {"pp_under_test"}
        assert loaded <= {"math"}, f"unexpected imports loaded: {sorted(loaded)}"
        held = {v.__name__ for v in vars(mod).values()
                if isinstance(v, types.ModuleType)}
        assert held <= {"math", "json"}, f"unexpected module refs: {sorted(held)}"
        print("PURE")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, policy_module.__file__],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PURE" in result.stdout
```

(The `Game` import is used by Task 2's tests in this same file — harmless now.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_policy.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'moonlander.core.policy'`

- [ ] **Step 3: Implement `src/moonlander/core/policy.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_policy.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/moonlander/core/policy.py tests/test_policy.py
git commit -m "feat: pure-stdlib MLP policy forward pass (core/policy.py)"
```

---

### Task 2: `Game.set_policy` / `step_policy` + CONTRACT §2/§11

**Files:**
- Modify: `src/moonlander/core/game.py` (imports ~line 15, `__init__` ~line 47, new methods after `_auto_controls` ~line 190)
- Modify: `docs/CONTRACT.md` (§2 API block + bullets; new §11)
- Test: `tests/test_policy.py` (append)

- [ ] **Step 1: Append the failing Game-integration tests to `tests/test_policy.py`**

```python
# ------------------------------------------------- Game.set_policy/step_policy

def fly_policy(game, seed, max_steps=7200):
    """Reset to ``seed``, step_policy until terminal; returns all frame JSONs."""
    game.reset(seed=seed)
    frames = [game.frame_json()]
    for _ in range(max_steps):
        frames.append(game.step_policy())
        if json.loads(frames[-1])["status"] != "flying":
            break
    return frames


def test_step_policy_deterministic_byte_identical_frames():
    # Always-thrust policy: climbs out of bounds in a few seconds — a short,
    # fully deterministic episode that exercises the thrust path.
    pj = policy_json(b2=[0.0, 0.0, 0.0, 1.0])
    runs = []
    for _ in range(2):
        g = Game(mode="classic", preset="cadet")
        g.set_policy(pj)
        runs.append(fly_policy(g, seed=11))
    assert runs[0] == runs[1]  # byte-identical, every frame


def test_step_policy_after_terminal_is_noop():
    g = Game(mode="classic", preset="cadet")
    g.set_policy(policy_json())  # all-noop policy: free fall -> crash
    last = fly_policy(g, seed=3)[-1]
    assert json.loads(last)["status"] != "flying"
    assert g.step_policy() == last  # §2: byte-identical no-op


def test_step_policy_without_policy_raises_runtime_error():
    g = Game(mode="classic")
    g.reset(seed=0)
    with pytest.raises(RuntimeError, match="set_policy"):
        g.step_policy()


def test_step_policy_raises_in_gym_mode():
    g = Game(mode="gym")
    g.set_policy(policy_json())
    g.reset(seed=0)
    with pytest.raises(NotImplementedError):
        g.step_policy()


def test_step_policy_raises_on_multi_lander_game():
    g = Game(mode="classic", n_landers=2)
    g.set_policy(policy_json())
    g.reset(seed=0)
    with pytest.raises(ValueError, match="one lander"):
        g.step_policy()


def test_set_policy_rejects_garbage():
    g = Game(mode="classic")
    with pytest.raises(ValueError):
        g.set_policy('{"format": "nope"}')


def test_policy_survives_reset():
    # §2: set once, fly many episodes (the web binge-watch path).
    g = Game(mode="classic")
    g.set_policy(policy_json())
    g.reset(seed=1)
    g.step_policy()
    g.reset(seed=2)
    g.step_policy()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_policy.py -q`
Expected: new tests FAIL with `AttributeError: 'Game' object has no attribute 'set_policy'`

- [ ] **Step 3: Implement in `src/moonlander/core/game.py`**

3a. Add the import after `from .physics import LanderState`:

```python
from .policy import Policy
```

3b. In `Game.__init__`, after `self.n_landers = n_landers`:

```python
        self.policy = None  # trained policy (§2 set_policy); survives reset
```

3c. Insert after `_auto_controls` (before `_tick`):

```python
    def set_policy(self, policy_json):
        """Attach a trained policy for ``step_policy`` (CONTRACT §2, schema §11).

        ``policy_json`` is the JSON string written by ``moonlander.train_cem``
        — the same artifact the browser fetches. Survives ``reset``;
        validation problems raise ValueError.
        """
        self.policy = Policy.from_json(policy_json)

    def step_policy(self):
        """One tick flown by the attached trained policy (CONTRACT §2).

        Classic mode only; single-lander games only; requires a prior
        ``set_policy``. Same no-op-after-terminal and JSON-string semantics
        as ``step``.
        """
        if self.mode != "classic":
            raise NotImplementedError(
                "step_policy is classic-mode only; the policy does not fly gym-mode engines"
            )
        if self.n_landers != 1:
            raise ValueError(
                f"step_policy() drives exactly one lander; this game has {self.n_landers}"
            )
        if self.policy is None:
            raise RuntimeError(
                "no policy attached — call set_policy(policy_json) first"
            )
        rotate, thrust = self.policy.act(self.obs(0))
        self._tick([(rotate, thrust, 0)])
        return self.frame_json()
```

- [ ] **Step 4: Run to verify pass + no regressions**

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS (92 existing + new)

- [ ] **Step 5: Update `docs/CONTRACT.md`** (same change, per the contract's own rule)

5a. §2 code block — add after the `step_auto_all` line:

```python
g.set_policy(policy_json)              # attach a trained policy (schema §11)
frame_json   = g.step_policy()         # n_landers == 1 trained-policy tick
```

5b. §2 bullets — add after the autopilot bullet:

```markdown
- Trained policy (classic only, NotImplementedError in gym mode; single-lander
  only, ValueError otherwise): `set_policy(policy_json)` attaches a §11 MLP —
  it survives `reset()` and rides the Game instance; bad payloads raise
  ValueError. `step_policy()` without a prior `set_policy` raises RuntimeError.
  Same no-op-after-terminal and JSON-string semantics as `step`.
```

5c. Append new §11 at the end of the file:

```markdown
## 11. Trained policy (AI PILOT)

The simplest learned pilot: a 14 → hidden(16) → 4 tanh MLP (~308 weights),
argmax over logits, trained by the cross-entropy method in
`python -m moonlander.train_cem` (numpy + gymnasium — training side only).
Inference is pure stdlib (`moonlander/core/policy.py`, math + json only), so
the artifact that trains is the artifact that flies in Pyodide.

**Artifact** `web/assets/policy.json` (committed — deploys with the site):

```json
{
  "format": "mlp-tanh-argmax/v1",
  "sizes": [14, 16, 4],
  "w1": [["... 14 floats"], "... x16"], "b1": ["... 16 floats"],
  "w2": [["... 16 floats"], "... x4"],  "b2": ["... 4 floats"],
  "meta": {
    "preset": "trainee", "pop": 64, "elite": 10, "episodes": 3, "gens": 60,
    "hidden": 16, "noise": 0.02, "seed": 0,
    "fitness_history": [{"gen": 1, "best": 0.0, "elite_mean": 0.0, "pop_mean": 0.0}],
    "eval": {"seeds": "0..29", "landed": 0, "perfect": 0}
  }
}
```

- `w1[j][i]` = weight from obs index `i` (§6 order) to hidden unit `j`;
  `w2[k][j]` likewise. Forward pass: `h = tanh(w1·obs + b1)`,
  `logits = w2·h + b2`, action = argmax (ties → lowest index), then the env's
  Discrete(4) mapping: 0 noop · 1 rotate left (+1) · 2 rotate right (−1) ·
  3 thrust.
- Validation (ValueError): format string must match, sizes `[14, h ≥ 1, 4]`,
  exact row/vector lengths, every entry a finite number.
- Trainer: fitness = mean **undiscounted** return over per-generation shared
  episode seeds (`game_seed`s — common random numbers), elite refit + a std
  noise floor. No discount factor anywhere — structurally immune to the §7
  γ-horizon trap. Same `--seed` → same run.
- `web/ml.html` ("THE MACHINERY") documents the network, the algorithm, and
  draws the learning curve from `meta.fitness_history`.
```

5d. Header paragraph — extend the revision sentence:

old:
```
The current revision
(**0.4.0**) adds terrain macro-variation (§3) and the vector stroke font +
docs page on the web side (§8/§9).
```
new:
```
The 0.4.0 revision added terrain macro-variation (§3) and the vector stroke
font + docs page on the web side (§8/§9). The current revision (**0.5.0**)
adds the trained AI pilot: `set_policy`/`step_policy` (§2), the `policy.json`
artifact + CEM trainer (§11), and the web AI PILOT mode + `ml.html` (§8).
```

- [ ] **Step 6: Commit**

```bash
git add src/moonlander/core/game.py tests/test_policy.py docs/CONTRACT.md
git commit -m "feat: Game.set_policy/step_policy — trained policy flies via the autopilot pattern (CONTRACT §2/§11)"
```

---

### Task 3: `moonlander/train_cem.py` — the CEM trainer

**Files:**
- Create: `src/moonlander/train_cem.py`
- Test: `tests/test_train_cem.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_cem.py`:

```python
"""CEM trainer unit pieces + one tiny end-to-end smoke run (CONTRACT §11)."""

import json

import numpy as np
import pytest

from moonlander import train_cem
from moonlander.core.game import Game
from moonlander.core.policy import Policy
from moonlander.train_cem import cem_update, n_params, unflatten


def test_n_params_counts_the_14_h_4_mlp():
    assert n_params(16) == 14 * 16 + 16 + 16 * 4 + 4  # == 308
    assert n_params(1) == 14 + 1 + 4 + 4


def test_unflatten_layout_is_w1_b1_w2_b2_row_major():
    hidden = 3
    theta = np.arange(n_params(hidden), dtype=float)
    w1, b1, w2, b2 = unflatten(theta, hidden)
    assert w1.shape == (3, 14) and b1.shape == (3,)
    assert w2.shape == (4, 3) and b2.shape == (4,)
    assert w1[0, 0] == 0.0 and w1[2, 13] == 41.0  # row-major w1 first
    assert b1[0] == 42.0                          # then b1
    assert w2[0, 0] == 45.0                       # then w2
    assert b2[0] == 57.0                          # then b2


def test_cem_update_refits_to_exactly_the_elite():
    samples = np.array([[0.0], [10.0], [20.0], [30.0]])
    fitnesses = np.array([0.0, 3.0, 2.0, 1.0])  # elite_k=2 -> rows 1 and 2
    mean, std = cem_update(samples, fitnesses, elite_k=2)
    assert mean[0] == pytest.approx(15.0)  # (10 + 20) / 2
    assert std[0] == pytest.approx(5.0)


def test_cem_converges_on_a_toy_quadratic():
    rng = np.random.default_rng(0)
    target = np.array([3.0, -2.0])
    mean, std = np.zeros(2), np.full(2, 2.0)
    for _ in range(30):
        samples = rng.normal(mean, std, size=(40, 2))
        fitnesses = -((samples - target) ** 2).sum(axis=1)
        mean, std = cem_update(samples, fitnesses, elite_k=8)
        std += 0.01
    assert np.allclose(mean, target, atol=0.3)


def test_smoke_one_tiny_run_writes_a_flyable_artifact(tmp_path):
    out = tmp_path / "policy.json"
    train_cem.main([
        "--pop", "6", "--elite", "2", "--episodes", "1", "--gens", "2",
        "--hidden", "4", "--seed", "0", "--eval-seeds", "2", "--out", str(out),
    ])
    s = out.read_text()
    Policy.from_json(s)  # §11 schema round-trips through the validator

    g = Game(mode="classic", preset="trainee")
    g.set_policy(s)
    g.reset(seed=0)
    frame = json.loads(g.step_policy())
    assert frame["t"] > 0  # it flies

    meta = json.loads(s)["meta"]
    assert [h["gen"] for h in meta["fitness_history"]] == [1, 2]
    assert set(meta["eval"]) == {"seeds", "landed", "perfect"}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_train_cem.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'moonlander.train_cem'`

- [ ] **Step 3: Implement `src/moonlander/train_cem.py`**

```python
"""CEM trainer for the tiny MLP pilot — the simplest ML that flies (§11).

The whole idea of the cross-entropy method: keep a Gaussian over ALL the
network's weights; each generation sample candidate networks, fly each one,
keep the elite, refit the Gaussian to the elite, repeat. No gradients, no
backprop — and no discount factor: fitness is the raw undiscounted episode
return, so the γ-horizon trap documented in DESIGN.md cannot exist here.

Run:   python -m moonlander.train_cem --seed 0
Deps:  numpy + gymnasium (the [env] extra) — training side only. Nothing
imports this module automatically, so ``import moonlander`` stays
stdlib-pure (tests/test_env.py enforces that).
"""

import argparse
import json

import numpy as np

from .env import MoonLanderEnv

OBS_DIM = 14
N_ACTIONS = 4
FORMAT = "mlp-tanh-argmax/v1"  # must match core/policy.py


def n_params(hidden):
    """Total weight count of the 14 -> hidden -> 4 MLP."""
    return OBS_DIM * hidden + hidden + hidden * N_ACTIONS + N_ACTIONS


def unflatten(theta, hidden):
    """Flat weight vector -> (w1, b1, w2, b2), row-major, in that order."""
    a, b = 0, OBS_DIM * hidden
    w1 = theta[a:b].reshape(hidden, OBS_DIM)
    a, b = b, b + hidden
    b1 = theta[a:b]
    a, b = b, b + hidden * N_ACTIONS
    w2 = theta[a:b].reshape(N_ACTIONS, hidden)
    b2 = theta[b:b + N_ACTIONS]
    return w1, b1, w2, b2


def act(parts, obs):
    """argmax action of the MLP — numpy mirror of core/policy.py (ties -> lowest)."""
    w1, b1, w2, b2 = parts
    h = np.tanh(w1 @ obs + b1)
    return int(np.argmax(w2 @ h + b2))


def episode_return(env, parts, seed):
    """Undiscounted return of one greedy episode on ``game_seed=seed``."""
    obs, _ = env.reset(options={"game_seed": seed})
    total = 0.0
    while True:
        obs, r, terminated, truncated, _ = env.step(act(parts, obs))
        total += r
        if terminated or truncated:
            return total


def cem_update(samples, fitnesses, elite_k):
    """One CEM step: (mean, std) of the elite_k best samples (higher = better)."""
    order = np.argsort(fitnesses)[::-1]
    elite = samples[order[:elite_k]]
    return elite.mean(axis=0), elite.std(axis=0)


def evaluate(env, parts, n_seeds):
    """(landed, perfect) over game_seeds 0..n_seeds-1 — the web-parity seeds."""
    landed = perfect = 0
    for seed in range(n_seeds):
        obs, _ = env.reset(options={"game_seed": seed})
        while True:
            obs, _r, terminated, truncated, info = env.step(act(parts, obs))
            if terminated or truncated:
                break
        kind = info.get("outcome", {}).get("kind")
        landed += kind in ("perfect", "hard")
        perfect += kind == "perfect"
    return landed, perfect


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Train the tiny MLP pilot with the cross-entropy method."
    )
    ap.add_argument("--pop", type=int, default=64, help="candidates per generation")
    ap.add_argument("--elite", type=int, default=10, help="winners refit each generation")
    ap.add_argument("--episodes", type=int, default=3,
                    help="episodes per candidate (seeds shared across the generation)")
    ap.add_argument("--gens", type=int, default=60, help="generations")
    ap.add_argument("--hidden", type=int, default=16, help="hidden layer width")
    ap.add_argument("--noise", type=float, default=0.02,
                    help="std floor added each generation (never stop exploring)")
    ap.add_argument("--seed", type=int, default=0, help="master seed for the whole run")
    ap.add_argument("--preset", default="trainee", help="difficulty preset to train on")
    ap.add_argument("--eval-seeds", type=int, default=30,
                    help="final-eval episode count (game_seeds 0..n-1)")
    ap.add_argument("--out", default="web/assets/policy.json", help="artifact path")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    env = MoonLanderEnv(preset=args.preset, frame_skip=4)
    n = n_params(args.hidden)
    mean, std = np.zeros(n), np.full(n, 1.0)

    history = []
    for gen in range(1, args.gens + 1):
        # Common random numbers: every candidate flies the SAME episode seeds
        # this generation — candidates compete on skill, not terrain luck.
        seeds = [int(s) for s in rng.integers(0, 2**31, size=args.episodes)]
        samples = rng.normal(mean, std, size=(args.pop, n))
        fitnesses = np.array([
            np.mean([episode_return(env, unflatten(th, args.hidden), s)
                     for s in seeds])
            for th in samples
        ])
        mean, std = cem_update(samples, fitnesses, args.elite)
        std += args.noise
        top = np.sort(fitnesses)[::-1]
        history.append({
            "gen": gen,
            "best": round(float(top[0]), 2),
            "elite_mean": round(float(top[:args.elite].mean()), 2),
            "pop_mean": round(float(fitnesses.mean()), 2),
        })
        h = history[-1]
        print(f"gen {gen:3d}/{args.gens}  best {h['best']:9.2f}  "
              f"elite {h['elite_mean']:9.2f}  pop {h['pop_mean']:9.2f}",
              flush=True)

    parts = unflatten(mean, args.hidden)
    landed, perfect = evaluate(env, parts, args.eval_seeds)
    print(f"eval: {landed}/{args.eval_seeds} landed ({perfect} perfect) on "
          f"{args.preset} game_seeds 0..{args.eval_seeds - 1}")

    w1, b1, w2, b2 = parts
    artifact = {
        "format": FORMAT,
        "sizes": [OBS_DIM, args.hidden, N_ACTIONS],
        "w1": w1.tolist(), "b1": b1.tolist(),
        "w2": w2.tolist(), "b2": b2.tolist(),
        "meta": {
            "preset": args.preset, "pop": args.pop, "elite": args.elite,
            "episodes": args.episodes, "gens": args.gens,
            "hidden": args.hidden, "noise": args.noise, "seed": args.seed,
            "fitness_history": history,
            "eval": {"seeds": f"0..{args.eval_seeds - 1}",
                     "landed": landed, "perfect": perfect},
        },
    }
    with open(args.out, "w") as f:
        json.dump(artifact, f, allow_nan=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_train_cem.py -q`
Expected: 5 PASS (smoke test takes ~1–3 s)

- [ ] **Step 5: Full suite, then commit**

Run: `.venv/bin/python -m pytest -q` — all PASS.

```bash
git add src/moonlander/train_cem.py tests/test_train_cem.py
git commit -m "feat: CEM trainer (python -m moonlander.train_cem) — no gradients, no gamma"
```

---

### Task 4: Version bump 0.4.0 → 0.5.0 (the four coupled spots) + wheel rebuild

**Files:**
- Modify: `pyproject.toml:7` (`version = "0.4.0"` → `"0.5.0"`)
- Modify: `src/moonlander/__init__.py` (`__version__ = "0.4.0"` → `"0.5.0"`)
- Modify: `web/app.js:477` (`assets/moonlander-0.4.0-py3-none-any.whl` → `0.5.0`)
- Modify: `docs/CONTRACT.md` §8 (wheel filename in the Files line AND the Boot line → `0.5.0`)

- [ ] **Step 1: Make all four edits** (exact strings above; CONTRACT §8 has TWO occurrences)
- [ ] **Step 2: Rebuild the wheel**

Run: `./scripts/build_web.sh`
Expected: `web/assets/moonlander-0.5.0-py3-none-any.whl` exists; the 0.4.0 wheel is gone (script `rm -f`s stale wheels). The wheel is gitignored — nothing to commit there.

- [ ] **Step 3: Run tests, commit**

Run: `.venv/bin/python -m pytest -q` — all PASS.

```bash
git add pyproject.toml src/moonlander/__init__.py web/app.js docs/CONTRACT.md
git commit -m "chore: bump version to 0.5.0 in all four coupled spots"
```

---

### Task 5: Run the real training → commit `web/assets/policy.json`

**Files:**
- Create (generated): `web/assets/policy.json`

- [ ] **Step 1: Train** (minutes; run in background and monitor)

Run: `.venv/bin/python -m moonlander.train_cem --seed 0`
Expected output: 60 `gen k/60 best … elite … pop …` lines with elite_mean clearly rising from large-negative (crashes ≈ −100±) toward positive; then `eval: N/30 landed (M perfect) on trainee game_seeds 0..29`; then `wrote web/assets/policy.json`.

- [ ] **Step 2: Acceptance gate — "visibly learned, lands sometimes"**

Accept if `landed >= 6` (≥ 20% on trainee) AND the fitness history shows clear improvement. If not: retry `--seed 1`, then `--seed 2`, then `--gens 100 --pop 96 --seed 0`. Ship the first run that passes; record which.

- [ ] **Step 3: Sanity-check the artifact against the validator**

Run: `.venv/bin/python -c "from moonlander.core.policy import Policy; Policy.from_json(open('web/assets/policy.json').read()); print('VALID')"`
Expected: `VALID`

- [ ] **Step 4: Commit the artifact**

```bash
git add web/assets/policy.json
git commit -m "feat: trained CEM policy artifact (trainee, seed <S>: <N>/30 landed, <M> perfect)"
```

(Fill `<S>/<N>/<M>` from the actual run.)

---

### Task 6: Web AI PILOT mode (`app.js`, `renderer.js`, `index.html`) + CONTRACT §8

**Files:**
- Modify: `web/app.js` (constants ~line 23, state ~line 40, `handleKeyPress` ~line 89, `bindButton` ~line 191, `drawButtonLabels` ~line 244, `ensureGame` ~line 270, `tick` ~line 331, render view ~line 443, `boot` ~line 466)
- Modify: `web/renderer.js` (view doc ~line 19, new draw fns after `drawSeed` ~line 298, `drawTitle` ~line 415, `render` ~line 451)
- Modify: `web/index.html` (CSS ~line 147, DOM ~line 216, footer ~line 222)
- Modify: `docs/CONTRACT.md` §8

There are no JS tests (project convention); CONTRACT documents behavior, Task 9 verifies in the browser.

- [ ] **Step 1: `app.js` — constants and state**

After `const PRESETS = [...]` add:

```js
  const NOTICE_TIME = 2.0;     // §8: NO POLICY notice duration, seconds
```

After `let preset = loadPreset();` add:

```js
  let aiPilot = false;     // §8/§11: AI PILOT — the trained policy flies lander 0
  let policyJson = null;   // raw §11 policy.json string fetched at boot, or null
  let notice = 0;          // seconds left on the NO POLICY notice
```

- [ ] **Step 2: `app.js` — key handling**

In `handleKeyPress`, insert between the `Digit1/2/3` block and the TITLE catch-all:

```js
    // §8 take-over: while AI PILOT flies, a fresh rotate/thrust PRESS hands
    // control straight back to the human — keydown already added the code to
    // `keys`, so the same tick that disengages also steers. Held keys from
    // before the toggle don't disengage (this is edge-triggered).
    if (state === "FLYING" && aiPilot &&
        (ROTATE_LEFT.includes(code) || ROTATE_RIGHT.includes(code) ||
         THRUST.includes(code))) {
      aiPilot = false;
      return;
    }
    // §8 AI PILOT toggle: P in TITLE/FLYING/ENDED (REVEAL returned above).
    // Like the digits, P NEVER counts as "any key".
    if (code === "KeyP") {
      toggleAiPilot();
      return;
    }
```

- [ ] **Step 3: `app.js` — `toggleAiPilot` helper** (insert after `selectPreset`)

```js
  // §8/§11: toggle the trained-policy pilot. With no policy artifact (fetch
  // failed / not trained yet) show the transient NO POLICY notice instead.
  function toggleAiPilot() {
    if (!policyJson) {
      notice = NOTICE_TIME;
      return;
    }
    aiPilot = !aiPilot;
  }
```

- [ ] **Step 4: `app.js` — touch wiring**

4a. In `bindButton`, update the data-code comment to `// BtnLeft | BtnRight | BtnThrust | BtnObs | BtnAi`, and insert after the `BtnObs` branch:

```js
      if (code === "BtnAi") {
        // §8 AI = touch parity with KeyP, same gating as the keyboard path
        // (no LOADING/ERROR/REVEAL) — and never "any key".
        if (state !== "LOADING" && state !== "ERROR" && state !== "REVEAL") {
          toggleAiPilot();
        }
        return;
      }
```

4b. Still in `bindButton`, right after `keys.add(code);` insert:

```js
      if (state === "FLYING" && aiPilot) aiPilot = false; // §8 take-over (touch)
```

4c. In `drawButtonLabels` add:

```js
    paint("label-ai", "AI", 26);
```

- [ ] **Step 5: `app.js` — `ensureGame` re-attaches the policy**

After `gamePreset = preset;` add:

```js
    if (policyJson) {
      // §11: re-attach the policy to every (re)built Game — AI PILOT must
      // survive preset changes and attract<->human rebuilds. Weights cross
      // the boundary as a JSON string, once per Game (§2).
      try {
        py.globals.set("policy_json", policyJson);
        py.runPython("game.set_policy(policy_json)");
      } catch (_) {
        policyJson = null; // corrupt artifact -> AI PILOT unavailable (§8)
      }
    }
```

- [ ] **Step 6: `app.js` — tick: policy path + notice countdown**

6a. At the top of `tick()`, after the LOADING/ERROR guard:

```js
    if (notice > 0) notice -= DT; // §8: NO POLICY notice countdown
```

6b. Replace the FLYING input block:

old:
```js
    if (state === "FLYING") {
      // Read held inputs (keys + arcade buttons), step Python, parse the frame.
      const left = ROTATE_LEFT.some((k) => keys.has(k));
      const right = ROTATE_RIGHT.some((k) => keys.has(k));
      const rotate = (left ? 1 : 0) - (right ? 1 : 0); // +1 CCW / tilt left (§2)
      const thrust = THRUST.some((k) => keys.has(k));

      frame = JSON.parse(game.step(rotate, thrust)); // n_landers == 1 (§2)
```
new:
```js
    if (state === "FLYING") {
      if (aiPilot && policyJson) {
        // §8/§11 AI PILOT: the trained policy flies lander 0 inside Python —
        // JS never even sees the action, only the resulting frame.
        frame = JSON.parse(game.step_policy());
      } else {
        // Read held inputs (keys + arcade buttons), step Python, parse the frame.
        const left = ROTATE_LEFT.some((k) => keys.has(k));
        const right = ROTATE_RIGHT.some((k) => keys.has(k));
        const rotate = (left ? 1 : 0) - (right ? 1 : 0); // +1 CCW / tilt left (§2)
        const thrust = THRUST.some((k) => keys.has(k));
        frame = JSON.parse(game.step(rotate, thrust)); // n_landers == 1 (§2)
      }
```

- [ ] **Step 7: `app.js` — render view + boot fetch**

7a. In the `Renderer.render({...})` call, after `overlay,` add:

```js
      aiPilot,                      // §8: AI PILOT indicator + title hint
      notice: notice > 0,           // §8: transient NO POLICY notice
```

7b. In `boot()`, after the wheel install (before `stage = "CREATING GAME";`):

```js
      stage = "FETCHING POLICY";
      try {
        // §8/§11: the trained-policy artifact is optional — without it the
        // game is fully playable and AI PILOT just reports NO POLICY.
        const resp = await fetch(
          new URL("assets/policy.json", location.href).href
        );
        if (resp.ok) policyJson = await resp.text();
      } catch (_) { /* file:// or missing artifact — AI PILOT unavailable */ }
```

- [ ] **Step 8: `renderer.js` — indicator, notice, title hint**

8a. View doc comment: after the `overlay:` line add:

```js
 *     aiPilot:       boolean  (AI PILOT engaged — §8 indicator + title hint),
 *     notice:        boolean  (transient NO POLICY notice — §8),
```

8b. After `drawSeed` insert:

```js
  // §8 AI PILOT indicator: blinking under the left HUD block while the
  // trained policy flies — unmistakably "the machine has the stick".
  function drawAiPilot() {
    glow(true);
    ctx.strokeStyle = "#fff";
    if (Effects.blink()) {
      VectorFont.draw(ctx, "AI PILOT", 40, 212, 20, { align: "left" });
    }
  }

  // §8 transient notice: P/AI pressed but no policy.json artifact exists.
  function drawNotice() {
    glow(true);
    ctx.strokeStyle = "#fff";
    centeredText("NO POLICY", 600, 28);
    ctx.strokeStyle = "#999";
    centeredText("TRAIN ONE:  PYTHON -M MOONLANDER.TRAIN_CEM", 642, 16);
  }
```

8c. `drawTitle` — change signature and add the armed line (copyright moves 530 → 538):

old:
```js
  function drawTitle(preset) {
```
new:
```js
  function drawTitle(preset, aiPilot) {
```
old:
```js
    drawPresetMenu(preset, 450);
    // Cabinet-style copyright line, the way 1979 did it.
    ctx.strokeStyle = "#777";
    centeredText("(C) 2026 BIJAN MEHR", 530, 15);
```
new:
```js
    drawPresetMenu(preset, 450);
    if (aiPilot) {
      // §8: armed on the title — the next episode starts with the policy flying.
      centeredText("AI PILOT ARMED", 496, 16);
    }
    // Cabinet-style copyright line, the way 1979 did it.
    ctx.strokeStyle = "#777";
    centeredText("(C) 2026 BIJAN MEHR", 538, 15);
```

8d. `render()` switch:
- TITLE case: `drawTitle(view.preset)` → `drawTitle(view.preset, view.aiPilot)`
- REVEAL case: `if (view.attract) drawTitle(view.preset);` → `if (view.attract) drawTitle(view.preset, view.aiPilot);`
- FLYING case, after the overlay line: `if (view.aiPilot) drawAiPilot();`
- ENDED case, after the overlay line: `if (view.aiPilot) drawAiPilot();`
- After the closing `}` of the switch, before `}` of `render`:

```js
    if (view.notice && (view.state === "TITLE" || view.state === "FLYING" ||
                        view.state === "ENDED")) {
      drawNotice(); // §8: NO POLICY, ~2 s
    }
```

- [ ] **Step 9: `index.html` — AI button (CSS + DOM + footer)**

9a. After the `#btn-obs { ... }` CSS block add:

```css
    #btn-ai {
      width: 84px;
      height: 48px;
      right: calc(112px + env(safe-area-inset-right, 0px));
      top: calc(14px + env(safe-area-inset-top, 0px));
    }
```

9b. Change `#btn-obs canvas { width: 55px; height: 22px; }` →
`#btn-obs canvas, #btn-ai canvas { width: 55px; height: 22px; }`

9c. After the `btn-obs` div in `#touch` add:

```html
    <div class="abtn" id="btn-ai" data-code="BtnAi">
      <canvas id="label-ai" width="110" height="44"></canvas>
    </div>
```

9d. Footer line: add ` &nbsp;&middot;&nbsp; P AI pilot` after `O agent view`, and
` &nbsp;&middot;&nbsp; <a href="ml.html">THE MACHINERY</a>` after the DOCS link.

- [ ] **Step 10: CONTRACT §8 updates** (same commit)

- Files line: add `web/ml.html` (machinery page) and `web/assets/policy.json` (trained-policy artifact, §11).
- Boot paragraph: after the micropip sentence add: `then fetch("assets/policy.json") — optional; on failure AI PILOT is unavailable and the game is unaffected. The string is passed to game.set_policy on every Game (re)build.`
- Keys line: add `· P AI PILOT toggle (TITLE/FLYING/ENDED, never "any key"; while it flies, a fresh rotate/thrust press hands control back to the human)`.
- Touch list: add an `AI` button bullet (top-right, left of OBS, parity with `P`, never "any key").
- Add a short **AI PILOT** paragraph: blinking `AI PILOT` HUD indicator while engaged; `AI PILOT ARMED` line on the title; transient `NO POLICY` notice when no artifact; the mode persists across episodes and preset changes.

- [ ] **Step 11: Manual smoke + commit**

Run: `./scripts/serve.sh` then open `http://localhost:8000` — title shows; press `P` (AI PILOT ARMED appears), any key → policy flies; arrow key takes over; `P` re-engages; `1/2/3` preset change keeps AI armed. Then:

```bash
git add web/app.js web/renderer.js web/index.html docs/CONTRACT.md
git commit -m "feat: AI PILOT mode — P key / AI arcade button, take-over, NO POLICY notice (CONTRACT §8)"
```

---

### Task 7: `web/ml.html` — THE MACHINERY page + cross-links

**Files:**
- Create: `web/ml.html`
- Modify: `web/docs.html` (controls list; new AI PILOT section after SENSOR MODES)

- [ ] **Step 1: Create `web/ml.html`** — same theme as docs.html (vectorfont-canvas headings, monospace body, no Pyodide, relative links). Full content:

Head: same favicon/meta pattern as docs.html, title `LUNAR LANDER — THE MACHINERY`, plus styles for the curve canvas (`#curve { width: 100%; max-width: 880px; }`) and `.num { color: #fff; }`.

Body sections (each heading a `<canvas data-h="...">`):

1. **masthead** `LUNAR LANDER — THE MACHINERY` + intro paragraph: a 308-number
   neural network flies the lander; press `P` in the game to watch it; this
   page explains every moving part — there is no black box.
2. **THE LOOP** — sense → think → act at 60 Hz; per tick: Python builds the
   14-float observation, the network turns it into 1 of 4 actions, physics
   integrates. `<pre>` diagram: `OBS(14) ──► MLP ──► ARGMAX ──► {NOOP, LEFT, RIGHT, THRUST} ──► PHYSICS ──► repeat`.
3. **WHAT IT SEES** — short version of the §6 table (link to docs.html for the
   full one); note the `O` overlay shows these live, also while the AI flies.
4. **THE NETWORK** — ASCII diagram of 14 → 16(tanh) → 4; "308 numbers total";
   the actual forward pass verbatim from `core/policy.py` in a `<pre>` block
   (the `act` method + ACTIONS tuple); a line on argmax determinism (ties →
   lowest index) and that inference is pure stdlib running in Pyodide.
5. **HOW IT LEARNS** — CEM in one breath: *guess 64 networks, fly each on the
   same 3 terrains, keep the 10 best, aim your next guesses at the winners,
   repeat 60 times.* The actual `cem_update` + generation loop verbatim in
   `<pre>`; bullets for the two tricks: common random numbers (same seeds per
   generation — skill, not terrain luck) and the `std += 0.02` noise floor
   (never stop exploring).
6. **WHY NO GRADIENTS, WHY NO GAMMA** — honest trade-offs (CEM scales badly to
   big networks; that's the point — this network is tiny); the audit story:
   at γ=0.99 hovering provably beat landing, PPO would have converged to it —
   CEM's fitness is the raw undiscounted return, so the trap cannot exist;
   pointer: PPO baseline is the roadmap's next phase.
7. **THE RESULTS** — `<canvas id="curve" width="880" height="300"></canvas>`
   + `<p id="curve-note">` fallback line; `<p id="eval-line">` filled from
   `meta.eval` (landed/perfect over seeds 0..29, vs the scripted autopilot's
   30/30 on trainee); "trained on TRAINEE — switch difficulty (`1/2/3`) and
   watch it struggle"; honesty note that imperfection is the teaching value.
8. **TRY IT** — press `P` (or the `AI` button) on
   <a href="index.html">the game</a>; `?seed=N` pins terrain for comparing
   your flying against the network's on identical ground.

Footer scripts: the same `canvas[data-h]` heading painter as docs.html, plus the curve loader:

```html
  <script src="vectorfont.js"></script>
  <script>
    // Headings (same painter as docs.html).
    for (const c of document.querySelectorAll("canvas[data-h]")) {
      const size = Number(c.dataset.size || 24);
      c.width = Math.ceil(VectorFont.measure(c.dataset.h, size)) + 16;
      c.height = Math.ceil(size * 1.6);
      const ctx = c.getContext("2d");
      ctx.strokeStyle = "#fff";
      ctx.shadowColor = "#fff";
      ctx.shadowBlur = 6;
      VectorFont.draw(ctx, c.dataset.h, 8, size * 1.3, size);
    }

    // Learning curve, drawn from the real artifact's meta.fitness_history.
    // fetch fails on file:// — the page stays readable, the note explains.
    const curve = document.getElementById("curve");
    const note = document.getElementById("curve-note");
    const evalLine = document.getElementById("eval-line");

    function drawCurve(meta) {
      const hist = meta.fitness_history;
      const ctx = curve.getContext("2d");
      const W = curve.width, H = curve.height;
      const PAD = { l: 70, r: 16, t: 16, b: 40 };
      const xs = hist.map((h) => h.gen);
      const series = [
        ["pop_mean", "#666"],
        ["elite_mean", "#aaa"],
        ["best", "#fff"],
      ];
      const all = hist.flatMap((h) => series.map(([k]) => h[k]));
      const lo = Math.min(...all), hi = Math.max(...all);
      const X = (g) => PAD.l + (g - xs[0]) / (xs[xs.length - 1] - xs[0]) *
                       (W - PAD.l - PAD.r);
      const Y = (v) => H - PAD.b - (v - lo) / (hi - lo) * (H - PAD.t - PAD.b);

      ctx.strokeStyle = "#333";
      ctx.lineWidth = 1;
      ctx.strokeRect(PAD.l, PAD.t, W - PAD.l - PAD.r, H - PAD.t - PAD.b);
      if (lo < 0 && hi > 0) {           // the zero line: crash vs not
        ctx.beginPath();
        ctx.moveTo(PAD.l, Y(0));
        ctx.lineTo(W - PAD.r, Y(0));
        ctx.stroke();
      }
      ctx.shadowColor = "#fff";
      for (const [key, color] of series) {
        ctx.strokeStyle = color;
        ctx.shadowBlur = key === "best" ? 6 : 0;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        hist.forEach((h, i) => {
          const x = X(h.gen), y = Y(h[key]);
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
      }
      ctx.shadowBlur = 6;
      ctx.strokeStyle = "#fff";
      VectorFont.draw(ctx, "RETURN", 8, 28, 13);
      VectorFont.draw(ctx, "GENERATION", W / 2, H - 8, 13, { align: "center" });
      VectorFont.draw(ctx, String(Math.round(hi)), PAD.l - 8, PAD.t + 12, 12,
                      { align: "right" });
      VectorFont.draw(ctx, String(Math.round(lo)), PAD.l - 8, H - PAD.b, 12,
                      { align: "right" });
      VectorFont.draw(ctx, "BEST", W - PAD.r - 8, PAD.t + 18, 12,
                      { align: "right" });
    }

    fetch("assets/policy.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!data) throw new Error("no artifact");
        drawCurve(data.meta);
        const ev = data.meta.eval;
        evalLine.innerHTML =
          `Final eval: <span class="num">${ev.landed}/30</span> landed ` +
          `(<span class="num">${ev.perfect}</span> perfect) on TRAINEE seeds ` +
          `${ev.seeds} — the scripted autopilot lands 30/30 there. ` +
          `The network earned its landings; the autopilot was told how.`;
        note.textContent =
          `TRAINED ${data.meta.gens} GENERATIONS × ${data.meta.pop} ` +
          `CANDIDATES × ${data.meta.episodes} EPISODES, SEED ${data.meta.seed}.`;
      })
      .catch(() => {
        note.textContent =
          "SERVE THE SITE TO SEE THE TRAINING CURVE (fetch is blocked on file://).";
      });
  </script>
```

(Implementer: keep body copy tight and honest; all code excerpts must be copied verbatim from the shipped `policy.py` / `train_cem.py`.)

- [ ] **Step 2: `web/docs.html` — controls + AI PILOT section**

2a. In HOW TO FLY, first `<li>`, after the `O` entry add: `&nbsp;&middot;&nbsp; <code>P</code> AI pilot`.

2b. After the SENSOR MODES section insert:

```html
    <canvas data-h="THE AI PILOT"></canvas>
    <p>
      Press <code>P</code> (or the <code>AI</code> button) and a trained
      308-parameter neural network flies the lander — the same pure-Python
      forward pass the training side used, running live in your browser.
      Touch any flight control and it hands the stick back. How it learned,
      line by line: <a href="ml.html">THE MACHINERY</a>.
    </p>
```

- [ ] **Step 3: Verify + commit**

Open `http://localhost:8000/ml.html` — headings render, curve draws from the committed artifact, eval line filled. Open `docs.html` — new section renders.

```bash
git add web/ml.html web/docs.html
git commit -m "feat: ml.html — THE MACHINERY explainer page with live learning curve"
```

---

### Task 8: README + DESIGN.md

**Files:**
- Modify: `README.md` (controls line ~63, Training section ~96, roadmap row 4 ~48)
- Modify: `docs/DESIGN.md` (Decisions table ~27, Roadmap item 4 ~77)

- [ ] **Step 1: README**

- Controls line: `O\` agent view` → `O\` agent view · \`P\` AI pilot`.
- Roadmap row 4: `| 4 | PPO baseline (SB3), in-browser trained-agent showcase | next |` → `| 4 | PPO baseline (SB3), in-browser trained-agent showcase | 🔶 CEM baseline shipped — PPO next |`
- After the multi-lander code block in **Training interface**, add:

```markdown
The simplest learned pilot ships first: a 308-parameter MLP trained with the
cross-entropy method — no gradients, no discount factor, numpy only:

​```bash
.venv/bin/python -m moonlander.train_cem      # minutes → web/assets/policy.json
​```

Press `P` in the web game to hand it the stick. Every moving part is explained
at **[ml.html](https://bijanmehr.github.io/MultiLander/ml.html)**.
```

- [ ] **Step 2: DESIGN.md**

- Decisions table, append row:
  `| First learned pilot (0.5.0) | CEM over a 308-param MLP; pure-stdlib forward pass in core; weights as JSON | simplest teachable baseline — no gradients, no γ (immune to the discount-horizon trap by construction); PPO stays phase 4 |`
- Roadmap item 4, prepend: `(0.5.0 shipped the warm-up: CEM baseline + AI PILOT mode + ml.html.)`

- [ ] **Step 3: Commit**

```bash
git add README.md docs/DESIGN.md
git commit -m "docs: README + DESIGN for the CEM AI pilot"
```

---

### Task 9: Full verification

- [ ] **Step 1:** `.venv/bin/python -m pytest -q` — ALL pass (expect 92 + ~21 new).
- [ ] **Step 2:** `./scripts/build_web.sh` — wheel `moonlander-0.5.0-…` rebuilt clean.
- [ ] **Step 3:** Serve and verify in a real browser (use the `verify`/`run` skill or manual): boot OK; attract OK; `P` → ARMED → episode → policy flies (HUD blinks AI PILOT); arrow-key take-over; `O` overlay during AI flight; `1/2/3` preset switch keeps mode; `ml.html` curve + eval line; `docs.html` new section; footer links.
- [ ] **Step 4:** Negative path: temporarily rename `web/assets/policy.json` → reload → `P` shows NO POLICY notice, game plays normally → restore the file.
- [ ] **Step 5:** `git status` clean except intended; final commit if anything remains.
