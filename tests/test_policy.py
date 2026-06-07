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


def test_failed_set_policy_preserves_the_active_policy():
    # Import-flow guarantee (§2): validation happens before assignment, so a
    # rejected payload leaves the previously attached policy flying.
    g = Game(mode="classic", preset="cadet")
    g.set_policy(policy_json(b2=[0.0, 0.0, 0.0, 1.0]))  # always-thrust
    g.reset(seed=11)
    before = g.step_policy()
    with pytest.raises(ValueError):
        g.set_policy('{"format": "nope"}')
    after = g.step_policy()  # still flying on the old policy, no RuntimeError
    assert json.loads(after)["t"] > json.loads(before)["t"]


def test_policy_survives_reset():
    # §2: set once, fly many episodes (the web binge-watch path).
    g = Game(mode="classic")
    g.set_policy(policy_json())
    g.reset(seed=1)
    g.step_policy()
    g.reset(seed=2)
    g.step_policy()


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
