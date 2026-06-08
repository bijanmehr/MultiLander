"""The training-stack template stays runnable and web-compatible (examples/).

Pins the two things that would silently rot the template: its forward pass
must agree with the real in-browser Policy, and `main` must still produce a
flyable artifact end to end.
"""

import json

import numpy as np
import pytest

from examples import train_template as T
from moonlander.core.game import Game
from moonlander.core.policy import Policy


def test_n_params_and_unflatten_match_the_web_mlp():
    assert T.n_params(16) == 14 * 16 + 16 + 16 * 4 + 4  # == 308
    theta = np.arange(T.n_params(3), dtype=float)
    w1, b1, w2, b2 = T.unflatten(theta, 3)
    assert w1.shape == (3, 14) and b1.shape == (3,)
    assert w2.shape == (4, 3) and b2.shape == (4,)


def test_template_forward_agrees_with_browser_policy():
    # The template's forward() and the shipped Policy.act() must pick the same
    # action for the same weights — otherwise "train here, fly in the browser"
    # is a lie. Check across random weights and observations.
    rng = np.random.default_rng(0)
    hidden = 5
    for _ in range(20):
        theta = rng.normal(size=T.n_params(hidden))
        policy = Policy.from_json(T.to_policy_json(theta, hidden))
        for _ in range(10):
            obs = rng.normal(size=14)
            mine = T.forward(theta, hidden, obs)
            theirs = policy.act(obs.tolist())
            assert T.OBS  # labels present
            # Policy returns (rotate, thrust); map back to the action index.
            from moonlander.core.policy import ACTIONS
            assert ACTIONS[mine] == theirs


def test_to_policy_json_round_trips_through_the_validator():
    theta = np.zeros(T.n_params(8))
    s = T.to_policy_json(theta, 8, meta={"algo": "test"})
    Policy.from_json(s)  # would raise on a bad payload
    assert json.loads(s)["sizes"] == [14, 8, 4]


def test_main_runs_end_to_end_and_writes_a_flyable_policy(tmp_path):
    out = tmp_path / "policy.json"
    T.main(["--hidden", "4", "--iters", "3", "--episodes", "1",
            "--seed", "0", "--out", str(out)])
    s = out.read_text()
    Policy.from_json(s)  # the §11 validator accepts it

    g = Game(mode="classic", preset="trainee")  # and it actually flies
    g.set_policy(s)
    g.reset(seed=0)
    frame = json.loads(g.step_policy())
    assert frame["t"] > 0
    assert "eval" in json.loads(s)["meta"]


def test_tour_runs_without_error(capsys):
    T.main(["--tour"])
    out = capsys.readouterr().out
    assert "action_space" in out and "pads" in out
