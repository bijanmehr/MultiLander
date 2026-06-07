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
