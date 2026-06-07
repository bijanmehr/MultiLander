"""Multi-lander (CONTRACT v2): collisions, spawns, step_all semantics, determinism."""

import json

import pytest

from moonlander.config import Config
from moonlander.core.game import Game

CFG = Config()

NOOP2 = json.dumps([[0, False], [0, False]])


def place(game, i, x, y, vx=0.0, vy=0.0):
    """Teleport lander i to a hand-built state (upright, no spin)."""
    s = game.landers[i].state
    s.x, s.y, s.vx, s.vy = x, y, vx, vy
    s.angle, s.ang_vel = 0.0, 0.0


def test_converging_flying_landers_both_crash():
    g = Game(mode="classic", n_landers=2)
    g.reset(seed=0)
    place(g, 0, 980.0, 600.0, vx=15.0)
    place(g, 1, 1040.0, 600.0, vx=-15.0)  # 60 apart, closing at 30/s

    raw = frame = None
    for _ in range(300):
        raw = g.step_all(NOOP2)
        frame = json.loads(raw)
        if frame["status"] != "flying":
            break
    assert frame["status"] == "done"
    assert frame["active"] == 0
    for ld in frame["landers"]:
        assert ld["status"] == "crashed"
        assert ld["outcome"]["kind"] == "crash"
        assert ld["outcome"]["reason"] == "collided with another lander"
        assert ld["outcome"]["points"] == 0

    # All landers terminal → stepping is a no-op returning the current frame.
    assert g.step_all(NOOP2) == raw


def test_flying_lander_crashes_into_landed_one_landed_keeps_points():
    g = Game(mode="classic", n_landers=2)
    g.reset(seed=1)
    pad = g.terrain.pads[0]
    cx = (pad["x0"] + pad["x1"]) / 2.0

    # Lander 1 hovers far away and high while lander 0 touches down.
    far_x = cx + 600.0 if cx + 600.0 < CFG.world_w - 20.0 else cx - 600.0
    place(g, 1, far_x, 700.0)
    place(g, 0, cx, pad["y"] + CFG.lander_bottom + 1.0, vy=-5.0)
    for _ in range(60):
        g.step_all(NOOP2)
        if g.landers[0].status == "landed":
            break
    assert g.landers[0].status == "landed"
    banked = g.landers[0].score
    assert banked == CFG.score_perfect * pad["mult"]

    # Now drop lander 1 right above the parked lander 0 — inside collision range.
    s0 = g.landers[0].state
    place(g, 1, s0.x, s0.y + CFG.lander_collision_dist - 6.0)
    frame = json.loads(g.step_all(NOOP2))

    ld0, ld1 = frame["landers"]
    assert ld1["status"] == "crashed"
    assert ld1["outcome"]["reason"] == "crashed into a landed lander"
    # Pad-blocking is legal strategy: the landed lander is unaffected.
    assert ld0["status"] == "landed"
    assert ld0["score"] == banked
    assert ld0["outcome"]["kind"] == "perfect"


def test_landers_beyond_collision_dist_never_collide():
    g = Game(mode="classic", n_landers=2)
    g.reset(seed=2)
    place(g, 0, 1000.0, 600.0)
    place(g, 1, 1000.0 + CFG.lander_collision_dist + 4.0, 600.0)
    frame = None
    for _ in range(10):  # both free-fall identically; the gap never shrinks
        frame = json.loads(g.step_all(NOOP2))
    for ld in frame["landers"]:
        assert ld["status"] == "flying"
    assert frame["active"] == 2


def test_spawn_min_separation_fuzz_200_seeds_n3():
    g = Game(mode="classic", n_landers=3)
    for seed in range(200):
        spawns = json.loads(g.reset(seed=seed))["spawns"]
        assert len(spawns) == 3
        xs = sorted(sp["x"] for sp in spawns)
        for a, b in zip(xs, xs[1:]):
            assert b - a >= CFG.spawn_min_separation - 1e-9, f"seed {seed}: {xs}"
        assert xs[0] >= CFG.spawn_x_min - 1e-9
        assert xs[-1] <= CFG.spawn_x_max + 1e-9
        for sp in spawns:
            assert sp["y"] == CFG.spawn_y


def test_step_all_ignores_controls_for_terminal_landers():
    g = Game(mode="classic", n_landers=2)
    g.reset(seed=3)
    # Crash lander 1: slam it into the ground far from lander 0.
    place(g, 0, 400.0, 700.0)
    place(g, 1, 1500.0, g._ground_y(1500.0) + CFG.lander_bottom + 1.0, vy=-100.0)
    frame = json.loads(g.step_all(NOOP2))
    assert frame["landers"][1]["status"] == "crashed"
    frozen = frame["landers"][1]

    # Full controls for the dead lander change nothing about it.
    controls = json.dumps([[0, False], [1, True]])
    for _ in range(5):
        frame = json.loads(g.step_all(controls))
        assert frame["landers"][1] == frozen  # frozen in place, fuel untouched
    assert frame["landers"][0]["status"] == "flying"  # the live one still flies


def test_three_lander_seeded_determinism_byte_identical_frames():
    def run(seed):
        g = Game(mode="classic", n_landers=3)
        terrain = g.reset(seed=seed)
        frames = []
        for k in range(240):
            controls = json.dumps(
                [[(k % 3) - 1, k % 2 == 0], [0, True], [1, False]]
            )
            frames.append(g.step_all(controls))
        return terrain, frames

    t1, f1 = run(7)
    t2, f2 = run(7)
    assert t1 == t2
    assert f1 == f2  # byte-identical at every step


def test_step_all_malformed_controls_raise_clear_value_error():
    g = Game(mode="classic", n_landers=2)
    g.reset(seed=0)
    with pytest.raises(ValueError, match=r"controls\[0\]"):
        g.step_all(json.dumps([0, 0]))  # entries must be [rotate, thrust] pairs
    with pytest.raises(ValueError, match=r"controls\[1\]"):
        g.step_all(json.dumps([[0, False], [0]]))  # short entry
    with pytest.raises(ValueError):  # not JSON at all (JSONDecodeError is a ValueError)
        g.step_all("not json")


def test_step_raises_value_error_on_multi_lander_game():
    g = Game(mode="classic", n_landers=3)
    g.reset(seed=0)
    with pytest.raises(ValueError):
        g.step(0, True)
