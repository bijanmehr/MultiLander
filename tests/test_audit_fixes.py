"""Audit-fix patch v2.1: angle wrap, allow_nan, frame_skip, terminal info,
game_seed, step-after-terminal, boundary semantics, frame-JSON-free training."""

import json
import math

import numpy as np
import pytest

from moonlander.config import Config
from moonlander.core import physics
from moonlander.core.game import Game
from moonlander.core.physics import LanderState
from moonlander.env import MoonLanderEnv

CFG = Config()


def make_state(**kw):
    base = dict(x=500.0, y=500.0, vx=0.0, vy=0.0, angle=0.0, ang_vel=0.0,
                fuel=CFG.fuel_init)
    base.update(kw)
    return LanderState(**base)


# ---------------------------------------------------------------- angle wrap


def test_angle_stays_wrapped_during_continuous_rotation():
    # CONTRACT §5: angle wrapped to (-pi, pi] after integration, both spin
    # directions, held well past a full revolution (300 ticks ~ 8 rad).
    for rotate in (1, -1):
        s = make_state()
        for _ in range(300):
            physics.step(s, CFG, "classic", rotate=rotate, thrust=False)
            assert -math.pi < s.angle <= math.pi


def test_gym_mode_angle_also_wrapped():
    s = make_state(ang_vel=CFG.ang_vel_max)  # saturated spin, no further torque
    for _ in range(300):
        physics.step(s, CFG, "gym", engine=0)
        assert -math.pi < s.angle <= math.pi


def test_full_revolution_upright_landing_grades_perfect():
    # The audit bug: an unwrapped 2*pi angle made a soft upright pad touchdown
    # grade "tipped over" while sin/cos obs were byte-identical to a perfect
    # landing. Direct state setup as in test_game.py.
    g = Game(mode="classic")
    g.reset(seed=11)
    pad = g.terrain.pads[0]

    L = g.landers[0].state
    L.x = (pad["x0"] + pad["x1"]) / 2.0
    L.y = pad["y"] + CFG.lander_bottom + 2.0
    L.vx, L.vy = 0.0, -5.0
    L.angle, L.ang_vel = 2.0 * math.pi, 0.0  # one full revolution, upright

    frame = None
    for _ in range(200):
        frame = json.loads(g.step(0, False))
        if frame["status"] != "flying":
            break

    ld = frame["landers"][0]
    assert ld["status"] == "landed"
    assert ld["outcome"]["kind"] == "perfect"
    assert -math.pi < ld["angle"] <= math.pi


# ----------------------------------------------------------------- allow_nan


def test_frame_json_rejects_non_finite_state():
    # CONTRACT §2: allow_nan=False — a NaN is a loud ValueError, never a NaN
    # token JSON.parse would reject.
    g = Game(mode="classic")
    g.reset(seed=3)
    g.landers[0].state.ang_vel = float("nan")
    with pytest.raises(ValueError):
        g.frame_json()


# ---------------------------------------------------------------- frame_skip


def run_to_terminal(action, frame_skip, game_seed=7, max_env_steps=5000):
    env = MoonLanderEnv(frame_skip=frame_skip)
    env.reset(options={"game_seed": game_seed})
    total, n = 0.0, 0
    info = {}
    for _ in range(max_env_steps):
        _, r, terminated, truncated, info = env.step(action)
        total += r
        n += 1
        if terminated or truncated:
            break
    return env, n, total, env.game.steps, info


def test_frame_skip_4_reaches_terminal_in_quarter_the_env_steps():
    _, n1, _, ticks1, info1 = run_to_terminal(action=0, frame_skip=1)
    _, n4, _, ticks4, info4 = run_to_terminal(action=0, frame_skip=4)
    assert info1["outcome"] == info4["outcome"]  # same underlying trajectory
    assert ticks1 == ticks4  # terminal at the identical physics tick
    assert n4 == (n1 + 3) // 4  # ceil(n1 / 4) env steps


@pytest.mark.parametrize("action", [0, 3])  # free fall; constant main thrust
def test_frame_skip_return_matches_skip1_within_float_tolerance(action):
    # Shaping is potential-based, so it telescopes over the underlying
    # trajectory: 10*(phi(end) - phi(start)) regardless of k. Fuel cost is
    # -0.06 per tick the main engine fired — identical tick count → identical
    # cost. Terminal bonus identical. Totals match to float tolerance.
    _, _, total1, ticks1, _ = run_to_terminal(action=action, frame_skip=1)
    _, _, total4, ticks4, _ = run_to_terminal(action=action, frame_skip=4)
    assert ticks1 == ticks4
    assert total4 == pytest.approx(total1, abs=1e-7)


def test_truncation_counts_ticks_not_env_steps():
    # CONTRACT §7: an episode is max_steps TICKS regardless of frame_skip.
    env = MoonLanderEnv(frame_skip=4, config=Config(max_steps=50))
    env.reset(seed=1)
    terminated = truncated = False
    info, n = {}, 0
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(0)
        n += 1
        assert n <= 50
    assert truncated and not terminated
    assert env.game.steps == 50  # stopped mid-skip exactly at the tick budget
    assert n == (50 + 3) // 4  # ceil(50 / 4) env steps
    assert info["is_success"] is False


# ------------------------------------------------------- step after terminal


def test_step_after_terminal_raises_runtime_error():
    env, _, _, _, _ = run_to_terminal(action=0, frame_skip=1)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


# ------------------------------------------------------------- terminal info


def test_crash_terminal_info_has_is_success_score_outcome():
    _, _, _, _, info = run_to_terminal(action=0, frame_skip=1)
    assert info["outcome"]["kind"] == "crash"
    assert info["is_success"] is False
    assert info["score"] == 0


def test_landed_terminal_info_is_success_true_with_score():
    env = MoonLanderEnv()
    env.reset(seed=11)
    pad = env.game.terrain.pads[0]
    L = env.game.landers[0].state
    L.x = (pad["x0"] + pad["x1"]) / 2.0
    L.y = pad["y"] + CFG.lander_bottom + 2.0
    L.vx, L.vy = 0.0, -5.0
    L.angle, L.ang_vel = 0.0, 0.0

    terminated, info = False, {}
    for _ in range(200):
        _, _, terminated, _, info = env.step(0)
        if terminated:
            break
    assert terminated
    assert info["is_success"] is True
    assert info["outcome"]["kind"] == "perfect"
    assert info["score"] == CFG.score_perfect * pad["mult"]


def test_info_outcome_is_a_copy_not_an_alias():
    env, _, _, _, info = run_to_terminal(action=0, frame_skip=1)
    info["outcome"]["kind"] = "corrupted"
    info["outcome"]["points"] = 10**6
    # Game state is untouched by mutating the info dict (the alias bug).
    assert env.game.landers[0].outcome["kind"] == "crash"
    assert env.game.landers[0].outcome["points"] == 0
    frame = json.loads(env.game.frame_json())
    assert frame["landers"][0]["outcome"]["kind"] == "crash"


# ------------------------------------------------------------------ game_seed


def test_reset_game_seed_option_seeds_game_directly():
    env = MoonLanderEnv()
    obs1, info1 = env.reset(options={"game_seed": 7})
    obs2, info2 = env.reset(options={"game_seed": 7})
    assert np.array_equal(obs1, obs2)
    assert info1["terrain"] == info2["terrain"]
    assert info1["terrain"]["seed"] == 7  # byte-identical to the web's ?seed=7


def test_plain_seed_keeps_np_random_derived_game_seed():
    env = MoonLanderEnv()
    _, info = env.reset(seed=7)
    assert info["terrain"]["seed"] != 7  # derived through the np_random chain


# ------------------------------------------------- boundary semantics (pinned)


def touchdown_with_vx(vx):
    g = Game(mode="classic")
    g.reset(seed=11)
    # Widest pad → largest feet window, immune to the one-tick x drift.
    pad = max(g.terrain.pads, key=lambda p: p["x1"] - p["x0"])
    L = g.landers[0].state
    L.x = (pad["x0"] + pad["x1"]) / 2.0
    L.y = pad["y"] + CFG.lander_bottom + 0.01  # contact on the very next tick
    L.vx, L.vy = vx, -5.0
    L.angle, L.ang_vel = 0.0, 0.0
    frame = json.loads(g.step(0, False))
    assert frame["status"] == "done"
    return frame["landers"][0]


def test_touchdown_vx_exactly_at_vx_perfect_is_perfect():
    ld = touchdown_with_vx(CFG.vx_perfect)  # |vx| == 12.0 exactly: inclusive
    assert ld["outcome"]["kind"] == "perfect"


def test_touchdown_vx_one_ulp_above_vx_perfect_is_hard():
    ld = touchdown_with_vx(math.nextafter(CFG.vx_perfect, math.inf))
    assert ld["outcome"]["kind"] == "hard"


def out_of_bounds_probe(y0):
    g = Game(mode="classic")
    g.reset(seed=11)
    L = g.landers[0].state
    L.x, L.y = 1000.0, y0
    # vy chosen so the gravity update zeroes it exactly: post-step y == y0.
    L.vx, L.vy = 0.0, CFG.gravity * CFG.dt
    L.angle, L.ang_vel = 0.0, 0.0
    return json.loads(g.step(0, False))["landers"][0]


def test_post_step_y_exactly_world_h_plus_10_is_still_flying():
    ld = out_of_bounds_probe(CFG.world_h + 10.0)  # y == 760.0: boundary is in
    assert ld["status"] == "flying"
    assert ld["y"] == CFG.world_h + 10.0


def test_post_step_y_one_ulp_above_world_h_plus_10_is_oob_crash():
    ld = out_of_bounds_probe(math.nextafter(CFG.world_h + 10.0, math.inf))
    assert ld["status"] == "crashed"
    assert ld["outcome"]["reason"] == "out of bounds"


# ----------------------------------------------------------------- perf sanity


def test_env_step_never_builds_frame_json(monkeypatch):
    # Keeps the speedup honest: training steps ride Game._tick, not the
    # JSON-emitting Game.step path.
    calls = {"n": 0}
    orig = Game.frame_json

    def counting(self):
        calls["n"] += 1
        return orig(self)

    monkeypatch.setattr(Game, "frame_json", counting)
    env = MoonLanderEnv(frame_skip=4)
    env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(0)
    assert calls["n"] == 0
