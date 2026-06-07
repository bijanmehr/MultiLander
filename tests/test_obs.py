"""Observation vector (CONTRACT §6) and shaping continuity (§7)."""

import math

from moonlander.config import Config
from moonlander.core.game import Game
from moonlander.env import MoonLanderEnv

CFG = Config()

# Crafted pad pair where horizontal-nearest and euclidean-nearest disagree:
# A is low (y=60), B is high (y=420).
PAD_A = {"x0": 845.0, "x1": 955.0, "y": 60.0, "mult": 2}     # cx 900
PAD_B = {"x0": 1062.5, "x1": 1137.5, "y": 420.0, "mult": 3}  # cx 1100


def test_obs_has_14_dims_with_contract_formulas():
    g = Game(mode="classic")
    g.reset(seed=5)
    s = g.landers[0].state
    s.x, s.y, s.vx, s.vy = 700.0, 600.0, -12.0, -30.0
    s.angle, s.ang_vel = 0.2, 0.5
    s.fuel = 500.0

    o = g.obs(0)
    assert len(o) == 14
    assert o[0] == 700.0 / CFG.world_w * 2.0 - 1.0
    assert o[1] == 600.0 / CFG.world_h * 2.0 - 1.0
    assert o[2] == -12.0 / 60.0
    assert o[3] == -30.0 / 60.0
    assert o[4] == math.sin(0.2)
    assert o[5] == math.cos(0.2)
    assert o[6] == 0.5 / 3.0
    assert o[7] == 500.0 / CFG.fuel_init

    pad = min(
        g.terrain.pads,
        key=lambda p: math.hypot((p["x0"] + p["x1"]) / 2.0 - s.x, p["y"] - s.y),
    )
    cx = (pad["x0"] + pad["x1"]) / 2.0
    assert o[8] == (cx - s.x) / CFG.world_w
    assert o[9] == (pad["y"] - s.y) / CFG.world_h
    assert o[10] == (pad["x1"] - pad["x0"]) / 2.0 / 100.0
    assert o[12] == pad["mult"] / 5.0
    assert o[13] == 1.0

    h = g.terrain.height
    ground = max(h(s.x - CFG.lander_half_w), h(s.x), h(s.x + CFG.lander_half_w))
    assert math.isclose(o[11], (s.y - CFG.lander_bottom - ground) / CFG.world_h)


def test_obs_normalizers_are_world_relative_not_hardcoded():
    cfg = Config(world_w=4000.0)
    g = Game(mode="classic", config=cfg)
    g.reset(seed=0)
    s = g.landers[0].state
    s.x, s.y, s.vx, s.vy = 3900.0, 600.0, 0.0, 0.0

    o = g.obs(0)
    assert o[0] == 3900.0 / 4000.0 * 2.0 - 1.0
    assert -1.0 <= o[0] <= 1.0  # would be 6.8 with a hardcoded /1000, 2.9 with /2000
    assert -1.0 <= o[8] <= 1.0


def test_radar_mode_masks_far_pads_and_reveals_near_ones():
    cfg = Config(pad_multipliers=(2,))  # a single pad makes "far from ALL pads" easy
    g = Game(mode="classic", obs_mode="radar", config=cfg)
    g.reset(seed=8)
    pad = g.terrain.pads[0]
    cx = (pad["x0"] + pad["x1"]) / 2.0
    s = g.landers[0].state

    # Near: well inside radar_range → the pad block is populated.
    s.x, s.y = cx, pad["y"] + cfg.radar_range / 2.0
    near = g.obs(0)
    assert near[13] == 1.0
    assert near[8] == (cx - s.x) / cfg.world_w
    assert near[10] == (pad["x1"] - pad["x0"]) / 2.0 / 100.0
    assert near[12] == pad["mult"] / 5.0

    # Far: euclidean distance > radar_range → 8, 9, 10, 12 zeroed and 13 = 0.
    far_x = cx + cfg.radar_range + 200.0
    if far_x > cfg.world_w:
        far_x = cx - cfg.radar_range - 200.0
    s.x, s.y = far_x, 700.0
    far = g.obs(0)
    assert far[8] == far[9] == far[10] == far[12] == far[13] == 0.0
    assert far[1] != 0.0  # truth channels stay live

    # Full mode never masks, even from the very same far state.
    g_full = Game(mode="classic", obs_mode="full", config=cfg)
    g_full.reset(seed=8)  # same seed → identical terrain
    sf = g_full.landers[0].state
    sf.x, sf.y = far_x, 700.0
    full = g_full.obs(0)
    assert full[13] == 1.0
    assert full[10] == (pad["x1"] - pad["x0"]) / 2.0 / 100.0
    assert full[12] == pad["mult"] / 5.0


def test_pad_targeting_is_euclidean_not_horizontal():
    g = Game(mode="classic")
    g.reset(seed=0)
    g.terrain.pads = [PAD_A, PAD_B]
    s = g.landers[0].state
    # Horizontally A is 50 away vs B 150 — but euclidean A ≈ 642 vs B ≈ 318.
    s.x, s.y, s.vx, s.vy = 950.0, 700.0, 0.0, 0.0

    o = g.obs(0)
    assert o[8] == (1100.0 - 950.0) / CFG.world_w  # targets B, not A
    assert o[9] == (420.0 - 700.0) / CFG.world_h
    assert o[12] == PAD_B["mult"] / 5.0


def _sweep(env, x0, x1):
    """Drift lander 0 rightward at low speed from x0 past x1; |reward| per step."""
    s = env.game.landers[0].state
    s.x = x0
    rewards = []
    while s.x < x1:
        s.y, s.vy, s.vx = 700.0, 0.0, 8.0  # re-pin: slow, level, upright drift
        s.angle, s.ang_vel = 0.0, 0.0
        _, r, terminated, truncated, _ = env.step(0)
        assert not (terminated or truncated)
        rewards.append(abs(r))
    return rewards


def test_shaping_reward_has_no_spike_across_nearest_pad_switch():
    # SHAPING CONTINUITY regression (§7): phi's min-over-pads euclidean form
    # must produce no single-step reward spike when the nearest pad switches.
    env = MoonLanderEnv()
    env.reset(seed=0)
    env.game.terrain.pads = [PAD_A, PAD_B]

    def nearest(x, y):
        return min(
            ((900.0, 60.0), (1100.0, 420.0)),
            key=lambda c: math.hypot(c[0] - x, c[1] - y),
        )

    # At y = 700 the euclidean-nearest pad flips from A to B near x = 172.
    assert nearest(150.0, 700.0)[0] == 900.0
    assert nearest(200.0, 700.0)[0] == 1100.0
    spikes = _sweep(env, 150.0, 200.0)  # crosses the euclidean switch boundary

    # ... and across x = 1000, where a horizontal-nearest phi (the v1 bug)
    # would jump by ~0.175 in dist and spike |r| ≈ 1.75.
    spikes += _sweep(env, 990.0, 1010.0)

    assert max(spikes) < 0.3, f"shaping spike at pad switch: max |r| = {max(spikes)}"
