"""Terrain generation: determinism, pad layout, flatness, bounds, spawns."""

import json
import math
import random

from moonlander.config import Config
from moonlander.core.game import Game
from moonlander.core.terrain import Terrain

CFG = Config()


def test_same_seed_identical_terrain_json():
    t1 = Game(mode="classic").reset(seed=42)
    t2 = Game(mode="classic").reset(seed=42)
    assert t1 == t2  # byte-identical strings


def test_terrain_has_257_points_across_world():
    terrain = json.loads(Game(mode="classic").reset(seed=7))
    points = terrain["points"]
    assert len(points) == CFG.terrain_points == 257
    assert points[0][0] == 0.0
    assert math.isclose(points[-1][0], CFG.world_w, abs_tol=1e-9)


def test_five_pads_with_contract_multipliers_and_widths():
    terrain = json.loads(Game(mode="classic").reset(seed=7))
    pads = terrain["pads"]
    assert len(pads) == 5
    assert tuple(p["mult"] for p in pads) == CFG.pad_multipliers == (2, 2, 3, 3, 5)
    for p in pads:
        assert math.isclose(p["x1"] - p["x0"], CFG.pad_widths[p["mult"]], abs_tol=1e-9)


def test_pads_exactly_flat():
    for seed in (0, 1, 99):
        t = Terrain(CFG, random.Random(seed))
        for p in t.pads:
            for k in range(26):  # sample across the full pad span
                x = p["x0"] + (p["x1"] - p["x0"]) * k / 25.0
                assert abs(t.height(x) - p["y"]) < 1e-9


def test_all_vertices_within_bounds():
    for seed in (3, 1234):
        t = Terrain(CFG, random.Random(seed))
        for _, y in t.points:
            assert CFG.terrain_y_min <= y <= CFG.terrain_y_max


def test_pads_do_not_overlap():
    for seed in (5, 6, 7, 8):
        t = Terrain(CFG, random.Random(seed))
        pads = sorted(t.pads, key=lambda p: p["x0"])
        # >= pad_margin from world edges
        assert pads[0]["x0"] >= CFG.pad_margin - 1e-9
        assert pads[-1]["x1"] <= CFG.world_w - CFG.pad_margin + 1e-9
        # >= pad_margin gap between consecutive pads (hence no overlap)
        for a, b in zip(pads, pads[1:]):
            assert b["x0"] - a["x1"] >= CFG.pad_margin - 1e-9


def test_stars_within_contract_band():
    t = Terrain(CFG, random.Random(12))
    assert len(t.stars) == CFG.n_stars
    for x, y in t.stars:
        assert 0.0 <= x <= CFG.world_w
        assert 0.6 * CFG.world_h <= y <= CFG.world_h - 10.0


def test_terrain_json_has_one_spawn_per_lander():
    for n in (1, 3):
        terrain = json.loads(Game(mode="classic", n_landers=n).reset(seed=4))
        spawns = terrain["spawns"]
        assert len(spawns) == n
        for sp in spawns:
            assert sp["y"] == CFG.spawn_y
            assert CFG.spawn_x_min <= sp["x"] <= CFG.spawn_x_max
