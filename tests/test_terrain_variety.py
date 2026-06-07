"""Macro-variation (CONTRACT §3): determinism + measured variety over seeds.

All variety assertions run on the FIXED seed list 0..49, preset CADET, and the
witnessing seeds are pinned in comments — measured once after implementing the
macro-variation scheme, so every assertion below is deterministic. If terrain
generation changes its rng draw order, re-measure and re-pin.
"""

import json
import random
import statistics

from moonlander.config import Config
from moonlander.core.game import Game
from moonlander.core.terrain import Terrain

SEEDS = range(50)  # the fixed seed list all variety facts below refer to
CFG = Config.preset("cadet")


def _terrains():
    return {seed: Terrain(CFG, random.Random(seed)) for seed in SEEDS}


def _sorted_pads(t):
    return sorted(t.pads, key=lambda p: p["x0"])


# ------------------------------------------------------------- determinism


def test_same_preset_and_seed_byte_identical_terrain_json():
    # Run twice from scratch: macro-variation must flow entirely through the
    # episode rng in a fixed draw order (CONTRACT §3/§5).
    for preset in ("trainee", "cadet", "commander"):
        for seed in (0, 19, 49):
            t1 = Game(mode="classic", preset=preset).reset(seed=seed)
            t2 = Game(mode="classic", preset=preset).reset(seed=seed)
            assert t1 == t2, (preset, seed)  # byte-identical strings


def test_terrain_json_still_parses_with_contract_schema():
    terrain = json.loads(Game(mode="classic", preset="cadet").reset(seed=29))
    assert set(terrain) == {"seed", "points", "pads", "stars", "spawns"}
    assert len(terrain["pads"]) == 5


# ----------------------------------------------------- pad placement variety


def test_some_seed_has_two_pads_closer_than_300():
    # (a) clustering happens. Measured: 50/50 seeds have an inter-pad edge
    # gap < 300 on cadet; tightest is seed 49 (gap ~60.9, seed 2 ~64.8).
    closest = min(
        b["x0"] - a["x1"]
        for t in _terrains().values()
        for a, b in zip(_sorted_pads(t), _sorted_pads(t)[1:])
    )
    assert closest < 300.0, f"no clustered pads in seeds 0..49: min gap {closest}"


def test_some_seed_has_an_inter_pad_gap_over_600():
    # (b) empty stretches happen. Measured witnesses on cadet: seeds 3, 5,
    # 22, 29, 33, 34, 37 — widest is seed 37 (gap ~799.7).
    widest = max(
        b["x0"] - a["x1"]
        for t in _terrains().values()
        for a, b in zip(_sorted_pads(t), _sorted_pads(t)[1:])
    )
    assert widest > 600.0, f"no empty stretch in seeds 0..49: max gap {widest}"


def test_pads_reach_both_world_edges_across_seeds():
    # (c) pads reach the edges. Measured: leftmost pad x0 min is ~65.5
    # (seed 29) and rightmost pad x1 max is ~1934.6 (seed 34) on cadet.
    terrains = _terrains()
    left = min(_sorted_pads(t)[0]["x0"] for t in terrains.values())
    right = max(_sorted_pads(t)[-1]["x1"] for t in terrains.values())
    assert left < 250.0, f"leftmost pad never below x=250: {left}"
    assert right > 1750.0, f"rightmost pad never above x=1750: {right}"


def test_pad_constraints_hold_under_rejection_sampling_fuzz():
    # Rejection sampling must still honour the §3 invariants every seed:
    # >= pad_margin from both edges and between every pair of pads.
    for seed, t in _terrains().items():
        pads = _sorted_pads(t)
        assert pads[0]["x0"] >= CFG.pad_margin - 1e-9, f"seed {seed}"
        assert pads[-1]["x1"] <= CFG.world_w - CFG.pad_margin + 1e-9, f"seed {seed}"
        for a, b in zip(pads, pads[1:]):
            assert b["x0"] - a["x1"] >= CFG.pad_margin - 1e-9, f"seed {seed}"


# --------------------------------------------------------- endpoint widening


def test_endpoint_heights_use_widened_range():
    # §3(a): endpoints drawn from [y_min + 20, y_max - 60] ([80, 420] on
    # cadet), not the old fixed [120, 300]. Measured on seeds 0..49: range
    # [~84.0, ~414.3]; 11 endpoint draws fall below 120 and 35 above 300.
    lo, hi = CFG.terrain_y_min + 20.0, CFG.terrain_y_max - 60.0
    ends = []
    for t in _terrains().values():
        ends += [t.points[0][1], t.points[-1][1]]
    assert all(lo - 1e-9 <= e <= hi + 1e-9 for e in ends), (min(ends), max(ends))
    assert min(ends) < 120.0, f"no endpoint below the old lo=120: min {min(ends)}"
    assert max(ends) > 300.0, f"no endpoint above the old hi=300: max {max(ends)}"


# ------------------------------------------------------------ zone variation


def test_zone_variation_spreads_terrain_roughness_across_seeds():
    # (d) zone-scaled displacement makes some seeds calm and others violent.
    # Measured on cadet: calmest is seed 31 (height stdev ~8.1), roughest is
    # seed 1 (~124.7) — a ~15x spread; we require only 1.5x.
    stdevs = [
        statistics.pstdev(y for _, y in t.points) for t in _terrains().values()
    ]
    lo, hi = min(stdevs), max(stdevs)
    assert hi >= 1.5 * lo, f"roughness too uniform: stdev range [{lo:.2f}, {hi:.2f}]"
