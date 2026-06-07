"""Difficulty presets (CONTRACT §2/§3/§7): exact table values, plumbing, terrain."""

import dataclasses
import json
import random

import pytest
from gymnasium.utils.env_checker import check_env

from moonlander.config import Config
from moonlander.core.game import Game
from moonlander.core.terrain import Terrain
from moonlander.env import MoonLanderEnv

# The CONTRACT §3 difficulty table, hardcoded LITERALLY — any drift in
# config.PRESETS (or the Config class defaults, which must equal CADET)
# fails here, not silently downstream.
TABLE = {
    "trainee": {
        "terrain_displacement": 140.0, "terrain_decay": 0.52, "terrain_y_max": 380.0,
        "pad_widths": {2: 130.0, 3: 95.0, 5: 60.0},
        "fuel_init": 1200.0, "spawn_vx_max": 15.0,
    },
    "cadet": {
        "terrain_displacement": 210.0, "terrain_decay": 0.62, "terrain_y_max": 480.0,
        "pad_widths": {2: 110.0, 3: 75.0, 5: 45.0},
        "fuel_init": 1000.0, "spawn_vx_max": 25.0,
    },
    "commander": {
        "terrain_displacement": 260.0, "terrain_decay": 0.68, "terrain_y_max": 560.0,
        "pad_widths": {2: 90.0, 3: 60.0, 5: 36.0},
        "fuel_init": 850.0, "spawn_vx_max": 40.0,
    },
}

PRESET_NAMES = ("trainee", "cadet", "commander")


# ----------------------------------------------------------------- §3 table


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_preset_values_match_contract_table_exactly(name):
    cfg = Config.preset(name)
    for field_name, want in TABLE[name].items():
        assert getattr(cfg, field_name) == want, (name, field_name)


def test_everything_not_listed_is_identical_across_presets():
    # CONTRACT §3: difficulty is terrain ruggedness, pad size, fuel budget and
    # spawn drift — NOT different physics. Every other field must match.
    listed = set(TABLE["cadet"])
    stripped = []
    for name in PRESET_NAMES:
        d = dataclasses.asdict(Config.preset(name))
        for k in listed:
            d.pop(k)
        stripped.append(d)
    assert stripped[0] == stripped[1] == stripped[2]


def test_class_defaults_equal_cadet_preset():
    assert Config() == Config.preset("cadet")


def test_unknown_preset_raises_value_error_listing_options():
    with pytest.raises(ValueError) as exc:
        Config.preset("ace")
    msg = str(exc.value)
    for option in PRESET_NAMES:
        assert option in msg


def test_preset_configs_do_not_share_the_pad_widths_dict():
    a, b = Config.preset("trainee"), Config.preset("trainee")
    assert a.pad_widths == b.pad_widths
    assert a.pad_widths is not b.pad_widths  # frozen Config, but dicts mutate


# ------------------------------------------------------------- Game plumbing


def test_explicit_config_overrides_preset_in_game():
    cfg = Config(fuel_init=123.0)
    g = Game(mode="classic", config=cfg, preset="commander")
    assert g.cfg is cfg  # CONTRACT §2: explicit config wins
    assert g.cfg.fuel_init == 123.0
    assert g.preset is None


def test_game_preset_selects_the_matching_config():
    for name in PRESET_NAMES:
        g = Game(mode="classic", preset=name)
        assert g.cfg == Config.preset(name)
        assert g.preset == name
    assert Game(mode="classic").cfg == Config.preset("cadet")  # default


def test_game_unknown_preset_raises():
    with pytest.raises(ValueError):
        Game(mode="classic", preset="ace")


def test_frame_json_schema_is_unchanged_by_presets():
    # Schema frozen (§4): the active preset name never leaks into frames.
    g = Game(mode="classic", preset="commander")
    g.reset(seed=1)
    frame = json.loads(g.step(0, False))
    assert set(frame) == {"t", "status", "active", "landers", "hud"}


# ----------------------------------------------------------- terrain safety


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_terrain_in_bounds_and_pads_flat_fuzz_100_seeds(name):
    cfg = Config.preset(name)
    for seed in range(100):
        t = Terrain(cfg, random.Random(seed))
        for _, y in t.points:
            assert cfg.terrain_y_min <= y <= cfg.terrain_y_max, f"seed {seed}"
        for p in t.pads:
            assert abs((p["x1"] - p["x0"]) - cfg.pad_widths[p["mult"]]) < 1e-9
            for k in range(26):  # sample across the full pad span
                x = p["x0"] + (p["x1"] - p["x0"]) * k / 25.0
                assert abs(t.height(x) - p["y"]) < 1e-9, f"seed {seed}"


def test_same_seed_different_presets_produce_different_terrain():
    jsons = {n: Game(mode="classic", preset=n).reset(seed=42) for n in PRESET_NAMES}
    assert len(set(jsons.values())) == 3  # by design (§2)
    points = {n: json.loads(j)["points"] for n, j in jsons.items()}
    assert points["trainee"] != points["cadet"]
    assert points["cadet"] != points["commander"]


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_same_seed_same_preset_is_byte_identical(name):
    t1 = Game(mode="classic", preset=name).reset(seed=7)
    t2 = Game(mode="classic", preset=name).reset(seed=7)
    assert t1 == t2


# -------------------------------------------------------------------- env §7


def test_env_accepts_preset_and_check_env_passes_on_commander():
    env = MoonLanderEnv(preset="commander")
    assert env.cfg == Config.preset("commander")
    check_env(env, skip_render_check=True)


def test_env_explicit_config_overrides_preset():
    cfg = Config(max_steps=50)
    env = MoonLanderEnv(preset="trainee", config=cfg)
    assert env.cfg is cfg
