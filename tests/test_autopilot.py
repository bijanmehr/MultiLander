"""Autopilot + step_auto (CONTRACT §2): landing rate, determinism, purity."""

import json
import subprocess
import sys
import textwrap

import pytest

from moonlander.core import autopilot
from moonlander.core.game import Game

MAX_STEPS = 7200  # 2 minutes of sim time — far beyond any sane episode


def fly_auto(game, seed):
    """Reset to ``seed`` and step_auto until terminal; returns the last frame JSON."""
    game.reset(seed=seed)
    frame_json = game.frame_json()
    for _ in range(MAX_STEPS):
        frame_json = game.step_auto()
        if json.loads(frame_json)["status"] != "flying":
            break
    return frame_json


def test_landing_rate_over_seeds_0_to_29():
    # Margin below the measured rate (30/30 perfect on the v2 2000-wide world
    # at time of writing) but comfortably above the contract's ~30% floor.
    g = Game(mode="classic")
    kinds = []
    for seed in range(30):
        frame = json.loads(fly_auto(g, seed))
        assert frame["status"] != "flying", f"seed {seed}: episode never ended"
        kinds.append(frame["landers"][0]["outcome"]["kind"])
    landed = sum(k in ("perfect", "hard") for k in kinds)
    assert landed >= 12, f"only {landed}/30 landed: {kinds}"
    assert kinds.count("perfect") >= 1, f"no perfect landing in 30 seeds: {kinds}"


def test_step_auto_is_deterministic():
    final1 = fly_auto(Game(mode="classic"), seed=7)
    final2 = fly_auto(Game(mode="classic"), seed=7)
    assert final1 == final2  # byte-identical final frame JSON


def test_step_auto_after_terminal_is_noop():
    g = Game(mode="classic")
    last = fly_auto(g, seed=3)
    status = json.loads(last)["status"]
    assert status != "flying"

    again = g.step_auto()
    assert again == last  # identical frame JSON
    assert json.loads(again)["status"] == status


def test_step_auto_raises_in_gym_mode():
    g = Game(mode="gym")
    g.reset(seed=0)
    with pytest.raises(NotImplementedError):
        g.step_auto()


def test_autopilot_module_is_pure_math_only():
    # Fresh interpreter: load the module file standalone and verify the only
    # import it pulls in (or holds a reference to) is math.
    code = textwrap.dedent(
        """
        import importlib.util, sys, types

        path = sys.argv[1]
        before = set(sys.modules)
        spec = importlib.util.spec_from_file_location("ap_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        loaded = set(sys.modules) - before - {"ap_under_test"}
        assert loaded <= {"math"}, f"unexpected imports loaded: {sorted(loaded)}"
        held = {v.__name__ for v in vars(mod).values() if isinstance(v, types.ModuleType)}
        assert held <= {"math"}, f"unexpected module references: {sorted(held)}"
        print("PURE")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, autopilot.__file__],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PURE" in result.stdout
