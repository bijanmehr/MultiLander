"""Gymnasium wrapper: env_checker, spaces, seeding, registry, terminal info."""

import subprocess
import sys
import textwrap

import gymnasium
import numpy as np
from gymnasium.utils.env_checker import check_env

import moonlander  # noqa: F401 — registers MoonLander-v0 as a side effect
from moonlander.config import Config
from moonlander.env import MoonLanderEnv


def test_check_env_passes_both_modes_and_obs_modes():
    check_env(MoonLanderEnv(), skip_render_check=True)
    check_env(MoonLanderEnv(mode="gym"), skip_render_check=True)
    check_env(MoonLanderEnv(obs_mode="radar"), skip_render_check=True)
    check_env(MoonLanderEnv(mode="gym", obs_mode="radar"), skip_render_check=True)


def test_obs_shape_and_dtype():
    env = MoonLanderEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (14,)
    assert obs.dtype == np.float32
    assert "terrain" in info and len(info["terrain"]["pads"]) == 5

    obs2, _, _, _, _ = env.step(0)
    assert obs2.shape == (14,) and obs2.dtype == np.float32


def test_gymnasium_make_works_after_plain_import():
    # `import moonlander` above already triggered the guarded registration.
    env = gymnasium.make("MoonLander-v0")
    obs, info = env.reset(seed=3)
    assert obs.shape == (14,)
    assert "terrain" in info
    env.close()


def test_package_import_does_not_load_env_module():
    # CONTRACT §7: registration is by entry-point STRING — `import moonlander`
    # must not import moonlander.env (and so not numpy via it).
    code = textwrap.dedent(
        """
        import sys
        import moonlander
        assert "moonlander.env" not in sys.modules, "moonlander.env imported eagerly"
        print("LAZY")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "LAZY" in result.stdout


def test_reset_seed_reproducible():
    env1, env2 = MoonLanderEnv(), MoonLanderEnv()
    obs1, info1 = env1.reset(seed=99)
    obs2, info2 = env2.reset(seed=99)
    assert np.array_equal(obs1, obs2)
    assert info1["terrain"] == info2["terrain"]

    # Re-seeding the same env reproduces the first obs too.
    obs3, _ = env1.reset(seed=99)
    assert np.array_equal(obs1, obs3)


def test_crash_episode_ends_with_big_negative_reward_and_outcome_info():
    env = MoonLanderEnv()
    env.reset(seed=4)
    reward, terminated = 0.0, False
    for _ in range(5000):  # noop → free fall → guaranteed crash
        _, reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    assert terminated
    # info["outcome"] is present and correct on the terminal step (§7).
    assert info["outcome"]["kind"] == "crash"
    assert info["outcome"]["points"] == 0
    assert info["status"] == "crashed"
    assert reward <= -50.0


def test_truncation_at_max_steps_when_hovering():
    env = MoonLanderEnv(config=Config(max_steps=50))
    env.reset(seed=1)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(0)
        steps += 1
        assert steps <= 50
    assert truncated and not terminated  # too short a fall to reach the ground
    assert steps == 50
