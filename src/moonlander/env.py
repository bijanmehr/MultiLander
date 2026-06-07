"""Gymnasium wrapper around the pure-stdlib Game core (CONTRACT §7).

This is the ONLY module in the package allowed to import gymnasium / numpy —
it never ships to the browser. Never import it from package __init__ (the
registry entry point references it as a string, so it loads lazily).
"""

import json
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core.game import Game


class MoonLanderEnv(gym.Env):
    """Discrete-action lunar lander (single agent — wraps Game(n_landers=1)).

    Actions (Discrete(4)):
      classic: 0 noop, 1 rotate left (+1 = CCW), 2 rotate right (-1), 3 thrust
      gym:     0 noop, 1 left thruster, 2 main engine, 3 right thruster
    """

    metadata = {"render_modes": []}

    def __init__(self, mode="classic", obs_mode="full", frame_skip=1,
                 preset="cadet", config=None):
        super().__init__()
        frame_skip = int(frame_skip)
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")
        self.mode = mode
        self.obs_mode = obs_mode
        self.frame_skip = frame_skip
        # CONTRACT §7: preset as in §2 — the Game resolves it (explicit
        # config= wins over preset), the env mirrors the resolved config.
        self.game = Game(mode=mode, n_landers=1, obs_mode=obs_mode,
                         config=config, preset=preset)
        self.cfg = self.game.cfg
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-10.0, 10.0, (14,), np.float32)

    # ---------------------------------------------------------------- helpers

    def _obs(self):
        return np.asarray(self.game.obs(0), dtype=np.float32)

    def _phi(self):
        """Potential for reward shaping (CONTRACT §7):
        phi(s) = -1.0*dist - 0.5*speed - 0.5*|sin angle|

        dist is the MIN over pads of euclidean distance to the pad center,
        normalized by world_w — continuous in state, so the shaping reward has
        no spike when the nearest pad switches.
        """
        L, cfg = self.game.landers[0].state, self.cfg
        dist = min(
            math.hypot((p["x0"] + p["x1"]) / 2.0 - L.x, p["y"] - L.y)
            for p in self.game.terrain.pads
        ) / cfg.world_w
        speed = math.hypot(L.vx, L.vy) / 60.0
        return -1.0 * dist - 0.5 * speed - 0.5 * abs(math.sin(L.angle))

    # -------------------------------------------------------------- gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random
        if options is not None and options.get("game_seed") is not None:
            # CONTRACT §7: reset(options={"game_seed": k}) seeds the Game
            # directly — terrain JSON seed == k, byte-identical to ?seed=k.
            game_seed = int(options["game_seed"])
        else:
            # Derive the Game seed from np_random so reset(seed=k) is
            # reproducible and unseeded resets follow the gymnasium RNG chain.
            game_seed = int(self.np_random.integers(0, 2**31))
        terrain_json = self.game.reset(seed=game_seed)
        return self._obs(), {"terrain": json.loads(terrain_json)}

    def step(self, action):
        lander = self.game.landers[0]
        if lander.status != "flying":
            # CONTRACT §7: silently re-paying the terminal reward corrupted
            # manual eval loops — stepping a finished episode is a hard error.
            raise RuntimeError(
                "step() called on a terminated episode "
                f"(lander status {lander.status!r}) — call reset() first"
            )
        action = int(action)
        phi_before = self._phi()

        if self.mode == "classic":
            rotate = 1 if action == 1 else (-1 if action == 2 else 0)
            controls = [(rotate, action == 3, 0)]
        else:
            controls = [(0, False, action)]

        # frame_skip (CONTRACT §7): repeat the action for k physics ticks via
        # the JSON-free Game._tick, stopping early on terminal or at the
        # max_steps tick budget. Fuel cost accrues per tick the main engine
        # actually fired; shaping is evaluated ONCE across the whole env step
        # (potential-based — it telescopes over the underlying trajectory).
        fuel_cost = 0.0
        for _ in range(self.frame_skip):
            self.game._tick(controls)
            if lander.thrust:
                fuel_cost += 0.06
            if lander.status != "flying" or self.game.steps >= self.cfg.max_steps:
                break

        reward = 10.0 * (self._phi() - phi_before) - fuel_cost

        terminated = lander.status != "flying"
        truncated = (not terminated) and self.game.steps >= self.cfg.max_steps

        info = {"status": lander.status}
        if terminated:
            kind = lander.outcome["kind"]
            if kind == "perfect":
                reward += 100.0 + 10.0 * lander.outcome["mult"]
            elif kind == "hard":
                reward += 30.0
            else:  # crash
                reward += -100.0
            # CONTRACT §7: outcome is a COPY (mutating info must not corrupt
            # game state), is_success follows the SB3 eval convention.
            info["outcome"] = dict(lander.outcome)
            info["is_success"] = lander.status == "landed"
            info["score"] = lander.score
        elif truncated:
            info["is_success"] = False
        return self._obs(), float(reward), terminated, truncated, info
