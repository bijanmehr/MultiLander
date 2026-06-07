"""Episode state machine and JSON interface (CONTRACT §2/§3/§4/§5/§6).

Pure stdlib. JS (via Pyodide) and the Gymnasium wrapper both drive the game
through this class; everything that crosses the Python⇄JS boundary is JSON.
v2: multi-lander — per-lander state, solid lander-lander collisions, the
``landers[]`` frame schema and sensor-filtered 14-dim observations.
"""

import json
import math
import random

from ..config import Config
from . import autopilot, physics
from .physics import LanderState
from .terrain import Terrain


class _Lander:
    """Per-lander episode state: physics body + status/score/outcome flags."""

    __slots__ = ("state", "status", "score", "outcome", "thrust", "side")

    def __init__(self, state):
        self.state = state
        self.status = "flying"  # "flying" | "landed" | "crashed"
        self.score = 0
        self.outcome = None
        self.thrust = False  # main engine fired this tick (for rendering)
        self.side = "none"  # gym-mode side engine fired this tick


class Game:
    """One multi-lander episode at a time. ``reset()`` starts a new one."""

    def __init__(self, mode="classic", n_landers=1, obs_mode="full", config=None):
        if mode not in ("classic", "gym"):
            raise ValueError(f"mode must be 'classic' or 'gym', got {mode!r}")
        if obs_mode not in ("full", "radar"):
            raise ValueError(f"obs_mode must be 'full' or 'radar', got {obs_mode!r}")
        n_landers = int(n_landers)
        if n_landers < 1:
            raise ValueError(f"n_landers must be >= 1, got {n_landers}")
        self.mode = mode
        self.obs_mode = obs_mode
        self.n_landers = n_landers
        self.cfg = config if config is not None else Config()
        self.reset()  # always hold a valid episode

    # ------------------------------------------------------------------ reset

    def reset(self, seed=None):
        """Start a new episode; returns the terrain JSON string (CONTRACT §3).

        Same seed → byte-identical terrain JSON and (given the same actions)
        byte-identical frames. seed=None draws fresh OS entropy (never
        wall-clock, never the global random module).
        """
        if seed is None:
            seed = random.SystemRandom().randrange(2**31)
        seed = int(seed)
        rng = random.Random(seed)  # the ONE rng for this episode

        self.seed = seed
        self.terrain = Terrain(self.cfg, rng, n_landers=self.n_landers)

        cfg = self.cfg
        self.landers = [
            _Lander(LanderState(
                x=sp["x"],
                y=sp["y"],
                vx=rng.uniform(-cfg.spawn_vx_max, cfg.spawn_vx_max),
                vy=0.0,
                angle=0.0,
                ang_vel=0.0,
                fuel=cfg.fuel_init,
            ))
            for sp in self.terrain.spawns
        ]
        self.steps = 0

        self._terrain_json = json.dumps(
            {
                "seed": seed,
                "points": self.terrain.points,
                "pads": self.terrain.pads,
                "stars": self.terrain.stars,
                "spawns": self.terrain.spawns,
            },
            allow_nan=False,  # CONTRACT §2: non-finite state is a loud error
        )
        return self._terrain_json

    # ------------------------------------------------------------------- step

    def step(self, rotate=0, thrust=False, engine=0):
        """One physics tick for a single-lander game (CONTRACT §2).

        Returns the frame JSON string (§4). ValueError when n_landers != 1 —
        multi-lander games must use ``step_all``.
        """
        if self.n_landers != 1:
            raise ValueError(
                f"step() drives exactly one lander; this game has {self.n_landers}"
                " — use step_all(controls_json)"
            )
        self._tick([(rotate, thrust, engine)])
        return self.frame_json()

    def step_all(self, controls_json):
        """One tick for ALL landers; ``controls_json`` is a JSON STRING (§2).

        classic: "[[rotate, thrust], ...]" — gym: "[[engine], ...]"; one entry
        per lander, in lander order. Entries for already-terminal landers are
        ignored (those landers stay frozen).
        """
        controls = json.loads(controls_json)
        if not isinstance(controls, list) or len(controls) != self.n_landers:
            raise ValueError(
                f"controls must be a JSON list with one entry per lander "
                f"({self.n_landers}), got {controls_json!r}"
            )
        parsed = []
        for k, entry in enumerate(controls):
            try:
                if self.mode == "classic":
                    rotate, thrust = entry
                    parsed.append((int(rotate), bool(thrust), 0))
                else:
                    (engine,) = entry
                    parsed.append((0, False, int(engine)))
            except (TypeError, ValueError) as exc:
                want = "[rotate, thrust]" if self.mode == "classic" else "[engine]"
                raise ValueError(
                    f"controls[{k}] must be {want}, got {entry!r}"
                ) from exc
        self._tick(parsed)
        return self.frame_json()

    def step_auto(self):
        """One tick flown by the built-in autopilot (CONTRACT §2, attract mode).

        Classic mode only; single-lander games only. Same no-op-after-terminal
        and JSON-string semantics as ``step``.
        """
        if self.mode != "classic":
            raise NotImplementedError(
                "step_auto is classic-mode only; the autopilot does not fly gym-mode engines"
            )
        if self.n_landers != 1:
            raise ValueError(
                f"step_auto() drives exactly one lander; this game has "
                f"{self.n_landers} — use step_auto_all()"
            )
        self._tick([self._auto_controls(self.landers[0])])
        return self.frame_json()

    def step_auto_all(self):
        """One tick with the autopilot flying every still-flying lander (§2).

        Classic mode only. No collision avoidance — drama is a feature.
        """
        if self.mode != "classic":
            raise NotImplementedError(
                "step_auto_all is classic-mode only; the autopilot does not fly gym-mode engines"
            )
        controls = [
            self._auto_controls(ln) if ln.status == "flying" else (0, False, 0)
            for ln in self.landers
        ]
        self._tick(controls)
        return self.frame_json()

    def _auto_controls(self, ln):
        """Autopilot decision for one lander → a (rotate, thrust, engine) triple."""
        s, cfg = ln.state, self.cfg
        clearance = s.y - cfg.lander_bottom - self._ground_y(s.x)
        rotate, thrust = autopilot.act(
            s.x, s.y, s.vx, s.vy, s.angle, clearance, self.terrain.pads,
            gravity=cfg.gravity, thrust_accel=cfg.thrust_accel,
        )
        return (rotate, thrust, 0)

    def _tick(self, controls):
        """Advance every still-flying lander one dt, then resolve collisions.

        ``controls``: one (rotate, thrust, engine) triple per lander. No-op
        when ALL landers are terminal (§2). This is the JSON-free fast path —
        the public step methods call it then append ``frame_json()``; the
        Gymnasium env calls it directly so training never pays for frame JSON.
        """
        if all(ln.status != "flying" for ln in self.landers):
            return
        self.steps += 1

        cfg = self.cfg
        for ln, (rotate, thrust, engine) in zip(self.landers, controls):
            if ln.status != "flying":
                continue  # terminal landers are frozen; their controls ignored
            ln.thrust, ln.side = physics.step(
                ln.state, cfg, self.mode, rotate=rotate, thrust=thrust, engine=engine
            )
            s = ln.state
            if s.y - cfg.lander_bottom - self._ground_y(s.x) <= 0.0:
                self._evaluate_landing(ln)
            elif s.x < 0.0 or s.x > cfg.world_w or s.y > cfg.world_h + 10.0:
                # CONTRACT §5: x < 0, x > world_w or y > world_h + 10 → crash
                ln.status = "crashed"
                ln.outcome = {"kind": "crash", "mult": 0, "points": 0,
                              "reason": "out of bounds"}

        self._resolve_collisions()

    # -------------------------------------------------------------- collisions

    def _resolve_collisions(self):
        """Lander-lander contact, checked after integration + terrain (§5).

        Deterministic pair order: i < j in lander order; statuses update as
        pairs resolve, so a lander crashed by an earlier pair this tick is no
        longer an obstacle for later pairs.
        """
        dist_max = self.cfg.lander_collision_dist
        n = len(self.landers)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.landers[i], self.landers[j]
                a_fly, b_fly = a.status == "flying", b.status == "flying"
                if not (a_fly or b_fly):
                    continue  # crashed are not obstacles; landed-landed inert
                if a.status == "crashed" or b.status == "crashed":
                    continue
                sa, sb = a.state, b.state
                if math.hypot(sb.x - sa.x, sb.y - sa.y) >= dist_max:
                    continue
                if a_fly and b_fly:
                    for ln in (a, b):
                        ln.status = "crashed"
                        ln.outcome = {"kind": "crash", "mult": 0, "points": 0,
                                      "reason": "collided with another lander"}
                else:  # one flying, one landed: flying crashes, landed unaffected
                    flyer = a if a_fly else b
                    flyer.status = "crashed"
                    flyer.outcome = {"kind": "crash", "mult": 0, "points": 0,
                                     "reason": "crashed into a landed lander"}

    # ------------------------------------------------------------- evaluation

    def _ground_y(self, x):
        """Terrain height under the lander: max over both feet and center."""
        h = self.terrain.height
        w = self.cfg.lander_half_w
        return max(h(x - w), h(x), h(x + w))

    def _evaluate_landing(self, ln):
        """Classify contact as perfect / hard / crash and set outcome + score."""
        s, cfg = ln.state, self.cfg

        # On pad: both feet (x ± half_w) inside one pad's [x0+2, x1-2].
        pad = None
        for p in self.terrain.pads:
            if p["x0"] + 2.0 <= s.x - cfg.lander_half_w and \
               s.x + cfg.lander_half_w <= p["x1"] - 2.0:
                pad = p
                break
        upright = abs(s.angle) <= cfg.angle_max

        if pad is not None and upright and \
                abs(s.vx) <= cfg.vx_perfect and abs(s.vy) <= cfg.vy_perfect:
            kind, points, reason = "perfect", cfg.score_perfect * pad["mult"], "perfect landing"
        elif pad is not None and upright and \
                abs(s.vx) <= cfg.vx_hard and abs(s.vy) <= cfg.vy_hard:
            kind, points, reason = "hard", cfg.score_hard * pad["mult"], "hard landing"
        else:
            kind, points = "crash", 0
            if pad is None:
                reason = "missed the pad"
            elif not upright:
                reason = "tipped over"
            else:
                reason = "came in too fast"

        ln.status = "landed" if kind != "crash" else "crashed"
        ln.score = points
        ln.outcome = {
            "kind": kind,
            "mult": pad["mult"] if (pad is not None and kind != "crash") else 0,
            "points": points,
            "reason": reason,
        }

    # ------------------------------------------------------------------ views

    def frame_json(self):
        """Current frame as JSON string, without stepping (CONTRACT §4)."""
        cfg = self.cfg
        active = sum(ln.status == "flying" for ln in self.landers)
        landers = []
        for i, ln in enumerate(self.landers):
            s = ln.state
            landers.append({
                "i": i,
                "x": s.x, "y": s.y, "vx": s.vx, "vy": s.vy,
                "angle": s.angle, "ang_vel": s.ang_vel,
                "thrust": ln.thrust, "side": ln.side,
                "fuel": s.fuel, "score": ln.score,
                "status": ln.status, "outcome": ln.outcome,
                "obs": self.obs(i),
            })
        s0 = self.landers[0].state  # hud always reflects lander 0 (§4)
        altitude = s0.y - cfg.lander_bottom - self._ground_y(s0.x)
        frame = {
            "t": self.steps * cfg.dt,
            "status": "flying" if active > 0 else "done",
            "active": active,
            "landers": landers,
            "hud": {
                "altitude": round(altitude),
                "hspeed": round(s0.vx),
                "vspeed": round(s0.vy),
            },
        }
        # CONTRACT §2: allow_nan=False — never emit NaN tokens JSON.parse rejects.
        return json.dumps(frame, allow_nan=False)

    def _nearest_pad(self, s):
        """(pad, euclidean distance) of the pad nearest to lander state ``s`` (§6)."""
        best, best_d = None, math.inf
        for p in self.terrain.pads:
            d = math.hypot((p["x0"] + p["x1"]) / 2.0 - s.x, p["y"] - s.y)
            if d < best_d:
                best, best_d = p, d
        return best, best_d

    def obs(self, i=0):
        """The 14-float observation vector for lander ``i`` (CONTRACT §6).

        Truth in core, perception as a filter: in "radar" obs_mode, pads
        farther than radar_range are invisible — indices 8, 9, 10, 12 zeroed
        and index 13 = 0.0.
        """
        ln = self.landers[i]
        s, cfg = ln.state, self.cfg
        pad, dist = self._nearest_pad(s)
        clearance = s.y - cfg.lander_bottom - self._ground_y(s.x)

        visible = self.obs_mode == "full" or dist <= cfg.radar_range
        if visible:
            pad_cx = (pad["x0"] + pad["x1"]) / 2.0
            pad_dx = (pad_cx - s.x) / cfg.world_w
            pad_dy = (pad["y"] - s.y) / cfg.world_h
            pad_hw = (pad["x1"] - pad["x0"]) / 2.0 / 100.0
            pad_mult = pad["mult"] / 5.0
            pad_seen = 1.0
        else:
            pad_dx = pad_dy = pad_hw = pad_mult = pad_seen = 0.0

        return [
            s.x / cfg.world_w * 2.0 - 1.0,
            s.y / cfg.world_h * 2.0 - 1.0,
            s.vx / 60.0,
            s.vy / 60.0,
            math.sin(s.angle),
            math.cos(s.angle),
            s.ang_vel / 3.0,
            s.fuel / cfg.fuel_init,
            pad_dx,
            pad_dy,
            pad_hw,
            clearance / cfg.world_h,
            pad_mult,
            pad_seen,
        ]
