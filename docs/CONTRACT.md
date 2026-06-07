# Python ⇄ JavaScript Contract — v2

This document is the **frozen interface** between the Python core (runs in
Pyodide) and the JS frontend (owns all rendering). Both sides implement
against this file. If you must deviate, update this file in the same change.

All constants referenced here live in `src/moonlander/config.py` (class `Config`).
v2 changes vs v1: 2000×750 world, 5 pads, **multi-lander** (`landers[]` schema,
solid collisions), config-relative observations, sensor models (full/radar),
euclidean pad targeting, wheel version **0.2.0**.

## 1. Coordinate conventions

- World: `x ∈ [0, world_w]` rightward, `y ∈ [0, world_h]` **upward**, origin
  bottom-left. Defaults: `world_w = 2000`, `world_h = 750`. **No dimension may
  be hardcoded** — everything derives from `Config`.
- `angle`: radians, `0` = upright, **positive = counter-clockwise** (tilts nose left).
- Thrust direction at angle `a`: unit vector `(-sin a, cos a)`.
- Canvas is y-down. JS converts through the camera (§10); at camera scale 1 and
  centered: `cx = x`, `cy = world_h - y`.
- World-CCW rotation appears as `ctx.rotate(-angle)` on canvas.

## 2. Python API (what JS calls through Pyodide)

```python
from moonlander.core.game import Game

g = Game(mode="classic", n_landers=1, obs_mode="full")
terrain_json = g.reset(seed=None)      # -> str (terrain JSON, §3)
frame_json   = g.step(rotate, thrust)  # n_landers == 1 only (ValueError otherwise)
frame_json   = g.step_all(controls_json)  # any n; one tick for ALL landers
frame_json   = g.step_auto()           # n_landers == 1 autopilot tick
frame_json   = g.step_auto_all()       # autopilot flies every still-flying lander
frame_json   = g.frame_json()          # current frame WITHOUT stepping
obs          = g.obs(i=0)              # list of 14 floats for lander i (§6)
```

- `mode`: `"classic"` (rotate+thrust) | `"gym"` (engines). `obs_mode`: `"full"` | `"radar"`.
- `step(rotate, thrust, engine=0)`: classic — `rotate ∈ {-1,0,+1}` (+1 = CCW), `thrust: bool`,
  simultaneous allowed; gym — `step(0, False, engine)`, `engine ∈ {0,1,2,3}` (noop/left/main/right).
- `step_all(controls_json)`: **JSON string** (everything crossing the boundary is a string):
  classic `"[[rotate, thrust], ...]"`, gym `"[[engine], ...]"` — one entry per lander,
  in lander order. Entries for already-terminal landers are ignored.
- Stepping when ALL landers are terminal is a no-op returning the current frame.
  Individual terminal landers are frozen in place (they keep appearing in `landers[]`).
- All JSON is emitted with `allow_nan=False` — a non-finite state is a loud Python
  error, never `NaN` tokens that `JSON.parse` would reject.
- `reset(seed=k)` fully deterministic: terrain, pads, stars, all spawns. `seed=None`: entropy.
- Autopilot (classic only, NotImplementedError in gym mode): simple stdlib proportional
  controller per lander targeting its nearest pad; no collision avoidance (drama is a feature).

## 3. Terrain JSON (returned by `reset`, fixed for the episode)

```json
{
  "seed": 42,
  "points": [[0.0, 212.0], "... terrain_points (257) [x,y] pairs, y-up"],
  "pads":   [{"x0": 310.0, "x1": 420.0, "y": 188.0, "mult": 2}, "... 5 pads"],
  "stars":  [[x, y], "... n_stars pairs, y ∈ [0.6*world_h, world_h-10]"],
  "spawns": [{"x": 420.0, "y": 690.0}, "... one per lander"]
}
```

- Vertices evenly spaced in x across `[0, world_w]`; midpoint displacement
  (`terrain_*` config), clamped to `[terrain_y_min, terrain_y_max]`.
- Pads: one per entry in `pad_multipliers = (2, 2, 3, 3, 5)`, widths from
  `pad_widths` (2X→110, 3X→75, 5X→45), random non-overlapping positions with
  ≥ `pad_margin` gaps; covered vertices flattened to pad `y`.
- Spawns: `n_landers` positions at `y = spawn_y`, x spread across
  `[spawn_x_min, spawn_x_max]` with ≥ `spawn_min_separation` between any two
  (slot + jitter placement); each lander gets independent `vx ~ U[-25, 25]`.

## 4. Frame JSON (returned by every step call / `frame_json`)

```json
{
  "t": 12.35,
  "status": "flying",
  "active": 2,
  "landers": [
    {"i": 0, "x": 512.3, "y": 401.7, "vx": -3.2, "vy": -41.0,
     "angle": 0.12, "ang_vel": 0.0, "thrust": true, "side": "none",
     "fuel": 873.2, "score": 0, "status": "flying", "outcome": null,
     "obs": ["... 14 floats, §6"]},
    "... one entry per lander, stable order"
  ],
  "hud": {"altitude": 391, "hspeed": -3, "vspeed": -41}
}
```

- Top-level `status`: `"flying"` while ≥1 lander flies, else `"done"`.
  `active` = count of landers with status `"flying"`.
- Per-lander `status`: `"flying" | "landed" | "crashed"`; `outcome` as v1:
  `null` or `{"kind": "perfect"|"hard"|"crash", "mult", "points", "reason"}`
  (crash → mult 0, points 0; reasons include `"came in too fast"`,
  `"missed the pad"`, `"tipped over"`, `"out of bounds"`,
  `"collided with another lander"`, `"crashed into a landed lander"`).
- `hud` always reflects **lander 0** (the human / focus lander).
- v1 top-level `lander`/`fuel`/`score`/`obs` fields are GONE — no aliases.

## 5. Physics & rules

Semi-implicit Euler per lander, `dt = 1/60`, exactly as v1:

```
classic: ang_vel = rotate * rot_rate          gym: ang_vel += engine_torque * dt
gym ang_vel is clamped to ±ang_vel_max (RCS saturation; keeps obs bounded)
angle += ang_vel * dt;  angle wrapped to (-π, π] after integration
  (sin/cos observations are modular — without the wrap, a full revolution would
  make two byte-identical observations carry opposite terminal rewards)
acc = (0, -gravity) + thrust_accel*(-sin a, cos a)[main, fuel>0]
    + side_accel*(±cos a, ±sin a)[gym side engines]
vel += acc*dt;  pos += vel*dt;  fuel -= burn*dt (clamp 0; engines dead at 0)
```

- Ground contact & landing grading: unchanged from v1 (3-sample ground under
  `x±lander_half_w`, both feet inside `[x0+2, x1-2]`, upright `|angle| ≤ angle_max`,
  perfect `|vx| ≤ 12 ∧ |vy| ≤ 18` → `50*mult`, hard `≤ 25/35` → `15*mult`, else crash).
- Out of bounds: `x < 0`, `x > world_w`, `y > world_h + 10` → crash.
- **Lander-lander collisions** (checked each tick after integration & terrain contact):
  - two **flying** landers with center distance < `lander_collision_dist` → BOTH crash,
    reason `"collided with another lander"`.
  - a **flying** lander within `lander_collision_dist` of a **landed** one → the flying
    one crashes (`"crashed into a landed lander"`); the landed one is unaffected
    (points stay banked — pad-blocking is legal strategy).
  - **crashed** landers are not obstacles.
  - Pair order must be deterministic (iterate i<j in lander order).
- Determinism: ALL randomness through one `random.Random(seed)` per episode.

## 6. Observation vector (14 floats per lander, sensor-model filtered)

Principle: **truth in core, perception as a filter.** The frame JSON always
carries truth; `obs` is what a policy sees, shaped by `obs_mode`.

Target pad = nearest by **euclidean distance** `hypot(pad_cx - x, pad_y - y)`
(NOT horizontal distance — euclidean keeps the shaping potential in §7
continuous when the nearest pad switches).

| idx | value                   | formula                                   |
|-----|-------------------------|-------------------------------------------|
| 0   | x (normalized)          | `x / world_w * 2 - 1`                      |
| 1   | y (normalized)          | `y / world_h * 2 - 1`                      |
| 2   | vx                      | `vx / 60`                                  |
| 3   | vy                      | `vy / 60`                                  |
| 4   | sin(angle)              | `sin(angle)`                               |
| 5   | cos(angle)              | `cos(angle)`                               |
| 6   | angular velocity        | `ang_vel / 3`                              |
| 7   | fuel fraction           | `fuel / fuel_init`                         |
| 8   | dx to target pad        | `(pad_cx - x) / world_w`                   |
| 9   | dy to target pad        | `(pad_y - y) / world_h`                    |
| 10  | pad half-width          | `(x1 - x0) / 2 / 100`                      |
| 11  | terrain clearance       | `(y - lander_bottom - ground_y(x)) / world_h` |
| 12  | pad multiplier          | `mult / 5`                                 |
| 13  | pad visible             | `1.0` or `0.0`                             |

- `obs_mode="full"`: indices 8–12 always populated, index 13 always `1.0`.
- `obs_mode="radar"`: if euclidean distance to the nearest pad > `radar_range`,
  indices 8, 9, 10, 12 are `0.0` and index 13 is `0.0` (the agent must explore).
  Pad targeting still uses the true nearest pad when visible.
- Future (documented, NOT implemented): `"lidar"` mode (terrain rays, no pad oracle),
  seeded sensor noise, other-lander slots + message channel for the multi-agent env.

## 7. Gymnasium env (`moonlander.env.MoonLanderEnv` — Python-side only)

- `MoonLanderEnv(mode="classic", obs_mode="full", frame_skip=1, config=None)` —
  single agent (wraps `Game(n_landers=1)`); the multi-agent PettingZoo wrapper is
  a later phase.
- `frame_skip=k`: each env step applies the action for k physics ticks (stopping
  early on terminal), sums per-tick fuel costs, and evaluates shaping across the
  whole env step. Truncation counts **ticks** (an episode is still `max_steps`
  ticks of physics regardless of k). The env does not bypass `Game` semantics.
- **Training note (audit-verified):** at k=1 a good landing is ~1300+ decisions, so
  with γ=0.99 the terminal reward is discounted to ~0.0002 and *hovering beats
  landing* in discounted return. Train with `frame_skip=4` and `gamma >= 0.997`
  (0.999 recommended). frame_skip up to 8 preserves 100% autopilot landability.
- Registered as **`MoonLander-v0`** via a guarded `gymnasium.register` in
  `moonlander/__init__.py` (`try/except ImportError` so the browser, which has no
  gymnasium, imports cleanly; `entry_point="moonlander.env:MoonLanderEnv"`).
- `action_space = Discrete(4)` (as v1). `observation_space = Box(-10, 10, (14,), float32)`.
- Reward:
  ```
  φ(s)  = -1.0 * dist - 0.5 * speed - 0.5 * |sin angle|
          dist  = (min over pads of euclidean distance to pad center) / world_w
          speed = hypot(vx, vy) / 60
  r_t   = 10 * (φ(s') - φ(s)) - 0.06 * (1 if main engine actually fired else 0)
  terminal: perfect → +100 + 10*mult;  hard → +30;  crash → -100
  ```
  φ's min-over-pads form is continuous in state (no fake reward when the nearest
  pad switches) and potential-based, so shaping is policy-invariant.
- `reset(seed=...)` → `(obs, info)`, `info["terrain"]` = parsed terrain dict.
  `seed=k` follows the gymnasium `np_random` chain (Game seed is derived, so the
  terrain JSON `seed` field ≠ k); `reset(options={"game_seed": k})` seeds the Game
  directly — terrain `seed` == k, byte-identical to the web app's `?seed=k`.
- On the terminal step: `info["outcome"]` = a **copy** of the outcome dict,
  `info["is_success"]` = landed (SB3 eval convention), `info["score"]` = points.
  On truncation: `info["is_success"] = False`. `truncated` at `config.max_steps` ticks.
- Calling `step()` after termination raises `RuntimeError` (the v1 behavior of
  silently re-paying the terminal reward corrupted manual eval loops).

## 8. Web frontend

Files: `web/index.html`, `web/app.js`, `web/renderer.js`, `web/effects.js`,
wheel at `web/assets/moonlander-0.2.0-py3-none-any.whl` (version matches pyproject).

**Boot** (as v1): pyodide v0.26.4 from jsdelivr, `loadPackage("micropip")`,
`micropip.install(new URL("assets/moonlander-0.2.0-py3-none-any.whl", location.href).href)`,
no numpy. Boot errors render on canvas.

**App states**: `LOADING` → `TITLE` (attract: `Game(n_landers=3)` flown by
`step_auto_all()`; collisions welcome) → `REVEAL` → `FLYING` (human:
`Game(n_landers=1)`) → `ENDED` → `REVEAL` …  Session score accumulates in JS
from lander 0 of human episodes only.

**URL param**: `?seed=123` → every human `reset` uses that seed (classroom: everyone
flies the same terrain; attract still uses fresh seeds). Invalid/absent → entropy.

**High score**: best single-episode points in `localStorage["moonlander.high"]`,
shown in the HUD line as `HIGH <n>` under SCORE, updated when beaten.

**Keys** (unchanged): `←→/AD` rotate · `↑/W/Space` thrust · `R` new episode ·
`O` agent-view overlay (now 14 labeled values incl. PAD MULT, PAD VISIBLE).

**Touch controls** (pointer events; shown only on coarse-pointer devices):
three translucent hold-zones — left third = rotate left, right third = rotate
right, bottom-center = thrust — feeding the same held-input set as the keyboard;
multi-touch works (rotate+thrust). `touch-action: none` on the canvas; in
portrait orientation show a "ROTATE YOUR DEVICE" line. Any tap leaves TITLE/ENDED.

**Page chrome** (retro presentation, §9 look): black page; header `LUNAR LANDER`
in glowing vector-style lettering with a small lander glyph; one-line tagline
under it (e.g. `A CLASSIC 1979 ARCADE — REBUILT AS A REINFORCEMENT-LEARNING ARENA`);
footer with control legend and a GitHub link placeholder. Inline SVG favicon
(white lander outline on black). OpenGraph/meta tags (title, description) for
link sharing. No external fonts/assets beyond the pyodide CDN.

## 9. Renderer (vector-monitor look)

- Canvas logical size = `world_w × world_h` (2000×750), CSS-scaled to fit width,
  aspect preserved. Black `#000`; white strokes `lineWidth ~1.5/cameraScale`;
  glow `shadowBlur 6` white. HUD/banners/overlay in screen space, font sizes
  scaled ~1.6× relative to v1 so they read at the wider canvas's CSS scale.
- Terrain polyline, pads (brighter double stroke + `"2X"` labels below), stars: as v1.
- **All landers** in `landers[]` are drawn (same §9-v1 polyline shape, per-lander
  flame); a crashed lander is hidden once its explosion has been spawned; a landed
  lander stays visible parked on its pad.
- Attract title, banners, blink, reveal: as v1. Banners describe lander 0 in
  human episodes.

## 10. Cosmetic animation layer (JS-side only)

As v1 (cosmetics may use browser RNG/clock; never feed back into Python), plus:

- **Explosions are per-lander**: triggered on each lander's flying→crashed
  transition (attract and human alike), debris from that lander's pose.
- **Arcade zoom**: engages only when exactly one lander is flying (`active == 1`)
  AND its altitude < 150 — target scale **3.0** (canvas is half-scale on screen
  now, so 3.0 ≈ v1's 2.2 apparent), centered between that lander and the ground;
  otherwise target the full-world view. Same exponential lerp (~3 s⁻¹), view
  clamped inside world bounds, frozen during ENDED.
- Terrain reveal traces the full 2000-wide polyline in the same ~0.7 s.
