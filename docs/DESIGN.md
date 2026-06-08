# Moon Lander — Design

*Started 2026-06-07. Decisions made interactively; the exact Python⇄JS interface
lives in `CONTRACT.md` (currently v2).*

## Vision

A 1979-Atari-style Lunar Lander that grows into a **multi-agent RL arena and
teaching platform**: train agents → compare algorithms side by side on identical
seeds → make them compete in one world (collisions, pad-blocking, comm channels)
→ eventually human + AI cooperative missions. The web showcase (GitHub Pages,
Pyodide) is a first-class deliverable: the same Python core that trains agents
runs live in visitors' browsers.

## Decisions

| Topic | Decision | Why |
|---|---|---|
| Mechanics | Both, configurable: **classic** (rotate+thrust) and **gym** (engines) | compare control schemes; classic is the showcase |
| Observation | State vector, **sensor-model filtered** (full / radar; lidar+noise later) | trains fast; PO is a config layer, not baked into physics — "truth in core, perception as a filter" |
| Web runtime | **Pyodide** (CPython on WASM) | single physics source of truth, no JS port |
| Rendering | **Python headless, JS-only visuals** | no desktop UI needed; matplotlib later if needed |
| Core deps | **Pure stdlib** | browser skips the numpy wheel; ~30 KB total |
| World (v2) | 2000×750, full view, 5 pads (2X 2X 3X 3X 5X), everything config-relative | room for multi-lander + radar-PO exploration |
| Multi-lander (v2) | **Solid**: flying-flying contact crashes both; flying-landed crashes the flyer (pad-blocking is legal strategy) | user choice — collision-on-contact is simple (no resolution physics) and enables adversarial play |
| Comms (future) | Message channel lives in the PettingZoo wrapper, zero core changes; pairs with radar PO (private info makes messages worth sending) | emergent-communication research needs PO to matter |
| Training | SB3 PPO (next phase), weights exported for in-browser inference | |
| Policy interface (0.5.0) | tiny MLP forward pass in core (stdlib, `policy.py`); weights as JSON (CONTRACT §11); `examples/train_template.py` to train+export; web loads via drag-drop / LOAD AI | ship the interface, not a particular learner — anyone's trainer that emits the JSON flies in the browser; the game ships no trained brain of its own |

## Pre-training audit (2026-06-07)

An 18-agent adversarial audit before any training. Confirmed and fixed:

- **Discount-horizon inversion** — at γ=0.99, hovering beat landing in discounted
  return (J(hover)=−2.53 vs J(land)=−3.22; ~1360-step episodes discount the
  terminal +130 to 0.0002). Fixed with `frame_skip` + a contract-documented
  γ≥0.997 constraint. *PPO with SB3 defaults would have converged to hovering —
  the globally optimal policy of the MDP we had specified.*
- **Unwrapped angle** — a full revolution made byte-identical sin/cos observations
  carry opposite terminal rewards. Fixed: wrap to (−π, π] after integration.
- **φ discontinuity at pad switches** (reward spikes ~20× shaping signal) — fixed
  structurally in v2: min-over-pads euclidean potential.
- Gym-mode ang_vel Box violation (adversarially demonstrated at 10.37) → physical clamp.
- Terminal `info` keys for SB3 eval, step-after-terminal guard, outcome aliasing,
  ~3× env.step speedup (frame-JSON skip), `allow_nan=False`.

Rejected as non-issues after refutation: pad-multiplier "invisibility" (reward is
provably a function of obs), 3-sample contact gaps (self-consistent MDP, pads
exact), input validation, obs scaling fears.

## Current scope (end of phase 3)

Env (single + multi-lander core) + human interface (keyboard/touch) + web viz +
attract autopilot + GitHub Pages pipeline. **No training stack yet** — phase 4.

## Architecture

```
src/moonlander/
  config.py        every constant; pure stdlib, ships to browser
  core/
    terrain.py     seeded terrain, 5 flat pads, spawns with min separation
    physics.py     both control modes, angle wrap, ang_vel clamp
    game.py        multi-lander state machine, collisions, JSON interface
    autopilot.py   stdlib proportional pilot (attract mode; lands ~100%)
  env.py           Gymnasium wrapper (numpy/gymnasium live ONLY here); frame_skip
web/               index.html · app.js · renderer.js · effects.js
tests/             determinism, physics, collisions, obs/sensor, env contract, audit pins
.github/workflows/ test + Pages deploy (wheel built fresh in CI)
```

Boundary rule: **JS never simulates game state; Python never draws.** Cosmetics
(debris, camera, blink) are JS-side and never feed back. Everything crossing the
boundary is a JSON string.

## Roadmap

4. PPO baseline (SB3, `frame_skip=4`, γ=0.999) + in-browser trained-agent showcase
   (numpy-free MLP forward pass, or replay-log playback — a full episode fits in
   ~128 URL-safe bytes). *0.5.0 shipped the interface: a stdlib MLP forward
   pass in core, AI PILOT mode (`P`) with drag-drop / LOAD AI policy import,
   and `examples/train_template.py` — the numpy-free MLP path is built and
   proven; a default trained pilot is still to come.*
5. Algorithm arena: N policies side by side on identical seeds
6. Multi-agent (PettingZoo): competition, pad-blocking, lidar/noise sensor models,
   message channels
7. Human + AI co-op missions

Backlog of audit-verified ideas: watch/kiosk mode (`?watch=1`), difficulty presets
(TRAINEE/CADET/COMMANDER as curriculum ladder), touchdown debrief panel, ghost race
on shared seed, fuel carry-over arcade mode, daily-challenge seed, overlay value bars.
