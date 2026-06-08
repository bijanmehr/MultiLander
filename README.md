<div align="center">

```
   __    _  _  __ _   __   ____    __    __   __ _  ____  ____  ____
  (  )  / )( \(  ( \ / _\ (  _ \  (  )  / _\ (  ( \(    \(  __)(  _ \
  / (_/\) \/ (/    //    \ )   /  / (_/\/    \/    / ) D ( ) _)  )   /
  \____/\____/\_)__)\_/\_/(__\_)  \____/\_/\_/\_)__)(____/(____)(__\_)
                                  ▲
                                 ▕ ▏        SCORE 0770   ALTITUDE  676
                                ◢ ▾ ◣       FUEL  2392   VSPEED   39↓
              ◢◣            ◢◣
          ◢◣◢◤◥◣    ◢◣   ◢◤  ◥◣   ◢◣        2X    5X      3X    2X
   ◢◤◥◣◢◤      ◥◤◥◣◢◤◥◣◢◤      ◥◣◢◤ ◥◣◢◣ ▔▔▔▔  ▔▔▔▔▔  ▔▔▔▔  ▔▔▔▔▔
```

**The 1979 arcade classic, rebuilt as a reinforcement-learning arena.**

*One pure-Python physics core — it trains your agents with Gymnasium
and runs live in the browser via WebAssembly. No port, no backend, no drift.*

</div>

---

## What is this?

Lunar Lander the way the arcade had it — vector terrain, score multipliers,
unforgiving fuel — built from scratch as a research and teaching platform:

- 🕹️ **Play it** — classic rotate-and-thrust flight, in your browser, on a phone, anywhere
- 🤖 **Train on it** — a clean Gymnasium env (`MoonLander-v0`), deterministic, fast, dependency-free core
- 📺 **Show it** — the whole thing deploys to GitHub Pages as static files; the title
  screen *is* a live demo (a built-in autopilot flying real episodes, 1979 attract-mode style)
- 🧑‍🏫 **Teach with it** — an agent-view overlay shows exactly what the policy observes,
  sensor models make partial observability tangible, and `?seed=` puts a whole
  classroom on the same terrain

And the part that makes it a platform rather than a toy: **multiple landers
share one world** — solid, collidable, each with its own fuel, score, and fate.

## The road ahead

| Phase | What | Status |
|---|---|---|
| 1 | Environment, physics, web visualization | ✅ done |
| 2 | Retro animation layer: explosions, attract mode, arcade zoom | ✅ done |
| 3 | Multi-lander world, sensor models (partial observability), GitHub Pages | ✅ this release |
| 4 | PPO baseline (SB3), in-browser trained-agent showcase | 🔶 CEM baseline shipped (`P` in the game) — PPO next |
| 5 | **Algorithm arena** — train different algorithms, watch them fly side by side on identical seeds | planned |
| 6 | **Competition** — multi-agent (PettingZoo), pad-blocking strategy, collision risk, comm channels for emergent cooperation | planned |
| 7 | **Human + AI co-op** — you fly one lander, the agent flies the other, shared mission | the dream |

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q          # the env's contract, enforced
./scripts/build_web.sh                 # wheel → web/assets/
./scripts/serve.sh                     # → http://localhost:8000
```

**Controls** — `←`/`→` rotate · `↑`/`Space` thrust · `R` new terrain ·
`O` agent view · `P` AI pilot · `1`/`2`/`3` difficulty (TRAINEE / CADET /
COMMANDER) · arcade buttons on mobile. Land slow and upright; narrow pads pay 5X.

## Training interface

```python
import gymnasium, moonlander

env = gymnasium.make("MoonLander-v0")            # classic rotate+thrust
# or: MoonLanderEnv(mode="gym")                  # LunarLander-style engines
# or: MoonLanderEnv(obs_mode="radar")            # partial observability: pads
#                                                # invisible beyond radar range

obs, info = env.reset(seed=42)                   # same seed = same world, always
obs, r, term, trunc, info = env.step(env.action_space.sample())
if term: print(info["outcome"])                  # {"kind": "perfect", "mult": 5, ...}
```

- **Observation** `Box(14,)` — pose, velocity, attitude, fuel, vector to nearest pad,
  pad width + multiplier, terrain clearance, pad-visible flag. All world-size-relative.
- **Actions** `Discrete(4)` — noop / rotate-left / rotate-right / thrust.
- **Reward** — potential-based shaping toward the pad (policy-invariant, continuous),
  small fuel cost, terminal +100·(multiplier bonus) / −100.
- **Difficulty = curriculum** — `MoonLanderEnv(preset="trainee" | "cadet" | "commander")`:
  rougher terrain, narrower pads, tighter fuel. Train up the same ladder the
  web game's `1`/`2`/`3` keys select.
- **Multi-lander core** (single-agent env wraps `n_landers=1`):

```python
from moonlander.core.game import Game
g = Game(n_landers=3)                            # one world, three landers
g.reset(seed=7)                                  # shared terrain, spread spawns
g.step_all('[[1, true], [0, false], [-1, true]]')  # solid: collisions crash both
```

The simplest learned pilot ships first: a 308-parameter MLP trained with the
cross-entropy method — no gradients, no discount factor, numpy only:

```bash
.venv/bin/python -m moonlander.train_cem      # minutes → web/assets/policy.json
```

Writing your own training stack? [`examples/train_template.py`](examples/train_template.py)
is an annotated, runnable skeleton — it tours every world function (the
14-input observation, the Gym env, the raw `Game` core, the reward) with a
working baseline and a `PLUG YOUR ALGORITHM HERE` section:

```bash
.venv/bin/python -m examples.train_template --tour    # see the whole API
.venv/bin/python -m examples.train_template           # train + export a flyable policy
```

Press `P` in the web game to hand it the stick — or **bring your own brain**:
drag any policy JSON onto the game (`LOAD AI` in the footer works too) and your
network flies instead. Every moving part, the import format, and a downloadable
80-parameter example live at
**[ml.html](https://bijanmehr.github.io/MultiLander/ml.html)**.

## Architecture (why this works in a browser)

```
            ┌────────────────────────  Python  ────────────────────────┐
            │  moonlander.core — physics, terrain, rules, autopilot    │
            │  PURE STDLIB: no numpy, no nothing. ~30 KB wheel.        │
            └──────────────┬────────────────────────┬──────────────────┘
              Gymnasium env│(numpy, training only)   │ Pyodide (CPython on WASM)
                           ▼                         ▼
                    your RL training          browser: JS draws what
                    (SB3, CleanRL, …)         Python computes, 60 Hz
```

**The boundary rule:** JS never simulates, Python never draws. Everything that
crosses is JSON, specified in [docs/CONTRACT.md](docs/CONTRACT.md). Same seed →
same episode, byte-for-byte, in CI, in training, and in your browser.

## Deploying to GitHub Pages

Already wired: `.github/workflows/pages.yml` tests, builds the wheel, and
publishes `web/` on every push to `main` (Pages source: GitHub Actions).
Live at **https://bijanmehr.github.io/MultiLander/** — embed it anywhere
with an `<iframe>`.
The site ships two pages: the game, and `docs.html` — a same-theme mini-manual
covering the controls, scoring, and the full RL interface.

## Project layout

```
src/moonlander/      config.py · core/ (terrain, physics, game, autopilot, policy) · env.py · train_cem.py
web/                 index.html · app.js · renderer.js · effects.js · ml.html — all rendering + the AI explainer
examples/            train_template.py — annotated training-stack starting point
tests/               determinism, physics, collisions, observations, env contract, policy
docs/                DESIGN.md (decisions) · CONTRACT.md (the frozen Py⇄JS interface)
scripts/             build_web.sh · serve.sh
.github/workflows/   test + Pages deploy
```
