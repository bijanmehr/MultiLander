# AI Pilot: CEM-trained MLP + in-game agent mode + explainer page — Design

*Approved 2026-06-07. Implements the simplest possible learned pilot for one lander,
a first-class AI PILOT mode in the web game, and `web/ml.html` explaining the machinery.
Ships as 0.5.0.*

## Goal

The most simplistic ML module that pilots one rocket:

- A tiny MLP policy (14 obs → 16 tanh → 4 logits → argmax, **308 parameters**) trained
  with the **cross-entropy method** (CEM) — no gradients, no ML framework, numpy only
  (already in the `[env]` extra; zero new dependencies).
- The trained pilot **flies in the browser**: weights export to JSON, the forward pass
  is pure-stdlib Python inside the existing Pyodide core (same pattern as the autopilot).
- An **`A` key / `AI` arcade button** toggles AI PILOT mode in the game.
- A **separate webpage `web/ml.html`** (docs.html retro theme) explains all the machinery.

Success bar (user choice): *visibly learned, lands sometimes* on TRAINEE. Training runs
in minutes on a laptop, single process. Honest imperfection is teaching material.

## Components

### 1. `src/moonlander/core/policy.py` — pure-stdlib MLP forward pass

Core rules apply: imports **only `math` and `json`** (enforced by a subprocess purity
test like the autopilot's, which allows only `math`).

- `class Policy` holding `w1` (h×14), `b1` (h), `w2` (4×h), `b2` (4) as lists of floats.
  `w1[j][i]` = weight from input *i* to hidden *j*; same orientation for `w2`.
- `Policy.from_json(s)` — parses and validates: `format == "mlp-tanh-argmax/v1"`,
  `sizes == [14, h, 4]` with `h >= 1`, all row/vector lengths consistent with `sizes`,
  every value a finite float. Any violation → `ValueError` naming the problem.
- `Policy.act(obs) -> (rotate, thrust)` — `h = tanh(W1·x + b1)`, `logits = W2·h + b2`,
  `action = argmax` (**first index wins ties** — deterministic), then the exact
  `Discrete(4)` mapping already used by `env.py`: 0 = noop, 1 = rotate-left,
  2 = rotate-right, 3 = thrust (implementation copies env.py's mapping; signs must match).
- The forward pass stays ~15 lines — it is printed verbatim on `ml.html`.

### 2. `core/game.py` — `set_policy` / `step_policy` (mirrors `step_auto`)

- `Game.set_policy(policy_json: str)` — attaches `Policy.from_json(policy_json)` to the
  game instance (per-instance, not module global; survives until the Game is rebuilt).
- `Game.step_policy() -> str` (frame JSON) — guards, in the same style as `step_auto`:
  - `ValueError` if `n_landers != 1` (mirror of `Game.step`)
  - `NotImplementedError` in gym mode (mirror of `step_auto`)
  - `RuntimeError` mentioning `set_policy` if no policy is attached
  - terminal episode → byte-identical no-op frame (same semantics as `step_auto`)
  - otherwise: `controls = policy.act(self.obs(0))` (obs respects the game's `obs_mode`),
    apply via the normal classic-mode tick, return frame JSON.
- Boundary rule intact: JS never sees the network; weights cross once as a JSON string,
  frames come back as JSON. JS still never simulates.

### 3. `src/moonlander/train_cem.py` — the trainer

Lives in the package (not `scripts/`) so its pure functions are unit-testable with a
plain import and it runs as `python -m moonlander.train_cem`. Top-level numpy/gymnasium
imports are fine here: nothing imports this module automatically, so `import moonlander`
still never pulls them in (the existing subprocess test keeps this honest). It ships in
the wheel unused, exactly like `env.py` (~4 KB; never imported in the browser).

Algorithm (classic CEM, ~40 lines of numpy for the core loop):

1. Gaussian over the flattened 308-weight vector: `mean = 0`, `std = 1.0`.
2. Each generation: sample `pop=64` candidate weight vectors; evaluate each by flying
   `episodes=3` full episodes on `MoonLanderEnv(preset="trainee", frame_skip=4)`;
   fitness = mean **undiscounted** return (sum of env rewards — no γ anywhere).
3. **Common random numbers:** the 3 episode seeds are drawn once per generation from the
   master rng and shared by all 64 candidates (fair comparison, lower variance — and a
   nice explainer-page point).
4. Keep the `elite=10` best; refit `mean`/`std` to the elite, then `std += 0.02` noise
   floor (prevents premature collapse).
5. Repeat `gens=60` (defaults; all CLI flags: `--pop --elite --episodes --gens --hidden
   --seed --preset --out`). `--seed` makes the whole run reproducible.
6. Final eval: greedy policy at the final mean, landing rate (perfect|hard) over seeds
   0..29 on the training preset — printed, and stored in the artifact meta.

Pure, unit-testable pieces factored out: `cem_update(samples, fitnesses, elite_k) ->
(mean, std)` and the flatten/unflatten between vector and `Policy` weights.

### 4. Artifact: `web/assets/policy.json`

```json
{
  "format": "mlp-tanh-argmax/v1",
  "sizes": [14, 16, 4],
  "w1": [[...14 floats] x16], "b1": [...16],
  "w2": [[...16 floats] x4],  "b2": [...4],
  "meta": {
    "preset": "trainee", "pop": 64, "elite": 10, "episodes": 3, "gens": 60, "seed": 0,
    "fitness_history": [{"gen": 1, "best": ..., "elite_mean": ..., "pop_mean": ...}, ...],
    "eval": {"seeds": "0..29", "landed": N, "perfect": M}
  }
}
```

**Committed to git** (small release artifact; `.gitignore` only excludes `*.whl`), so it
deploys to Pages automatically and `ml.html` can draw the learning curve from it.

### 5. Web game: AI PILOT mode (`app.js`, `renderer.js`, `index.html`)

- Boot: after Pyodide is ready, `fetch("assets/policy.json")` (non-fatal on failure);
  keep the raw string in JS; call `game.set_policy(...)` now and again after every
  `ensureGame` rebuild (preset/lander-count changes).
- **`A` key** toggles `aiPilot` in TITLE / FLYING / ENDED. Like `1/2/3` it is handled
  as a mode key and **never counts as "press any key"** (branch order in
  `handleKeyPress` is load-bearing — same pattern as presets, but allowed in FLYING too).
- Tick: when FLYING and `aiPilot` and policy loaded → `game.step_policy()`; otherwise
  the existing `game.step(rotate, thrust)` human path.
- **Take-over rule:** while AI is flying, any **new press event** (keydown /
  pointerdown) of a rotate/thrust control instantly disengages to human — edge-
  triggered, so a thumb already resting on THRUST when AI is engaged does not kick it
  out, but grabbing the stick does.
- Mode persists across episodes and resets (binge-watching); the `O` obs overlay works
  in AI mode and shows exactly the 14 numbers the net sees.
- HUD: renderer draws **AI PILOT** (vector font) while engaged.
- No policy loaded (fetch failed) → pressing A / tapping AI shows a transient
  `NO POLICY — RUN python -m moonlander.train_cem` screen-space message (~2 s).
- **Mobile:** fifth chamfered-octagon arcade button `AI` (top-right, by OBS), same
  `.abtn` CSS and wiring pattern as the OBS button (edge-triggered toggle); label drawn
  by VectorFont like the others.
- Title attract mode keeps the scripted autopilot (it is the better pilot — that's the
  cabinet demo). AI PILOT is the human lander's mode.

### 6. Explainer page: `web/ml.html`

Same theme and constraints as `docs.html`: loads **only `vectorfont.js`**, canvas
vector-font headings, monospace body, black/white, relative links, works from `file://`
(except the curve, see below). Sections:

1. **THE LOOP** — sense → think → act, every 1/60 s tick.
2. **WHAT IT SEES** — the 14 observation inputs (mirrors the CONTRACT §6 table).
3. **THE NETWORK** — 14→16→4 diagram (preformatted ASCII), "308 numbers", tanh, argmax,
   and the actual `policy.py` forward pass verbatim.
4. **HOW IT LEARNS** — CEM: guess 64 networks, fly them, keep the 10 best, re-center
   the guessing distribution on the winners, repeat; pseudocode plus the actual
   `cem_update` numpy code verbatim; the common-random-numbers and noise-floor tricks.
5. **WHY NO GRADIENTS, WHY NO γ** — trade-offs; the audit's hover-trap story (at γ=0.99
   hovering provably beat landing; CEM optimizes raw return so the trap cannot exist);
   pointers to REINFORCE/PPO as the roadmap's next steps.
6. **THE RESULTS** — learning curve drawn on a canvas from `policy.json`'s
   `fitness_history` (fetch; on failure — e.g. `file://` — show a fallback line:
   `SERVE THE SITE TO SEE THE TRAINING CURVE`); landing-rate eval vs the autopilot's;
   "trained on TRAINEE — switch difficulty to watch it struggle".
7. **TRY IT** — press `A` on the game page; link back.

Cross-links: game page footer and `docs.html` link to `ml.html`; `ml.html` links back.
`docs.html` controls section gains the `A` key.

### 7. Contract, docs, version (lockstep)

- **CONTRACT.md** updated in the same change (it is load-bearing; tests cite sections):
  `set_policy`/`step_policy` semantics + error contract in §2; the `policy.json` schema
  (new section); the `A` key, AI button, HUD indicator and take-over rule in §8;
  revision bumped to 0.5.0.
- **Version 0.4.0 → 0.5.0 in all four coupled spots** (the known footgun):
  `pyproject.toml`, `src/moonlander/__init__.py`, the hardcoded wheel URL in
  `web/app.js`, CONTRACT §8. Rebuild the wheel via `scripts/build_web.sh`.
- README: controls line gains `A`; Training section gains the one-liner
  `python -m moonlander.train_cem`; roadmap row updated (phase 4 partially delivered —
  simplest-possible baseline before PPO).
- DESIGN.md: decisions row (CEM chosen as the simplest teachable baseline; PPO still
  the phase-4 plan for a *strong* pilot).

### 8. Error handling summary

| Failure | Behavior |
|---|---|
| Bad/`NaN`/mis-shaped weights JSON | `Policy.from_json` → `ValueError` naming the problem |
| `step_policy` with no policy | `RuntimeError` mentioning `set_policy` |
| `step_policy` in gym mode | `NotImplementedError` (mirror autopilot) |
| `step_policy` with `n_landers != 1` | `ValueError` (mirror `Game.step`) |
| `step_policy` after terminal | byte-identical no-op frame (mirror `step_auto`) |
| `policy.json` fetch fails in browser | AI mode unavailable; A/AI shows `NO POLICY` message; game otherwise normal |
| `ml.html` curve fetch fails (`file://`) | fallback text, page still readable |

### 9. Tests (`tests/test_policy.py`, `tests/test_train_cem.py`)

Matching the suite's style (contract-section citations, byte-determinism, subprocess
purity, one-ulp spirit):

- Forward pass equals a hand-computed tiny network (e.g. h=2, fixed weights).
- Argmax tie-break: equal logits → action 0.
- Action mapping matches `env.py`'s `Discrete(4)` (same rotate/thrust tuples).
- `from_json` rejections: wrong format string, wrong sizes, ragged rows, non-finite
  values — each a `ValueError`.
- `set_policy` + `step_policy`: same seed + same weights → byte-identical terrain and
  frame JSON across two fresh Games.
- Guards: no-policy `RuntimeError`, gym-mode `NotImplementedError`, multi-lander
  `ValueError`, step-after-terminal no-op.
- Purity: subprocess test — `policy.py` standalone imports/holds only `math` + `json`.
- `cem_update` on a toy quadratic fitness converges to the known optimum in a few
  generations (fast, no env). Flatten/unflatten round-trips. **No slow training in CI.**
- End-to-end smoke: a 1-generation run with `--pop 4 --episodes 1` writes an artifact
  that parses with `Policy.from_json` and flies via `step_policy` (~5k physics ticks
  total — well under a second; runs in CI unmarked).

### 10. Out of scope (YAGNI)

- No PPO/SB3, no gradient methods (roadmap phase 4 proper).
- No multi-lander policy flying, no policy in attract mode.
- No in-browser training, no onnx/tfjs, no replay logs.
- No JS tests (project has none; CONTRACT documents web behavior).

## Build/run summary

```bash
.venv/bin/python -m moonlander.train_cem --seed 0     # → web/assets/policy.json (minutes)
.venv/bin/python -m pytest -q                          # contract enforced
./scripts/build_web.sh && ./scripts/serve.sh           # → press A
```
