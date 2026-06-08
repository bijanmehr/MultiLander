"""Training-stack TEMPLATE for MoonLander — copy this and make it yours.

A heavily-annotated tour of everything you need to train an agent on this
world, with a tiny WORKING baseline so it runs out of the box. Swap the
baseline (PART 5) for PPO / REINFORCE / CMA-ES / your idea and keep the rest.

It closes the loop: train here, export a `policy.json` (PART 6), then drop it
on the web game (drag-and-drop or the LOAD AI link, or commit it to
`web/assets/policy.json`) and watch your agent fly in the browser. The forward
pass you train is byte-for-byte the one Pyodide runs — no port, no drift.

Run the baseline:
    .venv/bin/python -m examples.train_template --iters 300 --out /tmp/mine.json
    # then: open the game, LOAD AI, pick /tmp/mine.json

Layout:
    PART 1  the observation — what your policy sees (14 floats)
    PART 2  the world, two ways — the Gym env, and the raw Game core
    PART 3  the policy — a tiny MLP forward pass (matches the web format)
    PART 4  rollouts — using the world to score a policy
    PART 5  >>> PLUG YOUR ALGORITHM HERE <<< (a runnable hill-climb baseline)
    PART 6  export — write policy.json the browser can fly

Deps: numpy + gymnasium (the `[env]` extra). The simulation core itself is
pure stdlib; only the training side needs these.
"""

import argparse
import json

import numpy as np

# The two faces of the world. Import whichever you need; the template shows both.
from moonlander.env import MoonLanderEnv        # Gymnasium interface (training)
from moonlander.core.game import Game           # raw core (full control + web export)
from moonlander.core.policy import Policy        # the exact in-browser forward pass


# ===========================================================================
# PART 1 — THE OBSERVATION  (what your policy sees: 14 floats, CONTRACT §6)
# ===========================================================================
#
# Every observation is world-size-relative (so a policy is not tied to the
# 2000x750 default) and mostly lands in [-1, 1]. The "target pad" is the one
# nearest by EUCLIDEAN distance to its center.
#
# idx  name                     formula
OBS = [
    ("x",            "x / world_w * 2 - 1"),            # 0  horizontal position
    ("y",            "y / world_h * 2 - 1"),            # 1  altitude (world frame)
    ("vx",           "vx / 60"),                        # 2  horizontal velocity
    ("vy",           "vy / 60"),                        # 3  vertical velocity
    ("sin(angle)",   "sin(angle)"),                     # 4  attitude, modular...
    ("cos(angle)",   "cos(angle)"),                     # 5  ...so 0 rad = upright
    ("ang_vel",      "ang_vel / 3"),                    # 6  angular velocity
    ("fuel",         "fuel / fuel_init"),               # 7  fuel fraction, 1 -> 0
    ("pad_dx",       "(pad_cx - x) / world_w"),         # 8  vector to target pad x
    ("pad_dy",       "(pad_y  - y) / world_h"),         # 9  vector to target pad y
    ("pad_hw",       "(x1 - x0) / 2 / 100"),            # 10 target pad half-width
    ("clearance",    "(y - lander_bottom - ground) / world_h"),  # 11 terrain gap
    ("pad_mult",     "mult / 5"),                       # 12 target pad payout (2/3/5)
    ("pad_seen",     "1.0 or 0.0"),                     # 13 pad visible? (radar mode)
]
OBS_DIM = len(OBS)        # 14
N_ACTIONS = 4             # classic: noop / rotate-left / rotate-right / thrust


def describe_obs(o):
    """Pretty-print one observation vector against its labels — for debugging."""
    return "\n".join(f"  [{i:2d}] {name:10s} {o[i]:+.3f}   ({formula})"
                     for i, (name, formula) in enumerate(OBS))


# ===========================================================================
# PART 2 — THE WORLD, TWO WAYS
# ===========================================================================
#
# (A) The Gymnasium env — MoonLanderEnv — is the training interface. It wraps
#     one lander and gives you the standard obs/action/reward/step contract
#     that SB3, CleanRL, etc. expect. This is what you train against.
#
# (B) The raw Game core — Game — is the whole simulation: multi-lander, terrain
#     you can read, the built-in autopilot to benchmark against, and the exact
#     per-tick stepping the browser uses. Use it for understanding, for honest
#     evaluation, and to confirm your exported policy flies as it will online.


def make_env(preset="trainee", frame_skip=4, obs_mode="full"):
    """(A) The training interface. Key knobs:

    - preset: "trainee" | "cadet" | "commander" — the difficulty curriculum.
      Same physics, tougher terrain/pads/fuel. Train trainee -> cadet -> commander.
    - frame_skip: physics ticks per env.step. Use >=4: at frame_skip=1 a landing
      is ~1300 decisions and a discount of gamma=0.99 makes hovering beat landing
      (a real, audited trap). With frame_skip=4 train at gamma >= 0.997.
    - obs_mode: "full" (pad always visible) | "radar" (pad hidden beyond range —
      partial observability; the agent must explore).
    - mode: "classic" (rotate+thrust, the default and what the web game flies) or
      "gym" (LunarLander-style engines). The web export below is classic-only.
    """
    return MoonLanderEnv(preset=preset, frame_skip=frame_skip, obs_mode=obs_mode)


def env_api_tour():
    """Everything the Gym env gives you, in one runnable snippet."""
    env = make_env()
    print("action_space     :", env.action_space)        # Discrete(4)
    print("observation_space:", env.observation_space)    # Box(-10,10,(14,),float32)

    # reset(seed=k) follows gymnasium's RNG chain (terrain seed != k).
    # reset(options={"game_seed": k}) seeds the world DIRECTLY — byte-identical
    # to the web game's ?seed=k, so you can compare training and browser.
    obs, info = env.reset(options={"game_seed": 0})
    print("terrain pads     :", len(info["terrain"]["pads"]), "pads")
    print("first obs:\n" + describe_obs(obs))

    # Actions (classic): 0 noop, 1 rotate-left (+1 CCW), 2 rotate-right, 3 thrust.
    obs, reward, terminated, truncated, info = env.step(3)  # fire main engine
    print(f"after thrust: reward={reward:+.3f} terminated={terminated}")

    # Reward = potential-based shaping + terminal bonus (CONTRACT §7):
    #   phi   = -1.0*dist - 0.5*speed - 0.5*|sin angle|
    #           dist  = min-over-pads euclidean distance to a pad center / world_w
    #           speed = hypot(vx, vy) / 60
    #   r_t   = 10*(phi' - phi) - 0.06*(1 if main engine fired this step else 0)
    #   terminal: perfect -> +100 + 10*mult ; hard -> +30 ; crash -> -100
    # The shaping is potential-based (policy-invariant) and continuous across
    # pad switches, so it telescopes — your return is frame_skip-invariant.

    # On the terminal step info carries the verdict (SB3 eval convention):
    #   info["outcome"] = {"kind": "perfect"|"hard"|"crash", "mult", "points", ...}
    #   info["is_success"], info["score"]
    # Truncation happens at config.max_steps PHYSICS TICKS (not env steps).
    # Stepping a finished episode raises RuntimeError — call reset() first.
    return env


def game_api_tour():
    """The raw core: world functions the Gym env hides from you."""
    g = Game(mode="classic", n_landers=1, obs_mode="full", preset="trainee")
    g.reset(seed=0)  # returns terrain JSON; same seed -> byte-identical world

    # The 14-vector for lander i — identical math to the env's obs.
    o = g.obs(0)
    assert len(o) == OBS_DIM

    # READ THE WORLD. The terrain is fully inspectable:
    t = g.terrain
    print("pads   :", [(round(p["x0"]), round(p["x1"]), p["mult"]) for p in t.pads])
    print("height at x=1000:", round(t.height(1000.0)))   # ground elevation, any x
    print("spawns :", t.spawns)                            # one per lander
    print("points :", len(t.points), "terrain vertices")  # the full polyline
    print("config :", f"gravity={g.cfg.gravity} thrust={g.cfg.thrust_accel} "
                       f"fuel={g.cfg.fuel_init}")

    # STEP THE WORLD. Single lander:
    g.step(rotate=0, thrust=True)              # -> frame JSON
    # ...or all landers at once (multi-lander), controls as a JSON string:
    #   g.step_all('[[1, true], [0, false], [-1, true]]')
    # ...or let the built-in autopilot fly (your baseline to beat — it lands
    # ~30/30 on trainee):
    #   g.step_auto()

    # READ A LANDER'S STATE directly (handy when scoring without JSON):
    s = g.landers[0].state          # LanderState: x, y, vx, vy, angle, ang_vel, fuel
    print(f"lander: x={s.x:.0f} y={s.y:.0f} vy={s.vy:.1f} fuel={s.fuel:.0f}")
    print("status:", g.landers[0].status, " score:", g.landers[0].score)
    return g


# ===========================================================================
# PART 3 — THE POLICY  (a tiny MLP — the thing you train)
# ===========================================================================
#
# obs(14) -> tanh(hidden) -> 4 logits -> argmax -> action. This IS the web
# format (CONTRACT §11): train these weights, export them in PART 6, and the
# browser flies this exact function. Keep this shape if you want web export;
# otherwise plug in any differentiable net (torch/jax) and only convert at
# export time.


def n_params(hidden):
    """Total weights of the 14 -> hidden -> 4 MLP."""
    return OBS_DIM * hidden + hidden + hidden * N_ACTIONS + N_ACTIONS


def unflatten(theta, hidden):
    """Flat vector -> (w1, b1, w2, b2), row-major, in that order."""
    a, b = 0, OBS_DIM * hidden
    w1 = theta[a:b].reshape(hidden, OBS_DIM)
    a, b = b, b + hidden
    b1 = theta[a:b]
    a, b = b, b + hidden * N_ACTIONS
    w2 = theta[a:b].reshape(N_ACTIONS, hidden)
    b2 = theta[b:b + N_ACTIONS]
    return w1, b1, w2, b2


def forward(theta, hidden, obs):
    """Action for one observation. Argmax ties pick the lowest index — exactly
    what moonlander.core.policy.Policy does, so training and browser agree."""
    w1, b1, w2, b2 = unflatten(theta, hidden)
    h = np.tanh(w1 @ obs + b1)
    return int(np.argmax(w2 @ h + b2))


# ===========================================================================
# PART 4 — ROLLOUTS  (use the world to turn a policy into a number)
# ===========================================================================


def episode_return(env, theta, hidden, seed):
    """Undiscounted return of one greedy episode on a fixed world seed.

    This is your TRAINING signal — fast, dense (shaping reward every step).
    Most algorithms optimize exactly this. Note: returns the shaped return,
    not the landing rate; they correlate but are not the same (see evaluate)."""
    obs, _ = env.reset(options={"game_seed": seed})
    total = 0.0
    while True:
        obs, r, terminated, truncated, _ = env.step(forward(theta, hidden, obs))
        total += r
        if terminated or truncated:
            return total


def evaluate(theta, hidden, preset="trainee", n_seeds=30):
    """Honest landing rate over the web-parity seeds, flown through the SAME
    per-tick path the browser uses (Game.step_policy via a built Policy). Report
    THIS, not the shaped return — it's what 'how good is my pilot' actually means.

    Returns (landed, perfect) over game_seeds 0..n_seeds-1."""
    g = Game(mode="classic", preset=preset)
    g.set_policy(to_policy_json(theta, hidden))
    landed = perfect = 0
    for seed in range(n_seeds):
        g.reset(seed=seed)
        frame = json.loads(g.frame_json())
        for _ in range(7200):
            frame = json.loads(g.step_policy())
            if frame["status"] != "flying":
                break
        kind = (frame["landers"][0]["outcome"] or {}).get("kind")
        landed += kind in ("perfect", "hard")
        perfect += kind == "perfect"
    return landed, perfect


# ===========================================================================
# PART 5 — >>> PLUG YOUR ALGORITHM HERE <<<
# ===========================================================================
#
# Below is a deliberately tiny baseline: (1+1) random-search hill-climbing.
# It LEARNS (poorly) and RUNS, so the template is end-to-end out of the box.
#
# To make this a real training stack, REPLACE the body of `train` with your
# method and keep everything around it:
#   - keep `episode_return` / `evaluate` (PART 4) — your reward + honest metric
#   - keep `unflatten` / `forward` (PART 3) — or swap in a torch/jax net and
#     only convert to (w1,b1,w2,b2) at export time
#   - keep `make_env` (PART 2) — set preset / frame_skip / gamma per the notes
#
# Drop-in ideas: REINFORCE (policy gradient on the logits), a population
# method like CMA-ES or the cross-entropy method, or SB3 PPO (the env is a
# standard gym.Env — `PPO("MlpPolicy", make_env()).learn(...)`, then read the
# trained weights into (w1,b1,w2,b2) for export).


def train(hidden, iters, episodes, sigma, seed, preset):
    """(1+1) hill-climb: perturb the weights, keep the change if it scored
    better on a fixed set of worlds. The simplest thing that is still 'training'.

    >>> THIS is the part to replace. Everything it calls is reusable. <<<
    """
    rng = np.random.default_rng(seed)
    env = make_env(preset=preset)
    theta = np.zeros(n_params(hidden))

    # Common random numbers: score every candidate on the SAME worlds, so we
    # compare skill, not terrain luck. (Re-roll them occasionally to avoid
    # overfitting a handful of seeds — left as an exercise.)
    score_seeds = [int(s) for s in rng.integers(0, 2**31, size=episodes)]
    fitness = lambda th: float(np.mean(
        [episode_return(env, th, hidden, s) for s in score_seeds]))

    best = fitness(theta)
    for it in range(1, iters + 1):
        candidate = theta + sigma * rng.normal(size=theta.shape)
        f = fitness(candidate)
        if f > best:
            theta, best = candidate, f
        if it % max(1, iters // 20) == 0 or it == iters:
            print(f"iter {it:4d}/{iters}  best shaped-return {best:8.2f}", flush=True)
    return theta


# ===========================================================================
# PART 6 — EXPORT  (write a policy.json the browser can fly)
# ===========================================================================


def to_policy_json(theta, hidden, meta=None):
    """Pack weights into the CONTRACT §11 format. The result is accepted by
    moonlander.core.policy.Policy.from_json — the in-browser validator — so if
    this round-trips, your agent will load in the game."""
    w1, b1, w2, b2 = unflatten(theta, hidden)
    artifact = {
        "format": "mlp-tanh-argmax/v1",
        "sizes": [OBS_DIM, hidden, N_ACTIONS],
        "w1": w1.tolist(), "b1": b1.tolist(),
        "w2": w2.tolist(), "b2": b2.tolist(),
    }
    if meta is not None:
        artifact["meta"] = meta
    return json.dumps(artifact, allow_nan=False)


def save_policy(theta, hidden, path, meta=None):
    s = to_policy_json(theta, hidden, meta)
    Policy.from_json(s)  # fail loudly here if it wouldn't load in the browser
    with open(path, "w") as f:
        f.write(s)
    return path


# ===========================================================================
# main — runs the baseline end to end and exports a flyable policy
# ===========================================================================


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hidden", type=int, default=16, help="MLP hidden width")
    ap.add_argument("--iters", type=int, default=300, help="hill-climb iterations")
    ap.add_argument("--episodes", type=int, default=5, help="worlds scored per candidate")
    ap.add_argument("--sigma", type=float, default=0.3, help="perturbation scale")
    ap.add_argument("--seed", type=int, default=0, help="master seed (reproducible)")
    ap.add_argument("--preset", default="trainee", help="trainee|cadet|commander")
    ap.add_argument("--tour", action="store_true", help="print the API tours and exit")
    ap.add_argument("--out", default="/tmp/policy.json", help="export path")
    args = ap.parse_args(argv)

    if args.tour:
        print("=== ENV API ===");  env_api_tour()
        print("\n=== GAME API ==="); game_api_tour()
        return

    theta = train(args.hidden, args.iters, args.episodes, args.sigma,
                  args.seed, args.preset)
    landed, perfect = evaluate(theta, args.hidden, args.preset)
    print(f"eval: {landed}/30 landed ({perfect} perfect) on {args.preset} "
          f"seeds 0..29 (per-tick, the browser path)")
    save_policy(theta, args.hidden, args.out, meta={
        "algo": "hill-climb (template baseline — replace me)",
        "hidden": args.hidden, "iters": args.iters, "episodes": args.episodes,
        "sigma": args.sigma, "seed": args.seed, "preset": args.preset,
        "eval": {"seeds": "0..29", "landed": landed, "perfect": perfect},
    })
    print(f"wrote {args.out}  ->  LOAD AI in the game, or copy to "
          f"web/assets/policy.json")


if __name__ == "__main__":
    main()
