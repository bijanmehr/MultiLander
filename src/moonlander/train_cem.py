"""CEM trainer for the tiny MLP pilot — the simplest ML that flies (§11).

The whole idea of the cross-entropy method: keep a Gaussian over ALL the
network's weights; each generation sample candidate networks, fly each one,
keep the elite, refit the Gaussian to the elite, repeat. No gradients, no
backprop — and no discount factor: fitness is the raw undiscounted episode
return, so the γ-horizon trap documented in DESIGN.md cannot exist here.

Run:   python -m moonlander.train_cem --seed 0
Deps:  numpy + gymnasium (the [env] extra) — training side only. Nothing
imports this module automatically, so ``import moonlander`` stays
stdlib-pure (tests/test_env.py enforces that).
"""

import argparse
import json

import numpy as np

from .env import MoonLanderEnv

OBS_DIM = 14
N_ACTIONS = 4
FORMAT = "mlp-tanh-argmax/v1"  # must match core/policy.py


def n_params(hidden):
    """Total weight count of the 14 -> hidden -> 4 MLP."""
    return OBS_DIM * hidden + hidden + hidden * N_ACTIONS + N_ACTIONS


def unflatten(theta, hidden):
    """Flat weight vector -> (w1, b1, w2, b2), row-major, in that order."""
    a, b = 0, OBS_DIM * hidden
    w1 = theta[a:b].reshape(hidden, OBS_DIM)
    a, b = b, b + hidden
    b1 = theta[a:b]
    a, b = b, b + hidden * N_ACTIONS
    w2 = theta[a:b].reshape(N_ACTIONS, hidden)
    b2 = theta[b:b + N_ACTIONS]
    return w1, b1, w2, b2


def act(parts, obs):
    """argmax action of the MLP — numpy mirror of core/policy.py (ties -> lowest)."""
    w1, b1, w2, b2 = parts
    h = np.tanh(w1 @ obs + b1)
    return int(np.argmax(w2 @ h + b2))


def episode_return(env, parts, seed):
    """Undiscounted return of one greedy episode on ``game_seed=seed``."""
    obs, _ = env.reset(options={"game_seed": seed})
    total = 0.0
    while True:
        obs, r, terminated, truncated, _ = env.step(act(parts, obs))
        total += r
        if terminated or truncated:
            return total


def cem_update(samples, fitnesses, elite_k):
    """One CEM step: (mean, std) of the elite_k best samples (higher = better)."""
    order = np.argsort(fitnesses)[::-1]
    elite = samples[order[:elite_k]]
    return elite.mean(axis=0), elite.std(axis=0)


def evaluate(env, parts, n_seeds):
    """(landed, perfect) over game_seeds 0..n_seeds-1 — the web-parity seeds."""
    landed = perfect = 0
    for seed in range(n_seeds):
        obs, _ = env.reset(options={"game_seed": seed})
        while True:
            obs, _r, terminated, truncated, info = env.step(act(parts, obs))
            if terminated or truncated:
                break
        kind = info.get("outcome", {}).get("kind")
        landed += kind in ("perfect", "hard")
        perfect += kind == "perfect"
    return landed, perfect


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Train the tiny MLP pilot with the cross-entropy method."
    )
    ap.add_argument("--pop", type=int, default=64, help="candidates per generation")
    ap.add_argument("--elite", type=int, default=10, help="winners refit each generation")
    ap.add_argument("--episodes", type=int, default=3,
                    help="episodes per candidate (seeds shared across the generation)")
    ap.add_argument("--gens", type=int, default=60, help="generations")
    ap.add_argument("--hidden", type=int, default=16, help="hidden layer width")
    ap.add_argument("--noise", type=float, default=0.02,
                    help="std floor added each generation (never stop exploring)")
    ap.add_argument("--seed", type=int, default=0, help="master seed for the whole run")
    ap.add_argument("--probes", type=int, default=6,
                    help="held-out episodes per generation used to pick the checkpoint")
    ap.add_argument("--preset", default="trainee", help="difficulty preset to train on")
    ap.add_argument("--eval-seeds", type=int, default=30,
                    help="final-eval episode count (game_seeds 0..n-1)")
    ap.add_argument("--out", default="web/assets/policy.json", help="artifact path")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    env = MoonLanderEnv(preset=args.preset, frame_skip=4)
    n = n_params(args.hidden)
    mean, std = np.zeros(n), np.full(n, 1.0)

    # Checkpoint: CEM's mean wanders — the FINAL generation is not necessarily
    # the best one. Probe each generation's mean on fixed held-out seeds
    # (disjoint from the 0..29 eval set, untouched by the master rng) and ship
    # the best report card, not the last one.
    best_mean, best_probe, best_gen = mean.copy(), -np.inf, 0
    probe_seeds = range(1000, 1000 + args.probes)

    history = []
    for gen in range(1, args.gens + 1):
        # Common random numbers: every candidate flies the SAME episode seeds
        # this generation — candidates compete on skill, not terrain luck.
        seeds = [int(s) for s in rng.integers(0, 2**31, size=args.episodes)]
        samples = rng.normal(mean, std, size=(args.pop, n))
        fitnesses = np.array([
            np.mean([episode_return(env, unflatten(th, args.hidden), s)
                     for s in seeds])
            for th in samples
        ])
        mean, std = cem_update(samples, fitnesses, args.elite)
        std += args.noise
        probe = np.mean([episode_return(env, unflatten(mean, args.hidden), s)
                         for s in probe_seeds])
        if probe > best_probe:
            best_probe, best_mean, best_gen = probe, mean.copy(), gen
        top = np.sort(fitnesses)[::-1]
        history.append({
            "gen": gen,
            "best": round(float(top[0]), 2),
            "elite_mean": round(float(top[:args.elite].mean()), 2),
            "pop_mean": round(float(fitnesses.mean()), 2),
            "probe": round(float(probe), 2),
        })
        h = history[-1]
        print(f"gen {gen:3d}/{args.gens}  best {h['best']:9.2f}  "
              f"elite {h['elite_mean']:9.2f}  pop {h['pop_mean']:9.2f}  "
              f"probe {h['probe']:9.2f}", flush=True)

    print(f"checkpoint: gen {best_gen} (probe {best_probe:.2f})")
    parts = unflatten(best_mean, args.hidden)
    landed, perfect = evaluate(env, parts, args.eval_seeds)
    print(f"eval: {landed}/{args.eval_seeds} landed ({perfect} perfect) on "
          f"{args.preset} game_seeds 0..{args.eval_seeds - 1}")

    w1, b1, w2, b2 = parts
    artifact = {
        "format": FORMAT,
        "sizes": [OBS_DIM, args.hidden, N_ACTIONS],
        "w1": w1.tolist(), "b1": b1.tolist(),
        "w2": w2.tolist(), "b2": b2.tolist(),
        "meta": {
            "preset": args.preset, "pop": args.pop, "elite": args.elite,
            "episodes": args.episodes, "gens": args.gens,
            "hidden": args.hidden, "noise": args.noise, "seed": args.seed,
            "probes": args.probes,
            "checkpoint": {"gen": best_gen, "probe": round(float(best_probe), 2)},
            "fitness_history": history,
            "eval": {"seeds": f"0..{args.eval_seeds - 1}",
                     "landed": landed, "perfect": perfect},
        },
    }
    with open(args.out, "w") as f:
        json.dump(artifact, f, allow_nan=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
