"""Seeded terrain generation: midpoint displacement + flat landing pads + stars.

Pure stdlib. All randomness flows through the ``random.Random`` instance the
caller passes in — never module-level random (CONTRACT §5 determinism rule).
"""

import math


class Terrain:
    """One episode's terrain: vertex heights, landing pads, stars and spawns.

    Attributes
    ----------
    points : list[[x, y]]   evenly spaced vertices across [0, world_w], y up
    pads   : list[dict]     {"x0", "x1", "y", "mult"} — one per pad_multipliers
                            entry, in that order (positions shuffled)
    stars  : list[[x, y]]   cosmetic, x in [0, world_w],
                            y in [0.6*world_h, world_h - 10]
    spawns : list[dict]     {"x", "y"} — one per lander (CONTRACT §3),
                            slot + jitter across [spawn_x_min, spawn_x_max]
    """

    def __init__(self, cfg, rng, n_landers=1):
        self.cfg = cfg
        self._dx = cfg.world_w / (cfg.terrain_points - 1)  # vertex spacing
        self._heights = self._midpoint_displacement(rng)
        self.pads = self._place_pads(rng)
        self.stars = self._make_stars(rng)
        self.spawns = self._make_spawns(rng, n_landers)
        self.points = [
            [i * self._dx, self._heights[i]] for i in range(cfg.terrain_points)
        ]

    # ------------------------------------------------------------------ build

    def _midpoint_displacement(self, rng):
        """Midpoint-displacement heightfield with macro-variation (CONTRACT §3).

        On top of the preset's base roughness: endpoint heights from the
        widened range [y_min + 20, y_max - 60], and the world is split into
        3-5 random zones each scaling displacement amplitude by an
        independent U[0.6, 1.5] factor (flat valleys next to violent ridges).

        FIXED rng draw order (determinism depends on it):
          1. left endpoint height        uniform [y_min + 20, y_max - 60]
          2. right endpoint height       uniform [y_min + 20, y_max - 60]
          3. zone count                  randint(3, 5)
          4. zone boundaries             (count - 1) x uniform [0, world_w]
          5. zone amplitude factors      count x uniform [0.6, 1.5]
          6. midpoint displacements      coarse-to-fine, left-to-right
        """
        cfg = self.cfg
        n = cfg.terrain_points  # must be 2^k + 1
        lo, hi = cfg.terrain_y_min + 20.0, cfg.terrain_y_max - 60.0
        h = [0.0] * n
        h[0] = rng.uniform(lo, hi)
        h[-1] = rng.uniform(lo, hi)

        n_zones = rng.randint(3, 5)
        bounds = sorted(rng.uniform(0.0, cfg.world_w) for _ in range(n_zones - 1))
        factors = [rng.uniform(0.6, 1.5) for _ in range(n_zones)]

        def factor_at(x):
            for b, f in zip(bounds, factors):
                if x < b:
                    return f
            return factors[-1]

        amp = cfg.terrain_displacement
        step = n - 1
        while step > 1:
            half = step // 2
            for i in range(half, n, step):
                mid = (h[i - half] + h[i + half]) / 2.0
                h[i] = mid + rng.uniform(-amp, amp) * factor_at(i * self._dx)
            amp *= cfg.terrain_decay
            step = half

        # CONTRACT §3: heights clamped to [terrain_y_min, terrain_y_max]
        return [min(max(y, cfg.terrain_y_min), cfg.terrain_y_max) for y in h]

    def _place_pads(self, rng):
        """Place one pad per pad_multipliers entry by seeded rejection sampling.

        CONTRACT §3 macro-variation: positions uniform over
        [pad_margin, world_w - pad_margin - width]; a candidate is accepted if
        it keeps >= pad_margin gap to every already-accepted pad. Clusters and
        large empty stretches are allowed and desirable.

        FIXED rng draw order (determinism depends on it):
          1. pad order shuffle (as before — which multiplier lands where)
          2. per pad, in shuffled order: uniform x0 candidates until accepted
             (at most _MAX_PAD_ATTEMPTS each)
        If any pad exhausts its attempts (unreachable in practice for 5 pads
        in a 2000-wide world, but never infinite-loop) we fall back to the old
        deterministic slot scheme for ALL pads.
        """
        cfg = self.cfg
        widths = [cfg.pad_widths[m] for m in cfg.pad_multipliers]
        n = len(widths)

        order = list(range(n))  # placement order of pad indices
        rng.shuffle(order)

        pads = self._sample_pads(rng, order, widths)
        if pads is None:  # deterministic fallback — never infinite-loop
            pads = self._slot_pads(rng, order, widths)

        # Flatten covered vertices to the pad y (terrain height at pad center).
        # We flatten one vertex beyond each edge so linear interpolation is
        # exactly flat across the full [x0, x1] span (CONTRACT §3).
        for p in pads:
            p["y"] = self.height((p["x0"] + p["x1"]) / 2.0)
            i0 = max(0, math.floor(p["x0"] / self._dx))
            i1 = min(self.cfg.terrain_points - 1, math.ceil(p["x1"] / self._dx))
            for i in range(i0, i1 + 1):
                self._heights[i] = p["y"]
        return pads

    _MAX_PAD_ATTEMPTS = 200  # per pad; cap so a pathological config can't hang

    def _sample_pads(self, rng, order, widths):
        """Rejection-sample pad positions; None if any pad exhausts attempts."""
        cfg = self.cfg
        pads = [None] * len(widths)
        placed = []  # accepted (x0, x1) intervals
        for idx in order:
            w = widths[idx]
            for _ in range(self._MAX_PAD_ATTEMPTS):
                x0 = rng.uniform(cfg.pad_margin, cfg.world_w - cfg.pad_margin - w)
                if all(x0 >= q1 + cfg.pad_margin or x0 + w <= q0 - cfg.pad_margin
                       for q0, q1 in placed):
                    break
            else:
                return None  # exhausted — caller falls back to slot scheme
            placed.append((x0, x0 + w))
            pads[idx] = {
                "x0": x0,
                "x1": x0 + w,
                "y": 0.0,  # filled by caller, before flattening
                "mult": cfg.pad_multipliers[idx],
            }
        return pads

    def _slot_pads(self, rng, order, widths):
        """Old slot scheme: split leftover slack randomly among the n+1 gaps.

        Guarantees >= pad_margin gaps by construction (fallback path only).
        """
        cfg = self.cfg
        n = len(widths)
        slack = cfg.world_w - 2 * cfg.pad_margin - sum(widths) - (n - 1) * cfg.pad_margin
        cuts = sorted(rng.uniform(0.0, slack) for _ in range(n))
        extras = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, n)] + [slack - cuts[-1]]

        pads = [None] * n
        x = cfg.pad_margin
        for k, idx in enumerate(order):
            x += extras[k]
            w = widths[idx]
            pads[idx] = {
                "x0": x,
                "x1": x + w,
                "y": 0.0,  # filled by caller, before flattening
                "mult": cfg.pad_multipliers[idx],
            }
            x += w + cfg.pad_margin
        return pads

    def _make_stars(self, rng):
        # CONTRACT §3: stars at x in [0, world_w], y in [0.6*world_h, world_h-10]
        cfg = self.cfg
        return [
            [rng.uniform(0.0, cfg.world_w),
             rng.uniform(0.6 * cfg.world_h, cfg.world_h - 10.0)]
            for _ in range(cfg.n_stars)
        ]

    def _make_spawns(self, rng, n):
        """n spawn positions at y = spawn_y, slot + jitter across the spawn span.

        The span [spawn_x_min, spawn_x_max] is split into n equal slots; each
        lander sits at its slot start plus a jitter of at most
        (slot - spawn_min_separation), which guarantees >= spawn_min_separation
        between any two landers (CONTRACT §3). A single lander keeps the v1
        behaviour: uniform over the whole span.
        """
        cfg = self.cfg
        if n == 1:
            xs = [rng.uniform(cfg.spawn_x_min, cfg.spawn_x_max)]
        else:
            slot = (cfg.spawn_x_max - cfg.spawn_x_min) / n
            if slot < cfg.spawn_min_separation:
                raise ValueError(
                    f"cannot place {n} landers >= {cfg.spawn_min_separation} apart "
                    f"in [{cfg.spawn_x_min}, {cfg.spawn_x_max}]"
                )
            jitter = slot - cfg.spawn_min_separation
            xs = [
                cfg.spawn_x_min + k * slot + rng.uniform(0.0, jitter)
                for k in range(n)
            ]
        return [{"x": x, "y": cfg.spawn_y} for x in xs]

    # ----------------------------------------------------------------- lookup

    def height(self, x):
        """Terrain height at x via linear interpolation between vertices."""
        cfg = self.cfg
        x = min(max(x, 0.0), cfg.world_w)
        i = min(int(x / self._dx), cfg.terrain_points - 2)
        t = (x - i * self._dx) / self._dx
        return self._heights[i] + t * (self._heights[i + 1] - self._heights[i])
