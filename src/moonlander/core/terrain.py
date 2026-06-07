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
        """Classic midpoint-displacement heightfield over terrain_points vertices."""
        cfg = self.cfg
        n = cfg.terrain_points  # must be 2^k + 1
        h = [0.0] * n
        h[0] = rng.uniform(cfg.terrain_init_lo, cfg.terrain_init_hi)
        h[-1] = rng.uniform(cfg.terrain_init_lo, cfg.terrain_init_hi)

        amp = cfg.terrain_displacement
        step = n - 1
        while step > 1:
            half = step // 2
            for i in range(half, n, step):
                mid = (h[i - half] + h[i + half]) / 2.0
                h[i] = mid + rng.uniform(-amp, amp)
            amp *= cfg.terrain_decay
            step = half

        # CONTRACT §3: heights clamped to [terrain_y_min, terrain_y_max]
        return [min(max(y, cfg.terrain_y_min), cfg.terrain_y_max) for y in h]

    def _place_pads(self, rng):
        """Place one pad per pad_multipliers entry at random non-overlapping x.

        Construction guarantees >= pad_margin gaps (pads to each other and to
        world edges) without rejection sampling: shuffle pad order, then split
        the leftover horizontal slack randomly among the n+1 gaps.
        """
        cfg = self.cfg
        widths = [cfg.pad_widths[m] for m in cfg.pad_multipliers]
        n = len(widths)

        order = list(range(n))  # left-to-right placement order of pad indices
        rng.shuffle(order)

        slack = cfg.world_w - 2 * cfg.pad_margin - sum(widths) - (n - 1) * cfg.pad_margin
        cuts = sorted(rng.uniform(0.0, slack) for _ in range(n))
        extras = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, n)] + [slack - cuts[-1]]

        pads = [None] * n
        x = cfg.pad_margin
        for k, idx in enumerate(order):
            x += extras[k]
            w = widths[order[k]]
            pads[idx] = {
                "x0": x,
                "x1": x + w,
                "y": 0.0,  # filled below, before flattening
                "mult": cfg.pad_multipliers[idx],
            }
            x += w + cfg.pad_margin

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
