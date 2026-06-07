"""All tunable constants for the Moon Lander game.

Single source of truth: physics, terrain, scoring and the Gymnasium env all
read from here. Values are gameplay-tuned, not physically exact. Pure stdlib —
this module ships to the browser via Pyodide.
"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Config:
    # World: x right, y UP, origin at bottom-left. Abstract units.
    world_w: float = 2000.0
    world_h: float = 750.0
    dt: float = 1.0 / 60.0  # physics timestep, seconds

    # Dynamics
    gravity: float = 25.0  # downward acceleration, units/s^2
    thrust_accel: float = 60.0  # main engine accel along body axis
    rot_rate: float = 1.6  # classic mode: angular speed while rotate held, rad/s
    side_accel: float = 15.0  # gym mode: lateral accel from a side engine
    side_torque: float = 3.0  # gym mode: angular accel from a side engine, rad/s^2
    ang_vel_max: float = 4.0  # gym mode: RCS saturation clamp, rad/s (keeps obs bounded)

    # Fuel
    fuel_init: float = 1000.0
    fuel_burn_main: float = 60.0  # units/s while main engine fires (~16.7 s total burn)
    fuel_burn_side: float = 6.0  # units/s while a side engine fires (gym mode)

    # Lander geometry (body frame, y up, center at origin)
    lander_half_w: float = 12.0  # foot x offset from center
    lander_bottom: float = 10.0  # feet sit this far below center
    lander_collision_dist: float = 26.0  # center-to-center contact distance (multi-lander)

    # Spawn
    spawn_x_min: float = 150.0
    spawn_x_max: float = 1850.0
    spawn_y: float = 690.0
    spawn_vx_max: float = 25.0  # |vx| uniform in [-max, +max]
    spawn_min_separation: float = 150.0  # min x-gap between landers at spawn

    # Sensor models (observation filtering, §6)
    radar_range: float = 600.0  # "radar" obs_mode: pads beyond this are invisible

    # Terrain (midpoint displacement)
    terrain_points: int = 257  # 2^8 + 1 vertices across [0, world_w]
    terrain_y_min: float = 60.0
    terrain_y_max: float = 480.0
    terrain_displacement: float = 210.0  # initial midpoint displacement amplitude
    terrain_decay: float = 0.62  # amplitude multiplier per subdivision level
    n_stars: int = 100

    # Landing pads: one pad per multiplier listed (shuffled positions each episode)
    pad_multipliers: tuple = (2, 2, 3, 3, 5)
    pad_widths: dict = field(default_factory=lambda: {2: 110.0, 3: 75.0, 5: 45.0})
    pad_margin: float = 60.0  # min horizontal gap between pads / world edges

    # Landing thresholds (units/s and rad)
    vx_perfect: float = 12.0
    vy_perfect: float = 18.0
    vx_hard: float = 25.0
    vy_hard: float = 35.0
    angle_max: float = 0.15  # max |angle| to count as upright

    # Scoring (points = base * pad multiplier)
    score_perfect: int = 50
    score_hard: int = 15

    # Episode
    max_steps: int = 2400  # Gymnasium truncation only (40 s); Game itself has no limit

    @classmethod
    def preset(cls, name):
        """A Config for difficulty preset ``name`` (CONTRACT §3 table).

        ``"trainee" | "cadet" | "commander"`` — class defaults == cadet.
        Difficulty is terrain ruggedness, pad size, fuel budget and spawn
        drift, never different physics. Unknown names raise ValueError.
        """
        if name not in PRESETS:
            raise ValueError(
                f"unknown preset {name!r} — options: {', '.join(PRESETS)}"
            )
        overrides = dict(PRESETS[name])
        overrides["pad_widths"] = dict(overrides["pad_widths"])  # never share
        return replace(cls(), **overrides)


# Difficulty presets (CONTRACT §3, EXACT values; Config class defaults == CADET).
PRESETS = {
    "trainee": dict(
        terrain_displacement=140.0, terrain_decay=0.52, terrain_y_max=380.0,
        pad_widths={2: 130.0, 3: 95.0, 5: 60.0},
        fuel_init=1200.0, spawn_vx_max=15.0,
    ),
    "cadet": dict(
        terrain_displacement=210.0, terrain_decay=0.62, terrain_y_max=480.0,
        pad_widths={2: 110.0, 3: 75.0, 5: 45.0},
        fuel_init=1000.0, spawn_vx_max=25.0,
    ),
    "commander": dict(
        terrain_displacement=260.0, terrain_decay=0.68, terrain_y_max=560.0,
        pad_widths={2: 90.0, 3: 60.0, 5: 36.0},
        fuel_init=850.0, spawn_vx_max=40.0,
    ),
}
