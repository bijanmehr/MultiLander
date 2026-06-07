"""Lander state and one semi-implicit Euler integration step (CONTRACT §5).

Pure stdlib. Both control modes:
  classic: direct angular-rate control (rotate in {-1, 0, +1}) + main thrust
  gym:     engine in {0: noop, 1: left thruster, 2: main, 3: right thruster};
           side thrusters accelerate along the body x axis and apply torque
"""

import math
from dataclasses import dataclass


@dataclass
class LanderState:
    x: float
    y: float
    vx: float
    vy: float
    angle: float  # rad, 0 = upright, positive = CCW
    ang_vel: float  # rad/s
    fuel: float


def step(state, cfg, mode, rotate=0, thrust=False, engine=0):
    """Advance ``state`` by one physics tick (cfg.dt), in place.

    Returns ``(main_on, side)``: whether the main engine actually fired this
    tick, and which side engine fired ("none" | "left" | "right"). Engines are
    dead when fuel == 0.
    """
    dt = cfg.dt
    has_fuel = state.fuel > 0.0

    # Resolve commands per mode.
    if mode == "classic":
        main_cmd = bool(thrust)
        side = "none"
    else:  # gym
        main_cmd = engine == 2
        side = "left" if engine == 1 else ("right" if engine == 3 else "none")

    main_on = main_cmd and has_fuel
    side_on = side != "none" and has_fuel
    if not side_on:
        side = "none"

    # Rotation.
    if mode == "classic":
        state.ang_vel = rotate * cfg.rot_rate  # arcade-style direct rate
    else:
        if side == "left":  # left thruster torques CCW (+side_torque)
            state.ang_vel += cfg.side_torque * dt
        elif side == "right":
            state.ang_vel -= cfg.side_torque * dt
        # CONTRACT §5: RCS saturation — keeps the obs ang_vel term bounded.
        state.ang_vel = max(-cfg.ang_vel_max, min(cfg.ang_vel_max, state.ang_vel))
    state.angle += state.ang_vel * dt
    # CONTRACT §5: wrap angle to (-pi, pi] after integration. The sin/cos
    # observations are modular — without the wrap, a full revolution makes two
    # byte-identical observations carry opposite landing grades (non-Markov
    # reward). The fmod wrap is exact at the branch point and leaves in-range
    # angles bit-identical.
    a = state.angle
    if a > math.pi or a <= -math.pi:
        a = math.fmod(a, 2.0 * math.pi)  # exact; result in (-2pi, 2pi)
        if a > math.pi:
            a -= 2.0 * math.pi
        elif a <= -math.pi:
            a += 2.0 * math.pi
        state.angle = a

    # Acceleration: gravity + engines (evaluated at the updated angle).
    ax, ay = 0.0, -cfg.gravity
    if main_on:  # thrust direction (-sin a, cos a): straight up when upright
        ax += cfg.thrust_accel * -math.sin(state.angle)
        ay += cfg.thrust_accel * math.cos(state.angle)
    if side_on:  # body +x axis is (cos a, sin a); left pushes toward +x
        sgn = 1.0 if side == "left" else -1.0
        ax += sgn * cfg.side_accel * math.cos(state.angle)
        ay += sgn * cfg.side_accel * math.sin(state.angle)

    # Semi-implicit Euler: velocity first, then position.
    state.vx += ax * dt
    state.vy += ay * dt
    state.x += state.vx * dt
    state.y += state.vy * dt

    # Fuel burn, clamped at zero.
    burn = 0.0
    if main_on:
        burn += cfg.fuel_burn_main
    if side_on:
        burn += cfg.fuel_burn_side
    state.fuel = max(0.0, state.fuel - burn * dt)

    return main_on, side
