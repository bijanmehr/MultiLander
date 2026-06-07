"""Built-in attract-mode autopilot (CONTRACT §2 ``step_auto``).

Pure stdlib — imports nothing beyond ``math``. A simple deterministic
proportional controller for CLASSIC mode: steer toward the nearest pad's
center, hold a clearance-scaled descent rate, and thrust early enough that
the available net upward acceleration can kill the fall before touchdown
(stopping-distance anticipation). Crashes are acceptable attract-mode drama.
"""

import math


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def act(x, y, vx, vy, angle, clearance, pads, gravity=25.0, thrust_accel=60.0):
    """One control decision for classic mode: ``(rotate, thrust)``.

    Parameters are raw world-frame state (CONTRACT §1) plus the terrain's pad
    list (§3 dicts) and the two physics constants the braking math needs
    (callers pass ``Config.gravity`` / ``Config.thrust_accel``).
    Deterministic: same inputs → same outputs.
    """
    # Target: nearest pad by euclidean distance (same rule as the obs vector, §6).
    pad = min(
        pads, key=lambda p: math.hypot((p["x0"] + p["x1"]) / 2.0 - x, p["y"] - y)
    )
    cx = (pad["x0"] + pad["x1"]) / 2.0

    # --- Horizontal: P-control on position → desired vx → desired tilt.
    if clearance < 80.0:
        vx_des = _clamp((cx - x) * 0.35, -6.0, 6.0)  # final approach: creep
    else:
        vx_des = _clamp((cx - x) * 0.25, -18.0, 18.0)
    a_des = _clamp(-(vx_des - vx) * 0.04, -0.45, 0.45)
    if clearance < 45.0:
        a_des = 0.0  # go upright for touchdown (|angle| <= 0.15 to count)
    da = a_des - angle
    rotate = 1 if da > 0.03 else (-1 if da < -0.03 else 0)

    # --- Vertical: clearance-scaled descent profile. The floor (11 u/s,
    # safely inside the perfect |vy| <= 18 gate) keeps the final approach
    # brisk — a slow creep hovers away the fuel budget and ends in a
    # dead-stick drop.
    vy_des = -_clamp(clearance * 0.18, 11.0, 28.0)
    # v2 retune (2000-wide world): when low but not yet centered over the pad,
    # slow — then stop — the descent while the horizontal creep finishes;
    # descending at full rate here touches down just short of the pad edge.
    if clearance < 80.0 and abs(cx - x) > 8.0:
        vy_des = -3.0 if clearance > 40.0 else 0.0
    thrust = vy < vy_des

    # ... plus stopping-distance anticipation: when falling faster than the
    # touchdown target, the altitude needed to brake from |vy| down to v_land
    # under the net upward accel (thrust_accel*cos(angle) - gravity,
    # ~35 u/s^2 upright) is
    #   d = (vy^2 - v_land^2) / (2 * a_net).
    # Start braking before clearance shrinks below that (plus a margin for
    # the tilt wobble of the horizontal controller). Only engages while
    # vy < -v_land — otherwise the margin term turns it into a hover trap
    # just above the ground.
    v_land = 10.0
    if vy < -v_land:
        a_net = thrust_accel * math.cos(angle) - gravity
        if a_net > 1.0:
            stop_dist = (vy * vy - v_land * v_land) / (2.0 * a_net)
            if stop_dist + 16.0 >= clearance:
                thrust = True

    return rotate, thrust
