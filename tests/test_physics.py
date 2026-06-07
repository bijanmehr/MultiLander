"""Physics laws: gravity, thrust, fuel exhaustion, classic rotation."""

import math

from moonlander.config import Config
from moonlander.core import physics
from moonlander.core.physics import LanderState

CFG = Config()


def make_state(**kw):
    base = dict(x=500.0, y=500.0, vx=0.0, vy=0.0, angle=0.0, ang_vel=0.0,
                fuel=CFG.fuel_init)
    base.update(kw)
    return LanderState(**base)


def test_free_fall_matches_gravity_accumulation():
    s = make_state()
    n = 60
    for _ in range(n):
        physics.step(s, CFG, "classic", rotate=0, thrust=False)
    assert math.isclose(s.vy, -CFG.gravity * n * CFG.dt, rel_tol=1e-9)
    assert s.vx == 0.0
    assert s.y < 500.0


def test_upright_thrust_reduces_descent_rate():
    free = make_state()
    powered = make_state()
    for _ in range(30):
        physics.step(free, CFG, "classic", thrust=False)
        physics.step(powered, CFG, "classic", thrust=True)
    assert powered.vy > free.vy  # thrust fights gravity (and here beats it)
    assert powered.vy - free.vy > 0.5 * CFG.thrust_accel * 30 * CFG.dt


def test_fuel_depletes_and_engine_dies_at_zero():
    # Enough fuel for ~2.5 ticks of main burn.
    s = make_state(fuel=CFG.fuel_burn_main * CFG.dt * 2.5)
    fired = []
    for _ in range(3):
        main_on, _ = physics.step(s, CFG, "classic", thrust=True)
        fired.append(main_on)
    assert fired == [True, True, True]  # fuel > 0 at the start of each tick
    assert s.fuel == 0.0  # clamped at zero

    # Tank empty: commanded thrust produces NO acceleration beyond gravity.
    vy_before = s.vy
    main_on, _ = physics.step(s, CFG, "classic", thrust=True)
    assert main_on is False
    assert math.isclose(s.vy - vy_before, -CFG.gravity * CFG.dt, rel_tol=1e-9)
    assert s.fuel == 0.0


def test_gym_ang_vel_clamped_to_ang_vel_max():
    # CONTRACT §5: gym-mode ang_vel saturates at ±ang_vel_max.
    s = make_state()
    n = int(2.0 * CFG.ang_vel_max / (CFG.side_torque * CFG.dt)) + 10
    for _ in range(n):  # hold the left thruster way past saturation
        physics.step(s, CFG, "gym", engine=1)
    assert s.ang_vel == CFG.ang_vel_max

    s = make_state()
    for _ in range(n):
        physics.step(s, CFG, "gym", engine=3)
    assert s.ang_vel == -CFG.ang_vel_max


def test_classic_rotation_changes_angle_at_rot_rate():
    s = make_state()
    physics.step(s, CFG, "classic", rotate=1)
    assert math.isclose(s.ang_vel, CFG.rot_rate, rel_tol=1e-12)
    assert math.isclose(s.angle, CFG.rot_rate * CFG.dt, rel_tol=1e-12)

    # Releasing the key stops rotation instantly (direct rate control).
    physics.step(s, CFG, "classic", rotate=0)
    assert s.ang_vel == 0.0

    physics.step(s, CFG, "classic", rotate=-1)
    assert math.isclose(s.ang_vel, -CFG.rot_rate, rel_tol=1e-12)
