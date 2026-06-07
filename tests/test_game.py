"""Game state machine: determinism, landing rules, crash, terminal no-op, schema."""

import json

from moonlander.config import Config
from moonlander.core.game import Game

CFG = Config()

# A mixed action script: rotate, thrust, both, idle.
SCRIPT = [(1, False)] * 10 + [(0, True)] * 40 + [(-1, True)] * 20 + [(0, False)] * 50


def run_script(seed):
    g = Game(mode="classic")
    terrain = g.reset(seed=seed)
    frames = [g.step(rotate=r, thrust=t) for r, t in SCRIPT]
    return terrain, frames


def test_determinism_same_seed_same_actions_identical_frames():
    terrain1, frames1 = run_script(seed=123)
    terrain2, frames2 = run_script(seed=123)
    assert terrain1 == terrain2
    assert frames1 == frames2  # byte-identical at every step, incl. step N


def test_frame_schema_matches_contract_v2():
    g = Game(mode="classic")
    g.reset(seed=9)
    frame = json.loads(g.step(0, False))

    assert set(frame) == {"t", "status", "active", "landers", "hud"}
    assert frame["status"] in ("flying", "done")
    assert frame["active"] == 1
    assert set(frame["hud"]) == {"altitude", "hspeed", "vspeed"}

    landers = frame["landers"]
    assert len(landers) == 1
    ld = landers[0]
    assert set(ld) == {"i", "x", "y", "vx", "vy", "angle", "ang_vel", "thrust",
                       "side", "fuel", "score", "status", "outcome", "obs"}
    assert ld["i"] == 0
    assert ld["status"] == "flying"
    assert ld["outcome"] is None
    assert len(ld["obs"]) == 14


def test_gentle_descent_on_pad_is_perfect_landing():
    g = Game(mode="classic")
    g.reset(seed=11)
    pad = g.terrain.pads[0]

    # Place the lander just above the pad center, drifting gently down.
    L = g.landers[0].state
    L.x = (pad["x0"] + pad["x1"]) / 2.0
    L.y = pad["y"] + CFG.lander_bottom + 2.0
    L.vx, L.vy = 0.0, -5.0
    L.angle, L.ang_vel = 0.0, 0.0

    frame = None
    for _ in range(200):
        frame = json.loads(g.step(0, False))
        if frame["status"] != "flying":
            break

    assert frame["status"] == "done"
    ld = frame["landers"][0]
    assert ld["status"] == "landed"
    assert ld["outcome"]["kind"] == "perfect"
    assert ld["outcome"]["mult"] == pad["mult"]
    assert ld["outcome"]["points"] == CFG.score_perfect * pad["mult"]
    assert ld["score"] == CFG.score_perfect * pad["mult"]


def test_fast_uncontrolled_drop_crashes():
    g = Game(mode="classic")
    g.reset(seed=2)
    frame = None
    for _ in range(5000):  # free fall from spawn always exceeds vy_hard
        frame = json.loads(g.step(0, False))
        if frame["status"] != "flying":
            break
    assert frame["status"] == "done"
    assert frame["active"] == 0
    ld = frame["landers"][0]
    assert ld["status"] == "crashed"
    assert ld["outcome"]["kind"] == "crash"
    assert ld["outcome"]["points"] == 0
    assert ld["score"] == 0


def test_step_after_terminal_is_noop():
    g = Game(mode="classic")
    g.reset(seed=2)
    last = None
    for _ in range(5000):
        last = g.step(0, False)
        if json.loads(last)["status"] != "flying":
            break
    assert json.loads(last)["status"] == "done"

    # Stepping again — even with active inputs — changes nothing.
    again = g.step(rotate=1, thrust=True)
    assert again == last
    assert g.frame_json() == last
