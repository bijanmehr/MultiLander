"use strict";
/*
 * effects.js — the cosmetic animation layer (CONTRACT §10). JS-side only.
 *
 * Boundary rule (§10): game STATE comes exclusively from Python; everything
 * here is cosmetic — camera, crash debris, terrain reveal, text blink — and
 * never influences anything sent back to Python. §10 exempts this layer from
 * determinism, so Math.random() / performance.now() are fine here.
 *
 * Exposes a script-global `Effects`:
 *   tick(opts)            — advance one fixed 1/60 s step (camera, debris, reveal)
 *                           opts: { frame, terrain, zoomLander, cameraFrozen }
 *                           zoomLander: the SINGLE still-flying lander when
 *                           active == 1, else null (§10 zoom gate)
 *   episodeReset(terrain) — clear debris, snap camera to full view, start reveal
 *   explode(lander)       — ADD crash debris for one lander + queue its flash
 *                           (per-lander, §10: simultaneous explosions stack)
 *   cameraView()          — { s, cx, cy } world camera for the renderer
 *   revealActive()        — terrain draw-in still running?
 *   revealView()          — { points, starAlpha, padsOn } partial scene, or null
 *   debrisView()          — { pieces, flashes }; flashes consumed (one render frame)
 *   blink()               — shared ~1.1 Hz text blink, on ~60% of the cycle
 */

const Effects = (() => {
  const W = 2000;            // world size == logical canvas size (§1/§9)
  const H = 750;
  const DT = 1 / 60;         // fixed cosmetic timestep, matches physics dt
  const GRAVITY = 25;        // §10: debris integrates with world gravity
  const LANDER_BOTTOM = 10;  // feet below center (Config.lander_bottom)

  // Camera (§10 arcade zoom)
  const CAM_RATE = 3;        // exponential lerp rate, s^-1
  const ZOOM_SCALE = 3.0;    // §10: target scale when one low lander remains
  const ZOOM_ALTITUDE = 150; // §10: altitude threshold engaging the zoom

  // Terrain reveal (§10 draw-in: full 2000-wide polyline in the same ~0.7 s)
  const TRACE_T = 0.7;       // polyline trace duration, seconds
  const PAD_BLINK_T = 0.4;   // pads + labels blink twice over this window

  // Crash debris (§10 explosion)
  const DEBRIS_COUNT = 8;    // pieces per exploding lander
  const DEBRIS_LIFE = 2.0;   // fade-out, seconds
  const KICK_MIN = 40;       // outward kick speed range, units/s
  const KICK_MAX = 110;
  const SPIN_MAX = 5;        // |spin| max, rad/s

  // Text blink (§10): ~1.1 Hz, on 60% of the cycle.
  const BLINK_HZ = 1.1;
  const BLINK_DUTY = 0.6;

  // ---------------------------------------------------------------- helpers

  // Terrain height under x by linear interpolation between the evenly spaced
  // vertices (§3). Cosmetic use only (camera framing, debris settling) — the
  // authoritative ground test lives in Python (§5).
  function terrainHeightAt(points, x) {
    const n = points.length;
    const spacing = W / (n - 1);
    const cx = Math.min(Math.max(x, 0), W);
    const i = Math.min(Math.floor(cx / spacing), n - 2);
    const [x0, y0] = points[i];
    const [x1, y1] = points[i + 1];
    const f = x1 > x0 ? (cx - x0) / (x1 - x0) : 0;
    return y0 + (y1 - y0) * f;
  }

  // ----------------------------------------------------------------- camera

  // Scale s and world-space center; default frames the full world exactly
  // like the pre-camera renderer (s=1, center (1000, 375)).
  const cam = { s: 1, cx: W / 2, cy: H / 2 };

  // Clamp a camera's view rectangle inside world bounds (§10).
  function clampCenter(c) {
    const hw = W / (2 * c.s);
    const hh = H / (2 * c.s);
    c.cx = Math.min(Math.max(c.cx, hw), W - hw);
    c.cy = Math.min(Math.max(c.cy, hh), H - hh);
  }

  function cameraSnapFull() {
    cam.s = 1;
    cam.cx = W / 2;
    cam.cy = H / 2;
  }

  function cameraTick(opts) {
    if (opts.cameraFrozen) return; // §10: freeze during ENDED / attract hold

    const target = { s: 1, cx: W / 2, cy: H / 2 }; // default: full world
    const lander = opts.zoomLander; // non-null only when active == 1 (§10)
    if (lander && opts.terrain) {
      // §10: zoom engages when that lander is < 150 above the ground.
      // Altitude is interpolated here cosmetically (hud.altitude only covers
      // lander 0); camera framing is exempt from the no-simulation rule.
      const gy = terrainHeightAt(opts.terrain.points, lander.x);
      if (lander.y - LANDER_BOTTOM - gy < ZOOM_ALTITUDE) {
        // Zoom centered between the lander and the ground beneath it.
        target.s = ZOOM_SCALE;
        target.cx = lander.x;
        target.cy = (lander.y + gy) / 2;
      }
    }
    clampCenter(target);

    // Exponential lerp at ~3 s^-1, frame-rate independent at fixed dt.
    const k = 1 - Math.exp(-CAM_RATE * DT);
    cam.s += (target.s - cam.s) * k;
    cam.cx += (target.cx - cam.cx) * k;
    cam.cy += (target.cy - cam.cy) * k;
    clampCenter(cam); // keep mid-lerp views inside the world too
  }

  const cameraView = () => ({ s: cam.s, cx: cam.cx, cy: cam.cy });

  // ----------------------------------------------------------------- debris

  // Segment pool: every polyline segment of the §9 lander shape, body frame
  // y-up. Crash pieces are drawn from these so the wreck reads as the lander.
  const LANDER_SEGMENTS = [
    [[-6, 2], [-6, 9]], [[-6, 9], [-3, 13]], [[-3, 13], [3, 13]], // cabin
    [[3, 13], [6, 9]], [[6, 9], [6, 2]], [[6, 2], [-6, 2]],
    [[-8, 2], [8, 2]],                                            // base
    [[-4, 2], [-2, -3]], [[-2, -3], [2, -3]], [[2, -3], [4, 2]],  // nozzle
    [[-6, 2], [-12, -10]], [[6, 2], [12, -10]],                   // legs
    [[-15, -10], [-9, -10]], [[9, -10], [15, -10]],               // feet
  ];

  let debris = [];     // live pieces (across ALL exploded landers, §10)
  let flashes = [];    // pending one-frame white flashes, one per explosion

  function debrisClear() {
    debris = [];
    flashes = [];
  }

  // Spawn ~8 pieces at one lander's crash pose (§10): each piece is a lander
  // segment that inherits the final velocity plus a random outward kick and
  // spin, then falls under gravity and settles on the terrain. Pieces are
  // APPENDED — per-lander explosions stack, so a lander-lander collision
  // (both crash in the same tick) yields two simultaneous explosions.
  function explode(lander) {
    const pool = LANDER_SEGMENTS.slice();
    for (let i = pool.length - 1; i > 0; i--) { // Fisher-Yates shuffle
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }

    const cosA = Math.cos(lander.angle);
    const sinA = Math.sin(lander.angle);

    for (const [a, b] of pool.slice(0, DEBRIS_COUNT)) {
      // Segment midpoint in body frame, then rotated into world (CCW, §1).
      const mx = (a[0] + b[0]) / 2;
      const my = (a[1] + b[1]) / 2;
      const ox = mx * cosA - my * sinA;
      const oy = mx * sinA + my * cosA;
      const d = Math.hypot(ox, oy) || 1;
      const kick = KICK_MIN + Math.random() * (KICK_MAX - KICK_MIN);
      debris.push({
        x: lander.x + ox,
        y: lander.y + oy,
        vx: lander.vx + (ox / d) * kick,
        vy: lander.vy + (oy / d) * kick,
        angle: lander.angle,
        spin: (Math.random() * 2 - 1) * SPIN_MAX,
        age: 0,
        settled: false,
        ends: [[a[0] - mx, a[1] - my], [b[0] - mx, b[1] - my]],
      });
    }

    flashes.push({ x: lander.x, y: lander.y }); // one-frame flash per crash
  }

  function debrisTick(terrain) {
    if (debris.length === 0) return;
    for (const p of debris) {
      p.age += DT;
      if (!p.settled) {
        // Semi-implicit Euler, same scheme as §5, gravity 25 (§10).
        p.vy -= GRAVITY * DT;
        p.x += p.vx * DT;
        p.y += p.vy * DT;
        p.angle += p.spin * DT;
        if (terrain) {
          const gy = terrainHeightAt(terrain.points, p.x);
          if (p.y <= gy) { // §10: pieces stop at terrain height
            p.y = gy;
            p.vx = 0;
            p.vy = 0;
            p.spin = 0;
            p.settled = true;
          }
        }
      }
    }
    debris = debris.filter((p) => p.age < DEBRIS_LIFE); // faded out
  }

  // Flashes are consumed here so each paints for exactly one rendered frame.
  function debrisView() {
    const f = flashes;
    flashes = [];
    return {
      pieces: debris.map((p) => ({
        x: p.x,
        y: p.y,
        angle: p.angle,
        alpha: Math.max(0, 1 - p.age / DEBRIS_LIFE), // ~2 s fade (§10)
        ends: p.ends,
      })),
      flashes: f,
    };
  }

  // ----------------------------------------------------------------- reveal

  let reveal = null; // {points, cum, total, t} while the draw-in runs

  // Start the terrain draw-in (§10): precompute cumulative polyline length
  // so the trace progresses by length, not by vertex count.
  function revealStart(terrain) {
    const points = terrain.points;
    const cum = [0];
    for (let i = 1; i < points.length; i++) {
      cum.push(cum[i - 1] + Math.hypot(points[i][0] - points[i - 1][0],
                                       points[i][1] - points[i - 1][1]));
    }
    reveal = { points, cum, total: cum[cum.length - 1], t: 0 };
  }

  function revealTick() {
    if (!reveal) return;
    reveal.t += DT;
    if (reveal.t >= TRACE_T + PAD_BLINK_T) reveal = null; // done; landers pop in
  }

  const revealActive = () => reveal !== null;

  function revealView() {
    if (!reveal) return null;

    // Partial polyline: every vertex within the traced length plus one
    // interpolated point exactly at the trace front.
    const frac = Math.min(reveal.t / TRACE_T, 1);
    const len = frac * reveal.total;
    const pts = reveal.points;
    const out = [pts[0]];
    for (let i = 1; i < pts.length; i++) {
      if (reveal.cum[i] <= len) {
        out.push(pts[i]);
      } else {
        const seg = reveal.cum[i] - reveal.cum[i - 1];
        const f = seg > 0 ? (len - reveal.cum[i - 1]) / seg : 0;
        out.push([pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * f,
                  pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * f]);
        break;
      }
    }

    // After the trace, pads + labels blink twice over ~0.4 s (§10):
    // two 0.2 s cycles, visible for the first half of each.
    let padsOn = false;
    if (reveal.t > TRACE_T) {
      padsOn = ((reveal.t - TRACE_T) % (PAD_BLINK_T / 2)) < PAD_BLINK_T / 4;
    }

    return { points: out, starAlpha: frac, padsOn };
  }

  // ------------------------------------------------------------------ blink

  // Shared blink for "press key" prompts (§10): ~1.1 Hz, on ~60% duty.
  function blink() {
    return ((performance.now() / 1000) * BLINK_HZ) % 1 < BLINK_DUTY;
  }

  // ------------------------------------------------------------ entry points

  // One fixed 1/60 s cosmetic step, called from the app's accumulator loop.
  function tick(opts) {
    revealTick();
    debrisTick(opts.terrain);
    cameraTick(opts);
  }

  // Episode boundary (every reset): old scene is gone, so hard-cut the
  // camera back to the full view, drop stale debris and start the draw-in.
  function episodeReset(terrain) {
    debrisClear();
    cameraSnapFull();
    revealStart(terrain);
  }

  return {
    tick,
    episodeReset,
    explode,
    cameraView,
    revealActive,
    revealView,
    debrisView,
    blink,
  };
})();
