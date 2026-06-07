"use strict";
/*
 * renderer.js — pure painting per CONTRACT §9/§10. No game logic, no Pyodide.
 *
 * Exposes a script-global `Renderer = { init(canvas), render(view) }` where view is:
 *   {
 *     state:         "LOADING" | "TITLE" | "REVEAL" | "FLYING" | "ENDED" | "ERROR",
 *     stage:         string   (loading progress line),
 *     message:       string   (error text),
 *     terrain:       parsed terrain JSON (§3) or null,
 *     frame:         parsed frame JSON (§4) or null — landers[] schema,
 *     sessionScore:  number   (session total, accumulated in app.js),
 *     high:          number   (best single-episode points, localStorage §8),
 *     seed:          number|null (?seed=N badge for human episodes, §8),
 *     overlay:       boolean  (agent-view obs panel toggle),
 *     attract:       boolean  (attract-mode episode -> title text over REVEAL),
 *     camera:        {s, cx, cy} world camera from Effects (§10),
 *     reveal:        {points, starAlpha, padsOn} partial scene or null (§10),
 *     debris:        {pieces, flashes} crash debris from Effects (§10),
 *     hiddenLanders: Set of lander indices destroyed (explosion spawned, §10),
 *     showBanner:    boolean  (ENDED outcome banner, delayed ~0.8 s on crash),
 *   }
 *
 * World-space drawing (terrain, pads, labels, stars, landers, debris) goes
 * through the camera transform; HUD, banners, title and overlay stay in
 * screen space (§9) with fonts ~1.6x v1 so they read at the wider canvas's
 * CSS scale. Blink timing comes from Effects.blink().
 *
 * Target look: 1979 Atari Lunar Lander vector monitor — pure black, thin
 * glowing white strokes, monospace uppercase text.
 */

const Renderer = (() => {
  // Logical canvas size equals world size (CONTRACT §1/§9).
  const W = 2000;
  const H = 750;

  let ctx = null;

  // Camera for the current render pass (§10). Defaults frame the full world,
  // where the transform reduces exactly to the classic wx=x, wy=H-y.
  let cam = { s: 1, cx: W / 2, cy: H / 2 };

  // ---------------------------------------------------------------- helpers

  // World -> canvas through the camera (§1 y-flip + §10 scale/center):
  // at s=1, cx=1000, cy=375 these are identical to x and H - y.
  const wx = (x) => (x - cam.cx) * cam.s + W / 2;
  const wy = (y) => (cam.cy - y) * cam.s + H / 2;

  const font = (px) =>
    `${px}px ui-monospace, Menlo, Consolas, "Courier New", monospace`;

  function glow(on) {
    ctx.shadowColor = "#fff";
    ctx.shadowBlur = on ? 6 : 0; // §9: glow via shadowBlur 6 white
  }

  function line(x0, y0, x1, y1, width) {
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }

  // Polyline through canvas-space points [[x,y],...]; optionally closed.
  function polyline(pts, closed) {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    if (closed) ctx.closePath();
    ctx.stroke();
  }

  function centeredText(text, y, size) {
    ctx.font = font(size);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, W / 2, y);
  }

  // Wrap an arbitrary message into lines of at most `max` characters.
  function wrapText(text, max) {
    const words = String(text).split(/\s+/);
    const lines = [];
    let cur = "";
    for (const w of words) {
      if (cur && cur.length + 1 + w.length > max) {
        lines.push(cur);
        cur = w;
      } else {
        cur = cur ? cur + " " + w : w;
      }
    }
    if (cur) lines.push(cur);
    return lines;
  }

  // ------------------------------------------------------------------ scene

  function clear() {
    glow(false);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, W, H);
  }

  function drawStars(stars) {
    glow(true);
    ctx.fillStyle = "#fff";
    // 2 world units per dot: the canvas is twice as wide as v1 at the same
    // CSS width, so this preserves the v1 apparent star size on screen.
    const r = 2 * cam.s;
    for (const [x, y] of stars) {
      ctx.fillRect(wx(x) - r / 2, wy(y) - r / 2, r, r); // glowing dot (§9)
    }
  }

  function drawTerrain(points) {
    glow(true);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(wx(points[0][0]), wy(points[0][1]));
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(wx(points[i][0]), wy(points[i][1]));
    }
    ctx.stroke();
  }

  function drawPads(pads) {
    glow(true);
    ctx.strokeStyle = "#fff";
    ctx.fillStyle = "#fff";
    for (const pad of pads) {
      const y = wy(pad.y);
      // Brighter double stroke on top of the terrain line (§9); the 3-unit
      // gap between the strokes is world-space, so it scales with the camera.
      line(wx(pad.x0), y, wx(pad.x1), y, 2.5);
      line(wx(pad.x0), y + 3 * cam.s, wx(pad.x1), y + 3 * cam.s, 1.5);
      // Multiplier label, centered under the pad at world y = pad.y - 25.
      // World-space text, ~1.6x the v1 size so it reads at the CSS scale.
      ctx.font = font(24 * cam.s);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`${pad.mult}X`, wx((pad.x0 + pad.x1) / 2), wy(pad.y - 25));
    }
  }

  function drawLander(lander) {
    ctx.save();
    ctx.translate(wx(lander.x), wy(lander.y));
    ctx.scale(cam.s, cam.s);   // body geometry is world-sized -> camera scales it
    ctx.rotate(-lander.angle); // §1: world CCW appears as rotate(-angle) on a y-down canvas
    ctx.scale(1, -1);          // body-frame coordinates below are y-up (§9)

    glow(true);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5 / cam.s; // constant ~1.5px beam on screen (§9 look)

    // Body-frame polylines, exactly per §9.
    polyline([[-6, 2], [-6, 9], [-3, 13], [3, 13], [6, 9], [6, 2]], true); // cabin
    polyline([[-8, 2], [8, 2]]);                                           // base
    polyline([[-4, 2], [-2, -3], [2, -3], [4, 2]]);                        // nozzle
    polyline([[-6, 2], [-12, -10]]);                                       // left leg
    polyline([[6, 2], [12, -10]]);                                         // right leg
    polyline([[-15, -10], [-9, -10]]);                                     // left foot
    polyline([[9, -10], [15, -10]]);                                       // right foot

    // Flickering thrust flame, per lander (cosmetic randomness only — never physics).
    if (lander.thrust) {
      const flicker = Math.random();
      polyline([[-3, -4], [0, -12 - 6 * flicker], [3, -4]]);
    }

    ctx.restore();
  }

  // Crash debris (§10): lander-segment pieces, world space through the camera.
  function drawDebris(pieces) {
    if (!pieces || pieces.length === 0) return;
    glow(true);
    ctx.strokeStyle = "#fff";
    for (const p of pieces) {
      ctx.save();
      ctx.globalAlpha = p.alpha;           // ~2 s fade-out
      ctx.translate(wx(p.x), wy(p.y));
      ctx.scale(cam.s, cam.s);
      ctx.rotate(-p.angle);                // same chirality fix as the lander (§1)
      ctx.scale(1, -1);
      ctx.lineWidth = 1.5 / cam.s;
      polyline(p.ends);
      ctx.restore();
    }
  }

  // One-frame white flash at a crash point (§10) — one per exploding lander,
  // so a lander-lander collision paints two simultaneous flashes.
  function drawFlash(pt) {
    glow(true);
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(wx(pt.x), wy(pt.y), 45 * cam.s, 0, Math.PI * 2);
    ctx.fill();
  }

  // Full or partially revealed scene. During REVEAL (view.reveal set) the
  // terrain traces in by length, stars fade alongside, pads blink and the
  // landers stay hidden until the reveal completes (§10).
  function drawScene(view) {
    const terrain = view.terrain;
    if (view.reveal) {
      ctx.globalAlpha = view.reveal.starAlpha;
      drawStars(terrain.stars);
      ctx.globalAlpha = 1;
      if (view.reveal.points.length >= 2) drawTerrain(view.reveal.points);
      if (view.reveal.padsOn) drawPads(terrain.pads);
    } else {
      drawStars(terrain.stars);
      drawTerrain(terrain.points);
      drawPads(terrain.pads);
      // §9: ALL landers are drawn — landed ones stay parked; a crashed one
      // disappears once its explosion has spawned (its debris remains).
      for (const lander of view.frame.landers) {
        if (!view.hiddenLanders || !view.hiddenLanders.has(lander.i)) {
          drawLander(lander);
        }
      }
    }
    if (view.debris) {
      drawDebris(view.debris.pieces);
      for (const f of view.debris.flashes) drawFlash(f);
    }
  }

  // -------------------------------------------------------------------- HUD

  // One HUD row: label left-aligned, value right-aligned (arcade style).
  function hudRow(label, value, labelX, valueX, y) {
    ctx.textAlign = "left";
    ctx.fillText(label, labelX, y);
    ctx.textAlign = "right";
    ctx.fillText(value, valueX, y);
  }

  // Screen-space, always the live frame values regardless of camera (§10).
  // hud reflects lander 0 (§4); FUEL comes from landers[0] likewise.
  function drawHud(frame, sessionScore, high) {
    glow(true);
    ctx.fillStyle = "#fff";
    ctx.font = font(27); // ~1.6x v1's 17px (§9)
    ctx.textBaseline = "alphabetic";

    // TIME as MM SS (§9).
    const mm = String(Math.floor(frame.t / 60)).padStart(2, "0");
    const ss = String(Math.floor(frame.t % 60)).padStart(2, "0");

    // Top-left: SCORE (session total) / HIGH (best episode, §8) / TIME / FUEL.
    hudRow("SCORE", String(sessionScore), 40, 440, 56);
    hudRow("HIGH", String(high), 40, 440, 94);
    hudRow("TIME", `${mm} ${ss}`, 40, 440, 132);
    hudRow("FUEL", String(Math.floor(frame.landers[0].fuel)), 40, 440, 170);

    // Top-right: ALTITUDE / HORIZONTAL SPEED / VERTICAL SPEED with
    // direction-of-motion arrows; magnitudes shown, arrow gives the sign.
    const hud = frame.hud;
    hudRow("ALTITUDE", String(hud.altitude), 1480, 1925, 56);
    hudRow("HORIZONTAL SPEED", String(Math.abs(hud.hspeed)), 1480, 1925, 94);
    hudRow("VERTICAL SPEED", String(Math.abs(hud.vspeed)), 1480, 1925, 132);

    ctx.textAlign = "left";
    if (hud.hspeed !== 0) ctx.fillText(hud.hspeed > 0 ? "→" : "←", 1942, 94);
    if (hud.vspeed !== 0) ctx.fillText(hud.vspeed > 0 ? "↑" : "↓", 1942, 132);
  }

  // Small "SEED N" badge in the bottom-left corner (§8 ?seed=N, human only).
  function drawSeed(seed) {
    glow(false);
    ctx.fillStyle = "#777";
    ctx.font = font(20);
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(`SEED ${seed}`, 40, H - 22);
  }

  // ------------------------------------------------------ agent-view overlay

  // Labels for the 14 observation values, in CONTRACT §6 order.
  const OBS_LABELS = [
    "X (NORMALIZED)",
    "Y (NORMALIZED)",
    "VX",
    "VY",
    "SIN(ANGLE)",
    "COS(ANGLE)",
    "ANGULAR VELOCITY",
    "FUEL FRACTION",
    "DX TO PAD CENTER",
    "DY TO PAD SURFACE",
    "PAD HALF-WIDTH",
    "CLEARANCE ABOVE TERRAIN",
    "PAD MULT",
    "PAD VISIBLE",
  ];

  function drawObsOverlay(obs) {
    const px = 30, py = 195, pw = 560;
    const rowH = 30;
    const ph = 24 + 38 + OBS_LABELS.length * rowH + 16;

    // Translucent panel with a thin glowing border.
    glow(false);
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
    ctx.fillRect(px, py, pw, ph);
    glow(true);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1;
    ctx.strokeRect(px, py, pw, ph);

    ctx.fillStyle = "#fff";
    ctx.textBaseline = "alphabetic";

    ctx.font = font(22); // ~1.6x v1 (§9)
    ctx.textAlign = "left";
    ctx.fillText("AGENT VIEW", px + 18, py + 36);

    ctx.font = font(21);
    for (let i = 0; i < OBS_LABELS.length; i++) {
      const v = obs[i];
      const y = py + 36 + 32 + i * rowH;
      ctx.textAlign = "left";
      ctx.fillText(`${i.toString().padStart(2, " ")} ${OBS_LABELS[i]}`, px + 18, y);
      ctx.textAlign = "right";
      ctx.fillText((v >= 0 ? "+" : "") + v.toFixed(3), px + pw - 18, y);
    }
  }

  // ---------------------------------------------------------------- banners

  function drawLoading(stage) {
    glow(true);
    ctx.fillStyle = "#fff";
    if (Effects.blink()) centeredText("INSERT COIN", 300, 64);
    centeredText("LOADING LUNAR MODULE...", 390, 32);
    ctx.fillStyle = "#aaa";
    centeredText(String(stage || "").toUpperCase(), 440, 22);
  }

  function drawError(message) {
    glow(true);
    ctx.fillStyle = "#fff";
    centeredText("BOOT FAILURE", 280, 52);
    ctx.fillStyle = "#ccc";
    const lines = wrapText(String(message || "UNKNOWN ERROR").toUpperCase(), 90);
    lines.slice(0, 6).forEach((l, i) => centeredText(l, 350 + i * 34, 22));
    ctx.fillStyle = "#888";
    centeredText("RUN scripts/build_web.sh THEN RELOAD", 620, 22);
  }

  // Title layout (§10): big LUNAR LANDER + blinking PRESS ANY KEY over the
  // attract gameplay. Screen space, untouched by the camera.
  function drawTitle() {
    // Dim the attract action behind the title text.
    glow(false);
    ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
    ctx.fillRect(0, 0, W, H);
    glow(true);
    ctx.fillStyle = "#fff";
    centeredText("LUNAR LANDER", 290, 90);
    if (Effects.blink()) centeredText("PRESS ANY KEY", 380, 32);
  }

  function drawOutcomeBanner(outcome) {
    glow(true);
    ctx.fillStyle = "#fff";
    if (outcome.kind === "perfect") {
      centeredText(`A PERFECT LANDING  +${outcome.points} POINTS`, 285, 42);
    } else if (outcome.kind === "hard") {
      centeredText(`A GOOD LANDING  +${outcome.points} POINTS`, 285, 42);
    } else {
      centeredText("YOU JUST DESTROYED A 100 MEGABUCK LANDER", 285, 42);
      centeredText(String(outcome.reason || "").toUpperCase(), 340, 26);
    }
    if (Effects.blink()) centeredText("PRESS SPACE TO FLY AGAIN", 425, 26);
  }

  // ------------------------------------------------------------- entry point

  function init(canvas) {
    ctx = canvas.getContext("2d");
  }

  function render(view) {
    cam = view.camera || { s: 1, cx: W / 2, cy: H / 2 };
    clear();
    switch (view.state) {
      case "LOADING":
        drawLoading(view.stage);
        break;

      case "ERROR":
        drawError(view.message);
        break;

      case "TITLE":
        if (view.terrain && view.frame) drawScene(view);
        drawTitle();
        break;

      case "REVEAL":
        if (view.terrain && view.frame) drawScene(view);
        if (view.attract) drawTitle(); // attract draw-in stays under the title
        break;

      case "FLYING":
        drawScene(view);
        drawHud(view.frame, view.sessionScore, view.high);
        if (view.seed !== null && view.seed !== undefined) drawSeed(view.seed);
        if (view.overlay) drawObsOverlay(view.frame.landers[0].obs);
        break;

      case "ENDED":
        drawScene(view);
        drawHud(view.frame, view.sessionScore, view.high);
        if (view.seed !== null && view.seed !== undefined) drawSeed(view.seed);
        if (view.overlay) drawObsOverlay(view.frame.landers[0].obs);
        // Banner describes lander 0 (§9), ~0.8 s after a crash (§10).
        if (view.showBanner) drawOutcomeBanner(view.frame.landers[0].outcome);
        break;
    }
  }

  return { init, render };
})();
