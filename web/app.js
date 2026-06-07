"use strict";
/*
 * app.js — boot, app state machine, fixed-60Hz loop, keyboard+touch, session score.
 *
 * Per CONTRACT §8 and the DESIGN boundary rule: JS never simulates physics.
 * Python (in Pyodide) is the only simulation; every value crossing the
 * boundary is a JSON string that we JSON.parse. All drawing is delegated to
 * Renderer (renderer.js); all cosmetics (camera, debris, reveal — §10) live
 * in Effects (effects.js) and never feed back into Python calls.
 *
 * v2 (CONTRACT §4): the frame carries a `landers[]` array — no top-level
 * lander/fuel/score/obs. Human episodes run Game(n_landers=1) and read
 * landers[0]; the attract mode flies Game(n_landers=3) via step_auto_all().
 */

(() => {
  const DT = 1 / 60;           // physics timestep (matches Python Config.dt)
  const MAX_ACC = 0.25;        // accumulator cap, survives tab switches (§8)
  const ATTRACT_HOLD = 1.5;    // §10: hold after an attract terminal, seconds
  const CRASH_BANNER_DELAY = 0.8; // §10: outcome banner delay after a crash
  const ATTRACT_LANDERS = 3;   // §8: attract flies three autopilots
  const HUMAN_LANDERS = 1;     // §8: human episodes are single-lander
  const PRESETS = ["trainee", "cadet", "commander"]; // §2/§8 difficulty ladder

  // App states (§8/§10): LOADING -> REVEAL/TITLE (attract) -> on any key
  // REVEAL -> FLYING -> ENDED -> REVEAL ... (ERROR on boot failure).
  let state = "LOADING";
  let stage = "BOOTING PYTHON RUNTIME";
  let errorMessage = "";

  let py = null;           // Pyodide handle (boot only constructs it once)
  let game = null;         // PyProxy game handle — the only PyProxy we keep
  let gameLanders = 0;     // n_landers of the current Game (attract 3 / human 1)
  let gamePreset = null;   // preset of the current Game (rebuild on mismatch, §8)
  let terrain = null;      // parsed terrain JSON (§3), fixed per episode
  let frame = null;        // parsed frame JSON (§4), the last frame — drawn as-is
  let sessionScore = 0;    // accumulates lander 0's score across human episodes
  let highScore = loadHigh(); // best single-episode points (localStorage, §8)
  let overlay = false;     // agent-view obs panel toggle
  let preset = loadPreset(); // §8 difficulty preset (localStorage, default cadet)

  let attract = true;      // current episode flown by step_auto_all() (§8 attract)?
  let revealNext = "TITLE"; // state once the REVEAL draw-in completes (§10)
  let attractHold = null;  // seconds left in the post-terminal attract hold, or null
  let bannerDelay = 0;     // seconds until the ENDED outcome banner shows

  // Per-lander explosion tracking (§10): explosions fire on each lander's own
  // flying->crashed transition; `exploded` hides a lander once its debris
  // has spawned. Both reset at every episode boundary.
  let landerWasFlying = []; // status === "flying" last tick, by lander index
  const exploded = new Set(); // lander indices already blown up (hidden)

  // §8: ?seed=N makes every HUMAN reset deterministic (classroom mode).
  // Attract episodes always use fresh entropy. Invalid/absent -> null.
  const urlSeed = (() => {
    const raw = new URLSearchParams(location.search).get("seed");
    return raw !== null && /^\d+$/.test(raw.trim()) ? parseInt(raw, 10) : null;
  })();

  // ---------------------------------------------------------------- keyboard

  const keys = new Set(); // currently-held input codes, read each physics tick

  // "Btn*" codes are synthetic — added/removed by the touch arcade buttons
  // so touch feeds the exact same held-input mechanism as the keyboard (§8).
  const ROTATE_LEFT = ["ArrowLeft", "KeyA", "BtnLeft"];   // rotate +1 = CCW = tilt left (§2)
  const ROTATE_RIGHT = ["ArrowRight", "KeyD", "BtnRight"]; // rotate -1
  const THRUST = ["ArrowUp", "KeyW", "Space", "BtnThrust"];
  const PREVENT = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Space"];
  const MODIFIERS = ["ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight",
                     "AltLeft", "AltRight", "MetaLeft", "MetaRight"];

  window.addEventListener("keydown", (e) => {
    if (PREVENT.includes(e.code)) e.preventDefault(); // no page scrolling
    // Ignore browser-shortcut combos (Cmd+R, Ctrl+T, ...). Crucially, macOS
    // browsers swallow keyup for letter keys while Cmd is held, so adding
    // e.g. Cmd+A's "KeyA" to the held set would leave it stuck forever.
    if (e.metaKey || e.ctrlKey) return;
    keys.add(e.code);
    if (!e.repeat) handleKeyPress(e.code);
  });
  window.addEventListener("keyup", (e) => keys.delete(e.code));
  window.addEventListener("blur", () => {
    keys.clear();      // no stuck keys on tab-away
    clearTouchState(); // and no stuck arcade buttons either (§8)
  });

  // Edge-triggered (non-held) key actions.
  function handleKeyPress(code) {
    if (state === "LOADING" || state === "ERROR") return;

    if (state === "REVEAL") {
      // §10: input other than R is ignored until the reveal completes.
      if (code === "KeyR") startEpisode(false);
      return;
    }
    if (code === "KeyO") {
      overlay = !overlay; // agent-view overlay, available even in human play
      return;
    }
    if (code === "KeyR") {
      startEpisode(false); // fresh human episode any time
      return;
    }
    // §8 difficulty select: 1/2/3 act in TITLE and ENDED ONLY, and a digit
    // NEVER counts as "any key" — this branch must sit above the TITLE
    // catch-all so a preset change keeps the player on the title screen.
    // (FLYING/REVEAL digits fall in here too and are swallowed as no-ops,
    // which is exactly the "never mid-flight" rule.)
    if (code === "Digit1" || code === "Digit2" || code === "Digit3") {
      if (state === "TITLE" || state === "ENDED") {
        selectPreset(PRESETS[Number(code.slice(-1)) - 1]);
      }
      return;
    }
    if (state === "TITLE" && !MODIFIERS.includes(code)) {
      startEpisode(false); // §10: any key starts a fresh human episode
      return;
    }
    if (state === "ENDED" && code === "Space") {
      startEpisode(false); // "PRESS SPACE TO FLY AGAIN"
    }
  }

  // ------------------------------------------------------------------- touch

  // §8 touch controls: arcade buttons (←/→ pair, wide THRUST, OBS toggle)
  // plus tap routing on the bare canvas. Wired only on coarse-pointer
  // devices (the CSS media query hides the buttons elsewhere; this JS check
  // keeps the listeners off entirely). Keyboard handling above is untouched
  // — buttons never see key events, so the O/R paths can't be swallowed.
  //
  // Each button tracks ITS OWN active pointers in a Set: the synthetic code
  // is held exactly while that set is non-empty, so two thumbs on one button
  // release cleanly and multi-touch across buttons (rotate+thrust, both
  // rotates = net 0) just works. setPointerCapture keeps a press through a
  // drift off the button (its pointerup still reaches the button); the
  // pointerleave handler covers engines where capture isn't in effect (it
  // never fires mid-press while capture holds). A press belongs to the
  // button it started on — sliding onto a neighbor never presses it.
  const touchButtons = []; // [{el, pointers}] so blur can clear everything

  function clearTouchState() {
    for (const b of touchButtons) {
      b.pointers.clear();
      b.el.classList.remove("pressed");
    }
  }

  // Pointer event -> logical canvas coords. The canvas is CSS-scaled with
  // the aspect preserved, so client px map to the 2000x750 logical grid by
  // one uniform factor per axis (both equal; computed per axis anyway).
  function canvasPoint(e, el) {
    const r = el.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) * (2000 / r.width),
      y: (e.clientY - r.top) * (750 / r.height),
    };
  }

  const inRect = (p, b) =>
    !!b && p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h;

  // §8 tap routing on the BARE canvas. The preset hit-tests run BEFORE the
  // any-key fallback: in TITLE a tap inside a preset-menu segment selects
  // that preset and does NOT start an episode (parity with 1/2/3); in ENDED
  // a tap on the persistent preset readout cycles trainee→cadet→commander.
  // Hitboxes come from the renderer's last-drawn layout (canvas coords).
  function handleCanvasTap(e) {
    if (state === "LOADING" || state === "ERROR") return;
    e.preventDefault();
    if (state === "TITLE") {
      const p = canvasPoint(e, e.currentTarget);
      for (const b of Renderer.presetMenuHitboxes || []) {
        if (inRect(p, b)) {
          selectPreset(b.preset); // select only — no episode start (§8)
          return;
        }
      }
      startEpisode(false); // bare-canvas tap = "any key"
    } else if (state === "ENDED") {
      const p = canvasPoint(e, e.currentTarget);
      if (inRect(p, Renderer.presetReadoutHitbox)) {
        selectPreset(PRESETS[(PRESETS.indexOf(preset) + 1) % PRESETS.length]);
        return; // cycle only — the next episode picks it up via ensureGame
      }
      startEpisode(false); // tap anywhere else = fly again
    }
  }

  function bindButton(el) {
    const code = el.dataset.code; // BtnLeft | BtnRight | BtnThrust | BtnObs
    const pointers = new Set();   // this button's active pointerIds
    touchButtons.push({ el, pointers });

    el.addEventListener("pointerdown", (e) => {
      e.preventDefault(); // no synthesized mouse events / focus changes
      try {
        el.setPointerCapture(e.pointerId); // keep the hold if the finger drifts
      } catch (_) { /* pointer already gone — release() still cleans up */ }
      pointers.add(e.pointerId);
      el.classList.add("pressed");
      if (code === "BtnObs") {
        // §8 OBS = touch parity with KeyO, same gating as handleKeyPress
        // (ignored in LOADING/ERROR/REVEAL) — and never "any key".
        if (state !== "LOADING" && state !== "ERROR" && state !== "REVEAL") {
          overlay = !overlay;
        }
        return;
      }
      keys.add(code); // same held-input set the keyboard uses (§8)
      // §8: arcade-button taps count as "any key" — a thumb already resting
      // on THRUST when the episode starts behaves like held Space.
      if (state === "TITLE" || state === "ENDED") startEpisode(false);
    });

    const release = (e) => {
      if (!pointers.delete(e.pointerId)) return; // not ours / already released
      if (pointers.size === 0) {
        keys.delete(code);
        el.classList.remove("pressed");
      }
    };
    el.addEventListener("pointerup", release);
    el.addEventListener("pointercancel", release);
    el.addEventListener("pointerleave", release); // only fires uncaptured
  }

  function initTouch() {
    const coarse = window.matchMedia &&
                   window.matchMedia("(pointer: coarse)").matches;
    if (!coarse) return;

    document.getElementById("screen")
            .addEventListener("pointerdown", handleCanvasTap);
    for (const el of document.querySelectorAll("#touch .abtn")) bindButton(el);
  }

  // §8: button labels are stroke-font canvases painted exactly ONCE at boot
  // (←/→ are font glyphs; THRUST and OBS are text) — never CSS text (§9,
  // zero fillText). Canvases are 2x their CSS size for a crisp beam, with
  // the same white glow as the tube. Drawn unconditionally: on fine-pointer
  // devices the buttons stay display:none, so this paints nothing visible.
  function drawButtonLabels() {
    const paint = (id, text, size) => {
      const c = document.getElementById(id);
      const lctx = c.getContext("2d");
      lctx.strokeStyle = "#fff";
      lctx.shadowColor = "#fff";
      lctx.shadowBlur = 6; // §9 glow
      VectorFont.draw(lctx, text, c.width / 2, (c.height + size) / 2, size,
                      { align: "center" });
    };
    paint("label-left", "←", 40);
    paint("label-right", "→", 40);
    paint("label-thrust", "THRUST", 36);
    paint("label-obs", "OBS", 26);
  }

  // --------------------------------------------------------- python boundary

  // Ensure `game` is a Game with the right lander count AND difficulty
  // preset (§2/§8). PROXY LIFECYCLE: switching attract (n=3) -> human (n=1)
  // or changing preset constructs a NEW Game. We explicitly destroy() the
  // old PyProxy FIRST (we never hold two), then reassign the Python global —
  // that reassignment drops Python's own last reference, so the old Game is
  // freed immediately rather than waiting on the JS GC / FinalizationRegistry.
  // `game` stays the single live PyProxy. `preset` is always one of PRESETS
  // (loadPreset validates), so interpolating it into Python source is safe.
  function ensureGame(nLanders) {
    if (game && gameLanders === nLanders && gamePreset === preset) return;
    if (game) game.destroy();
    py.runPython(
      `game = Game(mode="classic", n_landers=${nLanders}, preset="${preset}")`
    );
    game = py.globals.get("game");
    gameLanders = nLanders;
    gamePreset = preset;
  }

  // §8 difficulty change (TITLE/ENDED only — handleKeyPress gates states).
  // Persist, then restart the current context: TITLE gets a fresh attract
  // episode (startEpisode -> ensureGame rebuilds the Game with the new
  // preset, new terrain, reveal draw-in); ENDED stays put — the final frame
  // and outcome banner keep rendering from the already-parsed JSON, and the
  // next episode (R / Space / tap) rebuilds through the same ensureGame
  // check. Session score and high score are untouched (§8: one high score
  // across presets).
  function selectPreset(name) {
    if (name === preset) return; // re-pressing the active preset: no-op
    preset = name;
    savePreset(name);
    if (state === "TITLE") startEpisode(true);
  }

  // Fresh episode (human or attract): reset Python, restart the cosmetic
  // layer (debris cleared, camera snapped to full view, terrain draw-in)
  // and enter REVEAL (§10). Attract episodes return to TITLE afterwards.
  function startEpisode(isAttract) {
    attract = isAttract;
    attractHold = null;
    bannerDelay = 0;
    ensureGame(isAttract ? ATTRACT_LANDERS : HUMAN_LANDERS);
    // §8: human resets honor ?seed=N; attract always uses fresh entropy.
    terrain = JSON.parse(
      !isAttract && urlSeed !== null ? game.reset(urlSeed) : game.reset()
    );
    frame = JSON.parse(game.frame_json()); // current frame without stepping
    exploded.clear();
    landerWasFlying = frame.landers.map((l) => l.status === "flying");
    Effects.episodeReset(terrain);
    revealNext = isAttract ? "TITLE" : "FLYING";
    state = "REVEAL";
  }

  // Per-lander terminal handling (§10): spawn an explosion on each lander's
  // OWN flying->crashed transition — a two-lander collision in attract mode
  // transitions both in the same tick, so both explode simultaneously.
  // Cosmetic only — state is Python's.
  function trackExplosions() {
    for (const l of frame.landers) {
      if (landerWasFlying[l.i] && l.status === "crashed") {
        Effects.explode(l);
        exploded.add(l.i); // hidden from now on — the debris IS the lander
      }
      landerWasFlying[l.i] = l.status === "flying";
    }
  }

  // One fixed 1/60 s tick: step Python per state, then advance cosmetics.
  function tick() {
    if (state === "LOADING" || state === "ERROR") return;

    if (state === "FLYING") {
      // Read held inputs (keys + arcade buttons), step Python, parse the frame.
      const left = ROTATE_LEFT.some((k) => keys.has(k));
      const right = ROTATE_RIGHT.some((k) => keys.has(k));
      const rotate = (left ? 1 : 0) - (right ? 1 : 0); // +1 CCW / tilt left (§2)
      const thrust = THRUST.some((k) => keys.has(k));

      frame = JSON.parse(game.step(rotate, thrust)); // n_landers == 1 (§2)
      trackExplosions();

      if (frame.status !== "flying") { // §4: "done" — all landers terminal
        const human = frame.landers[0];
        sessionScore += human.score; // per-episode points -> session total (§8)
        if (human.score > highScore) { // §8: best single episode
          highScore = human.score;
          saveHigh(highScore);
        }
        bannerDelay = human.outcome && human.outcome.kind === "crash"
          ? CRASH_BANNER_DELAY : 0; // §10: explosion reads first
        state = "ENDED";
      }
    } else if (state === "TITLE") {
      // Attract mode (§8): three autopilot landers behind the title,
      // collisions welcome. These NEVER touch sessionScore.
      if (attractHold !== null) {
        attractHold -= DT; // post-terminal hold while debris settles
        if (attractHold <= 0) startEpisode(true); // fresh terrain, repeat
      } else {
        frame = JSON.parse(game.step_auto_all()); // synchronous, like step (§2)
        trackExplosions();
        if (frame.status !== "flying") attractHold = ATTRACT_HOLD;
      }
    } else if (state === "ENDED") {
      if (bannerDelay > 0) bannerDelay -= DT; // explosion reads first (§10)
    }

    // Cosmetic layer (§10): camera lerp, debris integration, reveal timer.
    // §10 zoom: only when EXACTLY ONE lander is flying — Effects checks that
    // lander's altitude (<150) against the terrain. Camera freezes during
    // ENDED (and the attract hold) so explosions stay framed.
    let zoomLander = null;
    if (frame && frame.active === 1) {
      zoomLander = frame.landers.find((l) => l.status === "flying") || null;
    }
    Effects.tick({
      frame,
      terrain,
      zoomLander,
      cameraFrozen: state === "ENDED" ||
                    (state === "TITLE" && attractHold !== null),
    });

    // REVEAL -> FLYING for human episodes, back to TITLE for attract (§10).
    if (state === "REVEAL" && !Effects.revealActive()) state = revealNext;
  }

  // ------------------------------------------------------------- high score

  // localStorage can throw (privacy modes) — treat it as best-effort (§8).
  function loadHigh() {
    try {
      return parseInt(localStorage.getItem("moonlander.high"), 10) || 0;
    } catch (_) {
      return 0;
    }
  }

  function saveHigh(value) {
    try {
      localStorage.setItem("moonlander.high", String(value));
    } catch (_) { /* best-effort */ }
  }

  // ----------------------------------------------------------------- preset

  // §8: persisted difficulty preset, validated against the three names —
  // anything else (tampered storage, old versions) falls back to "cadet".
  function loadPreset() {
    try {
      const raw = localStorage.getItem("moonlander.preset");
      return PRESETS.includes(raw) ? raw : "cadet";
    } catch (_) {
      return "cadet";
    }
  }

  function savePreset(name) {
    try {
      localStorage.setItem("moonlander.preset", name);
    } catch (_) { /* best-effort */ }
  }

  // -------------------------------------------------------------- game loop

  // Fixed-timestep accumulator at 60 Hz over requestAnimationFrame (§8).
  // Everything in here is synchronous — no awaits in the rAF loop.
  let last = performance.now();
  let acc = 0;

  function loop(now) {
    acc += (now - last) / 1000;
    last = now;
    if (acc > MAX_ACC) acc = MAX_ACC;

    while (acc >= DT) {
      tick();
      acc -= DT;
    }

    Renderer.render({
      state,
      stage,
      message: errorMessage,
      terrain,
      frame,
      sessionScore,
      high: highScore,
      seed: attract ? null : urlSeed, // §8: "SEED N" badge on human episodes
      preset,                         // §8: title menu + HUD readout
      overlay,
      attract,
      camera: Effects.cameraView(),
      reveal: Effects.revealView(), // null unless the draw-in is running
      debris: Effects.debrisView(), // flashes consumed: one rendered frame (§10)
      hiddenLanders: exploded,      // crashed landers hidden after debris (§9)
      showBanner: state === "ENDED" && bannerDelay <= 0,
    });
    requestAnimationFrame(loop);
  }

  // ------------------------------------------------------------------- boot

  async function boot() {
    try {
      stage = "BOOTING PYTHON RUNTIME";
      py = await loadPyodide(); // core is pure stdlib — no numpy (§8)

      stage = "LOADING MICROPIP";
      await py.loadPackage("micropip");

      stage = "INSTALLING MOONLANDER WHEEL";
      const micropip = py.pyimport("micropip");
      await micropip.install(
        new URL("assets/moonlander-0.4.0-py3-none-any.whl", location.href).href
      );

      stage = "CREATING GAME";
      py.runPython("from moonlander.core.game import Game");

      // First attract episode (§8): terrain draw-in, then three autopilots
      // fly behind the title until the player presses a key (or taps).
      startEpisode(true); // ensureGame builds Game(n_landers=3) here
    } catch (err) {
      errorMessage = err && err.message ? err.message : String(err);
      state = "ERROR";
    }
  }

  // ------------------------------------------------------------------ start

  Renderer.init(document.getElementById("screen"));
  drawButtonLabels(); // §8: arcade-button labels, painted once
  initTouch();
  requestAnimationFrame(loop); // render LOADING immediately
  boot();
})();
