"use strict";
/*
 * vectorfont.js — the Atari-vector stroke font (CONTRACT §9). No game logic.
 *
 * Every glyph is 1–3 polylines in a normalized cell: width 10 units, cap
 * height 14 units, baseline at y = 0, y-UP (descenders unused — the font is
 * uppercase-only; lowercase input is uppercased at draw time). Advance is 13
 * units (10-unit cell + 3-unit letter gap); the space advances 8.
 *
 * Letterforms are 1979-vector style: ONLY straight segments — corners are
 * chamfered at 45° where the beam would turn (O is a rectangle with all four
 * corners cut, S is a Z-like zigzag with chamfers, 8 is two stacked chamfered
 * boxes). No curves, no arcs. Digits read like a scoreboard.
 *
 * Exposes a script-global `VectorFont`:
 *   draw(ctx, text, x, y, size, {align, weight}) — paint `text` with x/y at
 *     the BASELINE; `size` is the cap height in px; align "left" (default) |
 *     "center" | "right"; weight scales lineWidth (default ~size/9, min
 *     1.25px). Inherits the ctx's strokeStyle and shadow — the painter sets
 *     the glow — and leaks no ctx state (save/restore around the stroke).
 *   measure(text, size) -> width in px.
 *
 * A character with no glyph renders as a hollow box — never a throw (§9).
 *
 * The GLYPHS block below is machine-checked (every coordinate inside the
 * cell): keep it JSON-shaped — quoted keys, flat [x0,y0, x1,y1, ...] arrays,
 * comments only at end-of-line.
 */

const VectorFont = (() => {
  const CAP = 14;       // cell cap height in units; size px == CAP units
  const ADV = 13;       // glyph advance: 10-unit cell + 3-unit letter gap
  const GAP = 3;        // the letter gap (trimmed from the last glyph)
  const SPACE_ADV = 8;  // the space character's full advance

  // Glyph table: char -> array of polylines, each a flat [x0,y0, x1,y1, ...]
  // run in cell units, y-up from the baseline.
  const GLYPHS = {
    "A": [[0,0, 0,9, 3,14, 7,14, 10,9, 10,0], [0,5, 10,5]],
    "B": [[0,7, 7,7, 10,10, 10,11, 7,14, 0,14, 0,0, 7,0, 10,3, 10,4, 7,7]],
    "C": [[10,3, 7,0, 3,0, 0,3, 0,11, 3,14, 7,14, 10,11]],
    "D": [[0,0, 0,14, 6,14, 10,10, 10,4, 6,0, 0,0]],
    "E": [[10,0, 0,0, 0,14, 10,14], [0,7, 7,7]],
    "F": [[0,0, 0,14, 10,14], [0,7, 7,7]],
    "G": [[10,11, 7,14, 3,14, 0,11, 0,3, 3,0, 7,0, 10,3, 10,6, 5,6]],
    "H": [[0,0, 0,14], [10,0, 10,14], [0,7, 10,7]],
    "I": [[2,0, 8,0], [5,0, 5,14], [2,14, 8,14]],
    "J": [[10,14, 10,3, 7,0, 3,0, 0,3]],
    "K": [[0,0, 0,14], [10,14, 0,7, 10,0]],
    "L": [[0,14, 0,0, 10,0]],
    "M": [[0,0, 0,14, 5,7, 10,14, 10,0]],
    "N": [[0,0, 0,14, 10,0, 10,14]],
    "O": [[3,0, 0,3, 0,11, 3,14, 7,14, 10,11, 10,3, 7,0, 3,0]],
    "P": [[0,0, 0,14, 7,14, 10,11, 10,9, 7,6, 0,6]],
    "Q": [[3,0, 0,3, 0,11, 3,14, 7,14, 10,11, 10,3, 7,0, 3,0], [5,5, 10,0]],
    "R": [[0,0, 0,14, 7,14, 10,11, 10,9, 7,6, 0,6], [4,6, 10,0]],
    "S": [[10,11, 7,14, 3,14, 0,11, 0,10, 3,7, 7,7, 10,4, 10,3, 7,0, 3,0, 0,3]],
    "T": [[0,14, 10,14], [5,14, 5,0]],
    "U": [[0,14, 0,3, 3,0, 7,0, 10,3, 10,14]],
    "V": [[0,14, 4,0, 6,0, 10,14]],
    "W": [[0,14, 2,0, 5,6, 8,0, 10,14]],
    "X": [[0,0, 10,14], [0,14, 10,0]],
    "Y": [[0,14, 5,7, 10,14], [5,7, 5,0]],
    "Z": [[0,14, 10,14, 0,0, 10,0]],
    "0": [[3,0, 0,3, 0,11, 3,14, 7,14, 10,11, 10,3, 7,0, 3,0]],
    "1": [[2,11, 5,14, 5,0], [2,0, 8,0]],
    "2": [[0,11, 3,14, 7,14, 10,11, 10,8, 0,0, 10,0]],
    "3": [[0,11, 3,14, 7,14, 10,11, 10,9, 8,7, 10,5, 10,3, 7,0, 3,0, 0,3],
          [5,7, 8,7]],
    "4": [[10,4, 0,4, 8,14, 8,0]],
    "5": [[10,14, 0,14, 0,8, 7,8, 10,5, 10,3, 7,0, 3,0, 0,3]],
    "6": [[10,11, 7,14, 3,14, 0,11, 0,3, 3,0, 7,0, 10,3, 10,5, 7,8, 0,8]],
    "7": [[0,14, 10,14, 10,11, 4,0]],
    "8": [[3,7, 1,9, 1,12, 3,14, 7,14, 9,12, 9,9, 7,7, 3,7],
          [2,0, 0,2, 0,5, 2,7, 8,7, 10,5, 10,2, 8,0, 2,0]],
    "9": [[0,3, 3,0, 7,0, 10,3, 10,11, 7,14, 3,14, 0,11, 0,9, 3,6, 10,6]],
    "-": [[2,7, 8,7]],
    "—": [[0,7, 10,7]],
    ".": [[4,0, 6,0, 6,2, 4,2, 4,0]],
    ",": [[6,3, 6,1, 4,0]],
    ":": [[4,2, 6,2, 6,4, 4,4, 4,2], [4,8, 6,8, 6,10, 4,10, 4,8]],
    ";": [[4,8, 6,8, 6,10, 4,10, 4,8], [6,4, 6,2, 4,0]],
    "/": [[1,0, 9,14]],
    "(": [[7,14, 4,11, 4,3, 7,0]],
    ")": [[3,14, 6,11, 6,3, 3,0]],
    "[": [[7,14, 3,14, 3,0, 7,0]],
    "]": [[3,14, 7,14, 7,0, 3,0]],
    "+": [[5,3, 5,11], [1,7, 9,7]],
    "×": [[2,4, 8,10], [2,10, 8,4]],
    "?": [[0,11, 3,14, 7,14, 10,11, 10,9, 5,6, 5,4], [4,0, 6,0, 6,2, 4,2, 4,0]],
    "!": [[5,14, 5,4], [4,0, 6,0, 6,2, 4,2, 4,0]],
    "'": [[5,14, 5,11]],
    "\"": [[3,14, 3,11], [7,14, 7,11]],
    "%": [[1,0, 9,14], [0,10, 3,10, 3,13, 0,13, 0,10], [7,1, 10,1, 10,4, 7,4, 7,1]],
    "=": [[1,5, 9,5], [1,9, 9,9]],
    ">": [[2,11, 8,7, 2,3]],
    "_": [[0,0, 10,0]],
    "↑": [[5,0, 5,14], [2,10, 5,14, 8,10]],
    "↓": [[5,0, 5,14], [2,4, 5,0, 8,4]],
    "←": [[0,7, 10,7], [4,11, 0,7, 4,3]],
    "→": [[0,7, 10,7], [6,11, 10,7, 6,3]]
  };

  // §9 missing-glyph fallback: a hollow box, never a throw.
  const MISSING = [[1,0, 9,0, 9,14, 1,14, 1,0]];

  const advance = (ch) => (ch === " " ? SPACE_ADV : ADV);

  // Width of `text` at cap height `size`, in px. The trailing 3-unit letter
  // gap is trimmed so aligned text ends exactly at its last beam.
  function measure(text, size) {
    const s = String(text);
    let units = 0;
    for (const ch of s) units += advance(ch.toUpperCase());
    return units > 0 ? (units - GAP) * (size / CAP) : 0;
  }

  // Paint `text` with (x, y) at the baseline. Uppercases as it goes, so
  // callers may pass raw strings (outcome reasons, error messages, …).
  function draw(ctx, text, x, y, size, opts) {
    const s = String(text);
    if (!s) return;
    const o = opts || {};
    const k = size / CAP; // px per cell unit
    let px = x;
    if (o.align === "center") px -= measure(s, size) / 2;
    else if (o.align === "right") px -= measure(s, size);

    ctx.save(); // inherit strokeStyle/shadow; leak no lineWidth/join/cap
    ctx.lineWidth = Math.max(1.25, (size / 9) * (o.weight || 1));
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath(); // all glyphs in one path: a single stroke() call
    for (const raw of s) {
      const ch = raw.toUpperCase();
      if (ch !== " ") {
        for (const poly of GLYPHS[ch] || MISSING) {
          ctx.moveTo(px + poly[0] * k, y - poly[1] * k);
          for (let i = 2; i < poly.length; i += 2) {
            ctx.lineTo(px + poly[i] * k, y - poly[i + 1] * k);
          }
        }
      }
      px += advance(ch) * k;
    }
    ctx.stroke();
    ctx.restore();
  }

  return { draw, measure };
})();
