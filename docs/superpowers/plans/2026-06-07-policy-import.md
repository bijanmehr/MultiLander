# Policy Import (Bring-Your-Own-Brain) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import a custom §11 policy JSON as the AI PILOT via drag & drop or a `LOAD AI` footer link, with a committed 80-parameter example brain (`web/assets/policy-tiny.json`).

**Architecture:** Pure web-side import surface; validation stays in Python (`game.set_policy`). The existing NO-POLICY notice generalizes to `{title, sub}` text. New `customPolicy` state flips the HUD indicator to `CUSTOM AI`.

**Tech Stack:** Vanilla JS (app.js/renderer.js/index.html), pytest for the one new core guarantee, CONTRACT/ml.html/docs/README updates.

Spec: addendum in `docs/superpowers/specs/2026-06-07-ml-pilot-design.md`.

### Task 1: Pin "failed set_policy preserves the old policy" (TDD)

- [ ] Append to `tests/test_policy.py`:

```python
def test_failed_set_policy_preserves_the_active_policy():
    # Import-flow guarantee (§2): validation happens before assignment, so a
    # rejected payload leaves the previously attached policy flying.
    g = Game(mode="classic", preset="cadet")
    g.set_policy(policy_json(b2=[0.0, 0.0, 0.0, 1.0]))  # always-thrust
    g.reset(seed=11)
    before = g.step_policy()
    with pytest.raises(ValueError):
        g.set_policy('{"format": "nope"}')
    after = g.step_policy()  # still flying on the old policy, no RuntimeError
    assert json.loads(after)["t"] > json.loads(before)["t"]
```

- [ ] Run `.venv/bin/python -m pytest tests/test_policy.py -q` — passes already
      (current code assigns after validation); this pins it. Commit.

### Task 2: app.js — import flow + dynamic notices

- [ ] Replace `let notice = 0` with `let notice = null;  // {title, sub, t} or null`
      and add `let customPolicy = null; // imported policy's file name, or null`.
- [ ] Add `setNotice(title, sub)` (sets `{title, sub, t: NOTICE_TIME}`);
      `toggleAiPilot` no-policy branch calls
      `setNotice("NO POLICY", "TRAIN ONE:  PYTHON -M MOONLANDER.TRAIN_CEM")`.
- [ ] Tick: `if (notice && (notice.t -= DT) <= 0) notice = null;`
- [ ] `importPolicyText(text, name)`:
      guards `if (!py || !game) return;`; try `py.globals.set("policy_json", text);
      py.runPython("game.set_policy(policy_json)")` → success: `policyJson = text;
      customPolicy = name; aiPilot = true; setNotice("CUSTOM POLICY LOADED",
      name.toUpperCase().slice(0, 40))`; catch: `setNotice("BAD POLICY FILE",
      pyErrorLine(err))` where `pyErrorLine` extracts the last `ValueError: ...`
      line of the Pyodide traceback, strips the prefix, uppercases, truncates.
- [ ] `importPolicyFile(file)`: null-guard, `file.size > 2e6` →
      `setNotice("BAD POLICY FILE", "OVER 2 MB")`, else `file.text().then(...)`.
- [ ] Drag & drop: window `dragover` + `drop` listeners (preventDefault both),
      drop takes `e.dataTransfer.files[0]`.
- [ ] `LOAD AI` link: `#load-ai` click → hidden `#policy-file` input `.click()`;
      input `change` → `importPolicyFile(input.files[0]); input.value = ""`.
- [ ] Render view: `notice: notice && {title: notice.title, sub: notice.sub}`,
      add `customAi: !!customPolicy`.

### Task 3: renderer.js — CUSTOM AI + dynamic notice

- [ ] `drawAiPilot(custom)` → text `custom ? "CUSTOM AI" : "AI PILOT"`;
      call sites pass `view.customAi`.
- [ ] `drawTitle(preset, aiPilot, customAi)` → armed line
      `customAi ? "CUSTOM AI ARMED" : "AI PILOT ARMED"`.
- [ ] `drawNotice(n)` → `centeredText(n.title, 600, 28)`; if `n.sub`,
      `centeredText(n.sub, 642, 16)` in `#999`. `render()` passes the object.
- [ ] View doc comment: notice is `{title, sub}|null`; add `customAi`.

### Task 4: index.html — LOAD AI link + hidden input

- [ ] Footer: ` · <a href="#" id="load-ai">LOAD AI</a>` after THE MACHINERY.
- [ ] Before `</body>` scripts: `<input type="file" id="policy-file"
      accept=".json,application/json" style="display:none">`.

### Task 5: CONTRACT + docs + example artifact

- [ ] CONTRACT §8: import surfaces paragraph (drag & drop + LOAD AI; Python-side
      validation; success auto-arms + `CUSTOM AI` indicator; failure keeps the
      previous policy + `BAD POLICY FILE` notice; session-only; >2 MB rejected).
- [ ] CONTRACT §11: note the schema doubles as the import format (any
      `[14, h≥1, 4]`); list `web/assets/policy-tiny.json` example artifact.
- [ ] ml.html: **BRING YOUR OWN BRAIN** section before TRY IT — format recap,
      download link to `assets/policy-tiny.json` (80 params, real eval stats),
      "drop it on the game or use LOAD AI", `train_cem.py` as reference exporter.
- [ ] docs.html AI PILOT section + README training section: one line each.
- [ ] Commit `web/assets/policy-tiny.json` from the `--hidden 4 --seed 0` run
      with its real eval numbers in the commit message.

### Task 6: Verify + ship

- [ ] `.venv/bin/python -m pytest -q` all green.
- [ ] Browser drive (puppeteer): synthesize a `drop` event with a DataTransfer
      File built from the tiny artifact → CUSTOM POLICY LOADED notice +
      CUSTOM AI armed; drop garbage JSON → BAD POLICY FILE, previous policy
      still active; LOAD AI link opens picker (presence check).
- [ ] Merge `policy-import` → main, push (explicitly authorized).
