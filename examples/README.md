# examples/

Starting points you copy and make your own.

## `train_template.py` — write a training stack

A heavily-annotated tour of the whole world API with a tiny working baseline,
so it runs end-to-end out of the box. Replace one function with your algorithm
and keep the rest.

```bash
# See every world function in action (no training):
.venv/bin/python -m examples.train_template --tour

# Run the baseline hill-climber and export a flyable policy:
.venv/bin/python -m examples.train_template --iters 300 --out /tmp/mine.json
```

Then open the game, click **LOAD AI** (or drag the file on), and your agent
flies — the forward pass you trained is the exact one Pyodide runs.

What it covers:

- **PART 1** the 14-float observation, every index with its formula
- **PART 2** both interfaces — the Gymnasium env (training) and the raw `Game`
  core (terrain you can read, the autopilot to beat, the per-tick browser path)
- **PART 3** the tiny MLP policy (the web export format)
- **PART 4** rollouts — turning a policy into a shaped return (training signal)
  and an honest landing rate (the metric that matters)
- **PART 5** `>>> PLUG YOUR ALGORITHM HERE <<<` — swap the baseline for PPO /
  REINFORCE / CMA-ES / CEM (see `src/moonlander/train_cem.py` for a
  population-based reference) and keep everything around it
- **PART 6** export to `policy.json` (CONTRACT §11) the browser can fly

The full machinery, the format, and a downloadable example brain live at
[ml.html](https://bijanmehr.github.io/MultiLander/ml.html).
