# archive/

Excluded from the active project — kept for reference, not deleted.

## Lower-level (follower) files

These are the standalone **lower-level (follower)** files. Nothing in the
active tree (`run.py`, `plot.py`, `output.py`, `model/bilevel_mpec.py`)
imports them; the follower is still fully present inside the bilevel model as
its embedded KKT block.

| file | what it is |
|---|---|
| `lower_level.py` | the follower's primal LP at a fixed leader price (`LowerLevel`) |
| `lower_level_kkt.py` | its KKT + Big-M reformulation (`LowerLevelKKT`), the standalone twin of the block embedded in `BilevelMPEC` |
| `run_lowerlevel.py` | entry point: build and solve the follower LP for one season/price |
| `plot_lowerlevel.py` | follower cost boxplot + per-scenario dispatch timelines |

### Restoring them

They are **not runnable from this folder as-is**: they import `config`,
`scenarios` and `model.lower_level` as top-level modules, so they expect to sit
in the repo root (scripts) and in `model/` (the two model files). To bring them
back:

```bash
git mv archive/lower_level.py archive/lower_level_kkt.py model/
git mv archive/run_lowerlevel.py archive/plot_lowerlevel.py .
```

Two things to re-check after restoring, since the active code moved on while
these sat here:

- `lower_level.py` / `lower_level_kkt.py` mirror the formulation in
  `model/bilevel_mpec.py` (scenario-dependent `qMGO[t,s]`). If the bilevel's
  follower block changes, these must change with it or the LP↔KKT validation
  is checking the wrong model.
- `run_lowerlevel.py --validate`, referenced in the docstrings, was removed
  before archiving — the flag no longer exists. The LP↔KKT comparison it ran is
  described in `lower_level_kkt.py` and would need to be re-added.

## `price_analysis.py` — leader-price sweep

Archived as obsolete. It pinned the leader price at each of `--steps` levels
from `P_ref` up to the cap (a one-entry menu, so the leader had no choice
left), once for the stochastic run and once per EDL outcome taken as certain,
and read the profit / cost / quantity curves and a per-outcome optimum off
that grid. Its last outputs (`output/price_analysis.xlsx`,
`output/price_analysis.png`, `output/price_analysis_scenarios.png`) were left
where they are.

It was also a stage of `run.py` (`--skip-analysis`, `--steps`); that stage and
both flags were removed when the file was archived.

### Restoring it

It imports `config`, `run`, `plot`, `scenarios` and `model.bilevel_mpec` as
top-level modules, so it has to sit in the repo root:

```bash
mv archive/price_analysis.py .
```

and, to run it from `run.py` again, re-add the stage in `run.main()`:
`import price_analysis; price_analysis.main([season], steps)`.

## `availability_analysis.py` — EDL availability sweep

Archived. It swept a continuous EDL availability profile — 11 energy shares
from 0 to 100% of daily demand × 6 timing patterns (`proportional`, `uniform`,
`peak`, `offpeak`, `announced`, `delayed`), each taken as certain — and
re-optimized the leader at every point, ~72 solves per season. It tested a
closed-form law: with suppression on the uncovered load, availability scales
the residual rather than shifting it, so it moves the price only through the
load the village shifts (`d x*/d sigma = -1/(2*alpha_1)`), and moves the profit
through the size of the uncovered market. Its last outputs
(`output/availability_analysis.xlsx`, `output/availability_<season>.png`) were
left where they are.

It was also the last stage of `run.py` (`--skip-availability`); that stage and
the flag were removed when the file was archived.

Known issues at the time of archiving, to fix if it comes back:

- the closed form assumes ONE constant marginal cost (diesel, 0.50 $/kWh). That
  holds for config's fleet (`CAP["PV"] = 0`) and not for any build with PV;
- the price-vs-sigma panel is still titled and documented as if its x axis were
  the older `B_eff/A`;
- `--levels` help still calls "the default 9" too coarse; the default is 65;
- the docstring's fit numbers (slope -1.08 vs -1.04, R^2 = 0.99) are hard-coded
  from an earlier winter run, not computed.

### Restoring it

It needs the parametric-profile helpers still in `scenarios.py`
(`AVAILABILITY_PATTERNS`, `availability_shape`, `synthetic_availability`),
which were deliberately left in place, and has to sit in the repo root:

```bash
mv archive/availability_analysis.py .
```

and, to run it from `run.py` again, re-add the stage in `run.main()`:
`import availability_analysis; availability_analysis.main([season])`.
