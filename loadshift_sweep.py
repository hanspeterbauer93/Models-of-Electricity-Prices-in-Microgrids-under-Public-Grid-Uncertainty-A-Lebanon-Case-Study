"""
How the village's load-shift capacity moves what EDL is worth to the leader.

The companion of :mod:`asset_sweep`: the same two runs, the same span, the same
figures and workbook -- with the fleet held fixed and the load shift swept
instead. The village may move up to ``shift_frac`` of each hour's demand into
other hours (sum_t dd_t = 0); this module sweeps ``shift_frac`` from
:data:`SHIFT_MIN` to :data:`SHIFT_MAX` in :data:`SHIFT_STEP` steps, and at every
value solves

    no supply    P(none) = 1: EDL is off all day
    commitment   the stochastic commitment: all six EDL outcomes at their real
                 probabilities, one price posted before the day starts

THE FLEET
The standard assets: ``config.CONFIG["CAP"]`` as it stands -- the 400 kW genset
and no microgrid PV, the case study every other module is built on. The 10%
point is therefore the genset-only row of
:mod:`stochastic_commitment_analysis`'s fleet comparison and the 0 kWp point of
:mod:`asset_sweep`. ``--pv`` / ``--dg`` sweep a different fleet.

THE SPAN
    span  = commitment - no supply           [$/kWh, or $/day]
    span% = 100 * span / (no supply)         [% of the EDL-off day]

WHAT TO LOOK FOR
The load shift enters on the VOLUME side: it lets the village move load between
hours, and under an EDL window that means moving it INTO the hours EDL serves
and out of the hours the microgrid would have sold into -- EDL costs the
village 0.10 $/kWh against the microgrid's ~0.66.

With the standard fleet the no-supply end should not move at all. Every kWh the
genset serves costs the same 0.50 $/kWh whatever the hour, and with EDL off and
a flat price what the village buys is fixed by the price alone -- so shifting
load between hours changes neither the microgrid's revenue nor its cost, as long
as the 400 kW genset never binds. (The village is then indifferent to how it
shifts, and the optimistic MPEC resolves that tie for the leader, but with one
cost in every hour there is nothing for the leader to gain from it.) The whole
span is then the stochastic commitment's response to the village's flexibility,
which is what makes this sweep cleaner to read than one with PV, where the
no-supply end moves too.

The shift ``dd_t`` is first stage in the stochastic commitment: the village
commits one shift profile across all six EDL outcomes, just as the leader
commits one price.

RUNTIME
Wider bands are a harder MILP -- the shift bounds' complementarity binaries stop
being slack -- so solve times climb with ``shift_frac``. With 400 kWp of PV the
50% commitment solve has exceeded 1800 s (summer); how hard the genset-only
fleet is at 50% is untested. A point that aborts is named in the console, in
the figure caption and in the workbook, not silently left out; raise
--time-limit if one does.

Outputs, under :data:`config.OUTPUT_DIR`:

    loadshift_sweep.xlsx         one row per (season, load shift) with both
                                 spans, and every solved run in full
    loadshift_sweep_price.png    price span against the load shift: absolute |
                                 % of the no-supply run
    loadshift_sweep_profit.png   profit span against the load shift, the same
                                 two panels

Usage:
    python loadshift_sweep.py                        # 3 seasons x 5 shifts
    python loadshift_sweep.py --seasons winter
    python loadshift_sweep.py --pv 400               # with the PV build
    python loadshift_sweep.py --time-limit 3600
"""

import argparse

import config
# All the machinery is the asset sweep's, so the two sweeps' figures and
# workbooks are built the same way and read side by side.
from asset_sweep import SweepAxis, run_sweep
from stochastic_commitment_analysis import TIME_LIMIT_S, Build

# The load-shift values swept, as fractions of each hour's demand. [ASSUMPTION]
# 10% is config's standard. The sweep stops at 30%: above it the commitment
# solves are the slowest in the project (tens of minutes each at 40-50%), and
# 5% steps still give the range five points. --shift-max widens it.
SHIFT_MIN: float = 0.10
SHIFT_MAX: float = 0.30
SHIFT_STEP: float = 0.05

# The standard assets, read from config rather than restated, so the sweep
# follows the case study if its fleet is ever changed there.
DG_STD: float = config.CONFIG["CAP"]["DG"]
PV_STD: float = config.CONFIG["CAP"]["PV"]

SHIFT_AXIS = SweepAxis(
    key="shift_pct", name="load shift", unit="%",
    label="Load-shift capacity  [% of each hour's demand]",
    driver="the load shift",
    columns=(("shift_pct", "load shift [%]"), ("dg_kw", "DG [kW]"),
             ("pv_kwp", "PV [kWp]")))


def shift_levels(lo: float = SHIFT_MIN, hi: float = SHIFT_MAX,
                 step: float = SHIFT_STEP) -> list[float]:
    """``lo``, ``lo + step``, ... up to and including ``hi``, as fractions."""
    n = int(round((hi - lo) / step))
    # Rounded, so 0.1 + 2*0.1 is 0.3 and not 0.30000000000000004 -- the value
    # is also the model's shift bound and the build's cache key.
    return [round(lo + k * step, 6) for k in range(n + 1)]


def _fleet(dg_kw: float, pv_kwp: float) -> str:
    return (f"{dg_kw:.0f} kW DG"
            + (f" + {pv_kwp:.0f} kWp PV" if pv_kwp > 0.0 else ", no PV"))


def build_at(frac: float, dg_kw: float = DG_STD,
             pv_kwp: float = PV_STD) -> Build:
    """The fixed fleet with the village allowed to shift ``frac`` of each
    hour's demand."""
    return Build(f"shift{100 * frac:g}",
                 f"{_fleet(dg_kw, pv_kwp)}, {100 * frac:.0f}% load shift",
                 dg_kw, pv_kwp, frac)


def main(seasons: list[str] | None = None, levels: int | None = None,
         shift_min: float = SHIFT_MIN, shift_max: float = SHIFT_MAX,
         shift_step: float = SHIFT_STEP, dg_kw: float = DG_STD,
         pv_kwp: float = PV_STD, time_limit: float = TIME_LIMIT_S) -> None:
    seasons = seasons or list(config.SEASONS.keys())
    fracs = shift_levels(shift_min, shift_max, shift_step)
    # The swept value is carried in percent, matching the summary rows'
    # ``shift_pct``, so the points, the axis and the workbook agree.
    points = [(100.0 * f, build_at(f, dg_kw, pv_kwp)) for f in fracs]
    fleet = _fleet(dg_kw, pv_kwp)
    std = ("standard assets" if (dg_kw, pv_kwp) == (DG_STD, PV_STD)
           else "non-standard fleet")
    subtitle = (f"{fleet} ({std}), load shift {100 * fracs[0]:.0f}–"
                f"{100 * fracs[-1]:.0f}%"
                + (f" in {100 * shift_step:.0f}% steps" if len(fracs) > 1
                   else "")
                + "; one line per season")
    run_sweep(seasons, points, SHIFT_AXIS, subtitle, "loadshift_sweep",
              levels, time_limit, f"fleet: {fleet} ({std}), swept load shift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep the village's load-shift capacity and report the "
                    "no-supply-to-stochastic-commitment span of the leader's "
                    "price and profit, per season, on the standard assets.")
    parser.add_argument("--seasons", nargs="+",
                        default=list(config.SEASONS.keys()),
                        choices=list(config.SEASONS.keys()),
                        help="Seasons to sweep (default: all three).")
    parser.add_argument("--shift-min", type=float, default=SHIFT_MIN,
                        metavar="FRAC",
                        help=f"Smallest load shift, as a fraction of each "
                             f"hour's demand (default {SHIFT_MIN:g}).")
    parser.add_argument("--shift-max", type=float, default=SHIFT_MAX,
                        metavar="FRAC",
                        help=f"Largest load shift (default {SHIFT_MAX:g}).")
    parser.add_argument("--shift-step", type=float, default=SHIFT_STEP,
                        metavar="FRAC",
                        help=f"Load-shift step (default {SHIFT_STEP:g}).")
    parser.add_argument("--pv", type=float, default=PV_STD, metavar="KWP",
                        help=f"PV capacity of the fixed fleet in kWp (default "
                             f"{PV_STD:g}, config's standard assets).")
    parser.add_argument("--dg", type=float, default=DG_STD, metavar="KW",
                        help=f"Diesel capacity in kW (default {DG_STD:g}, "
                             f"config's standard assets).")
    parser.add_argument("--levels", type=int, default=None,
                        help="Re-space the price menu to this many levels; "
                             "default keeps config's.")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_S,
                        metavar="S",
                        help=f"Solver time limit per solve in seconds (default "
                             f"{TIME_LIMIT_S:g}). Wide shift bands are the "
                             f"hardest solves in the project.")
    args = parser.parse_args()
    if not 0.0 < args.shift_min <= args.shift_max <= 1.0:
        parser.error("need 0 < --shift-min <= --shift-max <= 1: the shift is a "
                     "fraction of each hour's demand.")
    if args.shift_step <= 0.0:
        parser.error("--shift-step must be positive.")
    if args.pv < 0.0 or args.dg <= 0.0:
        parser.error("--pv must be >= 0 and --dg positive.")
    main(args.seasons, args.levels, args.shift_min, args.shift_max,
         args.shift_step, args.dg, args.pv, args.time_limit)
