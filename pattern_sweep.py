"""
How the leader's price and profit answer WHEN EDL supplies, and on what it rests.

:mod:`stochastic_analysis` sweeps the leader's belief in each of the six NAMED
outcomes. This module holds EDL's daily ENERGY fixed and moves only the hour of
day it arrives, over two synthetic patterns:

    peak       EDL on the highest-demand hours. Evening and night in every
               season -- summer's peak is 22-05 h, because rooftop PV flattens
               its midday -- so 0-2% of the day's sun falls inside it.
    offpeak    EDL on the lowest-demand hours, which are the sunny ones: 49%
               (summer), 58% (fall/spring), 83% (winter) of the day's sun.

:mod:`plot_scenarios` draws both patterns against the demand curve they are
ranked on, at no solver cost, and is the figure to read first.

WHAT IS REPORTED: THE SPAN AGAINST NO SUPPLY
Everything here is a pattern's distance from the run that is identical to it
except that EDL never arrives -- same season, same fleet, same load shift:

    price span  = p(pattern) - p(no supply)     [$/kWh, and in menu steps]
    profit span = E[profit](pattern) - profit(no supply)   [$/day, and in %]

The reference is re-solved at every point of every sweep, so a span never
carries the effect of the axis it is drawn against. What load shifting does to
the leader on its own stays in the reference and out of the span; what is left
is EDL timing alone.

THE TWO SWEEPS, AND THE FIGURES THEY SHARE

    shift_dg    DIESEL-ONLY fleet (config's 400 kW genset, no PV), across load
                shift 0 / 10 / 20%, at P(pattern) = 1.
    shift_pv    the same three shift levels with 200 kWp of PV added, so the
                pair isolates what the PV does.

The sweep over the leader's belief in the pattern lives in
:mod:`stochastic_sweep`, which reuses :func:`sweep` and :func:`solve_at_belief`
from here.

They are drawn as range lines in the language of :mod:`stochastic_analysis`: the
quantity on x, the series on y, each line one sweep collapsed to its two ends
with a tick at every point between. TWO figures, one per pattern, each with a
price panel and a profit panel sharing their rows and all three seasons in
both; the two fleets sit together because the contrast between them is the
argument. The axis is the LEVEL, not the span, so no supply is a POINT marked
on each line -- that row's own reference -- and the end labels give the span
from it in percent.

TWO CHANNELS, OPPOSITE SIGNS
The first two figures exist because the price responds to timing through two
mechanisms that push against each other, and only a fleet contrast separates
them:

  TECHNOLOGY (needs PV).  With one flat price and suppression on the uncovered
    load, p* ~ 0.5 * (P_ref / alpha1 + cbar), where cbar is the marginal cost of
    the hours the microgrid still serves. Timing cannot touch the first term, so
    it moves the price only by changing WHICH hours dominate cbar:
        peak     takes dark, diesel-served load out of the market and leaves
                 the PV output untouched -> cbar falls -> price FALLS
        offpeak  lands on the PV niche and displaces the PV itself, leaving a
                 diesel remainder -> cbar rises -> price RISES
    With a diesel-only fleet every hour costs the same, cbar cannot move, and
    this channel is exactly zero -- which is what the DG figure shows.

  SHIFT INTO SLACK EDL (needs shift > 0).  offpeak puts EDL's flat energy on
    hours whose demand is lower than it, so its cap goes SLACK there (571 kWh
    unused in winter, against 26 kWh under peak). The village can move load into
    those hours and buy it at P_EDL rather than from the microgrid, and the
    leader shaves its price in response. peak's cap binds in almost every hour
    it covers, so there is nothing to shift into.

So at 0% shift on a diesel-only fleet both spans must be ZERO -- the price is
then the closed form 0.5 * (P_ref/alpha1 + c_diesel) whatever EDL does with its
hours -- and each figure adds one channel back.

THE HOURS, TAKEN OVER COMPLETELY                                 [ASSUMPTION]
A pattern names 8 hours, and in them EDL's cap is M_phys: it covers whatever is
there, including whatever the village shifts in. Outside them there is nothing.

This is deliberately NOT equal energy. An earlier version handed both patterns
the case study's own daily energy and spread it over their hours, so the cap
bound inside the window and timing was the only difference between them. With
the cap released, `peak` takes over the highest-demand hours and `offpeak` the
lowest, so peak delivers far more kWh, and the question the pair answers is
which hours EDL taking over entirely costs the leader more -- not what pure
timing does at a fixed energy. Every figure reports each pattern's covered
share of demand, since the two are no longer comparable on energy.

It also hands load shifting a lever it did not have: with the cap gone, every
kWh the village moves into a covered hour is bought at P_EDL instead of from
the microgrid, and nothing exhausts that. The load shift sweep is where that
shows up.

Outputs, under :data:`config.OUTPUT_DIR`:

    pattern_peak.png            the peak pattern across load shift: the
                                leader's price and its profit, both fleets,
                                all three seasons
    pattern_offpeak.png         the same for offpeak
    pattern_sweep.xlsx          every run and every span behind them

Usage:
    python pattern_sweep.py
    python pattern_sweep.py --figures shift_dg shift_pv
    python pattern_sweep.py --seasons winter --shifts 0 0.1 0.2 0.3 0.4
    python pattern_sweep.py --replot          # redraw, no solving
"""

import argparse
import textwrap
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook, load_workbook
from pyomo.environ import TerminationCondition, value

import config
from asset_sweep import (AXIS, INK, MUTED, RULE, SEASON_COLOR, SEASON_LABEL,
                         _layout_issues, _write_rows, menu_step)
from data.input_data import load_data
from model.bilevel_mpec import BilevelMPEC
from run import build_instance
from scenarios import availability_shape, uncapped_availability
from stochastic_commitment_analysis import (DG_KW, TIME_LIMIT_S, Build,
                                            build_settings)

_OK = (TerminationCondition.optimal, TerminationCondition.feasible)
_TOL = 1e-6

PAT = "pattern"                   # the scenario the swept belief is put on
RESIDUAL = "none"                 # what carries 1 - w

PATTERNS: tuple[str, ...] = ("peak", "offpeak")
NONE = "no supply"

# The PV added in the shift_pv sweep (and in stochastic_sweep).   [ASSUMPTION]
# config carries none, and without PV the hours do not differ in cost, so the
# technology channel above is switched off entirely.
PV_KWP: float = 200.0

# Load shift levels of the two figures, as a fraction of hourly demand.
# 0% is the analytical anchor (no shift lever at all), 10% is config's own
# standard, 20% a village twice as flexible.                      [ASSUMPTION]
SHIFTS: tuple[float, ...] = (0.0, 0.10, 0.20)

FIELDS = [
    ("season",       "season"),
    ("fleet",        "fleet"),
    ("shift",        "load shift [-]"),
    ("series",       "series"),
    ("w",            "P(pattern) [-]"),
    ("w_none",       "P(no supply) [-]"),
    ("share",        "demand in the covered hours / daily demand [-]"),
    ("hours",        "hours supplied"),
    ("sun_share",    "% of the day's sun in those hours"),
    ("x",            "leader price [x P_ref]"),
    ("p",            "leader price [$/kWh]"),
    ("p_ref",        "price, no supply [$/kWh]"),
    ("d_p",          "price span [$/kWh]"),
    ("d_p_pct",      "price span [%]"),
    ("d_p_steps",    "price span [menu steps]"),
    ("profit",       "E[leader profit] [$/day]"),
    ("profit_ref",   "profit, no supply [$/day]"),
    ("d_profit",     "profit span [$/day]"),
    ("d_profit_pct", "profit span [%]"),
    ("qMGO",         "E[microgrid sales] [kWh]"),
    ("qPG",          "E[EDL delivered] [kWh]"),
    ("qsupp",        "E[suppressed demand] [kWh]"),
    ("qcri",         "E[unserved] [kWh]"),
    ("termination",  "termination"),
]


# --------------------------------------------------------------------------
# Solving
# --------------------------------------------------------------------------
def covered_share(season: str, pattern: str) -> float:
    """
    The share of the day's demand that falls in the hours EDL takes over.

    With an unlimited cap in those hours this replaces the old energy share as
    the size of the intervention: EDL covers them completely, so what it
    delivers is the demand that sits there (before the village shifts anything
    into them), not a quantity the pattern was handed.
    """
    data = build_instance(season)
    T, D = data["T"], data["D"]
    shape = availability_shape(season, pattern, D)
    covered = sum(D[t] for t in T if shape[t] > 0.0)
    return covered / sum(D.values())


def solve_at_belief(season: str, Qbar: dict, w: float, opts: dict,
                    levels: int | None) -> dict | None:
    """
    Re-optimize the leader believing ``Qbar`` arrives with probability ``w``,
    and nothing at all with probability 1 - w.

    At w = 1 the residual is dropped instead of being carried at probability
    zero, which would leave its recourse variables out of the objective. w = 0
    is never passed here: it is the no-supply run, solved as ``Qbar = 0`` at
    w = 1 and shared by both patterns.
    """
    data = dict(build_instance(season))
    T = data["T"]
    certain = w >= 1.0 - _TOL
    S = [PAT] if certain else [PAT, RESIDUAL]
    prob = {PAT: 1.0} if certain else {PAT: w, RESIDUAL: 1.0 - w}

    data["S"] = S
    data["prob"] = prob
    data["QbarPG"] = {(t, s): (Qbar[t] if s == PAT else 0.0)
                      for t in T for s in S}
    if levels:
        P_ref, Phat = data["P"][T[0]], data["Phat"][T[0]]
        data["Pi"] = [P_ref + (Phat - P_ref) * k / (levels - 1)
                      for k in range(levels)]

    mpec = BilevelMPEC(data)
    m = mpec.build()
    solver_name, res = mpec.solve("gurobi", options=opts)
    if solver_name is None:
        raise RuntimeError(
            "Gurobi not available (check the license and that gurobipy is "
            "installed).")

    tc = res.solver.termination_condition
    if tc not in _OK:
        print(f"    not solved ({tc})")
        return None

    def exp_sum(var):
        return sum(prob[s] * value(var[t, s]) for t in T for s in S)

    P_ref = data["P"][T[0]]
    qMGO = exp_sum(m.qMGO)
    p = value(m.p[T[0]])
    degenerate = qMGO <= _TOL        # sells nothing: its price is arbitrary

    return {
        "w": w,
        "w_none": 0.0 if certain else 1.0 - w,
        "p": None if degenerate else p,
        "x": None if degenerate else p / P_ref,
        "profit": value(m.profit),
        "qMGO": qMGO,
        "qPG": exp_sum(m.qPG),
        "qsupp": exp_sum(m.qsupp),
        "qcri": exp_sum(m.qcri),
        "termination": str(tc),
    }


def _span(rec: dict, ref: dict, step: float) -> None:
    """
    Fill in a record's span against the run it is compared with.

    The reference is that season's no-supply run under the SAME fleet and the
    SAME load shift, so the span is EDL timing alone.
    """
    rec["p_ref"], rec["profit_ref"] = ref.get("p"), ref.get("profit")
    for key, dk in (("p", "d_p"), ("profit", "d_profit")):
        a, b = ref.get(key), rec.get(key)
        rec[dk] = None if a is None or b is None else b - a
        rec[f"{dk}_pct"] = (None if rec[dk] is None or not a
                            else 100.0 * rec[dk] / a)
    rec["d_p_steps"] = None if rec["d_p"] is None else rec["d_p"] / step


def _fleet_label(dg_kw: float, pv_kwp: float) -> str:
    return (f"{dg_kw:.0f} kW DG" if pv_kwp <= 0.0
            else f"{dg_kw:.0f} kW DG + {pv_kwp:.0f} kWp PV")


def sweep(seasons: list[str], *, shifts: list[float], weights: list[float],
          dg_kw: float, pv_kwp: float, opts: dict, levels: int | None,
          step: float, cache: dict) -> list[dict]:
    """
    Every point of one sweep: both patterns and their reference, per season.

    One call is ONE figure. ``shifts`` and ``weights`` are both lists because
    the two sweep kinds differ only in which of them has length > 1: the shift
    figures here fix the belief at P(pattern) = 1, :mod:`stochastic_sweep`
    fixes the shift and sweeps the belief.
    ``cache`` is shared across calls, so a point the figures have in common --
    the 10% shift reference of the PV fleet, for instance -- is solved once.
    """
    bundle = load_data()
    fleet = _fleet_label(dg_kw, pv_kwp)
    rows = []

    for season in seasons:
        data = build_instance(season)
        D, T = data["D"], data["T"]
        phi = bundle["cap_factor"][season]
        yield_day = sum(phi.values()) or 1.0
        shares = {p: covered_share(season, p) for p in PATTERNS}
        print(f"[{season}] {fleet}; EDL covers its hours completely — "
              + ", ".join(f"{p} {100 * shares[p]:.1f}% of demand"
                          for p in PATTERNS))

        def solved(Qbar, series, w, shift, share=0.0):
            key = (season, dg_kw, pv_kwp, shift, series, w)
            if key not in cache:
                build = Build("fleet", fleet, dg_kw, pv_kwp, shift)
                with build_settings(build):
                    cache[key] = solve_at_belief(season, Qbar, w, opts, levels)
            rec = cache[key]
            if rec is None:
                return None
            hours = sorted(t for t in T if Qbar[t] > _TOL)
            rec = dict(rec) | {
                "season": season, "fleet": fleet, "shift": shift,
                "series": series, "share": share if hours else 0.0,
                "hours": len(hours),
                "sun_share": 100.0 * sum(phi[t] for t in hours) / yield_day}
            px = "   -  " if rec["p"] is None else f"{rec['p']:.4f}"
            print(f"    shift {100 * shift:3.0f}%  {series:>10} P={w:.2f}  "
                  f"p={px}  E[profit]={rec['profit']:8.2f}  "
                  f"E[qMGO]={rec['qMGO']:7.1f}  E[qPG]={rec['qPG']:7.1f}")
            return rec

        for shift in shifts:
            # The reference: the same run with EDL off all day. Re-solved at
            # every shift level, so the span never carries the shift's own
            # effect on the leader.
            ref = solved({t: 0.0 for t in T}, NONE, 1.0, shift)
            if ref is None:
                print(f"[{season}] no-supply reference did not solve at shift "
                      f"{100 * shift:.0f}% — skipped")
                continue
            ref["w"], ref["w_none"] = 0.0, 1.0
            _span(ref, ref, step)
            point_rows = [ref]

            for pattern in PATTERNS:
                for w in weights:
                    if w <= _TOL:
                        continue        # the w -> 0 limit IS the reference
                    rec = solved(uncapped_availability(season, pattern, D),
                                 pattern, w, shift, shares[pattern])
                    if rec is not None:
                        _span(rec, ref, step)
                        point_rows.append(rec)

            rows.extend(point_rows)
    return rows


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def report(rows: list[dict], xkey: str, xname: str, step: float) -> None:
    """The spans of one figure, as a table per season."""
    seasons = [s for s in config.SEASONS if any(r["season"] == s for r in rows)]
    for season in seasons:
        sel = [r for r in rows if r["season"] == season]
        print(f"\n[{season}] span against no supply, same fleet and shift")
        print(f"  {xname:>7} {'series':>10} {'price':>8} {'Δ price':>9} "
              f"{'steps':>6} {'profit':>9} {'Δ profit':>9} {'Δ %':>7}")
        for r in sorted(sel, key=lambda r: (r[xkey] if r[xkey] is not None
                                            else -1.0,
                                            r["series"])):
            x = "  -  " if r[xkey] is None else f"{100 * r[xkey]:5.0f}%"
            def f(v, spec):
                return "-" if v is None else format(v, spec)
            print(f"  {x:>7} {r['series']:>10} {f(r['p'], '8.4f'):>8} "
                  f"{f(r['d_p'], '+9.4f'):>9} "
                  f"{f(r['d_p_steps'], '+6.1f'):>6} "
                  f"{f(r['profit'], '9.2f'):>9} "
                  f"{f(r['d_profit'], '+9.2f'):>9} "
                  f"{f(r['d_profit_pct'], '+7.1f'):>7}")
        for x in sorted({r[xkey] for r in sel
                         if r[xkey] is not None and r["series"] in PATTERNS}):
            d = {r["series"]: r["d_p"] for r in sel
                 if r[xkey] == x and r["series"] in PATTERNS}
            if d.get("peak") is not None and d.get("offpeak") is not None:
                gap = d["offpeak"] - d["peak"]
                print(f"    at {xname} {100 * x:.0f}%: offpeak − peak = "
                      f"{gap:+.4f} $/kWh ({gap / step:+.1f} menu steps)")


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
# The figures speak the range-line language of :mod:`stochastic_analysis`: the
# quantity on x, the series on y, each line a whole sweep collapsed to its two
# ends with a tick at every point between.
#
# The axis carries the LEVEL -- the price the leader posts, the profit it makes
# -- not the span. Every line therefore lives in the same units whatever it
# sweeps, so load shift and belief can share a panel and be read against each
# other, and the reference is not an axis convention but a POINT: the no-supply
# run of that row's own season and fleet, marked on the line it belongs to. The
# span is then the distance from that mark to the line's ends, which the end
# labels give in percent alongside the level.
#
# Two figures, one per pattern, each carrying both sweeps and all three
# seasons in a price panel and a profit panel that share their rows. Read
# across a row, the price panel's short travel against the profit panel's long
# one is the module's claim.
@dataclass(frozen=True)
class Sweep:
    """One sweep, as the figures need to name it."""
    key: str          # figure name, and the workbook sheet it is written to
    xkey: str         # the swept field of a row
    label: str        # what distinguishes this sweep's rows, for the row label
    anchored: bool    # sweep starts at the no-supply run itself


SHIFT_SWEEPS: tuple[Sweep, ...] = (
    Sweep("shift_dg", "shift", "DG only", False),
    Sweep("shift_pv", "shift", "+200 kWp PV", False),
)

# The two panels of every figure.
QUANTITIES = (
    {"key": "p", "ref": "p_ref", "pct": "d_p_pct",
     "axis": "Leader price   [$ / kWh]", "fmt": "{:.4f}", "span": "{:+.4f}"},
    {"key": "profit", "ref": "profit_ref", "pct": "d_profit_pct",
     "axis": "Leader profit   [$ / day]", "fmt": "{:,.1f}",
     "span": "{:+,.1f}"},
)

REF_COLOR = "#333333"


def _references(recs: list[dict]) -> list[dict]:
    """
    The no-supply runs of one row worth drawing, one bar each.

    A sweep re-solves its reference at every point, so it normally hands the
    same run back several times over. Whether two of them are the SAME run is
    decided on the OBJECTIVE, because that is what the solver's gap bounds: two
    references whose profit agrees within MIPGap are one reference, and the
    price to keep is the one the better objective justifies.

    That last part matters. Profit is quadratic in the price, so near its
    optimum a whole menu step costs almost nothing -- 0.0095 $/day, 0.007%, in
    the fall/spring diesel-only run -- and the solver may stop a step off and
    still be inside its gap. Deciding on the price instead would draw that as a
    second reference, which is a solver artifact, not a reference that moved.
    """
    pairs = sorted(((r["profit_ref"], r.get("p_ref")) for r in recs
                    if r.get("profit_ref") is not None), key=lambda t: t[0])
    out: list[dict] = []
    for profit, price in pairs:
        if out and abs(profit - out[-1]["profit"]) <= max(
                1e-9, config.CONFIG["MIPGap"] * abs(profit)):
            out[-1] = {"profit": profit, "p": price}   # the better solve
            continue
        out.append({"profit": profit, "p": price})
    return out


def _range(recs: list[dict], xkey: str, q: dict, anchored: bool,
           refs: list[float]) -> dict | None:
    """
    One sweep collapsed to a range: its two ends, and every point between.

    The relative reading travels with the level, so an end can be labelled in
    both without the panel carrying a second scale. ``refs`` are the row's
    no-supply levels in this quantity, already collapsed by :func:`_references`;
    ``anchored`` sweeps start at the no-supply run itself -- P(pattern) = 0 IS
    that run -- so it is prepended as the line's own first point rather than
    solved twice.
    """
    key, pct = q["key"], q["pct"]
    pts = sorted(((r[xkey], r[key], r.get(pct)) for r in recs
                  if r.get(key) is not None and r.get(xkey) is not None),
                 key=lambda t: t[0])
    if not pts:
        return None
    xs = [t[0] for t in pts]
    vs = [t[1] for t in pts]
    ps = [t[2] for t in pts]
    if anchored and xs[0] > _TOL and refs:
        xs, vs, ps = [0.0] + xs, [refs[0]] + vs, [0.0] + ps
    return {"xs": xs, "vs": vs, "ps": ps, "start": vs[0], "end": vs[-1],
            "start_pct": ps[0], "end_pct": ps[-1], "n": len(vs), "refs": refs}


def _rows(per_figure: dict[str, list[dict]], specs: list[dict]) -> list[dict]:
    """
    One y row per spec, carrying both quantities' ranges.

    A spec names a season, a sweep and a pattern; the colour is the season's,
    as everywhere else in the project, and the label carries the sweep.
    """
    rows = []
    for spec in specs:
        recs = [r for r in per_figure.get(spec["sweep"].key, [])
                if r["season"] == spec["season"]
                and r["series"] == spec["pattern"]]
        if not recs:
            continue
        # One reference per row, decided once on the objective, so the price
        # panel and the profit panel mark the same no-supply run.
        refs = _references(recs)
        ranges = {q["key"]: _range(recs, spec["sweep"].xkey, q,
                                   spec["sweep"].anchored,
                                   [ref[q["key"]] for ref in refs
                                    if ref.get(q["key"]) is not None])
                  for q in QUANTITIES}
        if all(r is None for r in ranges.values()):
            continue
        rows.append({"label": spec["label"], "ranges": ranges,
                     "season": spec["season"], "pattern": spec["pattern"],
                     "color": SEASON_COLOR.get(spec["season"], "#444444")})
    return rows


def _draw(ax, rows: list[dict], ys: list[int], quantity: dict) -> None:
    """One panel: a range line per row, with no supply marked on the line."""
    for row, y in zip(rows, ys):
        rng = row["ranges"].get(quantity["key"])
        if rng is None:
            continue
        color = row["color"]
        start, end = rng["start"], rng["end"]
        lo, hi = min(rng["vs"]), max(rng["vs"])

        # Where the sweep leaves the interval its two ends span, the excursion
        # goes behind the line, so the line is never read as the full range.
        if hi - lo > abs(end - start) + _TOL:
            ax.hlines(y, lo, hi, color=color, alpha=0.25, linewidth=9, zorder=1)
        ax.hlines(y, start, end, color=color, linewidth=2.6, zorder=2)
        # The points between the two ends -- the 10% shift, on a three-level
        # sweep -- as full dots, so each solved level has its own marker.
        if rng["n"] > 2:
            ax.plot(rng["vs"][1:-1], [y] * (rng["n"] - 2), linestyle="none",
                    marker="o", markersize=7.0, markerfacecolor=color,
                    markeredgecolor="#222222", markeredgewidth=0.7, zorder=4)

        # No supply, on the line it belongs to: the row's own reference, so the
        # distance from here to an end IS that end's span. A sweep that starts
        # at the reference (belief) has it under its opening marker.
        for ref in rng["refs"]:
            ax.plot([ref], [y], marker="|", markersize=20, markeredgewidth=2.4,
                    color=REF_COLOR, zorder=3.5)

        ax.plot([start], [y], marker="o", markersize=7.5,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=2.0, zorder=4)
        ax.plot([end], [y], marker="D", markersize=7.5, markerfacecolor=color,
                markeredgecolor="#222222", markeredgewidth=0.7, zorder=4)

        # Each end in the axis unit, on the outer side of its own marker, so
        # the two never collide however short the line.
        right = end >= start
        for value, is_end in ((start, False), (end, True)):
            outward = 1 if (is_end == right) else -1
            ax.annotate(quantity["fmt"].format(value), (value, y),
                        xytext=(9 * outward, 0), textcoords="offset points",
                        ha="left" if outward > 0 else "right", va="center",
                        fontsize=8, color="#222222")

        # What the sweep moved, above the middle of the line it moved: the
        # distance between the two ends, absolute and against the no-supply
        # level the row is measured from.
        moved = end - start
        text = quantity["span"].format(moved)
        ref = rng["refs"][0] if rng["refs"] else None
        if ref:
            text += f"  ({100.0 * moved / abs(ref):+.1f}%)"
        ax.annotate(text, (0.5 * (start + end), y), xytext=(0, 10),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color="#222222")

    ax.set_xlabel(quantity["axis"], fontsize=10, color=INK)
    ax.grid(axis="x", color=RULE, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.margins(x=0.20)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.tick_params(axis="y", length=0, labelcolor=INK, labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def plot_figure(per_figure: dict[str, list[dict]], specs: list[dict], out_path,
                *, order: list[str], title: str, levels: list[str]) -> None:
    """
    One figure: a price panel and a profit panel sharing their rows.

    A row is one series under one sweep; its line runs from the sweep's first
    solved point to its last, with a tick at every point between, and the bar on
    it is that row's own no-supply run -- so the distance from the bar to an end
    is the span, and the end labels give it in percent beside the level.
    """
    rows = _rows(per_figure, specs)
    if not rows:
        print(f"  nothing to draw — skipped {out_path}")
        return

    ys = list(range(len(rows) - 1, -1, -1))       # first spec at the top
    fig, axes = plt.subplots(1, 2, sharey=True,
                             figsize=(15.0, 1.1 * len(rows) + 3.0))
    for ax, quantity in zip(axes, QUANTITIES):
        _draw(ax, rows, ys, quantity)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r["label"] for r in rows])
    axes[0].set_ylim(-0.8, len(rows) - 0.2)

    handles = [
        plt.Line2D([], [], linestyle="none", marker="|", markersize=16,
                   markeredgewidth=2.4, color=REF_COLOR,
                   label="no supply (EDL off all day)"),
        plt.Line2D([], [], linestyle="none", marker="o", markersize=7.5,
                   markerfacecolor="white", markeredgecolor="#444444",
                   markeredgewidth=2.0, label=levels[0]),
    ] + ([plt.Line2D([], [], linestyle="none", marker="o", markersize=7.0,
                     markerfacecolor="#444444", markeredgecolor="#222222",
                     markeredgewidth=0.7, label=" / ".join(levels[1:-1]))]
         if len(levels) > 2 else []) + [
        plt.Line2D([], [], linestyle="none", marker="D", markersize=7.5,
                   markerfacecolor="#444444", markeredgecolor="#222222",
                   label=levels[-1]),
    ] + [plt.Line2D([], [], color=SEASON_COLOR[s], linewidth=2.6,
                    label=SEASON_LABEL.get(s, s)) for s in order]

    fig_w = fig.get_size_inches()[0]
    ttl = fig.suptitle(textwrap.fill(title, width=int(8.2 * fig_w)),
                       x=0.01, y=0.988, ha="left", va="top", fontsize=13,
                       color=INK, fontweight="bold")
    ttl_lines = ttl.get_text().count("\n")
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.968 - 0.040 * (ttl_lines + 1)),
                     ncol=len(handles), frameon=False, fontsize=8.5)

    fig.tight_layout(rect=(0, 0.01, 1, 0.94 - 0.040 * (ttl_lines + 1)))
    for msg in _layout_issues(fig, [ttl, leg]):
        print(f"  layout: {msg}  [{out_path}]")
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_all(per_figure: dict[str, list[dict]], out_dir,
             seasons: list[str], shifts: list[float]) -> None:
    """
    Two figures: the load shift sweeps once per pattern, as range lines.

    A pattern's shift figure puts the two fleets in one panel, because the
    contrast between them IS the argument -- one fleet has a cost channel and
    the other cannot. The belief sweep has its own module,
    :mod:`stochastic_sweep`.
    """
    order = [s for s in config.SEASONS if s in seasons]
    shift_txt = (f"{' / '.join(f'{100 * s:.0f}' for s in shifts)}% load shift, "
                 f"at prob = 1")
    levels = [f"{100 * s:.0f}%" for s in shifts]
    for pattern in PATTERNS:
        where = "highest" if pattern == "peak" else "lowest"
        specs = [{"season": season, "sweep": sw, "pattern": pattern,
                  "label": f"{SEASON_LABEL.get(season, season)}\n{sw.label}"}
                 for sw in SHIFT_SWEEPS for season in order]
        plot_figure(
            per_figure, specs, out_dir / f"pattern_{pattern}.png",
            order=order, levels=levels,
            title=f"EDL on the {where}-demand hours ({pattern}): what it does "
                  f"to the leader, across {shift_txt}")



def load_rows(path) -> dict[str, list[dict]]:
    """
    Every run of a previous sweep, read back from its workbook.

    The solving is what costs -- 78 MILPs, most of an hour -- so redrawing must
    not need it. The sheet names are the figure keys, and the header row maps
    back through FIELDS, so the workbook is a complete record of a run.
    """
    wb = load_workbook(path, data_only=True)
    key_of = {label: key for key, label in FIELDS}
    out: dict[str, list[dict]] = {}
    for ws in wb:
        table = list(ws.iter_rows(values_only=True))
        if not table:
            continue
        header = [key_of.get(cell) for cell in table[0]]
        recs = []
        for row in table[1:]:
            if row[0] is None:            # the trailing caption line
                continue
            recs.append({k: v for k, v in zip(header, row) if k is not None})
        out[ws.title] = recs
    return out
# --------------------------------------------------------------------------
def save_analysis(per_figure: dict[str, list[dict]], out_path,
                  levels: int | None) -> None:
    """Every run of every figure, with its span, one sheet each."""
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in per_figure.items():
        ws = wb.create_sheet(name[:31])
        _write_rows(ws, [lbl for _, lbl in FIELDS],
                    ([r.get(k) for k, _ in FIELDS] for r in rows))
        ws.append([])
        ws.append([f"Price-menu step {menu_step(levels):.5f} $/kWh. Each "
                   f"pattern row is a two-point belief: P(pattern) = w, "
                   f"P(no supply) = 1 - w, so its quantities are expectations "
                   f"at that w. Spans are against the no-supply row of the "
                   f"SAME season, fleet and load shift."])
    wb.save(out_path)
    print(f"  wrote {out_path}")


FIGURES: tuple[str, ...] = ("shift_dg", "shift_pv")


def main(seasons: list[str] | None = None, figures: list[str] | None = None,
         shifts: list[float] | None = None, pv_kwp: float = PV_KWP,
         dg_kw: float = DG_KW, levels: int | None = None,
         time_limit: float = TIME_LIMIT_S, replot: bool = False) -> None:
    seasons = seasons or list(config.SEASONS.keys())
    figures = figures or list(FIGURES)
    shifts = sorted(shifts or SHIFTS)
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = {"MIPGap": config.CONFIG["MIPGap"], "TimeLimit": time_limit}
    step = menu_step(levels)
    cache: dict = {}
    per_figure: dict[str, list[dict]] = {}

    # Redrawing must not need the solver: the workbook of a previous run is a
    # complete record of it, and the solving is what costs.
    if replot:
        book = out_dir / "pattern_sweep.xlsx"
        per_figure = load_rows(book)
        print(f"replot from {book}: "
              + ", ".join(f"{k} {len(v)} runs" for k, v in per_figure.items()))
        plot_all(per_figure, out_dir, seasons, shifts)
        return

    n = len(seasons) * len(figures) * len(shifts) * (1 + 2)
    print(f"up to {n} solves: {len(seasons)} season(s), figures "
          f"{' '.join(figures)}; shared points are solved once")

    if "shift_dg" in figures:
        print(f"\n=== load shift sweep, diesel only "
              f"({_fleet_label(dg_kw, 0.0)}), prob = 1")
        rows = sweep(seasons, shifts=shifts, weights=[1.0], dg_kw=dg_kw,
                     pv_kwp=0.0, opts=opts, levels=levels, step=step,
                     cache=cache)
        if rows:
            per_figure["shift_dg"] = rows
            report(rows, "shift", "shift", step)

    if "shift_pv" in figures:
        print(f"\n=== load shift sweep, with PV "
              f"({_fleet_label(dg_kw, pv_kwp)}), prob = 1")
        rows = sweep(seasons, shifts=shifts, weights=[1.0], dg_kw=dg_kw,
                     pv_kwp=pv_kwp, opts=opts, levels=levels, step=step,
                     cache=cache)
        if rows:
            per_figure["shift_pv"] = rows
            report(rows, "shift", "shift", step)

    # One figure per pattern.
    plot_all(per_figure, out_dir, seasons, shifts)


    if per_figure:
        save_analysis(per_figure, out_dir / "pattern_sweep.xlsx", levels)
    else:
        print("Nothing solved — no figures or workbook written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Price and profit span of the two synthetic EDL timing "
                    "patterns (peak, offpeak) against no supply: across load "
                    "shift on a diesel-only and on a PV fleet.")
    parser.add_argument("--seasons", nargs="+",
                        default=list(config.SEASONS.keys()),
                        choices=list(config.SEASONS.keys()))
    parser.add_argument("--figures", nargs="+", default=list(FIGURES),
                        choices=list(FIGURES),
                        help=f"Figures to draw (default: both).")
    parser.add_argument("--shifts", nargs="+", type=float, default=None,
                        metavar="F",
                        help=f"Load shift levels of the two shift figures "
                             f"(default {' '.join(f'{s:g}' for s in SHIFTS)}).")
    parser.add_argument("--pv", type=float, default=PV_KWP, metavar="KWP",
                        help=f"PV of the shift_pv figure (default "
                             f"{PV_KWP:g}). shift_dg is always diesel only.")
    parser.add_argument("--dg", type=float, default=DG_KW, metavar="KW",
                        help=f"Diesel capacity (default {DG_KW:g}).")
    parser.add_argument("--levels", type=int, default=None,
                        help="Re-space the price menu to this many levels.")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_S,
                        metavar="S", help="Solver time limit per solve.")
    parser.add_argument("--replot", action="store_true",
                        help="Redraw the figures from the existing workbook "
                             "without solving anything.")
    args = parser.parse_args()
    if args.shifts and not all(0.0 <= s <= 1.0 for s in args.shifts):
        parser.error("--shifts must lie in [0, 1]: it is a fraction of D_t.")
    if args.pv < 0.0 or args.dg <= 0.0:
        parser.error("--pv must be >= 0 and --dg positive.")
    main(args.seasons, args.figures, args.shifts, args.pv, args.dg,
         args.levels, args.time_limit, args.replot)
