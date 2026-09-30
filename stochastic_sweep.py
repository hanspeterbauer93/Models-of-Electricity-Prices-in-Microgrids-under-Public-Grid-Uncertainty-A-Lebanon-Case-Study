"""
How the leader's BELIEF in an EDL timing pattern moves its price, and what
information is worth to it.

Moved out of :mod:`pattern_sweep`, whose solve it reuses. EDL takes over the 8
highest- (peak) or the 8 lowest-demand (offpeak) hours of the season completely
-- its cap there is M_phys -- or does not come at all, with

    prob(pattern) = w ,   prob(no supply) = 1 - w ,   w in WEIGHTS

THREE INFORMATION CASES, at every w

    balanced             neither side knows the outcome. The leader posts one
                         price and the village commits one load-shift profile,
                         both against prob; only the dispatch follows the
                         outcome. The committed run of the stochastic model.
    leader's advantage   the leader learns the outcome before it prices and
                         posts that outcome's own price (``leaderInformed`` in
                         :class:`model.bilevel_mpec.BilevelMPEC`); the village
                         still schedules against prob, knowing only the
                         contingent tariff "p_s if outcome s".
    follower uninformed  the leader knows the outcome; the village plans for no
                         supply, where it has no reason to move load, so it
                         does not shift at all. Each outcome is then its own
                         deterministic run at 0% load shift, and the expected
                         profit is w * profit(pattern) + (1 - w) * profit(none).

At w = 0 all three are the no-supply run. At w = 1 balanced and leader's
advantage are the same certain run; follower uninformed is that run without the
load shift. The fourth combination -- the village informed, the leader not --
is left out: the village learning EDL's schedule before the operator does is
not a realistic setting.

Leader's advantage is not guaranteed to beat balanced: a contingent tariff also
changes how the village schedules. Any case where it falls short is printed.

THE LEADER'S EVPI
What it is worth to the LEADER to learn the outcome before it prices, with the
village's behaviour held as it is -- one committed price against one price per
outcome:

    village schedules against prob   EVPI = leader's advantage - balanced
    village plans for no supply      EVPI = follower uninformed - naive

where "naive" is the committed run at 0% load shift: the leader prices against
prob and the village does not shift (the naive_* sheets). Balanced and leader's
advantage are the two sides of the first pair, so they share one EVPI; follower
uninformed has the second. The second is non-negative by construction: without
a shift the village has no first-stage decision, each outcome's response is
independent, and one price can never beat a price per outcome. The first is
not guaranteed -- the village's first-stage shift reacts to the tariff menu --
so any negative value is printed.

THE GAIN FROM RE-PRICING FOR AN INFORMED VILLAGE
The village learns the outcome before it schedules; the leader does not, and
still posts ONE price. Two runs of that setting:

    held(w)      = w * profit(pattern | p*) + (1 - w) * profit(none | p*)
                   the balanced price p* kept as posted -- each term a certain
                   run with the menu reduced to p* (the follower_* sheets)
    adjusted(w)  = the leader re-optimizes its one price knowing the village is
                   informed (``followerInformed`` in
                   :class:`model.bilevel_mpec.BilevelMPEC`; adjusted_* sheets)

    gain(w)      = adjusted(w) - held(w)                             [$/day]

p* is one of the prices the adjusted run can choose, so the gain is
non-negative up to the solver gap; any value below that is printed. It is zero
at w = 0 and 1. The workbook also keeps VFIL = balanced - held, the change in
the leader's profit from informing the village at the price it already posted.

THE VALUE OF THE STOCHASTIC SOLUTION
What planning against the two-point belief is worth to the leader, over
planning against its mean:

    EV(w)    the certain run on the MEAN day: in the pattern's hours EDL's cap
             is w * (1 + shift_frac) * D_t, the expected delivery of a pattern
             that covers the hour and everything shifted into it. The literal
             mean cap, w * M_phys, is still unlimited and would make EV the
             certain pattern run at every w. At w = 1 the mean cap covers the
             whole hour, so EV is the certain run itself.
    EEV(w)   the balanced model at belief w with the menu reduced to EV's
             price: the leader posts the mean-day price, the village schedules
             against prob as in balanced
    VSS(w)   = balanced(w) - EEV(w)                                  [$/day]

EV's price is on balanced's menu, so VSS is non-negative up to the solver gap;
any value below that is printed. It is zero at w = 0 and 1, and zero wherever
the two prices coincide (no EEV solve is needed there). The vss_* sheets.

TWO SWEEPS

    belief_dg   config's 400 kW genset alone, at BELIEF_SHIFT load shift
    belief_pv   the same genset + pattern_sweep.PV_KWP of PV

Colour carries the PATTERN in both figures, because that is the comparison; the
season is a shade of it, summer lightest and winter darkest, which is also the
order of their sun. In the information figure the line style carries the case.

Outputs, under :data:`config.OUTPUT_DIR`:

    stochastic_sweep_price.png        the balanced committed price against
                                      prob: diesel only left, with PV right
    stochastic_sweep_information.png  the leader's expected profit in the three
                                      cases against prob: peak on top, offpeak
                                      below; diesel only left, with PV right
    stochastic_sweep_evpi.png         the leader's EVPI for both village
                                      behaviours, the same panels
    stochastic_sweep_repricing.png    the gain from re-pricing for an informed
                                      village, the same panels
    stochastic_sweep_vss.png          the value of the stochastic solution,
                                      the same panels
    stochastic_sweep_shift.png        the energy the village shifts in the
                                      balanced run, prob 0-2% in 0.5% steps
                                      (only with --shift-sweep; shift_*
                                      sheets)
    stochastic_sweep.xlsx             belief_* (balanced), uninformed_* (the
                                      0%-shift deterministic runs), naive_*
                                      (committed at 0% shift), leader_*
                                      (leader's advantage), follower_*
                                      (balanced price held, village
                                      informed), adjusted_* (one price
                                      re-optimized for an informed village),
                                      vss_* (the mean-day price and its EEV)
                                      and information
                                      (the profits and EVPIs side by side)

Usage:
    python stochastic_sweep.py
    python stochastic_sweep.py --reuse           # solve only what the workbook lacks (e.g. the VSS)
    python stochastic_sweep.py --figures belief_pv
    python stochastic_sweep.py --reuse --shift-sweep   # + the shift, prob 0-2%
    python stochastic_sweep.py --replot          # redraw, no solving
"""

import argparse
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from pyomo.environ import TerminationCondition, value

import config
from asset_sweep import (AXIS, INK, MUTED, RULE, SEASON_LABEL, _end_labels,
                         _layout_issues, _write_rows, menu_step)
from model.bilevel_mpec import BilevelMPEC
# The solve, the span, the summary and the workbook are pattern_sweep's own, so
# a prob = 1 point here and a shift-figure point there are the same model.
from pattern_sweep import (NONE, PAT, PATTERNS, PV_KWP, RESIDUAL, Sweep, _TOL,
                           _fleet_label, _rows, load_rows, report,
                           save_analysis, sweep)
from run import build_instance
from scenarios import uncapped_availability
from stochastic_commitment_analysis import (DG_KW, TIME_LIMIT_S, Build,
                                            build_settings)

_OK = (TerminationCondition.optimal, TerminationCondition.feasible)

# Belief weights, prob(pattern). 0 is the no-supply run itself -- the shared
# limit both patterns start from -- so it is solved once.        [ASSUMPTION]
WEIGHTS: tuple[float, ...] = (0.0, 0.20, 0.40, 0.60, 0.80, 1.00)

# The build the belief is swept on: config's standard shift.     [ASSUMPTION]
BELIEF_SHIFT: float = 0.10

AVAIL_SWEEPS: tuple[Sweep, ...] = (
    Sweep("belief_pv", "w", "+200 kWp PV", True),
    Sweep("belief_dg", "w", "DG only", True),
)
FIGURES: tuple[str, ...] = tuple(sw.key for sw in AVAIL_SWEEPS)
# Left to right in both figures: the diesel-only fleet first.
PANEL_ORDER: tuple[str, ...] = ("belief_dg", "belief_pv")

# The information cases: (key, label, line style, marker).
CASES: tuple[tuple[str, str, object, str], ...] = (
    ("balanced",   "balanced",            "-",          "o"),
    ("leader",     "leader's advantage",  (0, (5, 2)),  "^"),
    ("uninformed", "follower uninformed", (0, (1, 1.5)), "v"),
)
WHERE = {"peak": "highest", "offpeak": "lowest"}

# Okabe-Ito, through the palette validator (adjacent CVD dE >= 11, all >= 3:1 on
# white).
COLOR = {"peak": "#D55E00", "offpeak": "#009E73", NONE: "#666666"}
MARKER = {"peak": "o", "offpeak": "s"}

# The figures colour by PATTERN and shade by season, since here the pattern is
# the comparison and the season only repeats it. Summer is the lightest and
# winter the darkest, which is also the order of their sun.
SEASON_SHADE = {"summer": 0.38, "fall_spring": 0.0, "winter": -0.32}

LEADER_FIELDS = [
    ("season",      "season"),
    ("fleet",       "fleet"),
    ("shift",       "load shift [-]"),
    ("series",      "series"),
    ("w",           "prob(pattern) [-]"),
    ("p_pattern",   "price if the pattern arrives [$/kWh]"),
    ("p_none",      "price if no supply [$/kWh]"),
    ("profit",      "E[leader profit] [$/day]"),
    ("qMGO",        "E[microgrid sales] [kWh]"),
    ("qPG",         "E[EDL delivered] [kWh]"),
    ("qsupp",       "E[suppressed demand] [kWh]"),
    ("termination", "termination"),
]

INFO_FIELDS = [
    ("sweep",      "sweep"),
    ("fleet",      "fleet"),
    ("season",     "season"),
    ("series",     "series"),
    ("w",          "prob(pattern) [-]"),
    ("balanced",   "E[profit], balanced [$/day]"),
    ("leader",     "E[profit], leader's advantage [$/day]"),
    ("uninformed", "E[profit], follower uninformed [$/day]"),
    ("naive",       "E[profit], one committed price, no village shift [$/day]"),
    ("evpi_belief", "leader's EVPI, leader's advantage [$/day]"),
    ("evpi_naive",  "leader's EVPI, follower uninformed [$/day]"),
    ("follower",    "E[profit], balanced price held, village informed [$/day]"),
    ("vfil",        "VFIL = balanced - village informed [$/day]"),
    ("adjusted",    "E[profit], one price re-optimized, village informed [$/day]"),
    ("repricing",   "gain from re-pricing = adjusted - held [$/day]"),
    ("eev",         "EEV: E[profit] at the mean-day price [$/day]"),
    ("vss",         "VSS = balanced - EEV [$/day]"),
]

# The VSS has one line per pattern and season.
VSS_CASES: tuple[tuple[str, str, object, str], ...] = (
    ("vss", "value of the stochastic solution", "-", "o"),
)

VSS_FIELDS = [
    ("season",      "season"),
    ("fleet",       "fleet"),
    ("shift",       "load shift [-]"),
    ("series",      "series"),
    ("w",           "prob(pattern) [-]"),
    ("p_ev",        "mean-day (EV) price [$/kWh]"),
    ("p_rp",        "balanced price [$/kWh]"),
    ("ev",          "EV: profit on the mean day [$/day]"),
    ("eev",         "EEV: E[profit] at the EV price [$/day]"),
    ("rp",          "balanced E[profit] [$/day]"),
    ("vss",         "VSS = balanced - EEV [$/day]"),
    ("shift_ev",    "energy shifted on the mean day [kWh]"),
    ("shift_eev",   "energy shifted at the EV price [kWh]"),
    ("termination", "termination (EV / EEV)"),
]

# The leader's EVPI of the two cases where the leader is informed: (key, label,
# style, marker). Styles and markers are those cases' own in CASES.
EVPI_CASES: tuple[tuple[str, str, object, str], ...] = (
    ("evpi_belief", "leader's advantage",  (0, (5, 2)),   "^"),
    ("evpi_naive",  "follower uninformed", (0, (1, 1.5)), "v"),
)

# The re-pricing gain has one line per pattern and season.
REPRICE_CASES: tuple[tuple[str, str, object, str], ...] = (
    ("repricing", "one price re-optimized for an informed village", "-", "o"),
)

ADJUSTED_FIELDS = [
    ("season",        "season"),
    ("fleet",         "fleet"),
    ("shift",         "load shift [-]"),
    ("series",        "series"),
    ("w",             "prob(pattern) [-]"),
    ("p",             "re-optimized price [$/kWh]"),
    ("p_held",        "balanced price [$/kWh]"),
    ("profit",        "E[profit], village informed [$/day]"),
    ("shift_pattern", "energy shifted if the pattern arrives [kWh]"),
    ("shift_none",    "energy shifted if no supply [kWh]"),
    ("termination",   "termination"),
]

# The shift sweep: prob 0.5-2% in 0.5% steps. The village switches to its full
# shift already below 1% -- where prob * (p - P_EDL) first exceeds the 2 * eps
# a moved kWh costs it -- so the sweep zooms in there. prob = 0 is the
# no-supply run, solved once per season and shared by both
# patterns.                                                        [ASSUMPTION]
SHIFT_WEIGHTS: tuple[float, ...] = tuple(round(0.005 * k, 3)
                                         for k in range(1, 5))

SHIFT_FIELDS = [
    ("season",       "season"),
    ("fleet",        "fleet"),
    ("shift",        "load shift [-]"),
    ("series",       "series"),
    ("w",            "prob(pattern) [-]"),
    ("p",            "balanced price [$/kWh]"),
    ("profit",       "E[leader profit] [$/day]"),
    ("shifted",      "energy shifted, 1/2 sum |dd| [kWh]"),
    ("into_covered", "net energy moved into the covered hours [kWh]"),
    ("termination",  "termination"),
]

FOLLOWER_FIELDS = [
    ("season",         "season"),
    ("fleet",          "fleet"),
    ("shift",          "load shift [-]"),
    ("series",         "series"),
    ("w",              "prob(pattern) [-]"),
    ("p",              "balanced price, held [$/kWh]"),
    ("profit_pattern", "profit if the pattern arrives, village informed [$/day]"),
    ("profit_none",    "profit if no supply, village informed [$/day]"),
    ("shift_pattern",  "energy shifted if the pattern arrives [kWh]"),
    ("shift_none",     "energy shifted if no supply [kWh]"),
    ("profit",         "E[profit], village informed [$/day]"),
    ("termination",    "termination (pattern / no supply)"),
]

# What the three cases are, for the captions of both case figures.
CASE_NOTE = (
    "balanced: one price and one village shift, both set against prob.\n"
    "leader's advantage: the leader learns the outcome and posts its own price "
    "for it; the village schedules against prob.\n"
    "follower uninformed: the leader learns the outcome; the village plans for "
    "no supply and does not shift. Colour: pattern, shaded by season (summer "
    "light, winter dark).")


def _shade(color: str, amount: float):
    """``color`` blended toward white (amount > 0) or black (amount < 0)."""
    r, g, b = mcolors.to_rgb(color)
    if amount >= 0.0:
        return tuple(c + (1.0 - c) * amount for c in (r, g, b))
    return tuple(c * (1.0 + amount) for c in (r, g, b))


# --------------------------------------------------------------------------
# Solving: leader's advantage
# --------------------------------------------------------------------------
def solve_leader_informed(season: str, Qbar: dict, w: float, opts: dict,
                          levels: int | None) -> dict | None:
    """
    The two-point belief of :func:`pattern_sweep.solve_at_belief`, with the
    leader told the outcome before it prices: one price per outcome, one
    village shift for both.
    """
    data = dict(build_instance(season))
    T = data["T"]
    S = [PAT, RESIDUAL]
    prob = {PAT: w, RESIDUAL: 1.0 - w}
    data["S"], data["prob"] = S, prob
    data["QbarPG"] = {(t, s): (Qbar[t] if s == PAT else 0.0)
                      for t in T for s in S}
    data["leaderInformed"] = True
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

    return {"w": w, "p_pattern": value(m.p[T[0], PAT]),
            "p_none": value(m.p[T[0], RESIDUAL]), "profit": value(m.profit),
            "qMGO": exp_sum(m.qMGO), "qPG": exp_sum(m.qPG),
            "qsupp": exp_sum(m.qsupp), "termination": str(tc)}


def leader_sweep(seasons: list[str], weights: list[float], *, dg_kw: float,
                 pv_kwp: float, opts: dict, levels: int | None) -> list[dict]:
    """Leader's advantage at every interior weight. w = 1 has one scenario and
    is the balanced certain run itself, so it is not solved again."""
    fleet = _fleet_label(dg_kw, pv_kwp)
    build = Build("leader", fleet, dg_kw, pv_kwp, BELIEF_SHIFT)
    rows = []
    for season in seasons:
        D = build_instance(season)["D"]
        print(f"[{season}] {fleet}, leader's advantage")
        for pattern in PATTERNS:
            Qbar = uncapped_availability(season, pattern, D)
            for w in weights:
                if w >= 1.0 - _TOL:
                    continue
                with build_settings(build):
                    rec = solve_leader_informed(season, Qbar, w, opts, levels)
                if rec is None:
                    continue
                rec |= {"season": season, "fleet": fleet,
                        "shift": BELIEF_SHIFT, "series": pattern}
                rows.append(rec)
                print(f"    {pattern:>8} prob={w:.2f}  "
                      f"p|pattern={rec['p_pattern']:.4f}  "
                      f"p|none={rec['p_none']:.4f}  "
                      f"E[profit]={rec['profit']:8.2f}")
    return rows


# --------------------------------------------------------------------------
# Solving: balanced price held, village informed (VFIL)
# --------------------------------------------------------------------------
def solve_fixed_price(season: str, Qbar: dict, scen: str, price: float,
                      opts: dict) -> dict | None:
    """
    One outcome, certain, with the leader's price fixed at ``price``: the menu
    is that one level, so the leader cannot re-price, while the village knows
    the outcome and schedules for it.
    """
    data = dict(build_instance(season))
    T = data["T"]
    data["S"], data["prob"] = [scen], {scen: 1.0}
    data["QbarPG"] = {(t, scen): Qbar[t] for t in T}
    data["Pi"] = [price]

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
    return {"profit": value(m.profit),
            # Energy moved between hours: half of sum |dd|, since it sums to 0.
            "shifted": 0.5 * sum(abs(value(m.dd[t])) for t in T),
            "termination": str(tc)}


def follower_sweep(seasons: list[str], balanced: list[dict], *, dg_kw: float,
                   pv_kwp: float, opts: dict) -> list[dict]:
    """Both outcomes at every interior weight, at that weight's balanced price.
    Runs are cached by (season, outcome, price): prices repeat across weights
    and the no-supply outcome is shared by both patterns."""
    fleet = _fleet_label(dg_kw, pv_kwp)
    build = Build("vfil", fleet, dg_kw, pv_kwp, BELIEF_SHIFT)
    cache: dict = {}
    rows = []
    balanced = _real(balanced)
    for season in seasons:
        D = build_instance(season)["D"]
        off_q = {t: 0.0 for t in D}
        print(f"[{season}] {fleet}, balanced price held, village informed")

        def run(series, scen, q, p):
            key = (series, round(p, 6))
            if key not in cache:
                with build_settings(build):
                    cache[key] = solve_fixed_price(season, q, scen, p, opts)
            return cache[key]

        for pattern in PATTERNS:
            on_q = uncapped_availability(season, pattern, D)
            pts = sorted((r for r in balanced
                          if r["season"] == season and r["series"] == pattern
                          and _TOL < r["w"] < 1.0 - _TOL
                          and r.get("p") is not None), key=lambda r: r["w"])
            for r in pts:
                w, p = r["w"], r["p"]
                on = run(pattern, PAT, on_q, p)
                off = run(NONE, RESIDUAL, off_q, p)
                if on is None or off is None:
                    print(f"    {pattern:>8} prob={w:.2f}  a run did not solve "
                          f"— no VFIL here")
                    continue
                rows.append({
                    "season": season, "fleet": fleet, "shift": BELIEF_SHIFT,
                    "series": pattern, "w": w, "p": p,
                    "profit_pattern": on["profit"],
                    "profit_none": off["profit"],
                    "shift_pattern": on["shifted"],
                    "shift_none": off["shifted"],
                    "profit": w * on["profit"] + (1.0 - w) * off["profit"],
                    "termination": f"{on['termination']} / "
                                   f"{off['termination']}"})
                print(f"    {pattern:>8} prob={w:.2f}  p={p:.4f}  "
                      f"profit|pattern={on['profit']:8.2f} "
                      f"(shift {on['shifted']:5.1f} kWh)  "
                      f"profit|none={off['profit']:8.2f} "
                      f"(shift {off['shifted']:5.1f} kWh)  "
                      f"VFIL={r['profit'] - rows[-1]['profit']:+7.2f}")
        cache.clear()                   # prices are per season
    return rows


# --------------------------------------------------------------------------
# Solving: one price re-optimized for an informed village
# --------------------------------------------------------------------------
def solve_follower_informed(season: str, Qbar: dict, w: float, opts: dict,
                            levels: int | None) -> dict | None:
    """
    The two-point belief of :func:`pattern_sweep.solve_at_belief`, with the
    village told the outcome before it schedules: one leader price, one
    village shift per outcome.
    """
    data = dict(build_instance(season))
    T = data["T"]
    S = [PAT, RESIDUAL]
    data["S"], data["prob"] = S, {PAT: w, RESIDUAL: 1.0 - w}
    data["QbarPG"] = {(t, s): (Qbar[t] if s == PAT else 0.0)
                      for t in T for s in S}
    data["followerInformed"] = True
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
    return {"w": w, "p": value(m.p[T[0]]), "profit": value(m.profit),
            "shift_pattern": 0.5 * sum(abs(value(m.dd[t, PAT])) for t in T),
            "shift_none": 0.5 * sum(abs(value(m.dd[t, RESIDUAL])) for t in T),
            "termination": str(tc)}


def adjusted_sweep(seasons: list[str], weights: list[float],
                   balanced: list[dict], *, dg_kw: float, pv_kwp: float,
                   opts: dict, levels: int | None) -> list[dict]:
    """The re-optimized price for an informed village at every interior
    weight, with the balanced price it moved from."""
    fleet = _fleet_label(dg_kw, pv_kwp)
    build = Build("adjusted", fleet, dg_kw, pv_kwp, BELIEF_SHIFT)
    held = {(r["season"], r["series"], round(r["w"], 6)): r.get("p")
            for r in _real(balanced)}
    rows = []
    for season in seasons:
        D = build_instance(season)["D"]
        print(f"[{season}] {fleet}, one price re-optimized, village informed")
        for pattern in PATTERNS:
            Qbar = uncapped_availability(season, pattern, D)
            for w in weights:
                if w >= 1.0 - _TOL:
                    continue            # one outcome: nothing to learn
                with build_settings(build):
                    rec = solve_follower_informed(season, Qbar, w, opts,
                                                  levels)
                if rec is None:
                    continue
                rec |= {"season": season, "fleet": fleet,
                        "shift": BELIEF_SHIFT, "series": pattern,
                        "p_held": held.get((season, pattern, round(w, 6)))}
                rows.append(rec)
                print(f"    {pattern:>8} prob={w:.2f}  p={rec['p']:.4f} "
                      f"(balanced {rec['p_held'] or float('nan'):.4f})  "
                      f"E[profit]={rec['profit']:8.2f}  shift "
                      f"{rec['shift_pattern']:5.1f} / {rec['shift_none']:5.1f} "
                      f"kWh")
    return rows


# --------------------------------------------------------------------------
# Solving: the value of the stochastic solution
# --------------------------------------------------------------------------
def _solve_or_none(mpec: BilevelMPEC, opts: dict):
    """Solve; the termination condition, or None if it did not solve."""
    solver_name, res = mpec.solve("gurobi", options=opts)
    if solver_name is None:
        raise RuntimeError(
            "Gurobi not available (check the license and that gurobipy is "
            "installed).")
    tc = res.solver.termination_condition
    if tc not in _OK:
        print(f"    not solved ({tc})")
        return None
    return tc


def solve_ev(season: str, Qbar: dict, w: float, opts: dict,
             levels: int | None) -> dict | None:
    """
    The certain run on the mean day: in the hours ``Qbar`` covers, EDL's cap
    is w * (1 + shift_frac) * D_t -- what a pattern that covers the hour and
    all load shifted into it delivers in expectation -- and 0 elsewhere.
    """
    data = dict(build_instance(season))
    T, D = data["T"], data["D"]
    cover = w * (1.0 + data["shiftFrac"])
    data["S"], data["prob"] = [PAT], {PAT: 1.0}
    data["QbarPG"] = {(t, PAT): (cover * D[t] if Qbar[t] > 0.0 else 0.0)
                      for t in T}
    if levels:
        P_ref, Phat = data["P"][T[0]], data["Phat"][T[0]]
        data["Pi"] = [P_ref + (Phat - P_ref) * k / (levels - 1)
                      for k in range(levels)]

    mpec = BilevelMPEC(data)
    m = mpec.build()
    tc = _solve_or_none(mpec, opts)
    if tc is None:
        return None
    return {"p": value(m.p[T[0]]), "profit": value(m.profit),
            "shifted": 0.5 * sum(abs(value(m.dd[t])) for t in T),
            "termination": str(tc)}


def solve_eev(season: str, Qbar: dict, w: float, price: float,
              opts: dict) -> dict | None:
    """
    The balanced two-point belief of :func:`pattern_sweep.solve_at_belief`
    with the menu reduced to ``price``: the leader cannot re-price, the village
    schedules against prob.
    """
    data = dict(build_instance(season))
    T = data["T"]
    S = [PAT, RESIDUAL]
    data["S"], data["prob"] = S, {PAT: w, RESIDUAL: 1.0 - w}
    data["QbarPG"] = {(t, s): (Qbar[t] if s == PAT else 0.0)
                      for t in T for s in S}
    data["Pi"] = [price]

    mpec = BilevelMPEC(data)
    m = mpec.build()
    tc = _solve_or_none(mpec, opts)
    if tc is None:
        return None
    return {"profit": value(m.profit),
            "shifted": 0.5 * sum(abs(value(m.dd[t])) for t in T),
            "termination": str(tc)}


def vss_sweep(seasons: list[str], weights: list[float], balanced: list[dict],
              *, dg_kw: float, pv_kwp: float, opts: dict,
              levels: int | None) -> list[dict]:
    """EV and EEV at every interior weight, against that weight's balanced
    run. Where the EV price is the balanced price, EEV is balanced itself and
    is not solved."""
    fleet = _fleet_label(dg_kw, pv_kwp)
    build = Build("vss", fleet, dg_kw, pv_kwp, BELIEF_SHIFT)
    rp = {(r["season"], r["series"], round(r["w"], 6)): r
          for r in _real(balanced)}
    rows = []
    for season in seasons:
        D = build_instance(season)["D"]
        print(f"[{season}] {fleet}, value of the stochastic solution")
        for pattern in PATTERNS:
            Qbar = uncapped_availability(season, pattern, D)
            for w in weights:
                if w >= 1.0 - _TOL:
                    continue            # the mean day is the certain run
                bal = rp.get((season, pattern, round(w, 6)))
                if bal is None or bal.get("p") is None:
                    print(f"    {pattern:>8} prob={w:.2f}  no balanced run "
                          f"— no VSS here")
                    continue
                with build_settings(build):
                    ev = solve_ev(season, Qbar, w, opts, levels)
                if ev is None:
                    continue
                if abs(ev["p"] - bal["p"]) <= 1e-6:
                    eev = {"profit": bal["profit"], "shifted": None,
                           "termination": "same price as balanced"}
                else:
                    with build_settings(build):
                        eev = solve_eev(season, Qbar, w, ev["p"], opts)
                    if eev is None:
                        continue
                rows.append({
                    "season": season, "fleet": fleet, "shift": BELIEF_SHIFT,
                    "series": pattern, "w": w, "p_ev": ev["p"],
                    "p_rp": bal["p"], "ev": ev["profit"],
                    "eev": eev["profit"], "rp": bal["profit"],
                    "vss": bal["profit"] - eev["profit"],
                    "shift_ev": ev["shifted"], "shift_eev": eev["shifted"],
                    "termination": f"{ev['termination']} / "
                                   f"{eev['termination']}"})
                print(f"    {pattern:>8} prob={w:.2f}  p_EV={ev['p']:.4f} "
                      f"(balanced {bal['p']:.4f})  EV={ev['profit']:8.2f}  "
                      f"EEV={eev['profit']:8.2f}  "
                      f"VSS={rows[-1]['vss']:+7.2f}")
    return rows


# --------------------------------------------------------------------------
# Solving: the village's shift across prob (balanced)
# --------------------------------------------------------------------------
def solve_shift(season: str, Qbar: dict, w: float, opts: dict,
                levels: int | None) -> dict | None:
    """
    The balanced run at belief w -- the two-point belief of
    :func:`pattern_sweep.solve_at_belief` -- with the village's committed
    hourly shift read out. w = 1 and w = 0 are single-scenario runs.
    """
    data = dict(build_instance(season))
    T = data["T"]
    if w >= 1.0 - _TOL:
        S, prob = [PAT], {PAT: 1.0}
    elif w <= _TOL:
        S, prob = [RESIDUAL], {RESIDUAL: 1.0}
    else:
        S, prob = [PAT, RESIDUAL], {PAT: w, RESIDUAL: 1.0 - w}
    data["S"], data["prob"] = S, prob
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
    dd = {t: value(m.dd[t]) for t in T}
    return {"w": w, "p": value(m.p[T[0]]), "profit": value(m.profit),
            # Energy moved between hours: half of sum |dd|, since it sums to 0.
            "shifted": 0.5 * sum(abs(v) for v in dd.values()),
            "dd": dd, "termination": str(tc)}


def shift_sweep_runs(seasons: list[str], *, dg_kw: float, pv_kwp: float,
                     opts: dict, levels: int | None,
                     have: list[dict] = ()) -> list[dict]:
    """The balanced run at every prob in SHIFT_WEIGHTS, per season and
    pattern, with the energy the village shifts. Points already in ``have``
    (same season, pattern and prob) are not solved again; only the new rows
    are returned."""
    fleet = _fleet_label(dg_kw, pv_kwp)
    build = Build("shift", fleet, dg_kw, pv_kwp, BELIEF_SHIFT)
    done = {(r["season"], r["series"], round(r["w"], 6))
            for r in _real(list(have))}
    rows = []
    for season in seasons:
        D = build_instance(season)["D"]
        T = sorted(D)
        todo = [(p, w) for p in PATTERNS for w in (0.0,) + SHIFT_WEIGHTS
                if (season, p, round(w, 6)) not in done]
        if not todo:
            print(f"[{season}] {fleet}: every point already in the workbook")
            continue
        print(f"[{season}] {fleet}, village shift across prob "
              f"({len(todo)} points to solve)")
        ref = None
        if any(w == 0.0 for _, w in todo):
            with build_settings(build):
                ref = solve_shift(season, {t: 0.0 for t in T}, 0.0, opts,
                                  levels)
        for pattern in PATTERNS:
            Qbar = uncapped_availability(season, pattern, D)
            covered = [t for t in T if Qbar[t] > 0.0]
            recs = ([ref] if ref is not None
                    and (pattern, 0.0) in todo else [])
            for w in SHIFT_WEIGHTS:
                if (pattern, w) not in todo:
                    continue
                with build_settings(build):
                    rec = solve_shift(season, Qbar, w, opts, levels)
                if rec is not None:
                    recs.append(rec)
            for rec in recs:
                row = {k: v for k, v in rec.items() if k != "dd"} | {
                    "season": season, "fleet": fleet, "shift": BELIEF_SHIFT,
                    "series": pattern,
                    "into_covered": sum(rec["dd"][t] for t in covered)}
                rows.append(row)
                print(f"    {pattern:>8} prob={100 * row['w']:4.1f}%  "
                      f"p={row['p']:.4f}  shifted={row['shifted']:6.1f} kWh  "
                      f"into EDL hours={row['into_covered']:+6.1f} kWh")
    return rows


# --------------------------------------------------------------------------
# The three cases side by side
# --------------------------------------------------------------------------
def _real(rows: list[dict]) -> list[dict]:
    """Runs only: a sheet's caption line reads back as a record with no series."""
    return [r for r in rows if r.get("series") is not None]


def information_rows(key: str, balanced: list[dict], leader: list[dict],
                     uninformed: list[dict], naive: list[dict],
                     follower: list[dict], adjusted: list[dict],
                     vss: list[dict] = ()) -> list[dict]:
    """The leader's expected profit in each case, its EVPI, the VFIL, the
    gain from re-pricing and the VSS, per season, pattern and w."""
    out = []
    balanced, leader = _real(balanced), _real(leader)
    uninformed, naive = _real(uninformed), _real(naive)
    follower, adjusted = _real(follower), _real(adjusted)
    vss = _real(list(vss))
    for season in config.SEASONS:
        b = [r for r in balanced if r["season"] == season]
        ref = next((r for r in b if r["series"] == NONE), None)
        if ref is None:
            continue
        u = [r for r in uninformed if r["season"] == season]
        u_none = next((r for r in u if r["series"] == NONE), None)
        for pattern in PATTERNS:
            pts = {round(r["w"], 6): r for r in b if r["series"] == pattern}
            lead = {round(r["w"], 6): r for r in leader
                    if r["season"] == season and r["series"] == pattern}
            u_pat = next((r for r in u if r["series"] == pattern), None)
            nv = {round(r["w"], 6): r for r in naive
                  if r["season"] == season and r["series"] == pattern}
            fl = {round(r["w"], 6): r for r in follower
                  if r["season"] == season and r["series"] == pattern}
            ad = {round(r["w"], 6): r for r in adjusted
                  if r["season"] == season and r["series"] == pattern}
            vs = {round(r["w"], 6): r for r in vss
                  if r["season"] == season and r["series"] == pattern}
            for w in [0.0] + sorted(pts):
                bal = ref["profit"] if w <= _TOL else pts[w]["profit"]
                if w <= _TOL or w >= 1.0 - _TOL:
                    ldr = bal                   # one scenario: the same model
                else:
                    ldr = lead[w]["profit"] if w in lead else None
                unf = (None if u_none is None or u_pat is None
                       else w * u_pat["profit"] + (1.0 - w) * u_none["profit"])
                # One committed price with a village that does not shift. With
                # one outcome (w = 0 or 1) that is the uninformed run itself.
                if w <= _TOL or w >= 1.0 - _TOL:
                    nav = unf
                else:
                    nav = nv[w]["profit"] if w in nv else None
                # The balanced price held, the village informed. With one
                # outcome there is nothing to learn: it is balanced itself.
                if w <= _TOL or w >= 1.0 - _TOL:
                    inf_f = bal
                else:
                    inf_f = fl[w]["profit"] if w in fl else None
                # One price re-optimized for the informed village; with one
                # outcome it is balanced itself.
                if w <= _TOL or w >= 1.0 - _TOL:
                    adj = bal
                else:
                    adj = ad[w]["profit"] if w in ad else None
                # The mean-day price; with one outcome the mean day is that
                # outcome and EEV is balanced itself.
                if w <= _TOL or w >= 1.0 - _TOL:
                    eev = bal
                else:
                    eev = vs[w]["eev"] if w in vs else None
                out.append({
                    "eev": eev,
                    "vss": None if eev is None else bal - eev,
                    "adjusted": adj,
                    "repricing": (None if adj is None or inf_f is None
                                  else adj - inf_f),
                    "sweep": key, "fleet": ref["fleet"], "season": season,
                    "series": pattern, "w": w, "balanced": bal,
                    "leader": ldr, "uninformed": unf, "naive": nav,
                    "evpi_belief": None if ldr is None else ldr - bal,
                    "evpi_naive": (None if unf is None or nav is None
                                   else unf - nav),
                    "follower": inf_f,
                    "vfil": None if inf_f is None else bal - inf_f})
    return out


def report_information(info: list[dict]) -> None:
    """The three profits as a table per fleet, with shortfalls flagged."""
    def f(v):
        return f"{'-':>10}" if v is None else f"{v:10.2f}"

    for key in PANEL_ORDER:
        sel = [r for r in info if r["sweep"] == key]
        if not sel:
            continue
        print(f"\n=== E[leader profit] [$/day] by information case, "
              f"{sel[0]['fleet']}")
        print(f"  {'season':>12} {'pattern':>8} {'prob':>5}"
              f"{'balanced':>10}{'leader':>10}{'uninf.':>10}")
        for r in sel:
            print(f"  {r['season']:>12} {r['series']:>8} {r['w']:5.2f}"
                  f"{f(r['balanced'])}{f(r['leader'])}{f(r['uninformed'])}")
        print(f"\n=== leader's EVPI [$/day], {sel[0]['fleet']}  (leader's "
              f"advantage - balanced; follower uninformed - naive)")
        print(f"  {'season':>12} {'pattern':>8} {'prob':>5}{'naive':>10}"
              f"{'leader':>10}{'uninf.':>10}")
        for r in sel:
            print(f"  {r['season']:>12} {r['series']:>8} {r['w']:5.2f}"
                  f"{f(r['naive'])}{f(r['evpi_belief'])}"
                  f"{f(r['evpi_naive'])}")
        print(f"\n=== VFIL [$/day] = balanced - village informed at the "
              f"balanced price, {sel[0]['fleet']}")
        print(f"  {'season':>12} {'pattern':>8} {'prob':>5}{'balanced':>10}"
              f"{'informed':>10}{'VFIL':>10}")
        for r in sel:
            print(f"  {r['season']:>12} {r['series']:>8} {r['w']:5.2f}"
                  f"{f(r['balanced'])}{f(r['follower'])}{f(r['vfil'])}")
        print(f"\n=== gain from re-pricing for an informed village [$/day] = "
              f"adjusted - held, {sel[0]['fleet']}")
        print(f"  {'season':>12} {'pattern':>8} {'prob':>5}{'held':>10}"
              f"{'adjusted':>10}{'gain':>10}")
        for r in sel:
            print(f"  {r['season']:>12} {r['series']:>8} {r['w']:5.2f}"
                  f"{f(r['follower'])}{f(r['adjusted'])}{f(r['repricing'])}")
        print(f"\n=== value of the stochastic solution [$/day] = balanced - "
              f"EEV, {sel[0]['fleet']}")
        print(f"  {'season':>12} {'pattern':>8} {'prob':>5}{'balanced':>10}"
              f"{'EEV':>10}{'VSS':>10}")
        for r in sel:
            print(f"  {r['season']:>12} {r['series']:>8} {r['w']:5.2f}"
                  f"{f(r['balanced'])}{f(r['eev'])}{f(r['vss'])}")
    for r in info:
        # The EV price is on balanced's menu: a hit means a failed solve.
        if r.get("vss") is not None and r["vss"] < -max(
                1e-6, config.CONFIG["MIPGap"] * abs(r["balanced"])):
            print(f"  !! negative VSS, {r['vss']:+.2f} $/day: {r['fleet']}, "
                  f"{r['season']}, {r['series']}, prob {r['w']:.2f}")
        if r["leader"] is not None and r["leader"] < r["balanced"] - max(
                1e-6, config.CONFIG["MIPGap"] * abs(r["balanced"])):
            print(f"  !! leader's advantage below balanced by "
                  f"{r['balanced'] - r['leader']:.2f} $/day: {r['fleet']}, "
                  f"{r['season']}, {r['series']}, prob {r['w']:.2f}")
        # Cannot happen but for the solver gap, so a hit means a failed solve.
        if r["evpi_naive"] is not None and r["evpi_naive"] < -max(
                1e-6, config.CONFIG["MIPGap"] * abs(r["uninformed"])):
            print(f"  !! negative EVPI with a non-shifting village, "
                  f"{r['evpi_naive']:+.2f} $/day: {r['fleet']}, "
                  f"{r['season']}, {r['series']}, prob {r['w']:.2f}")
        # The held price is one the adjusted run could choose, so this too is
        # a failed solve if it ever shows.
        if r["repricing"] is not None and r["repricing"] < -max(
                1e-6, config.CONFIG["MIPGap"] * abs(r["adjusted"])):
            print(f"  !! negative gain from re-pricing, "
                  f"{r['repricing']:+.2f} $/day: {r['fleet']}, "
                  f"{r['season']}, {r['series']}, prob {r['w']:.2f}")


# --------------------------------------------------------------------------
# Workbook
# --------------------------------------------------------------------------
def _read_sheet(book, name: str, fields: list) -> list[dict]:
    """A sheet written with ``fields``, read back; [] if it is not there."""
    wb = load_workbook(book, data_only=True)
    if name not in wb.sheetnames:
        return []
    table = list(wb[name].iter_rows(values_only=True))
    if not table:
        return []
    key_of = {label: key for key, label in fields}
    header = [key_of.get(cell) for cell in table[0]]
    return _real([{k: v for k, v in zip(header, row) if k is not None}
                  for row in table[1:]])


def _write_sheet(wb, name: str, fields: list, rows: list[dict],
                 note: str) -> None:
    if name in wb.sheetnames:
        wb.remove(wb[name])
    ws = wb.create_sheet(name[:31])
    _write_rows(ws, [lbl for _, lbl in fields],
                ([r.get(k) for k, _ in fields] for r in rows))
    ws.append([])
    ws.append([note])


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def _chrome(ax) -> None:
    ax.grid(color=RULE, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def _prob_axis(ax, weights: list[float]) -> None:
    ax.set_xticks([100.0 * w for w in sorted(set([0.0] + list(weights)))])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlim(-4.0, 104.0)


def _price_series(balanced: dict[str, list[dict]], key: str,
                  order: list[str]) -> list[dict]:
    """One line per pattern and season: the committed price against prob,
    with its change from prob = 0 as the end label."""
    sw = next(s for s in AVAIL_SWEEPS if s.key == key)
    specs = [{"season": season, "sweep": sw, "pattern": pattern,
              "label": f"{pattern} · {SEASON_LABEL.get(season, season)}"}
             for pattern in PATTERNS for season in order]
    out = []
    for row in _rows(balanced, specs):
        rng = row["ranges"].get("p")
        if rng is None:
            continue
        out.append({"pattern": row["pattern"], "season": row["season"],
                    "xs": rng["xs"], "ys": rng["vs"],
                    "end": (None if rng["ps"][-1] is None
                            else f"{rng['ps'][-1]:+.1f}%")})
    return out


def _title_legend(fig, title: str, handles: list, caption: str, out_path,
                  ncol: int) -> None:
    """Title, legend under it, caption at the foot, layout check, save."""
    fig_w = fig.get_size_inches()[0]
    fig_h = fig.get_size_inches()[1]
    line = 0.28 / fig_h                    # one title line, in figure fraction
    ttl = fig.suptitle(textwrap.fill(title, width=int(8.2 * fig_w)),
                       x=0.01, y=0.985, ha="left", va="top", fontsize=12.5,
                       color=INK, fontweight="bold")
    ttl_lines = ttl.get_text().count("\n") + 1
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.975 - line * ttl_lines),
                     ncol=ncol, frameon=False, fontsize=8.5)
    n_cap = caption.count("\n") + 1
    cap = fig.text(0.01, 0.012, caption, fontsize=8, color=MUTED, ha="left",
                   va="bottom")
    fig.tight_layout(rect=(0, 0.02 + 0.2 * n_cap / fig_h, 1,
                           0.95 - line * (ttl_lines + 1.2)))
    issues = _layout_issues(fig, [ttl, leg, cap])
    for msg in issues:
        print(f"  layout: {msg}  [{out_path}]")
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}"
          + ("" if issues else "  (layout check: no overlaps)"))


def plot_price(balanced: dict[str, list[dict]], keys: list[str],
               names: dict[str, str], out_path, *, seasons: list[str],
               weights: list[float]) -> None:
    """The balanced committed price, one panel per fleet on one price scale."""
    order = [s for s in config.SEASONS if s in seasons]
    fig, axes = plt.subplots(1, len(keys), figsize=(14.5, 6.6), sharey=True,
                             squeeze=False)
    for ax, key in zip(axes[0], keys):
        ax.set_title(names[key], loc="left", fontsize=11, color=INK)
        ends = []
        for s in _price_series(balanced, key, order):
            color = _shade(COLOR[s["pattern"]],
                           SEASON_SHADE.get(s["season"], 0.0))
            xs = [100.0 * x for x in s["xs"]]
            ax.plot(xs, s["ys"], color=color, linewidth=2.0,
                    marker=MARKER[s["pattern"]], markersize=6,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=3)
            if s["end"]:
                ends.append((xs[-1], s["ys"][-1], s["end"]))
        _chrome(ax)
        _prob_axis(ax, weights)
        ax.set_xlabel("prob   —   no supply takes the rest", fontsize=10,
                      color=INK)
        ax.set_ylabel("Committed price   [$ / kWh]", fontsize=10, color=INK)
        # Shared y hides the right panel's numbers; each keeps its scale.
        ax.tick_params(axis="y", labelleft=True)
        _end_labels(ax, ends)                 # the % change, spread apart

    handles = [plt.Line2D([], [], color=_shade(COLOR[p], SEASON_SHADE.get(s, 0)),
                          linewidth=2.0, marker=MARKER[p], markersize=6,
                          markeredgecolor="white",
                          label=f"{p} · {SEASON_LABEL.get(s, s)}")
               for p in PATTERNS for s in order]
    _title_legend(
        fig,
        f"The price the leader commits to against prob, balanced "
        f"information, at {100 * BELIEF_SHIFT:.0f}% load shift. prob = 0 is "
        f"the no-supply run, so the figure at each line's right end is its "
        f"change from there",
        handles,
        "The leader posts one flat price before the day; prob is the weight "
        "it puts on the pattern arriving, no supply taking the rest.",
        out_path, ncol=len(handles) // 2)


def plot_information(info: list[dict], keys: list[str], names: dict[str, str],
                     out_path, *, seasons: list[str], weights: list[float],
                     field: str, ylabel: str, title: str, caption: str,
                     zero: bool = False, cases: tuple = CASES) -> None:
    """
    One quantity of the three information cases -- ``field`` with ``{case}``
    in it, e.g. the profit itself or its EVPI: a row per pattern, a column per
    fleet, all on one scale. Colour is the pattern shaded by season, the line
    style is the case.
    """
    order = [s for s in config.SEASONS if s in seasons]
    fig, axes = plt.subplots(len(PATTERNS), len(keys), figsize=(14.5, 10.5),
                             sharex=True, sharey=True, squeeze=False)
    for i, pattern in enumerate(PATTERNS):
        for j, key in enumerate(keys):
            ax = axes[i, j]
            for season in order:
                sel = sorted((r for r in info if r["sweep"] == key
                              and r["season"] == season
                              and r["series"] == pattern),
                             key=lambda r: r["w"])
                color = _shade(COLOR[pattern], SEASON_SHADE.get(season, 0.0))
                for case, _, style, marker in cases:
                    k = field.format(case=case)
                    pts = [(100.0 * r["w"], r[k]) for r in sel
                           if r.get(k) is not None]
                    if len(pts) < 2:
                        continue
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, color=color, linestyle=style,
                            linewidth=1.8, marker=marker, markersize=5,
                            markeredgecolor="white", markeredgewidth=0.8,
                            zorder=3)
            _chrome(ax)
            if zero:
                ax.axhline(0.0, color="#999999", linewidth=0.9, zorder=1)
            _prob_axis(ax, weights)
            ax.set_title(f"{names[key]}  ·  EDL on the {WHERE[pattern]}-demand "
                         f"hours ({pattern})", loc="left", fontsize=10.5,
                         color=INK)
            # Shared axes hide inner tick labels; every panel keeps its scales.
            ax.tick_params(axis="x", labelbottom=True)
            ax.tick_params(axis="y", labelleft=True)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=10, color=INK)
            if i == len(PATTERNS) - 1:
                ax.set_xlabel("prob   —   no supply takes the rest",
                              fontsize=10, color=INK)

    grey = "#666666"
    handles = ([plt.Line2D([], [], color=grey, linestyle=style, linewidth=1.8,
                           marker=marker, markersize=5, markeredgecolor="white",
                           label=label) for _, label, style, marker in cases]
               + [plt.Line2D([], [], color=_shade(grey, SEASON_SHADE.get(s, 0)),
                             linewidth=4.0, label=SEASON_LABEL.get(s, s))
                  for s in order])
    _title_legend(fig, title, handles, caption, out_path, ncol=len(handles))


def plot_shift(shifts: dict[str, list[dict]], keys: list[str],
               names: dict[str, str], out_path, *, seasons: list[str]) -> None:
    """
    The energy the village shifts in the balanced run against prob: a row per
    pattern, a column per fleet, one scale. Colour is the pattern shaded by
    season.
    """
    order = [s for s in config.SEASONS if s in seasons]
    fig, axes = plt.subplots(len(PATTERNS), len(keys), figsize=(14.5, 10.5),
                             sharex=True, sharey=True, squeeze=False)
    for i, pattern in enumerate(PATTERNS):
        for j, key in enumerate(keys):
            ax = axes[i, j]
            # The workbook keeps every prob ever solved; the figure shows the
            # swept range only.
            rows = [r for r in _real(shifts.get(f"shift_{key.split('_')[-1]}",
                                                []))
                    if r["w"] <= max(SHIFT_WEIGHTS) + _TOL]
            for season in order:
                pts = sorted((100.0 * r["w"], r["shifted"]) for r in rows
                             if r["season"] == season
                             and r["series"] == pattern)
                if len(pts) < 2:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys,
                        color=_shade(COLOR[pattern],
                                     SEASON_SHADE.get(season, 0.0)),
                        linewidth=2.0, marker=MARKER[pattern], markersize=4.5,
                        markeredgecolor="white", markeredgewidth=0.8,
                        zorder=3)
            _chrome(ax)
            # The axis is the swept prob, a tick at every step.
            ticks = [0.0] + [100.0 * w for w in SHIFT_WEIGHTS]
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}%")
            ax.set_xlim(-0.03 * ticks[-1], 1.03 * ticks[-1])
            ax.set_ylim(bottom=0.0)
            ax.set_title(f"{names[key]}  ·  EDL on the {WHERE[pattern]}-demand "
                         f"hours ({pattern})", loc="left", fontsize=10.5,
                         color=INK)
            # Shared axes hide inner tick labels; every panel keeps its scales.
            ax.tick_params(axis="x", labelbottom=True)
            ax.tick_params(axis="y", labelleft=True)
            if j == 0:
                ax.set_ylabel("Shifted energy   [kWh / day]", fontsize=10,
                              color=INK)
            if i == len(PATTERNS) - 1:
                ax.set_xlabel("prob   —   no supply takes the rest",
                              fontsize=10, color=INK)

    handles = [plt.Line2D([], [], color=_shade(COLOR[p], SEASON_SHADE.get(s, 0)),
                          linewidth=2.0, marker=MARKER[p], markersize=5,
                          markeredgecolor="white",
                          label=f"{p} · {SEASON_LABEL.get(s, s)}")
               for p in PATTERNS for s in order]
    _title_legend(
        fig,
        f"How much energy the village shifts in the balanced run, against "
        f"prob, at {100 * BELIEF_SHIFT:.0f}% load shift",
        handles,
        "Shifted energy = ½ Σ_t |Δd_t|: the load the village moves between "
        "hours, one schedule committed for both outcomes.\n"
        "prob from 0 to 2% in 0.5% steps, around where the village switches to "
        "its full shift; prob = 0 is the no-supply run.",
        out_path, ncol=len(handles) // 2)


def plot_all(balanced: dict[str, list[dict]], info: list[dict], out_dir,
             seasons: list[str], weights: list[float],
             shifts: dict[str, list[dict]] | None = None) -> None:
    """Both figures. Panel names are read off the runs themselves, so a redraw
    from the workbook labels itself correctly."""
    keys = [k for k in PANEL_ORDER if _real(balanced.get(k, []))]
    if not keys:
        print("  nothing to draw")
        return
    names = {k: _real(balanced[k])[0]["fleet"] for k in keys}
    plot_price(balanced, keys, names, out_dir / "stochastic_sweep_price.png",
               seasons=seasons, weights=weights)
    shift = f"{100 * BELIEF_SHIFT:.0f}% load shift"
    plot_information(
        info, keys, names, out_dir / "stochastic_sweep_information.png",
        seasons=seasons, weights=weights, field="{case}",
        ylabel="Expected leader profit   [$ / day]",
        title=f"The leader's expected profit under three information cases, "
              f"against prob, at {shift}",
        caption=CASE_NOTE)
    plot_information(
        info, keys, names, out_dir / "stochastic_sweep_evpi.png",
        seasons=seasons, weights=weights, field="{case}", cases=EVPI_CASES,
        zero=True, ylabel="Leader's EVPI   [$ / day]",
        title=f"What learning the EDL outcome before pricing is worth to the "
              f"leader, against prob, at {shift}",
        caption="Leader's EVPI = its expected profit with one price per "
                "outcome minus with one committed price, the village behaving "
                "the same in both.\n"
                "leader's advantage: minus balanced (the village schedules "
                "against prob in both).\n"
                "follower uninformed: minus one committed price with the same "
                "village, which plans for no supply and does not shift.\n"
                "Colour: pattern, shaded by season (summer light, winter "
                "dark).")
    plot_information(
        info, keys, names, out_dir / "stochastic_sweep_repricing.png",
        seasons=seasons, weights=weights, field="{case}",
        cases=REPRICE_CASES, zero=True,
        ylabel="Gain from re-pricing   [$ / day]",
        title=f"What re-optimizing its price for a perfectly informed village "
              f"is worth to the leader, against prob, at {shift}",
        caption="The village learns the outcome before it schedules; the "
                "leader does not and posts one price.\n"
                "Gain = E[profit] with that one price re-optimized − E[profit] "
                "with the balanced price kept as posted. Non-negative by "
                "construction.\n"
                "Colour: pattern, shaded by season (summer light, winter "
                "dark).")
    if any(r.get("vss") is not None and _TOL < r["w"] < 1.0 - _TOL
           for r in info):
        plot_information(
            info, keys, names, out_dir / "stochastic_sweep_vss.png",
            seasons=seasons, weights=weights, field="{case}",
            cases=VSS_CASES, zero=True,
            ylabel="VSS   [$ / day]",
            title=f"What pricing against the belief, rather than against the "
                  f"mean day, is worth to the leader, against prob, at {shift}",
            caption="VSS = balanced E[profit] − EEV. EEV: the balanced model "
                    "with the price fixed at the mean-day (EV) price; the "
                    "village schedules against prob in both.\n"
                    "Mean day: in the pattern's hours EDL's cap is prob · "
                    "(1 + shift_frac) · D_t. Non-negative by construction; 0 "
                    "at prob = 0 and 1.\n"
                    "Colour: pattern, shaded by season (summer light, winter "
                    "dark).")
    if shifts and any(_real(v) for v in shifts.values()):
        plot_shift(shifts, keys, names, out_dir / "stochastic_sweep_shift.png",
                   seasons=seasons)


# --------------------------------------------------------------------------
def main(seasons: list[str] | None = None, figures: list[str] | None = None,
         weights: list[float] | None = None, pv_kwp: float = PV_KWP,
         dg_kw: float = DG_KW, levels: int | None = None,
         time_limit: float = TIME_LIMIT_S, replot: bool = False,
         reuse: bool = False, shift_sweep: bool = False) -> None:
    seasons = seasons or list(config.SEASONS.keys())
    figures = figures or list(FIGURES)
    weights = sorted(w for w in (weights or WEIGHTS) if w > _TOL)
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    book = out_dir / "stochastic_sweep.xlsx"

    # --replot draws only from the workbook; --reuse takes whatever runs it
    # already holds, as they are, and solves the rest.
    old = load_rows(book) if (replot or reuse) and book.exists() else {}
    if replot and not old:
        print(f"no workbook at {book} — nothing to redraw")
        return

    opts = {"MIPGap": config.CONFIG["MIPGap"], "TimeLimit": time_limit}
    step = menu_step(levels)
    cache: dict = {}
    balanced: dict[str, list[dict]] = {}
    uninformed: dict[str, list[dict]] = {}
    leader: dict[str, list[dict]] = {}
    naive: dict[str, list[dict]] = {}
    follower: dict[str, list[dict]] = {}
    adjusted: dict[str, list[dict]] = {}
    shifts: dict[str, list[dict]] = {}
    vss: dict[str, list[dict]] = {}

    for sw in AVAIL_SWEEPS:
        if sw.key not in figures:
            continue
        tag = sw.key.split("_")[-1]
        pv = pv_kwp if tag == "pv" else 0.0
        fleet = _fleet_label(dg_kw, pv)

        rows = _real(old.get(sw.key, []))
        if not rows and not replot:
            print(f"\n=== balanced, {fleet}, load shift "
                  f"{100 * BELIEF_SHIFT:.0f}%")
            rows = sweep(seasons, shifts=[BELIEF_SHIFT], weights=weights,
                         dg_kw=dg_kw, pv_kwp=pv, opts=opts, levels=levels,
                         step=step, cache=cache)
            if rows:
                report(rows, "w", "prob", step)
        if rows:
            balanced[sw.key] = rows

        name = f"uninformed_{tag}"
        rows = _real(old.get(name, []))
        if not rows and not replot:
            print(f"\n=== follower uninformed: each outcome at 0% load shift, "
                  f"{fleet}")
            rows = sweep(seasons, shifts=[0.0], weights=[1.0], dg_kw=dg_kw,
                         pv_kwp=pv, opts=opts, levels=levels, step=step,
                         cache=cache)
        if rows:
            uninformed[name] = rows

        # Follower uninformed's counterpart with an uninformed leader: one
        # committed price, a village that does not shift. prob = 1 and the
        # no-supply reference are the uninformed runs themselves, so they seed
        # the cache and only the interior weights are solved.
        name = f"naive_{tag}"
        rows = _real(old.get(name, []))
        if not rows and not replot:
            for r in _real(uninformed.get(f"uninformed_{tag}", [])):
                cache.setdefault((r["season"], dg_kw, pv, 0.0, r["series"],
                                  1.0), r)
            print(f"\n=== one committed price, village does not shift, "
                  f"{fleet}")
            interior = [w for w in weights if w < 1.0 - _TOL]
            rows = sweep(seasons, shifts=[0.0], weights=interior or [1.0],
                         dg_kw=dg_kw, pv_kwp=pv, opts=opts, levels=levels,
                         step=step, cache=cache)
        if rows:
            naive[name] = rows

        name = f"leader_{tag}"
        rows = _read_sheet(book, name, LEADER_FIELDS) if old else []
        if not rows and not replot:
            print(f"\n=== leader's advantage, {fleet}, load shift "
                  f"{100 * BELIEF_SHIFT:.0f}%")
            rows = leader_sweep(seasons, weights, dg_kw=dg_kw, pv_kwp=pv,
                                opts=opts, levels=levels)
        if rows:
            leader[name] = rows

        name = f"follower_{tag}"
        rows = _read_sheet(book, name, FOLLOWER_FIELDS) if old else []
        if not rows and not replot and balanced.get(sw.key):
            print(f"\n=== balanced price held, village informed, {fleet}")
            rows = follower_sweep(seasons, balanced[sw.key], dg_kw=dg_kw,
                                  pv_kwp=pv, opts=opts)
        if rows:
            follower[name] = rows

        name = f"adjusted_{tag}"
        rows = _read_sheet(book, name, ADJUSTED_FIELDS) if old else []
        if not rows and not replot:
            print(f"\n=== one price re-optimized for an informed village, "
                  f"{fleet}")
            rows = adjusted_sweep(seasons, weights, balanced.get(sw.key, []),
                                  dg_kw=dg_kw, pv_kwp=pv, opts=opts,
                                  levels=levels)
        if rows:
            adjusted[name] = rows

        name = f"vss_{tag}"
        rows = _read_sheet(book, name, VSS_FIELDS) if old else []
        if not rows and not replot and balanced.get(sw.key):
            print(f"\n=== value of the stochastic solution, {fleet}")
            rows = vss_sweep(seasons, weights, balanced[sw.key], dg_kw=dg_kw,
                             pv_kwp=pv, opts=opts, levels=levels)
        if rows:
            vss[name] = rows

        # The shift sweep runs only when asked, and solves only the points the
        # workbook lacks; once in the workbook it is always carried on.
        name = f"shift_{tag}"
        rows = _read_sheet(book, name, SHIFT_FIELDS) if old else []
        if shift_sweep and not replot:
            print(f"\n=== village shift at prob "
                  f"{', '.join(f'{100 * w:g}%' for w in SHIFT_WEIGHTS)}, "
                  f"{fleet}")
            rows = rows + shift_sweep_runs(seasons, dg_kw=dg_kw, pv_kwp=pv,
                                           opts=opts, levels=levels,
                                           have=rows)
        if rows:
            shifts[name] = rows

    if not balanced:
        print("Nothing solved or found — no figures or workbook written.")
        return

    info = []
    for key in PANEL_ORDER:
        if key in balanced:
            tag = key.split("_")[-1]
            info += information_rows(key, balanced[key],
                                     leader.get(f"leader_{tag}", []),
                                     uninformed.get(f"uninformed_{tag}", []),
                                     naive.get(f"naive_{tag}", []),
                                     follower.get(f"follower_{tag}", []),
                                     adjusted.get(f"adjusted_{tag}", []),
                                     vss.get(f"vss_{tag}", []))

    if not replot:
        save_analysis(balanced | uninformed | naive, book, levels)
        wb = load_workbook(book)
        for name, rows in shifts.items():
            _write_sheet(wb, name, SHIFT_FIELDS, rows,
                         "Balanced run at every prob solved so far. Energy "
                         "shifted = "
                         "1/2 sum_t |dd_t|; into the covered hours = sum of dd "
                         "over the hours the pattern names.")
        for name, rows in vss.items():
            _write_sheet(wb, name, VSS_FIELDS, rows,
                         "EV: certain run on the mean day, EDL cap prob * (1 "
                         "+ shift_frac) * D_t in the pattern's hours. EEV: "
                         "balanced model with the menu reduced to the EV "
                         "price. VSS = balanced - EEV. prob = 1 is the certain "
                         "run (VSS = 0) and is not repeated here.")
        for name, rows in adjusted.items():
            _write_sheet(wb, name, ADJUSTED_FIELDS, rows,
                         "One leader price re-optimized, the village informed "
                         "(shift per outcome). prob = 1 has one outcome and is "
                         "the balanced certain run.")
        for name, rows in follower.items():
            _write_sheet(wb, name, FOLLOWER_FIELDS, rows,
                         "Balanced price held, village informed: each outcome "
                         "a certain run with the menu reduced to the balanced "
                         "price. E[profit] = prob * profit if the pattern "
                         "arrives + (1 - prob) * profit if no supply.")
        for name, rows in leader.items():
            _write_sheet(wb, name, LEADER_FIELDS, rows,
                         "Leader's advantage: the leader learns the outcome "
                         "before pricing (one price per outcome); the village "
                         "schedules against prob. prob = 1 is the balanced "
                         "certain run and is not repeated here.")
        _write_sheet(wb, "information", INFO_FIELDS, info,
                     "Follower uninformed = prob * profit(pattern, 0% shift) "
                     "+ (1 - prob) * profit(no supply, 0% shift), from the "
                     "uninformed_* sheets. Leader's EVPI: by prob = leader's "
                     "advantage - balanced; no supply = follower uninformed - "
                     "naive (naive_* sheets).")
        wb.save(book)
        print(f"  wrote the leader and information sheets of {book}")
    else:
        weights = sorted({r["w"] for rows in balanced.values()
                          for r in _real(rows) if r["w"] > _TOL}) or weights

    report_information(info)
    plot_all(balanced, info, out_dir, seasons, weights, shifts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The leader's committed price against its belief "
                    "prob(pattern) in the synthetic EDL timing patterns (peak, "
                    "offpeak), and its expected profit under three information "
                    "cases, on a diesel-only and on a PV fleet.")
    parser.add_argument("--seasons", nargs="+",
                        default=list(config.SEASONS.keys()),
                        choices=list(config.SEASONS.keys()))
    parser.add_argument("--figures", nargs="+", default=list(FIGURES),
                        choices=list(FIGURES),
                        help="Fleets to run and draw (default: both).")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        metavar="W",
                        help=f"Belief weights prob(pattern), in (0, 1] "
                             f"(default "
                             f"{' '.join(f'{w:g}' for w in WEIGHTS if w > 0)}).")
    parser.add_argument("--pv", type=float, default=PV_KWP, metavar="KWP",
                        help=f"PV of the belief_pv sweep (default {PV_KWP:g}). "
                             f"belief_dg is always diesel only.")
    parser.add_argument("--dg", type=float, default=DG_KW, metavar="KW",
                        help=f"Diesel capacity (default {DG_KW:g}).")
    parser.add_argument("--levels", type=int, default=None,
                        help="Re-space the price menu to this many levels.")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_S,
                        metavar="S", help="Solver time limit per solve.")
    parser.add_argument("--reuse", action="store_true",
                        help="Take the runs the existing workbook already "
                             "holds and solve only the missing ones.")
    parser.add_argument("--shift-sweep", action="store_true",
                        help="Also solve the balanced run at prob 0.5-2%% in "
                             "0.5%% steps and draw the energy the village "
                             "shifts; points already in the workbook are not "
                             "solved again.")
    parser.add_argument("--replot", action="store_true",
                        help="Redraw the figures from the existing workbook "
                             "without solving anything.")
    args = parser.parse_args()
    if args.weights and not all(0.0 <= w <= 1.0 for w in args.weights):
        parser.error("--weights must lie in [0, 1]: w is a probability.")
    if args.pv < 0.0 or args.dg <= 0.0:
        parser.error("--pv must be >= 0 and --dg positive.")
    main(args.seasons, args.figures, args.weights, args.pv, args.dg,
         args.levels, args.time_limit, args.replot, args.reuse,
         args.shift_sweep)
