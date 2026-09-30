"""
How the leader's BELIEF about EDL moves its price and its profit.

:mod:`scenario_analysis` asks what committing to the wrong outcome costs.
This module asks something upstream of that: how much does the leader's belief
matter at all? It sweeps the weight the leader puts on one EDL outcome and
re-optimizes at every point.

THE BELIEF BEING SWEPT
Each supply scenario s is given a probability w, and NO SUPPLY absorbs the
residual:

    P(s) = w ,   P(none) = 1 - w ,   w in WEIGHTS = 0.20, 0.50, 1.00

so the leader prices against a two-point belief that slides from "EDL will
probably deliver nothing" to "outcome s is certain". At w = 1 this is the
deterministic run of :mod:`scenario_analysis`'s diagonal; the w -> 0 limit is
the pure no-supply run, the same for every curve, which is solved once and
drawn as a reference.

Three weights, not a fine grid. The price is very nearly linear in w -- that is
this module's own finding, not an assumption imported into it -- so intermediate
points restate what the two ends already say, at one full re-optimization each.
The interior weight is what keeps the claim falsifiable: it is where a curve
that is NOT linear would show, and where ``summarize``'s convexity check reads
the profit curve against its chord. Widen :data:`WEIGHTS` (or pass --weights)
if a curve ever looks suspicious there.

``none`` is not itself swept: it IS the residual, so its curve would be the
constant P(none) = 1 and carry no information.

THE STOCHASTIC COMMITMENT
Every sweep above is a counterfactual -- what the leader WOULD do believing one
outcome. The run it actually makes is the stochastic commitment: all six outcomes at their
real probabilities, one price committed before the day starts. It is solved
once per season and drawn on both figures as a heavy dash-dotted line, so each
counterfactual range can be read as a distance from the real decision. It is a
single point, not a range, and is deliberately not given a row of its own.

Only the two outcomes in play are put in the model's scenario set. Carrying the
other four at probability zero would leave their recourse variables out of the
objective entirely -- feasible but unpriced, so the solver could return any
dispatch for them -- and would cost binaries for nothing.

WHAT TO LOOK FOR
Under the residual-based demand response (see :mod:`model.bilevel_mpec`) EDL
availability scales the MGO's residual demand rather than shifting it, so it
moves the VOLUME the microgrid sells without moving the price that maximizes
the margin on it. The belief sweep is the sharpest test of that: if the law
holds, the profit curves fan out strongly while the price curves stay nearly
flat. The summary reports the price span of each curve so this is a number
rather than an impression.

Expected profit is a maximum over p of a family linear in w, hence CONVEX in w:
the curves may bend upward, never downward.

Outputs, under :data:`config.OUTPUT_DIR`:

    stochastic_analysis.xlsx        one sheet per season with every swept point,
                                    plus a summary of price span and profit range
    stochastic_price_range_<season>.png
    stochastic_profit_range_<season>.png
                                    the span of each outcome as one horizontal
                                    range, over the swept weights. Built the
                                    same way from the same sweep, so the price
                                    figure's stub lines and the profit figure's
                                    long ones ARE the module's claim, side by
                                    side.

Usage:
    python stochastic_analysis.py                      # all three seasons
    python stochastic_analysis.py --seasons winter
    python stochastic_analysis.py --weights 0.2 0.4 0.6 0.8 1.0
    python stochastic_analysis.py --scenarios announced delayed --levels 65
"""

import argparse

import matplotlib
matplotlib.use("Agg")             # headless / file output
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from pyomo.environ import value, TerminationCondition

import config
from model.bilevel_mpec import BilevelMPEC
from plot import SCENARIO_COLOR, STOCHASTIC, STOCHASTIC_LABEL
from run import build_instance
from scenarios import SCENARIOS, SCENARIO_LABEL

_OK = (TerminationCondition.optimal, TerminationCondition.feasible)
_TOL = 1e-6

NONE = "none"                     # the residual outcome: EDL delivers nothing
# The belief weights actually solved. The sweep is not looking for structure
# BETWEEN the points -- the price moves almost linearly in w, which is the
# module's own finding -- so a fine even grid buys resolution nobody reads and
# costs a full re-optimization per point. Three weights pin the two ends and
# one interior point, which is exactly what the range figure draws and still
# enough to see the profit curve bend away from its chord.
WEIGHTS: tuple[float, ...] = (0.20, 0.50, 1.00)

# Every scenario except the residual itself.
SWEPT: tuple[str, ...] = tuple(sc.key for sc in SCENARIOS if sc.key != NONE)

# Colours follow the SCENARIO, never the plotting order, and are shared with
# every other figure through :data:`plot.SCENARIO_COLOR`.
_SCENARIO_COLOR = {k: SCENARIO_COLOR[k] for k in SWEPT}

FIELDS = [
    ("scenario",    "scenario"),
    ("label",       "outcome"),
    ("w",           "P(scenario) [-]"),
    ("w_none",      "P(no supply) [-]"),
    ("x",           "leader price [x P_ref]"),
    ("p",           "leader price [$/kWh]"),
    ("p_min",       "min hourly price [$/kWh]"),
    ("p_max",       "max hourly price [$/kWh]"),
    ("profit",      "leader profit [$]"),
    ("revenue",     "leader revenue [$]"),
    ("gen_cost",    "generation cost [$]"),
    ("qMGO",        "E[microgrid sales] [kWh]"),
    ("qPG",         "E[EDL delivered] [kWh]"),
    ("qsupp",       "E[suppressed demand] [kWh]"),
    ("qcri",        "E[unserved] [kWh]"),
    ("D",           "sum D [kWh]"),
    ("supp_share",  "E[qsupp] / D [%]"),
    ("degenerate",  "price arbitrary (no sales)"),
    ("termination", "termination"),
]


# --------------------------------------------------------------------------
# Solving
# --------------------------------------------------------------------------
def _refine_menu(data: dict, levels: int | None) -> None:
    """
    Re-space the price menu over [P_ref, Phat] with ``levels`` entries.

    The sweep reads a price SPAN off the curves, so the menu step is the
    resolution of the very quantity under test; ``None`` keeps the configured
    menu.
    """
    if not levels:
        return
    T = data["T"]
    P_ref, Phat = data["P"][T[0]], data["Phat"][T[0]]
    data["Pi"] = [P_ref + (Phat - P_ref) * k / (levels - 1)
                  for k in range(levels)]


def _instance(season: str, key: str, w: float, levels: int | None) -> dict:
    """
    Instance data for ``season`` under the belief P(key) = w, P(none) = 1 - w.

    At w = 1 the residual is dropped rather than carried at probability zero,
    which would leave its recourse variables outside the objective.
    """
    data = dict(build_instance(season))
    T = data["T"]
    base = data["QbarPG"]

    S = [key] if w >= 1.0 - _TOL else [key, NONE]
    prob = {key: 1.0} if len(S) == 1 else {key: w, NONE: 1.0 - w}

    data["S"] = S
    data["prob"] = prob
    data["QbarPG"] = {(t, s): base[(t, s)] for t in T for s in S}
    _refine_menu(data, levels)
    return data


def _record(m, data: dict, key: str, w: float | None) -> dict:
    """
    Leader and follower totals of one solved belief point.

    ``w`` is ``None`` for the stochastic commitment, which is not a point on any belief
    sweep: it carries the real scenario probabilities rather than a weight put
    on one outcome.
    """
    T, S, prob = data["T"], data["S"], data["prob"]
    P_ref = data["P"][T[0]]

    def exp_sum(var):
        return sum(prob[s] * value(var[t, s]) for t in T for s in S)

    qMGO, qPG = exp_sum(m.qMGO), exp_sum(m.qPG)
    qsupp, qcri = exp_sum(m.qsupp), exp_sum(m.qcri)
    sD = sum(data["D"][t] for t in T)

    p_vals = [value(m.p[t]) for t in T]
    p = sum(p_vals) / len(p_vals)          # == p_t itself under a flat tariff
    degenerate = qMGO <= _TOL              # sells nothing: its price is arbitrary

    revenue = sum(prob[s] * value(m.p[t]) * value(m.qMGO[t, s])
                  for t in T for s in S)
    gen_cost = sum(prob[s] * (data["cdiesel"][t] * value(m.qdiesel[t, s])
                              + data["cPV"][t] * value(m.qPV[t, s]))
                   for t in T for s in S)

    return {
        "scenario": key,
        "label": SCENARIO_LABEL.get(key, key),
        "w": w,
        "w_none": None if w is None else 1.0 - w,
        "x": None if degenerate else p / P_ref,
        "p": None if degenerate else p,
        "p_min": None if degenerate else min(p_vals),
        "p_max": None if degenerate else max(p_vals),
        "profit": value(m.profit),
        "revenue": revenue,
        "gen_cost": gen_cost,
        "qMGO": qMGO, "qPG": qPG, "qsupp": qsupp, "qcri": qcri, "D": sD,
        "supp_share": 100.0 * qsupp / sD if sD else 0.0,
        "degenerate": degenerate,
        "termination": "",
    }


def solve_at_belief(season: str, key: str, w: float, opts: dict,
                    levels: int | None) -> dict | None:
    """Re-optimize the leader against one belief, or ``None`` if it did not
    solve."""
    data = _instance(season, key, w, levels)

    mpec = BilevelMPEC(data)
    m = mpec.build()
    solver_name, res = mpec.solve("gurobi", options=opts)
    if solver_name is None:
        raise RuntimeError(
            "Gurobi not available (check the license and that gurobipy is "
            "installed)."
        )

    tc = res.solver.termination_condition
    if tc not in _OK:
        print(f"  {key} @ w={w:.2f}: not solved ({tc})")
        return None

    rec = _record(m, data, key, w)
    rec["termination"] = str(tc)
    return rec


def sweep(season: str, scenarios: list[str], weights: list[float], opts: dict,
          levels: int | None) -> dict[str, list[dict]]:
    """Sweep the belief weight for each scenario."""
    out: dict[str, list[dict]] = {}
    for key in scenarios:
        print(f"[{season}] scenario {key} ...")
        rows = []
        for w in sorted(weights):
            rec = solve_at_belief(season, key, w, opts, levels)
            if rec is None:
                continue
            rows.append(rec)
            x = "   -  " if rec["x"] is None else f"{rec['x']:6.3f}"
            print(f"  P({key})={w:4.2f}  P(none)={1 - w:4.2f}  x*={x}  "
                  f"profit={rec['profit']:8.2f}  E[qMGO]={rec['qMGO']:7.1f}  "
                  f"E[qPG]={rec['qPG']:7.1f}")
        if rows:
            out[key] = rows
    return out


def no_supply_reference(season: str, opts: dict,
                        levels: int | None) -> dict | None:
    """
    The pure no-supply outcome: the w -> 0 limit shared by every curve.

    Solved as P(none) = 1, which is what ``_instance`` produces for the
    residual itself at w = 1.
    """
    print(f"[{season}] no-supply reference (P(none) = 1) ...")
    rec = solve_at_belief(season, NONE, 1.0, opts, levels)
    if rec is not None:
        rec["w"], rec["w_none"] = 0.0, 1.0     # placed at the w -> 0 limit
        x = "   -  " if rec["x"] is None else f"{rec['x']:6.3f}"
        print(f"  x*={x}  profit={rec['profit']:8.2f}")
    return rec


def commitment_run(season: str, opts: dict, levels: int | None) -> dict | None:
    """
    The leader's ACTUAL commitment: all six outcomes at their real
    probabilities from :data:`scenarios.SCENARIOS`.

    This is the run the thesis reports, and the only one the MGO can really
    make -- it commits a single price before knowing which outcome arrives. The
    belief sweeps around it are counterfactuals: each asks what the leader would
    do if it believed one outcome instead. Drawing the commitment point on the same
    axis is what turns those counterfactuals into a statement about the real
    decision, by showing how far from it any single-outcome belief would pull.

    Not a point on any sweep, so it carries no weight ``w``.
    """
    print(f"[{season}] stochastic commitment (all outcomes at their real P) ...")
    data = dict(build_instance(season))
    _refine_menu(data, levels)

    mpec = BilevelMPEC(data)
    m = mpec.build()
    solver_name, res = mpec.solve("gurobi", options=opts)
    if solver_name is None:
        raise RuntimeError(
            "Gurobi not available (check the license and that gurobipy is "
            "installed)."
        )

    tc = res.solver.termination_condition
    if tc not in _OK:
        print(f"  stochastic commitment: not solved ({tc})")
        return None

    rec = _record(m, data, STOCHASTIC, None)
    rec["label"] = STOCHASTIC_LABEL
    rec["termination"] = str(tc)
    x = "   -  " if rec["x"] is None else f"{rec['x']:6.3f}"
    print(f"  x*={x}  profit={rec['profit']:8.2f}  "
          f"E[qMGO]={rec['qMGO']:7.1f}")
    return rec


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def summarize(rows: list[dict]) -> dict:
    """
    Price span and profit range of one scenario's curve.

    ``price_span`` is the headline: how far the leader's price moves across the
    whole belief range, in units of P_ref. Under the residual-based demand
    response it should be small next to the profit's own range, because
    availability scales the residual instead of shifting it.

    ``curvature`` is the largest gap between the profit curve and the straight
    line joining its endpoints, signed so that positive means bending UPWARD --
    the convexity that a maximum over p of a w-linear family must have. A
    materially negative value would mean a point failed to reach its optimum.
    """
    xs = [r["x"] for r in rows if r["x"] is not None]
    ws = np.array([r["w"] for r in rows], dtype=float)
    pf = np.array([r["profit"] for r in rows], dtype=float)

    curvature = None
    if len(ws) > 2 and ws[-1] > ws[0]:
        chord = pf[0] + (pf[-1] - pf[0]) * (ws - ws[0]) / (ws[-1] - ws[0])
        dev = pf - chord
        curvature = float(dev[np.argmax(np.abs(dev))])

    return {
        "scenario": rows[0]["scenario"],
        "label": rows[0]["label"],
        "x_min": min(xs) if xs else None,
        "x_max": max(xs) if xs else None,
        "price_span": (max(xs) - min(xs)) if xs else None,
        "profit_lo": float(pf.min()),
        "profit_hi": float(pf.max()),
        "profit_range": float(pf.max() - pf.min()),
        "profit_at_wmin": float(pf[0]),
        "profit_at_wmax": float(pf[-1]),
        "curvature": curvature,
        "n": len(rows),
    }


def swept_weights(results: dict[str, list[dict]]) -> list[float]:
    """The belief weights actually present in a result set, ascending."""
    return sorted({r["w"] for rows in results.values() for r in rows})


def print_summary(season: str, results: dict[str, list[dict]],
                  ref: dict | None, commitment: dict | None = None) -> None:
    ws = swept_weights(results)
    grid = ", ".join(f"{w:.2f}" for w in ws) if ws else "none"
    print(f"\n[{season}] belief sweep summary "
          f"(P(scenario) = {grid}; no supply as residual)")
    if ref is not None:
        rx = "  -  " if ref["x"] is None else f"{ref['x']:.3f}"
        print(f"  reference, P(no supply) = 1: x* = {rx}, "
              f"profit = {ref['profit']:.2f} $/day")
    if commitment is not None:
        hx = ("  -  " if commitment["x"] is None
              else f"{commitment['x']:.3f}")
        print(f"  stochastic commitment:      x* = {hx}, "
              f"profit = {commitment['profit']:.2f} $/day   <- the actual "
              f"commitment")
    head = (f"  {'outcome':<17}{'x* range':>18}{'span':>8}"
            f"{f'profit @{ws[0]:.2f}':>13}{f'profit @{ws[-1]:.2f}':>13}"
            f"{'range':>10}")
    print(head)
    for key in results:
        s = summarize(results[key])
        rng = ("     -  " if s["x_min"] is None
               else f"{s['x_min']:.3f} - {s['x_max']:.3f}")
        span = "   -  " if s["price_span"] is None else f"{s['price_span']:.3f}"
        print(f"  {s['label']:<17}{rng:>18}{span:>8}"
              f"{s['profit_at_wmin']:>13.2f}{s['profit_at_wmax']:>13.2f}"
              f"{s['profit_range']:>10.2f}")

    spans = [summarize(r)["price_span"] for r in results.values()]
    spans = [s for s in spans if s is not None]
    ranges = [summarize(r)["profit_range"] for r in results.values()]
    if spans:
        print(f"  widest price span {max(spans):.3f} x P_ref against a profit "
              f"range up to {max(ranges):.2f} $/day")


# --------------------------------------------------------------------------
# Excel
# --------------------------------------------------------------------------
def _write_rows(ws, header: list, rows) -> None:
    ws.append(header)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in rows:
        ws.append(list(row))
    ws.freeze_panes = "A2"
    for i, label in enumerate(header, start=1):
        ws.column_dimensions[get_column_letter(i)].width = min(len(str(label)) + 3, 30)


def save_analysis(per_season: dict[str, dict], out_path) -> None:
    """One sheet per season, plus the summary of price span and profit range."""
    wb = Workbook()
    wb.remove(wb.active)

    # The swept grid is a run-time choice, so the two end columns are named
    # after the weights actually solved rather than after a fixed 0.1/1.0.
    all_results = {k: v for blob in per_season.values()
                   for k, v in blob["results"].items()}
    ends = swept_weights(all_results) or [0.0, 1.0]

    ws = wb.create_sheet("summary")
    header = ["season", "scenario", "outcome", "min x* [x P_ref]",
              "max x* [x P_ref]", "price span [x P_ref]",
              f"profit at P={ends[0]:.2f} [$]", f"profit at P={ends[-1]:.2f} [$]",
              "min profit [$]", "max profit [$]", "profit range [$]",
              "convexity gap vs chord [$]", "points"]
    keys = ["scenario", "label", "x_min", "x_max", "price_span",
            "profit_at_wmin", "profit_at_wmax", "profit_lo", "profit_hi",
            "profit_range", "curvature", "n"]
    rows = []
    for season, blob in per_season.items():
        for key in blob["results"]:
            s = summarize(blob["results"][key])
            rows.append([season] + [s[k] for k in keys])
        # The two single runs have no span of their own: one value repeated
        # across the range columns, so they line up with the swept rows.
        for rec, label in ((blob["reference"], "No supply (reference, P = 1)"),
                           (blob.get("commitment"),
                            f"{STOCHASTIC_LABEL} (real probabilities)")):
            if rec is None:
                continue
            rows.append([season, rec["scenario"], label,
                         rec["x"], rec["x"], 0.0, rec["profit"], rec["profit"],
                         rec["profit"], rec["profit"], 0.0, None, 1])
        rows.append([])
    _write_rows(ws, header, rows)

    for season, blob in per_season.items():
        rows = [r for key in blob["results"] for r in blob["results"][key]]
        # Both single runs first, so the sheet opens on the two points the
        # swept rows are read against.
        rows = [r for r in (blob.get("commitment"), blob["reference"])
                if r is not None] + rows
        _write_rows(wb.create_sheet(f"{season}"[:31]),
                    [lbl for _, lbl in FIELDS],
                    ([r[k] for k, _ in FIELDS] for r in rows))

    wb.save(out_path)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def _range_rows(results: dict[str, list[dict]],
                ykey: str) -> tuple[list[dict], list[str]]:
    """
    One range record per outcome, plus the outcomes that had nothing to draw.

    A record collapses that outcome's belief sweep to the two ends the figure
    needs -- the value at the lowest swept weight and at the highest -- while
    keeping every intermediate value, so the drawing can show that the ends are
    not the only samples.
    """
    rows, empty = [], []
    for key in [k for k in SWEPT if k in results]:
        pts = sorted((r["w"], r[ykey]) for r in results[key]
                     if r[ykey] is not None)
        if not pts:
            empty.append(SCENARIO_LABEL.get(key, key))
            continue
        rows.append({
            "key": key,
            "label": SCENARIO_LABEL.get(key, key),
            "w_start": pts[0][0],  "v_start": pts[0][1],
            "w_end":   pts[-1][0], "v_end":   pts[-1][1],
            "v_all": [v for _, v in pts],
            "n": len(pts),
            "drop": len(results[key]) - len(pts),
        })
    return rows, empty


def _plot_range(season: str, blob: dict, ykey: str, out_path, *,
                xlabel: str, title: str, fmt: str,
                marks: list[tuple] = ()) -> None:
    """
    One horizontal range per EDL outcome: the quantity on x, the outcome on y.

    Each line is that outcome's whole belief sweep collapsed to its two ends --
    it STARTS at the value the leader lands on at the lowest swept weight (no
    supply taking the rest) and RUNS to the value it lands on at the highest,
    normally certainty. The length of the line is therefore the span the summary
    reports, read directly off the figure, and the direction says which way
    believing in the outcome pushes the quantity.

    The swept grid is read off the data rather than assumed, so the figure
    labels itself correctly whatever :data:`WEIGHTS` is set to.

    Drawn for the price and for the profit from the same data, which is the
    point: side by side, the price lines are stubs and the profit lines cross
    the chart. That contrast IS the module's claim, and putting the two figures
    in one visual language is what makes it legible at a glance.

    The swept points in between are drawn as ticks on the line, because the ends
    alone would hide a non-monotone curve. Where the curve leaves the interval
    spanned by its two ends, that excursion is drawn as a faint band behind the
    line, so the line is never mistaken for the full range.

    ``marks`` are vertical reference lines, each ``(value, label, linestyle)``.
    Degenerate points (the MGO sells nothing, so its price is arbitrary) carry
    no price and are dropped from the price figure; an outcome with no value at
    all is named in a footnote rather than drawn as an empty row. Profit is
    defined at every point, so its figure keeps all of them.
    """
    rows, empty = _range_rows(blob["results"], ykey)
    if not rows:
        print(f"  no {ykey} value in any outcome — skipping {out_path}")
        return

    ws = swept_weights(blob["results"])
    w_lo, w_hi = ws[0], ws[-1]

    fig, ax = plt.subplots(figsize=(10, 1.05 * len(rows) + 3.0))
    ys = list(range(len(rows) - 1, -1, -1))       # first outcome at the top

    for row, y in zip(rows, ys):
        color = _SCENARIO_COLOR.get(row["key"], "#444444")
        start, end = row["v_start"], row["v_end"]
        lo, hi = min(row["v_all"]), max(row["v_all"])

        # The excursion beyond the two ends, if the curve is not monotone.
        if hi - lo > abs(end - start) + _TOL:
            ax.hlines(y, lo, hi, color=color, alpha=0.25, linewidth=9,
                      zorder=1)
        ax.hlines(y, start, end, color=color, linewidth=2.6, zorder=2)
        # Every swept weight, so the ends are not read as the only samples.
        ax.plot(row["v_all"], [y] * row["n"], linestyle="none", marker="|",
                markersize=9, markeredgewidth=1.2, color="#333333", alpha=0.7,
                zorder=3)
        ax.plot([start], [y], marker="o", markersize=8, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=2.0, zorder=4)
        ax.plot([end], [y], marker="D", markersize=8, markerfacecolor=color,
                markeredgecolor="#222222", markeredgewidth=0.7, zorder=4)

        # Value at each end, always on the outer side of its own marker.
        right = end >= start
        for value_, is_end in ((start, False), (end, True)):
            outward = 1 if (is_end == right) else -1
            ax.annotate(fmt.format(value_), (value_, y),
                        xytext=(10 * outward, 0), textcoords="offset points",
                        ha="left" if outward > 0 else "right", va="center",
                        fontsize=8.5, color="#222222")

        note = f"span {fmt.format(abs(end - start))}   n={row['n']}"
        if row["drop"]:
            note += f" (+{row['drop']} degen.)"
        # A dropped end means the line stops short of the swept range: say
        # where it actually stops rather than let the legend speak for it.
        if row["w_start"] > w_lo + _TOL or row["w_end"] < w_hi - _TOL:
            note += f"\nprob = {row['w_start']:.2f} … {row['w_end']:.2f}"
        ax.annotate(note, (0.5 * (start + end), y), xytext=(0, 11),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.5, color="#333333")

    # Vertical references, drawn under the ranges. The commitment gets its own
    # weight and colour because it is not a reference the sweep passes through
    # -- it is the commitment the leader actually makes.
    for value_, label, style, color, lw in marks:
        if value_ is None:
            continue
        ax.axvline(value_, color=color, linewidth=lw, linestyle=style,
                   zorder=0)
        if label:
            ax.annotate(label, (value_, 0.0),
                        xycoords=("data", "axes fraction"), xytext=(4, 6),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=8, color=color)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("EDL supply outcome")
    ax.set_title(
        f"{title} — {season}\n"
        f"each line runs from prob = {w_lo:.2f} to {w_hi:.2f} "
        f"({len(ws)} weights solved), "
        "no supply taking the rest", fontsize=12)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.margins(x=0.16)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D([], [], linestyle="none", marker="o", markersize=8,
                   markerfacecolor="white", markeredgecolor="#444444",
                   markeredgewidth=2.0,
                   label=f"prob = {w_lo:.2f}"),
        plt.Line2D([], [], linestyle="none", marker="D", markersize=8,
                   markerfacecolor="#444444", markeredgecolor="#222222",
                   label=f"prob = {w_hi:.2f}"
                         + ("  (certain)" if w_hi >= 1.0 - _TOL else "")),
        plt.Line2D([], [], linestyle="none", marker="|", markersize=9,
                   markeredgewidth=1.2, color="#333333",
                   label="intermediate weights"),
    ]
    ax.legend(handles=handles, fontsize=8, framealpha=0.9,
              loc="lower right", ncol=3)

    if empty:
        ax.annotate("no value (MGO sells nothing): " + ", ".join(empty),
                    (0.01, 0.01), xycoords="axes fraction", fontsize=7.5,
                    color="#999999")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_ranges(season: str, blob: dict, price_path, profit_path) -> None:
    """
    The belief sweep as two range figures: the leader's price, and its profit.

    Same construction, same season, same outcomes -- so the price figure's short
    lines and the profit figure's long ones can be read against each other
    directly. Each carries the other's headline number in its axis label, so a
    slide showing only one of them still states the comparison.
    """
    results = blob["results"]
    ref, commitment = blob["reference"], blob.get("commitment")
    P_ref = config.CONFIG["P_ref"]

    spans = [s for s in (summarize(r)["price_span"] for r in results.values())
             if s is not None]
    ranges = [summarize(r)["profit_range"] for r in results.values()]
    # price_span is measured in multiples of P_ref; the figures are in $/kWh.
    span_txt = f"{max(spans) * P_ref:.3f}" if spans else "n/a"
    range_txt = f"{max(ranges):,.0f}" if ranges else "n/a"

    def _mark(rec, key, fmt, label):
        """A vertical reference at ``rec[key]``, or a no-op if it has none."""
        return (None if rec is None or rec[key] is None else rec[key],
                None if rec is None or rec[key] is None
                else f"{label}  {fmt.format(rec[key])}")

    # Grey for the limits the sweep runs between, near-black and heavier for the
    # stochastic commitment: it is the decision, not a reference.
    GREY, INK = "#666666", "#111111"

    p_commit = _mark(commitment, "p", "{:.3f}", "commitment:")
    _plot_range(
        season, blob, "p", price_path,
        xlabel=("Leader price $\\tilde p$  [\\$ / kWh]"
                f"      widest span {span_txt} \\$/kWh, against a profit range "
                f"up to {range_txt} \\$/day"),
        title="Where the leader's belief moves the price",
        fmt="{:.3f}",
        marks=[(P_ref, f"$P_{{ref}}$ = {P_ref:.3f}", ":", GREY, 1.1),
               (*_mark(ref, "p", "{:.3f}", "no supply, prob = 1:"), "--", GREY,
                1.1),
               (*p_commit, "-.", INK, 2.0)])

    _plot_range(
        season, blob, "profit", profit_path,
        xlabel=("Leader profit  [\\$ / day]"
                f"      widest range {range_txt} \\$/day, against a price span "
                f"of only {span_txt} \\$/kWh"),
        title="Where the leader's belief moves the profit",
        fmt="{:,.1f}",
        marks=[(0.0, None, "-", GREY, 0.8),
               (*_mark(ref, "profit", "{:,.1f}", "no supply, prob = 1:"), "--",
                GREY, 1.1),
               (*_mark(commitment, "profit", "{:,.1f}", "commitment:"), "-.",
                INK,
                2.0)])



# --------------------------------------------------------------------------
def main(seasons: list[str] | None = None,
         scenarios: list[str] | None = None,
         weights: list[float] | None = None, levels: int | None = None,
         reference: bool = True, commitment: bool = True) -> None:
    seasons = seasons or list(config.SEASONS.keys())
    scenarios = scenarios or list(SWEPT)
    weights = sorted(weights or WEIGHTS)
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = {"MIPGap": config.CONFIG["MIPGap"],
            "TimeLimit": config.CONFIG["TimeLimit"]}

    extra = int(reference) + int(commitment)
    n = len(seasons) * (len(scenarios) * len(weights) + extra)
    print(f"{n} solves: {len(seasons)} season(s) x "
          f"({len(scenarios)} outcomes x {len(weights)} weights"
          + (f" + {extra} single run(s)" if extra else "") + ")")

    per_season: dict[str, dict] = {}
    for season in seasons:
        results = sweep(season, scenarios, weights, opts, levels)
        if not results:
            print(f"[{season}] nothing solved.")
            continue

        blob = {
            "results": results,
            "reference": (no_supply_reference(season, opts, levels)
                          if reference else None),
            "commitment": (commitment_run(season, opts, levels)
                           if commitment else None),
        }
        per_season[season] = blob

        print_summary(season, results, blob["reference"],
                      blob["commitment"])
        plot_ranges(season, blob,
                    out_dir / f"stochastic_price_range_{season}.png",
                    out_dir / f"stochastic_profit_range_{season}.png")

    if not per_season:
        print("No season solved — nothing to write.")
        return
    save_analysis(per_season, out_dir / "stochastic_analysis.xlsx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep the leader's belief in each EDL supply outcome over "
                    "a few weights, with no supply as the residual.")
    parser.add_argument("--seasons", nargs="+",
                        default=list(config.SEASONS.keys()),
                        choices=list(config.SEASONS.keys()),
                        help="Seasons to sweep (default: all three). Each is a "
                             "full re-optimization per belief point, so one "
                             "season is much quicker.")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        choices=list(SWEPT),
                        help="Supply outcomes to sweep (default: all). "
                             f"{NONE!r} is the residual and cannot be swept.")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        metavar="W",
                        help="Belief weights P(outcome) to solve, in (0, 1] "
                             f"(default {' '.join(f'{w:g}' for w in WEIGHTS)}). "
                             "Each weight is a full re-optimization per "
                             "outcome and season, so this is the main cost "
                             "knob.")
    parser.add_argument("--levels", type=int, default=None,
                        help="Re-space the price menu to this many levels; "
                             "default keeps the configured menu.")
    parser.add_argument("--no-reference", action="store_true",
                        help="Skip the pure no-supply reference solve.")
    parser.add_argument("--no-commitment", action="store_true",
                        help="Skip the stochastic commitment (all outcomes at their real "
                             "probabilities), the commitment the sweeps are "
                             "read against.")
    args = parser.parse_args()
    if args.weights is not None and not all(0.0 < w <= 1.0
                                            for w in args.weights):
        parser.error("--weights must lie in (0, 1]: w is a probability, and "
                     "w = 0 is the no-supply reference, solved separately.")
    main(args.seasons, args.scenarios, args.weights, args.levels,
         reference=not args.no_reference,
         commitment=not args.no_commitment)
