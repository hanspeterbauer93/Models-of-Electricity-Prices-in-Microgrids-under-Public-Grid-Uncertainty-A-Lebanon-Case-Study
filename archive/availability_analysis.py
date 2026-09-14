"""
How public-grid (EDL) availability moves the leader's price and profit.

The six scenarios in :mod:`scenarios` are the case study: named, discrete
outcomes. To show the influence of availability GENERALLY this module sweeps a
continuous knob instead -- an availability profile carrying a chosen share of
daily demand, laid out over the day by a chosen pattern -- and re-optimizes the
leader at every point. The result is a comparative-statics curve, not a
sensitivity around one solution.

THE LAW BEING TESTED
Suppression applies to the uncovered BASELINE load Rbar_t = [D_t - Qbar_t]^+,
so in an hour the MGO serves, what it actually sells is

    q^{cov,MGO}_t = (1 - alpha1*(x - o)) * (D_t - Qbar_t)  +  dd_t

with x = p/P_ref and o = 1 under the "deviation" response, 0 under "level".
Availability therefore SCALES the residual instead of shifting its intercept,
and load shifting enters additively beside it. Over the hours the MGO still
serves (H), with A = sum_H D_t, B = sum_H Qbar_t, W = A - B and
shifted = -sum_H dd_t, maximizing (x*P_ref - c) * sum_t q^{cov,MGO}_t gives

    x*  = 0.5 * (o + 1/alpha1 + c/P_ref)  -  (1/(2*alpha1)) * sigma
    pi* = (P_ref*W / (4*alpha1)) * (G - sigma)^2

    sigma = shifted / W          the shifted load, per unit of uncovered market
    G     = (1 + o*alpha1) - alpha1*c/P_ref

This is the substantive difference from a response tied to gross demand. There,
availability was additive in the residual, so the price fell linearly in B/A.
Here it is multiplicative, and the maximizer of a SCALED function is the
function's own maximizer: availability no longer moves the price directly at
all. What is left of the price effect runs through two indirect channels:

    EXTENSIVE  which hours are in H, and hence W -- availability that covers an
               hour outright removes it from the MGO's market, shrinking the
               profit but not the price
    SHIFTING   sigma -- the follower moves load out of the served hours into the
               covered ones, and that IS priced in, at the familiar slope

    slope      d x* / d sigma = -1/(2*alpha1)
    threshold  sigma = G, where profit reaches zero and x* has fallen exactly to
               c/P_ref (price = marginal cost)

With no shifting at all (sigma = 0) the leader posts x* = 0.5*(o + 1/alpha1 +
c/P_ref) whatever EDL does, and only its VOLUME responds. That is the headline.

WHAT THE SWEEP ADDS
The law is exact only while no hour leaves the MGO's market. Real, window-
shaped availability kinks: each time an hour flips out of H, A and B jump and
the curve switches to a new quadratic. That deviation from the smooth law IS
the effect of TIMING, which is why the sweep runs several patterns at equal
energy and plots them against the raw share (what EDL delivers), against B_eff/A
(what it takes from the microgrid, kept for continuity with the policy view) and
against sigma, which is what the law above actually predicts on.

Unlike the additive form, the extensive margin no longer depends on the price:
an hour leaves H when Qbar_t >= D_t, whatever x is.

WHERE THE LAW APPLIES
Differentiating in x above holds the shifted load fixed, while the follower
actually re-optimizes it against every price. That is harmless while shifted is
small next to W, which covers the sweep wherever the MGO still has a market:
over those points the winter fit gives slope -1.08 against a predicted -1.04,
intercept 1.670 against 1.669, R^2 = 0.99. It fails once availability has
squeezed W to a sliver and the follower starts shifting load INTO the served
hours (sigma < 0): the additive term then dominates the multiplicative one and
x* stays pinned at x_max instead of rising as the formula would have it.
:func:`fit_slope` therefore fits on sigma >= 0 and reports how many points it
set aside.

Outputs, under :data:`config.OUTPUT_DIR`:

    availability_analysis.xlsx      one sheet per pattern, the case-study
                                    scenarios placed on the same axes, and a
                                    summary of fitted vs predicted slope and
                                    threshold
    availability_<season>.png       price and profit, against the raw share and
                                    against sigma, with the closed form overlaid

Usage:
    python availability_analysis.py
    python availability_analysis.py --seasons winter --steps 11 --levels 65
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
from plot import COLORS
from run import build_instance
from scenarios import (AVAILABILITY_PATTERNS, SCENARIOS, SCENARIO_LABEL,
                       synthetic_availability)

_OK = (TerminationCondition.optimal, TerminationCondition.feasible)
_TOL = 1e-6
AVAIL = "avail"                   # the single, certain outcome under test

# Colours follow the PATTERN, in a fixed order, never the plotting order.
_PATTERN_COLOR = dict(zip(
    AVAILABILITY_PATTERNS,
    [COLORS["grid"], COLORS["diesel"], COLORS["shifted"], COLORS["supp"],
     COLORS["unserved"], COLORS["dp"]],
))

FIELDS = [
    ("pattern",     "pattern"),
    ("share",       "EDL energy OFFERED / demand [-]"),
    ("delivered",   "EDL energy DELIVERED / demand [-]"),
    ("retained",    "A / sum D: demand still in the MGO's market [-]"),
    ("ratio",       "B_eff/A: taken from the MGO's own market [-]"),
    ("W",           "W = A - B: uncovered load of the served hours [kWh]"),
    ("sigma",       "sigma = shifted/W: what the closed form predicts on [-]"),
    ("avail_served", "of which EDL availability [-]"),
    ("shift_out",   "of which load shifted away [-]"),
    ("x",           "optimal p [x P_ref]"),
    ("p",           "optimal p [$/kWh]"),
    ("profit",      "leader profit [$]"),
    ("x_theory",    "closed-form p [x P_ref]"),
    ("profit_theory", "closed-form profit [$]"),
    ("qMGO",        "microgrid sales [kWh]"),
    ("qPG",         "EDL delivered [kWh]"),
    ("qsupp",       "suppressed demand [kWh]"),
    ("qcri",        "unserved [kWh]"),
    ("served",      "hours served by the MGO"),
    ("degenerate",  "price arbitrary (no sales)"),
    ("termination", "termination"),
]


# --------------------------------------------------------------------------
# The closed form
# --------------------------------------------------------------------------
def marginal_cost() -> float:
    """Diesel marginal cost c [$/kWh]: fuel + variable O&M."""
    cfg = config.CONFIG
    return cfg["C_fuel_L"] * cfg["alpha_fuel"] + cfg["c_OV"]["DG"]


def theory(season: str) -> dict:
    """
    Slope, intercept and threshold of the closed form for ``season``.

    Both demand responses give the same SHAPE -- the residual is linear in the
    price either way, so x* is linear in sigma and the profit quadratic -- but
    they shift the intercept and the exit threshold by exactly the 1 that
    distinguishes p/P_ref from (p-P_ref)/P_ref:

        "level"      x* = 0.5*(    1/a1 + c/P_ref) - (1/(2*a1))*sigma
                     exit at sigma = 1     - a1*c/P_ref
        "deviation"  x* = 0.5*(1 + 1/a1 + c/P_ref) - (1/(2*a1))*sigma
                     exit at sigma = (1+a1) - a1*c/P_ref

    The slope -1/(2*a1) is common to both: it depends only on how fast demand
    responds, not on where the response is anchored.

    Note what the slope is taken WITH RESPECT TO. Against a gross-demand
    response this was d x*/d(B_eff/A), and availability moved the price
    directly. Basing suppression on the uncovered load makes availability
    multiplicative, so it drops out of x* entirely; the same slope now applies
    to sigma = shifted/(A - B), the load-shifting channel alone.
    """
    cfg = config.CONFIG
    a1, P_ref = cfg["alpha_1"], cfg["P_ref"]
    c = marginal_cost()
    offset = 1.0 if cfg["demand_response"] == "deviation" else 0.0
    return {
        "x_max": 0.5 * (offset + 1 / a1 + c / P_ref),   # x* with no shifting
        "slope": -1.0 / (2 * a1),                       # d x* / d sigma
        # Q* = (W/2) * [1 + offset*a1 - a1*c/P_ref - sigma], so the bracket is
        # what hits zero -- the intercept differs between the two forms by a1,
        # not by the 1 that separates their x_max.
        "threshold": (1.0 + offset * a1) - a1 * c / P_ref,
        "c_over_Pref": c / P_ref,
    }


def theory_price(season: str, sigma: float) -> float:
    th = theory(season)
    return th["x_max"] + th["slope"] * sigma


def theory_profit(season: str, W: float, sigma: float) -> float:
    """``W = A - B`` is the uncovered load of the hours the MGO still serves."""
    cfg = config.CONFIG
    a1, P_ref = cfg["alpha_1"], cfg["P_ref"]
    k = theory(season)["threshold"] - sigma
    return (P_ref * W / (4 * a1)) * k * k if k > 0 else 0.0


# --------------------------------------------------------------------------
# Solving
# --------------------------------------------------------------------------
def _refine_menu(data: dict, levels: int) -> None:
    """
    Re-space the price menu over [P_ref, Phat] with ``levels`` entries.

    The sweep measures a SLOPE in the price, which the default menu is far too
    coarse to resolve: the whole decision is one scalar, so the menu step is
    the precision of every price reported here.
    """
    T = data["T"]
    P_ref, Phat = data["P"][T[0]], data["Phat"][T[0]]
    data["Pi"] = [P_ref + (Phat - P_ref) * k / (levels - 1)
                  for k in range(levels)]


def _record(m, data: dict, Qbar: dict, season: str, key: str) -> dict:
    """
    Totals of a solved single-outcome model, plus its availability measures.

    Two channels have to be separated, because they act differently:

    EXTENSIVE -- hours EDL covers outright leave the MGO's market, and energy it
        takes out of the hours the MGO still serves shrinks what is left there.
        Both act on W = A - B, the uncovered load of the served hours: they
        scale the profit but, with suppression based on the uncovered load,
        leave the optimal price alone.
    SHIFTING -- the follower moves load away from the served hours into the
        covered ones. That enters the residual ADDITIVELY rather than as a
        scaling, so it is the one channel that still moves the price. Hence

            sigma = shifted / W,   shifted = -sum_H dd_t

        with dd_t < 0 in the served hours. ``ratio`` = B_eff/A is kept beside it
        for continuity with the policy-facing view, but sigma is what the closed
        form predicts on.
    """
    T, D = data["T"], data["D"]
    P_ref = data["P"][T[0]]
    total_D = sum(D.values())

    qMGO = {t: value(m.qMGO[t, AVAIL]) for t in T}
    dd = {t: value(m.dd[t]) for t in T}
    served = [t for t in T if qMGO[t] > _TOL]

    A = sum(D[t] for t in served)
    avail = sum(Qbar[t] for t in served)
    shifted = -sum(dd[t] for t in served)      # load moved out of those hours
    ratio = (avail + shifted) / A if A > _TOL else float("nan")
    W = A - avail                              # uncovered load still addressable
    sigma = shifted / W if W > _TOL else float("nan")

    sales = sum(qMGO.values())
    degenerate = sales <= _TOL          # sells nothing: its price is arbitrary
    p = value(m.p[T[0]])
    qPG = sum(value(m.qPG[t, AVAIL]) for t in T)

    return {
        "pattern": key,
        "share": sum(Qbar.values()) / total_D,
        "delivered": qPG / total_D,
        "retained": A / total_D,
        "ratio": ratio,
        "sigma": sigma,
        "W": W,
        "avail_served": avail / A if A > _TOL else float("nan"),
        "shift_out": shifted / A if A > _TOL else float("nan"),
        "x": None if degenerate else p / P_ref,
        "p": None if degenerate else p,
        "profit": value(m.profit),
        "x_theory": None if degenerate else theory_price(season, sigma),
        "profit_theory": (None if degenerate
                          else theory_profit(season, W, sigma)),
        "qMGO": sales,
        "qPG": qPG,
        "qsupp": sum(value(m.qsupp[t, AVAIL]) for t in T),
        "qcri": sum(value(m.qcri[t, AVAIL]) for t in T),
        "served": len(served),
        "degenerate": degenerate,
        "termination": "",
    }


def solve_at_availability(season: str, Qbar: dict, key: str, opts: dict,
                          levels: int) -> dict | None:
    """Re-optimize the leader against one availability profile, taken as certain."""
    data = dict(build_instance(season))
    T = data["T"]
    data["S"] = [AVAIL]
    data["prob"] = {AVAIL: 1.0}
    data["QbarPG"] = {(t, AVAIL): Qbar[t] for t in T}
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
        print(f"  {key}: not solved ({tc})")
        return None

    rec = _record(m, data, Qbar, season, key)
    rec["termination"] = str(tc)
    return rec


def sweep(season: str, patterns: list[str], shares: list[float], opts: dict,
          levels: int) -> dict[str, list[dict]]:
    """Sweep the availability share for each pattern."""
    D = build_instance(season)["D"]
    out: dict[str, list[dict]] = {}
    for pattern in patterns:
        print(f"[{season}] pattern {pattern} ...")
        rows = []
        for share in shares:
            Qbar = synthetic_availability(season, share, pattern, D)
            rec = solve_at_availability(season, Qbar, pattern, opts, levels)
            if rec is None:
                continue
            rec["share"] = share
            rows.append(rec)
            x = "   -  " if rec["x"] is None else f"{rec['x']:6.3f}"
            print(f"  share={share:5.2f}  deliv={rec['delivered']:5.2f}  "
                  f"kept={rec['retained']:5.2f}  B_eff/A={rec['ratio']:6.3f}  "
                  f"x*={x}  profit={rec['profit']:8.2f}")
        if rows:
            out[pattern] = rows
    return out


def scenario_points(season: str, opts: dict, levels: int) -> list[dict]:
    """The six case-study scenarios placed on the same axes as the sweep."""
    base = build_instance(season)
    T = base["T"]
    print(f"[{season}] case-study scenarios ...")
    rows = []
    for sc in SCENARIOS:
        Qbar = {t: base["QbarPG"][(t, sc.key)] for t in T}
        rec = solve_at_availability(season, Qbar, sc.key, opts, levels)
        if rec is None:
            continue
        rec["label"] = SCENARIO_LABEL[sc.key]
        rows.append(rec)
        x = "   -  " if rec["x"] is None else f"{rec['x']:6.3f}"
        print(f"  {sc.key:>12}: share={rec['share']:5.2f}  "
              f"deliv={rec['delivered']:5.2f}  kept={rec['retained']:5.2f}  "
              f"B_eff/A={rec['ratio']:6.3f}  x*={x}  "
              f"profit={rec['profit']:8.2f}")
    return rows


# --------------------------------------------------------------------------
# Fitting the law
# --------------------------------------------------------------------------
def fit_slope(results: dict[str, list[dict]]) -> dict | None:
    """
    Least-squares slope of x* against sigma, over the regime the law describes.

    Points with sigma < 0 are EXCLUDED, and their count is reported as
    ``n_excluded`` rather than dropped quietly. Negative sigma means the
    follower shifted load INTO the hours the MGO serves instead of out of them,
    which happens only once availability has collapsed W = A - B to a sliver --
    and there the closed form's premise fails. It treats the shifted load as
    given when differentiating in x, whereas the follower re-optimizes its
    shifting against every price; while shifted is small next to W that is a
    harmless approximation, but at small W the additive term dominates the
    multiplicative one and the approximation goes with it. Pooling both regimes
    would report a slope belonging to neither (-0.17 against a predicted -1.04
    on the winter sweep, R^2 = 0.18, versus -1.08 and R^2 = 0.99 here).
    """
    pts = [(r["sigma"], r["x"]) for rows in results.values() for r in rows
           if r["x"] is not None and np.isfinite(r["sigma"])]
    kept = [p for p in pts if p[0] >= -_TOL]
    if len(kept) < 2:
        return None
    xs, ys = np.array([p[0] for p in kept]), np.array([p[1] for p in kept])
    slope, intercept = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + intercept)
    ss_tot = ((ys - ys.mean()) ** 2).sum()
    return {"slope": slope, "intercept": intercept, "n": len(kept),
            "n_excluded": len(pts) - len(kept),
            "r2": 1 - (resid ** 2).sum() / ss_tot if ss_tot else float("nan")}


def zero_crossing(rows: list[dict]) -> dict:
    """
    Where the MGO's market dies, observed two ways.

    ``exit_ratio`` extrapolates sigma to the point of zero profit. Since pi* is
    proportional to (threshold - sigma)^2, its SQUARE ROOT is linear in sigma,
    so the root is read off a straight line rather than off the flat tail of a
    parabola -- which is also why looking for a sign change finds nothing:
    profit reaches zero and stays there, it never goes negative.

    ``dead_share`` is the blunter, policy-facing companion: the smallest
    offered share at which the MGO sells nothing at all. A window-shaped
    pattern never gets there -- it leaves the uncovered hours untouched -- so
    both may be ``None``.
    """
    live = [(r["sigma"], r["profit"]) for r in rows
            if r["profit"] > _TOL and np.isfinite(r["sigma"] or np.nan)]
    dead = [r["share"] for r in rows if r["qMGO"] <= _TOL]

    out = {"exit_ratio": None,
           "dead_share": min(dead) if dead else None,
           "last_live_share": max((r["share"] for r in rows
                                   if r["qMGO"] > _TOL), default=None)}
    if len(live) < 2:
        return out

    live.sort(key=lambda kv: kv[0])
    peak = max(p for _, p in live)
    # Only extrapolate from a tail that actually approaches the root.
    if live[-1][1] > 0.25 * peak:
        return out
    (r0, p0), (r1, p1) = live[-2], live[-1]
    s0, s1 = np.sqrt(p0), np.sqrt(p1)
    if s0 <= s1:
        return out
    out["exit_ratio"] = r0 + (r1 - r0) * s0 / (s0 - s1)
    return out


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
    """One sheet per season+pattern, the scenarios, and the fitted summary."""
    wb = Workbook()
    wb.remove(wb.active)

    ws = wb.create_sheet("summary")
    rows = []
    for season, blob in per_season.items():
        th, ft = blob["theory"], blob["fit"]
        rows.append([season, "predicted slope  d x*/d sigma", th["slope"]])
        if ft:
            rows.append([season, "fitted slope", ft["slope"]])
            rows.append([season, "fit R^2", ft["r2"]])
            rows.append([season, "fit points", ft["n"]])
            rows.append([season, "points excluded (sigma < 0, law's premise "
                                 "fails)", ft["n_excluded"]])
        rows.append([season, "predicted x* at sigma = 0", th["x_max"]])
        rows.append([season, "predicted exit sigma (profit = 0)",
                     th["threshold"]])
        rows.append([season, "c / P_ref (price floor)", th["c_over_Pref"]])
        for pattern, rws in blob["results"].items():
            zc = zero_crossing(rws)
            rows.append([season, f"observed exit sigma [{pattern}]",
                         zc["exit_ratio"]])
            rows.append([season, f"offered share where sales stop [{pattern}]",
                         zc["dead_share"]])
        rows.append([])
    _write_rows(ws, ["season", "quantity", "value"], rows)

    for season, blob in per_season.items():
        for pattern, rws in blob["results"].items():
            _write_rows(wb.create_sheet(f"{season}_{pattern}"[:31]),
                        [lbl for _, lbl in FIELDS],
                        ([r[k] for k, _ in FIELDS] for r in rws))

        if blob["scenarios"]:
            _write_rows(
                wb.create_sheet(f"{season}_scenarios"[:31]),
                ["scenario", "outcome"] + [lbl for _, lbl in FIELDS],
                ([r["pattern"], r["label"]] + [r[k] for k, _ in FIELDS]
                 for r in blob["scenarios"]),
            )

    wb.save(out_path)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def _series(rows: list[dict], xkey: str, ykey: str):
    """x/y arrays with degenerate points masked out (NaN breaks the line)."""
    x = np.array([np.nan if rw[xkey] is None else rw[xkey] for rw in rows],
                 dtype=float)
    y = np.array([np.nan if rw[ykey] is None else rw[ykey] for rw in rows],
                 dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def plot_availability(season: str, blob: dict, out_path) -> None:
    """
    Four panels: how price and profit respond, and whether the law holds.

    Top row is the price, bottom the profit. The left column is the
    policy-facing view (energy EDL actually delivers); the right column tests
    the closed form -- the price against B_eff/A, where the patterns should
    COLLAPSE onto one line, and the profit against its own prediction, where
    every point should land on the 45 deg line.
    """
    results, scen, th = blob["results"], blob["scenarios"], blob["theory"]
    P_ref = config.CONFIG["P_ref"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for pattern, rows in results.items():
        color = _PATTERN_COLOR.get(pattern, "#444444")
        kw = dict(color=color, marker="o", markersize=4, linewidth=2)
        x, y = _series(rows, "delivered", "x")
        axes[0][0].plot(x, y, label=pattern, **kw)
        x, y = _series(rows, "delivered", "profit")
        axes[1][0].plot(x, y, **kw)
        x, y = _series(rows, "sigma", "x")
        axes[0][1].plot(x, y, **kw)
        # Parity: the closed form evaluated at this point's own A and B_eff.
        px = np.array([np.nan if r["profit_theory"] is None else
                       r["profit_theory"] for r in rows], dtype=float)
        py = np.array([r["profit"] for r in rows], dtype=float)
        axes[1][1].plot(px, py, linestyle="none", marker="o", markersize=7,
                        color=color, alpha=0.85)

    if scen:
        for ax, xk, yk in ((axes[0][0], "delivered", "x"),
                           (axes[1][0], "delivered", "profit"),
                           (axes[0][1], "sigma", "x")):
            sx = np.array([np.nan if r[xk] is None else r[xk] for r in scen],
                          dtype=float)
            sy = np.array([np.nan if r[yk] is None else r[yk] for r in scen],
                          dtype=float)
            ax.plot(sx, sy, linestyle="none", marker="*", markersize=15,
                    markerfacecolor="#FFFFFF", markeredgecolor="#222222",
                    markeredgewidth=1.0,
                    label="case-study scenarios" if ax is axes[0][0] else None)
        for r in scen:
            if r["x"] is not None and np.isfinite(r["sigma"]):
                axes[0][1].annotate(r["label"], (r["sigma"], r["x"]),
                                    xytext=(0, -13), textcoords="offset points",
                                    ha="center", fontsize=7, color="#555555")

    # ---- the closed form, on the right column ----------------------------
    grid = np.linspace(0.0, th["threshold"] * 1.05, 100)
    axes[0][1].plot(grid, [theory_price(season, r) for r in grid],
                    color="#222222", linewidth=2, linestyle="--",
                    label="closed form")
    lim = max([r["profit"] for rows in results.values() for r in rows]
              + [1.0]) * 1.05
    axes[1][1].plot([0, lim], [0, lim], color="#222222", linewidth=2,
                    linestyle="--", label="parity")

    axes[0][0].set_xlabel("EDL energy delivered  [fraction of daily demand]")
    axes[1][0].set_xlabel("EDL energy delivered  [fraction of daily demand]")
    axes[0][1].set_xlabel("$\\sigma$ — load shifted out, per unit of $W=A-B$")
    axes[1][1].set_xlabel("Closed-form profit at this point's $W$, $\\sigma$"
                          "  [\\$ / day]")
    axes[0][0].set_ylabel("Leader price $\\tilde p$  [$\\times P_{ref}$]")
    axes[1][0].set_ylabel("Leader profit  [\\$ / day]")
    axes[0][1].set_ylabel("Leader price $\\tilde p$  [$\\times P_{ref}$]")
    axes[1][1].set_ylabel("Model profit  [\\$ / day]")

    axes[0][0].set_title("price vs what EDL delivers", fontsize=10)
    axes[1][0].set_title("profit vs what EDL delivers", fontsize=10)
    axes[0][1].set_title("price vs what EDL takes  —  should collapse",
                         fontsize=10)
    axes[1][1].set_title("profit vs its own prediction  —  should be parity",
                         fontsize=10)

    for j in (0, 1):
        axes[0][j].axhline(th["c_over_Pref"], color="#999999", linewidth=1.0,
                           linestyle=":")
    axes[1][0].axhline(0, color="#666666", linewidth=0.8)
    for row in axes:
        for ax in row:
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    axes[0][0].annotate(f"marginal cost  $c/P_{{ref}}$ = {th['c_over_Pref']:.2f}",
                        (0.99, th["c_over_Pref"]),
                        xycoords=("axes fraction", "data"), xytext=(0, 4),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=8, color="#777777")
    axes[0][1].axvline(th["threshold"], color="#999999", linewidth=1.0,
                       linestyle=":")
    axes[0][1].annotate(f"exit  $\\sigma$ = {th['threshold']:.3f}",
                        (th["threshold"], 0.03),
                        xycoords=("data", "axes fraction"), xytext=(-5, 0),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=8, color="#777777")

    fit = blob["fit"]
    sub = (f"fitted slope {fit['slope']:.3f} vs predicted {th['slope']:.3f} "
           f"($R^2$ = {fit['r2']:.4f}, n = {fit['n']}"
           + (f", {fit['n_excluded']} excluded with $\\sigma<0$)"
              if fit["n_excluded"] else ")") if fit else "")
    fig.suptitle(f"EDL availability → leader price and profit — {season}"
                 f"   ($P_{{ref}}$ = {P_ref:.3f} \\$/kWh)\n{sub}", fontsize=12)

    h, l = axes[0][0].get_legend_handles_labels()
    h2, l2 = axes[0][1].get_legend_handles_labels()
    axes[0][0].legend(h + h2, l + l2, fontsize=8, framealpha=0.9,
                      loc="lower left", ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
def main(seasons: list[str] | None = None,
         patterns: list[str] | None = None,
         steps: int = 11, max_share: float = 1.0, levels: int = 65) -> None:
    seasons = seasons or ["winter"]
    patterns = patterns or list(AVAILABILITY_PATTERNS)
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = {"MIPGap": config.CONFIG["MIPGap"],
            "TimeLimit": config.CONFIG["TimeLimit"]}
    shares = [max_share * k / (steps - 1) for k in range(steps)]

    per_season: dict[str, dict] = {}
    for season in seasons:
        th = theory(season)
        print(f"[{season}] closed form: x* = {th['x_max']:.4f} "
              f"{th['slope']:+.4f} * sigma,  profit = 0 at sigma = "
              f"{th['threshold']:.4f}")

        results = sweep(season, patterns, shares, opts, levels)
        if not results:
            print(f"[{season}] nothing solved.")
            continue

        blob = {
            "results": results,
            "scenarios": scenario_points(season, opts, levels),
            "theory": th,
            "fit": fit_slope(results),
            "A_total": sum(build_instance(season)["D"].values()),
        }
        per_season[season] = blob

        fit = blob["fit"]
        if fit:
            print(f"\n[{season}] fitted slope {fit['slope']:.4f} vs predicted "
                  f"{th['slope']:.4f}  (R^2 = {fit['r2']:.4f}, n = {fit['n']}"
                  f", {fit['n_excluded']} excluded with sigma < 0)")
        for pattern, rows in results.items():
            zc = zero_crossing(rows)
            er, ds = zc["exit_ratio"], zc["dead_share"]
            print(f"  {pattern:>13}: exit sigma = "
                  f"{'n/a' if er is None else f'{er:.3f}'} "
                  f"(predicted {th['threshold']:.3f});  sales stop at offered "
                  f"share {'never' if ds is None else f'{ds:.2f}'}")

        plot_availability(season, blob, out_dir / f"availability_{season}.png")

    if not per_season:
        print("No season solved — nothing to write.")
        return
    save_analysis(per_season, out_dir / "availability_analysis.xlsx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep EDL availability and re-optimize the leader.")
    parser.add_argument("--seasons", nargs="+", default=["winter"],
                        choices=list(config.SEASONS.keys()))
    parser.add_argument("--patterns", nargs="+", default=None,
                        choices=list(AVAILABILITY_PATTERNS),
                        help="Availability patterns (default: all).")
    parser.add_argument("--steps", type=int, default=11,
                        help="Availability shares from 0 to --max-share.")
    parser.add_argument("--max-share", type=float, default=1.0,
                        help="Largest EDL energy share of daily demand.")
    parser.add_argument("--levels", type=int, default=65,
                        help="Price menu levels; the sweep measures a slope, "
                             "so the default 9 is far too coarse.")
    args = parser.parse_args()
    main(args.seasons, args.patterns, args.steps, args.max_share, args.levels)
