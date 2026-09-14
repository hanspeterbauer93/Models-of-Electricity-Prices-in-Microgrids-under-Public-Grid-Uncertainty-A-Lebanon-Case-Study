"""
Plotting for the bilevel MPEC microgrid model.

Solves the model **deterministically once per public-grid (EDL) scenario**
(``prob = 1`` for the active scenario) so that every scenario gets its own
leader decisions. This makes revenue, profit and dispatch genuinely vary across
scenarios (in a single stochastic solve the leader's price/generation are
first-stage and identical for all scenarios).

Two figures are produced (saved under :data:`config.OUTPUT_DIR`):

    plot1_revenue_boxplot.png
        Boxplot of the leader's hourly revenue (p_t * qMGO_t), one box per
        scenario outcome plus the stochastic commitment, with the price
        charged in each on a second scale.

    plot2_timeline_<scenario>.png   (one per scenario)
        24-hour timeline of dispatched energy (diesel + PV), public-grid
        energy, suppressed demand and unserved load, with the demand curve and
        the shifted demand overlaid. The leader's price and daily profit are
        reported as numbers in the panel: the tariff is flat over the day, so
        there is no hourly shape to plot.

Usage:
    python plot.py                 # winter (default)
    python plot.py --season summer
"""

import argparse

import matplotlib
matplotlib.use("Agg")             # headless / file output
import matplotlib.pyplot as plt
import numpy as np
from pyomo.environ import value, TerminationCondition

import config
from run import build_instance
from model.bilevel_mpec import BilevelMPEC
from scenarios import SCENARIO_PROB, SCENARIO_LABEL

# --------------------------------------------------------------------------
# Consistent, colour-blind-friendly palette (shared across both figures).
# --------------------------------------------------------------------------
COLORS = {
    "diesel":   "#D55E00",   # vermillion
    "pv":       "#F0E442",   # yellow
    "grid":     "#0072B2",   # blue      (public grid / EDL)
    "supp":     "#CC79A7",   # magenta   (suppressed demand)
    "unserved": "#999999",   # grey      (critical unserved load)
    "demand":   "#000000",   # black     (demand curve)
    "shifted":  "#009E73",   # green     (shifted demand)
    "profit":   "#0072B2",
    "profit_neg": "#D55E00",
    "dp":       "#000000",   # Δp price deviation
}

# One colour per EDL outcome, keyed by SCENARIO rather than by plotting order,
# so an outcome keeps its colour across every figure in the thesis and the
# slides. Five Okabe-Ito hues; ``none`` is deliberately absent, because the
# outcome draws nothing (EDL is off all day) and so never needs a fill.
#
# Checked with the palette validator at CVD deltaE >= 6 on all pairs and a
# normal-vision floor of 15.6 -- the pure grey #999999 this used to give
# full_supply failed both the chroma floor and its pair with the magenta.
SCENARIO_COLOR: dict[str, str] = {
    "announced":   "#0072B2",   # blue
    "partial":     "#D55E00",   # vermillion
    "delayed":     "#009E73",   # green
    "bonus":       "#CC79A7",   # magenta
    "full_supply": "#E69F00",   # orange
}

# Human-readable scenario labels come from the GridScenario definitions in
# scenarios.py, so the two stay in sync. The stochastic commitment is not
# one of them.
STOCHASTIC = "stochastic"
STOCHASTIC_LABEL = "Stochastic commitment"


def label_of(key: str) -> str:
    """Human-readable name of a scenario outcome or of the stochastic
    commitment."""
    return (STOCHASTIC_LABEL if key == STOCHASTIC
            else SCENARIO_LABEL.get(key, key))


# --------------------------------------------------------------------------
# Solve
# --------------------------------------------------------------------------
def _extract(m, data: dict, key: str) -> dict:
    """
    Per-hour arrays from a solved model, as EXPECTATIONS over its scenario set.

    qMGO / qdiesel / qPV / qPG / qcri / qsupp are recourse (scenario-indexed), so they
    are averaged over the scenarios at their probabilities. With a single
    certain outcome (prob = 1) that average IS that outcome's value, so the
    deterministic runs and the stochastic commitment yield directly
    comparable series.
    """
    T, S, prob = data["T"], data["S"], data["prob"]

    def exp_t(var):
        return np.array([sum(prob[s] * value(var[t, s]) for s in S) for t in T])

    def per_t(var):
        return np.array([value(var[t]) for t in T])

    p, P_ref = per_t(m.p), per_t(m.P)
    qMGO, qdiesel, qPV = exp_t(m.qMGO), exp_t(m.qdiesel), exp_t(m.qPV)
    cdiesel, cPV = per_t(m.cdiesel), per_t(m.cPV)

    revenue = p * qMGO
    cost = cdiesel * qdiesel + cPV * qPV

    return {
        "scenario": key,
        "T": np.array(T),
        "p": p, "P_ref": P_ref, "dp": p - P_ref,   # Δp = price offer − reference
        "qMGO": qMGO, "qdiesel": qdiesel, "qPV": qPV,
        "qsupp": exp_t(m.qsupp), "dd": per_t(m.dd),
        "qPG": exp_t(m.qPG), "qcri": exp_t(m.qcri),
        "D": per_t(m.D),
        "revenue": revenue, "cost": cost, "profit": revenue - cost,
    }


def _solve(data: dict, key: str) -> dict | None:
    """Build and solve, returning the extracted series (or ``None``)."""
    mpec = BilevelMPEC(data)
    m = mpec.build()

    gurobi_opts = {
        "MIPGap": config.CONFIG["MIPGap"],
        "TimeLimit": config.CONFIG["TimeLimit"],
    }
    solver_name, res = mpec.solve("gurobi", options=gurobi_opts)
    if solver_name is None:
        raise RuntimeError(
            "Gurobi not available (check the license and that gurobipy is "
            "installed)."
        )

    tc = res.solver.termination_condition
    if tc not in (TerminationCondition.optimal, TerminationCondition.feasible):
        print(f"  [{key}] not solved: termination = {tc}")
        return None
    return _extract(m, data, key)


def solve_deterministic(season: str, scenario: str) -> dict | None:
    """
    Solve the model for a single EDL scenario treated as certain (prob = 1).

    Returns a dict of per-hour numpy arrays, or ``None`` if the solve did not
    reach an optimal/feasible point (or Gurobi is unavailable).
    """
    data = dict(build_instance(season))
    T = data["T"]

    # Restrict the scenario set to this single outcome, certain to occur.
    data["S"] = [scenario]
    data["prob"] = {scenario: 1.0}
    data["QbarPG"] = {(t, scenario): data["QbarPG"][(t, scenario)] for t in T}

    return _solve(data, scenario)


def solve_stochastic(season: str) -> dict | None:
    """
    Solve the model ONCE over all EDL outcomes at their own probabilities.

    This is the schedule the model actually recommends: the leader commits one
    price before knowing which outcome occurs, so the series returned are
    expectations over the outcomes rather than any single one of them.
    """
    return _solve(build_instance(season), STOCHASTIC)


# --------------------------------------------------------------------------
# Plot 1 — boxplot of leader revenue per scenario
# --------------------------------------------------------------------------
def plot_revenue_boxplot(results: dict[str, dict], season: str, out_path):
    """
    Boxplot of hourly leader revenue, one box per scenario outcome, with the
    price that produced it on a second scale.

    Revenue [$ /h] and price [$/kWh] are different units, so the price needs
    its own axis. The x axis is categorical and the price is drawn as discrete
    markers rather than a line, so the two scales cannot be misread as a trend.

    A stochastic-commitment entry in ``results`` is placed last, behind a divider
    and drawn hatched: it is not an EDL outcome but the one schedule the leader
    commits to across all of them, so its box is an expectation, not a
    realization.
    """
    scenarios = [s for s in SCENARIO_PROB if s in results]
    keys = scenarios + ([STOCHASTIC] if STOCHASTIC in results else [])
    data = [results[s]["revenue"] for s in keys]
    labels = [label_of(s) for s in keys]
    totals = [results[s]["revenue"].sum() for s in keys]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, showmeans=True,
        medianprops=dict(color="#222222", linewidth=1.5),
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="#222222", markersize=6),
        flierprops=dict(marker="o", markersize=4, markerfacecolor="#999999",
                        markeredgecolor="none", alpha=0.6),
    )
    cmap = plt.get_cmap("Blues")
    for i, box in enumerate(bp["boxes"]):
        if keys[i] == STOCHASTIC:
            # Texture, not another hue: the commitment is a different kind of
            # object, and a hatch says so without colour.
            box.set_facecolor("#FFFFFF")
            box.set_hatch("//")
        else:
            box.set_facecolor(cmap(0.35 + 0.5 * i / max(1, len(scenarios) - 1)))
        box.set_edgecolor("#333333")
        box.set_alpha(0.9)

    if STOCHASTIC in results:
        ax.axvline(len(scenarios) + 0.5, color="#BBBBBB", linewidth=1.2)

    # Annotate each box with the daily total revenue.
    ymax = max((d.max() for d in data), default=1.0)
    for i, tot in enumerate(totals, start=1):
        ax.text(i, ymax * 1.02, f"Σ {tot:,.0f}", ha="center", va="bottom",
                fontsize=8, color="#333333")

    ax.set_ylabel("Leader revenue per hour  [$]")
    ax.set_xlabel("EDL scenario outcome  (each priced for)"
                  + ("   |   stochastic commitment" if STOCHASTIC in results else ""))
    ax.set_title(f"Leader revenue by scenario — {season}\n"
                 f"(box = distribution over 24 h; Σ = daily total)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.margins(y=0.18)
    ax.tick_params(axis="x", labelrotation=12)

    # ---- second scale: the price the leader charges in that scenario ------
    x = np.arange(1, len(keys) + 1)
    price = np.array([results[s]["p"].mean() for s in keys])
    P_ref = results[keys[0]]["P_ref"][0]

    axp = ax.twinx()
    axp.plot(x, price, linestyle="none", marker="D", markersize=9,
             markerfacecolor=COLORS["diesel"], markeredgecolor="white",
             markeredgewidth=1.5)
    # With an hourly tariff (flat_price = False) the marker is the daily mean,
    # so show the spread it hides.
    for xi, s in zip(x, keys):
        p = results[s]["p"]
        if p.max() - p.min() > 1e-9:
            axp.vlines(xi, p.min(), p.max(), color=COLORS["diesel"],
                       linewidth=1.5, alpha=0.6)
    axp.axhline(P_ref, color="#666666", linewidth=1.0, linestyle=":")
    axp.annotate(f"reference tariff $P_{{ref}}$ = {P_ref:.3f}",
                 (len(keys) + 0.45, P_ref), textcoords="offset points",
                 xytext=(0, 4), ha="right", va="bottom", fontsize=8,
                 color="#666666")

    for xi, pi in zip(x, price):
        axp.annotate(f"{pi:.3f}", (xi, pi), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=8, color="#222222")

    # Anchored at 0 so a marker's height reads as the price itself, and topped
    # at the cap so it also reads against the leader's room to move.
    axp.set_ylim(0, max(price.max(), config.CONFIG["phat_factor"] * P_ref) * 1.12)
    # No legend box: the right axis is drawn in the marker's colour, which ties
    # the two together without stealing space from the boxes.
    axp.set_ylabel("Leader price $\\tilde p$  [\\$/kWh]  ◆")
    axp.tick_params(axis="y", colors=COLORS["diesel"])
    axp.yaxis.label.set_color(COLORS["diesel"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
# Plot 2 — per-scenario 24 h timeline (dispatch + profit)
# --------------------------------------------------------------------------
def _stacked_bars(ax, x, series, width=0.8):
    """
    Stacked bar chart that tolerates mixed signs: positive parts stack
    upward from 0, negative parts stack downward from 0.
    """
    pos_bottom = np.zeros(len(x))
    neg_bottom = np.zeros(len(x))
    for label, values, color in series:
        values = np.asarray(values, dtype=float)
        pos = np.where(values > 0, values, 0.0)
        neg = np.where(values < 0, values, 0.0)
        if np.any(pos):
            ax.bar(x, pos, width, bottom=pos_bottom, color=color,
                   label=label, edgecolor="white", linewidth=0.3)
            pos_bottom += pos
        if np.any(neg):
            # Only add the label once (attach to the positive part if present).
            lbl = None if np.any(pos) else label
            ax.bar(x, neg, width, bottom=neg_bottom, color=color,
                   label=lbl, edgecolor="white", linewidth=0.3, hatch="//")
            neg_bottom += neg


def plot_timeline(res: dict, season: str, out_path):
    """
    24 h dispatch/demand timeline for a single scenario.

    The leader's price and profit are daily figures, not hourly shapes -- the
    tariff is flat over the day -- so they are reported as numbers in the panel
    rather than as a second set of axes.
    """
    T = res["T"]
    scenario = res["scenario"]
    label = SCENARIO_LABEL.get(scenario, scenario)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # ---- dispatch / demand balance ---------------------------------------
    # Balance:  D + dd  ==  qMGO(=diesel+PV) + qPG + qsupp + qcri
    _stacked_bars(ax1, T, [
        ("MGO coverage $q^{cov,MGO}_t$", res["qMGO"], COLORS["diesel"]),
        ("Public grid (EDL)", res["qPG"],     COLORS["grid"]),
        ("Unserved (critical)", res["qcri"],  COLORS["unserved"]),
        ("Suppressed demand", res["qsupp"],   COLORS["supp"]),
    ])
    ax1.plot(T, res["D"], color=COLORS["demand"], marker="o", markersize=3,
             linewidth=1.6, linestyle="--", label="Demand $D_t$")
    ax1.plot(T, res["D"] + res["dd"], color=COLORS["shifted"], marker="s",
             markersize=3, linewidth=1.8, label="Shifted demand $D_t+\\Delta d_t$")

    ax1.axhline(0, color="#666666", linewidth=0.8)
    ax1.set_ylabel("Energy  [kWh]")
    ax1.set_xlabel("Hour of day")
    ax1.set_xticks(list(T))
    ax1.set_title(f"Dispatch & demand — {season} / {label}")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.margins(y=0.18)          # headroom for the legend and the figures box
    ax1.legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.9)

    # ---- the leader's two headline numbers -------------------------------
    p, P_ref = res["p"], res["P_ref"]
    if np.allclose(p, p[0]):
        price = f"{p[0]:.3f} \\$/kWh   ({p[0] / P_ref[0]:.2f} × $P_{{ref}}$)"
    else:                        # hourly tariff (config flat_price = False)
        price = (f"{p.min():.3f}–{p.max():.3f} \\$/kWh   "
                 f"(mean {p.mean():.3f})")

    ax1.text(0.99, 0.97,
             f"Leader price  $\\tilde p$ = {price}\n"
             f"Leader profit  Σ = {res['profit'].sum():,.1f} \\$/day",
             transform=ax1.transAxes, ha="right", va="top", fontsize=10,
             linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#cccccc",
                       alpha=0.95))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
def main(season: str = "winter") -> None:
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Solving one deterministic model per scenario [{season}] ...")
    results: dict[str, dict] = {}
    for scenario in SCENARIO_PROB:
        print(f"- scenario: {scenario}")
        res = solve_deterministic(season, scenario)
        if res is not None:
            results[scenario] = res

    if not results:
        print("No scenario solved successfully — nothing to plot.")
        return

    # The stochastic commitment: one solve over all outcomes, for the boxplot only. The
    # timelines stay per-outcome -- a dispatch that is an expectation over six
    # different grid profiles is not a dispatch anyone ever operates.
    print("- stochastic commitment solve")
    stochastic = solve_stochastic(season)

    print("Plotting ...")
    plot_revenue_boxplot(
        results | ({STOCHASTIC: stochastic} if stochastic else {}),
        season, out_dir / "plot1_revenue_boxplot.png")
    for scenario, res in results.items():
        plot_timeline(
            res, season, out_dir / f"plot2_timeline_{scenario}.png")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot leader revenue and per-scenario dispatch/profit "
                    "timelines for the bilevel MPEC model.")
    parser.add_argument("--season", default="winter",
                        choices=list(config.SEASONS.keys()),
                        help="Representative season to solve and plot.")
    args = parser.parse_args()
    main(args.season)
