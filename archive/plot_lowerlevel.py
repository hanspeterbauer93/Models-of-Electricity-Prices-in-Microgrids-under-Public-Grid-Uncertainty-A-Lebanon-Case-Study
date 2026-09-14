"""
Plotting for the lower level (follower) of the bilevel microgrid model.

The counterpart of :mod:`plot`, for the follower LP at a FIXED leader price.

Unlike the bilevel case this needs only ONE solve. There the leader's price and
generation are first-stage and therefore identical across scenarios, so
``plot.py`` re-solves a deterministic model per scenario to make the dispatch
vary. Here the price is a parameter and everything that responds to the public
grid -- qMGO, qPG, qcri -- is recourse, so a single stochastic solve already
yields a genuinely different dispatch per scenario. Only the suppressed demand
qsupp and the load shift Δd are first-stage and hence shared by all scenarios.

Costs are reported CONDITIONAL on the scenario (no ``prob`` weight), i.e. what
the village pays if that scenario realizes; weighting them by ``prob`` and
summing recovers the LP objective.

Two figures are produced (saved under :data:`config.OUTPUT_DIR`):

    plot1_lowerlevel_cost_boxplot.png
        Boxplot of the follower's hourly supply cost, one box per scenario.

    plot2_lowerlevel_timeline_<scenario>.png   (one per scenario)
        24-hour timeline with two stacked subplots:
          (top)    microgrid coverage, public-grid energy, suppressed demand
                   and unserved load, with the demand curve, the shifted demand
                   and the EDL capacity overlaid.
          (bottom) the follower's hourly cost split by supply source.

Usage:
    python plot_lowerlevel.py                          # winter, price = P_ref
    python plot_lowerlevel.py --season summer --price 0.4
"""

import argparse

import matplotlib
matplotlib.use("Agg")             # headless / file output
import matplotlib.pyplot as plt
import numpy as np
from pyomo.environ import value, TerminationCondition

import config
from model.lower_level import LowerLevel
from run_lowerlevel import build_instance, solver_options
from scenarios import SCENARIO_PROB, SCENARIO_LABEL
# Shared styling, so both figure sets look like one family.
from plot import COLORS, _stacked_bars


# --------------------------------------------------------------------------
# Solve
# --------------------------------------------------------------------------
def solve_follower(season: str, price=None) -> dict | None:
    """
    Solve the follower LP once and unpack it into per-scenario arrays.

    Returns ``{scenario: {...per-hour numpy arrays...}}`` plus the shared
    first-stage series, or ``None`` if the solve did not reach an
    optimal/feasible point.
    """
    data = build_instance(season, price)
    T, S = data["T"], data["S"]

    llp = LowerLevel(data)
    m = llp.build()

    solver_name, res = llp.solve("gurobi", options=solver_options())
    if solver_name is None:
        raise RuntimeError(
            "Gurobi not available (check the license and that gurobipy is "
            "installed)."
        )

    tc = res.solver.termination_condition
    if tc not in (TerminationCondition.optimal, TerminationCondition.feasible):
        print(f"  not solved: termination = {tc}")
        return None

    # ---- first stage: shared by every scenario ----
    p     = np.array([value(m.p[t])     for t in T])
    P_ref = np.array([value(m.P[t])     for t in T])
    D     = np.array([value(m.D[t])     for t in T])
    dd    = np.array([value(m.dd[t])    for t in T])
    qsupp = np.array([value(m.qsupp[t]) for t in T])
    pPG   = np.array([value(m.pPG[t])   for t in T])
    pVOLL = value(m.pVOLL)

    results: dict[str, dict] = {}
    for s in S:
        qMGO   = np.array([value(m.qMGO[t, s])   for t in T])
        qPG    = np.array([value(m.qPG[t, s])    for t in T])
        qcri   = np.array([value(m.qcri[t, s])   for t in T])
        QbarPG = np.array([value(m.QbarPG[t, s]) for t in T])

        # Conditional on s: what the village pays if this scenario realizes.
        c_mgo = p * qMGO
        c_pg  = pPG * qPG
        c_cri = pVOLL * qcri

        results[s] = {
            "scenario": s, "prob": data["prob"][s],
            "T": np.array(T),
            "p": p, "P_ref": P_ref, "D": D, "dd": dd, "qsupp": qsupp,
            "qMGO": qMGO, "qPG": qPG, "qcri": qcri, "QbarPG": QbarPG,
            "c_mgo": c_mgo, "c_pg": c_pg, "c_cri": c_cri,
            "cost": c_mgo + c_pg + c_cri,
        }

    expected = sum(r["prob"] * r["cost"].sum() for r in results.values())
    print(f"  solved: expected supply cost = {llp.objective_value():,.4f} "
          f"(scenario costs recombine to {expected:,.4f})")
    return results


# --------------------------------------------------------------------------
# Plot 1 — boxplot of follower cost per scenario
# --------------------------------------------------------------------------
def plot_cost_boxplot(results: dict[str, dict], season: str, out_path):
    """Boxplot of hourly follower supply cost (one box per scenario outcome)."""
    scenarios = [s for s in SCENARIO_PROB if s in results]
    data = [results[s]["cost"] for s in scenarios]
    labels = [f"{SCENARIO_LABEL.get(s, s)}\n($\\pi$ = {results[s]['prob']:.2f})"
              for s in scenarios]
    totals = [results[s]["cost"].sum() for s in scenarios]

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
        box.set_facecolor(cmap(0.35 + 0.5 * i / max(1, len(data) - 1)))
        box.set_edgecolor("#333333")
        box.set_alpha(0.9)

    # Annotate each box with the daily total cost.
    ymax = max((d.max() for d in data), default=1.0)
    for i, tot in enumerate(totals, start=1):
        ax.text(i, ymax * 1.02, f"Σ {tot:,.0f}", ha="center", va="bottom",
                fontsize=8, color="#333333")

    price = results[scenarios[0]]["p"]
    ax.set_ylabel("Follower supply cost per hour  [$]")
    ax.set_xlabel("EDL scenario outcome")
    # NB: write the unit as USD/kWh -- a bare "$" here would open a second
    # mathtext region and the whole title would render literally.
    ax.set_title(f"Follower supply cost by scenario — {season}\n"
                 f"(box = distribution over 24 h; Σ = daily total; "
                 f"$\\tilde p$ = {price.mean():.3f} USD/kWh)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
# Plot 2 — per-scenario 24 h timeline (dispatch + cost)
# --------------------------------------------------------------------------
def plot_timeline(res: dict, season: str, out_path):
    """Two-panel 24 h timeline for a single scenario."""
    T = res["T"]
    scenario = res["scenario"]
    label = SCENARIO_LABEL.get(scenario, scenario)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # ---- top: dispatch / demand balance ----------------------------------
    # Balance:  D + dd  ==  qMGO + qPG + qsupp + qcri
    _stacked_bars(ax1, T, [
        ("MGO coverage $q^{cov,MGO}_{t,s}$", res["qMGO"], COLORS["diesel"]),
        ("Public grid (EDL)", res["qPG"],     COLORS["grid"]),
        ("Unserved (critical)", res["qcri"],  COLORS["unserved"]),
        ("Suppressed demand", res["qsupp"],   COLORS["supp"]),
    ])
    ax1.plot(T, res["D"], color=COLORS["demand"], marker="o", markersize=3,
             linewidth=1.6, linestyle="--", label="Demand $D_t$")
    ax1.plot(T, res["D"] + res["dd"], color=COLORS["shifted"], marker="s",
             markersize=3, linewidth=1.8, label="Shifted demand $D_t+\\Delta d_t$")
    # EDL availability envelope for this scenario.
    ax1.step(T, res["QbarPG"], where="mid", color=COLORS["grid"],
             linewidth=1.2, linestyle=":",
             label="EDL capacity $\\bar Q^{PG}_{t,s}$")

    ax1.axhline(0, color="#666666", linewidth=0.8)
    ax1.set_ylabel("Energy  [kWh]")
    ax1.set_title(f"Follower dispatch & demand — {season} / {label} "
                  f"($\\pi$ = {res['prob']:.2f})")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.9)

    # ---- bottom: hourly cost split by supply source ----------------------
    _stacked_bars(ax2, T, [
        ("Microgrid  $\\tilde p_t q^{cov,MGO}$", res["c_mgo"], COLORS["diesel"]),
        ("Public grid  $p^{PG}_t q^{cov,PG}$",   res["c_pg"],  COLORS["grid"]),
        ("Unserved  $p^{VOLL} q^{cri}$",         res["c_cri"], COLORS["unserved"]),
    ])

    ax2b = ax2.twinx()
    ax2b.plot(T, res["p"], color=COLORS["dp"], marker="o", markersize=3,
              linewidth=1.5, label="Leader price $\\tilde p_t$")
    ax2b.plot(T, res["P_ref"], color=COLORS["dp"], linewidth=1.0,
              linestyle="--", label="Reference price $P_{ref}$")

    ax2.axhline(0, color="#666666", linewidth=0.8)
    ax2.set_ylabel("Hourly cost  [$]")
    ax2b.set_ylabel("Price  [USD/kWh]")
    ax2.set_xlabel("Hour of day")
    ax2.set_xticks(list(T))
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    # Merge the two legends of the bottom panel.
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, fontsize=8,
               framealpha=0.9)

    total = res["cost"].sum()
    ax2.text(0.99, 0.05, f"Daily cost  Σ = {total:,.1f} $",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
def main(season: str = "winter", price=None) -> None:
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Solving the follower LP [{season}] ...")
    results = solve_follower(season, price)
    if not results:
        print("Nothing to plot.")
        return

    print("Plotting ...")
    plot_cost_boxplot(
        results, season, out_dir / "plot1_lowerlevel_cost_boxplot.png")
    for scenario, res in results.items():
        plot_timeline(
            res, season, out_dir / f"plot2_lowerlevel_timeline_{scenario}.png")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot follower cost and per-scenario dispatch timelines "
                    "for the lower-level LP.")
    parser.add_argument("--season", default="winter",
                        choices=list(config.SEASONS.keys()),
                        help="Representative season to solve and plot.")
    parser.add_argument("--price", type=float, default=None,
                        help="Fixed leader price [$/kWh] (default: the season's P_ref).")
    args = parser.parse_args()
    main(args.season, args.price)
