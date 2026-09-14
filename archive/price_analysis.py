"""
Price analysis: sweep the leader price upwards from the reference tariff.

For each price level the leader's offer is PINNED (the discrete menu is
collapsed to the single value under test and the cap is moved to match), so the
model reduces to: "given this price, what does the village do, and what does
the MGO earn?". The follower still responds through its full KKT system and the
MGO still dispatches optimally, so every curve below is an equilibrium
response, not a partial calculation.

Swept over ``p = f * P_ref`` for ``f`` from 1.0 (the reference tariff) up to
``config.CONFIG["phat_factor"]`` (the model's own price cap), for winter and
summer, using the basic scenario set from :mod:`scenarios`.

The same sweep is then repeated once per EDL outcome with that outcome treated
as CERTAIN (weight 1), which gives a profit-maximizing price per scenario. In
the stochastic run there is no such thing: the leader commits one price
schedule before the scenario realizes, so the per-outcome optima show how far
apart the stochastic commitment would pull if the MGO knew what EDL was going
to do.

Outputs, under :data:`config.OUTPUT_DIR`:

    price_analysis.xlsx         one sheet per season, a combined sheet, the
                                per-scenario optima and their full curves
    price_analysis.png          profit / cost and quantity curves per season
    price_analysis_scenarios.png
                                profit vs price for every outcome taken as
                                certain, with each optimum marked

Usage:
    python price_analysis.py
    python price_analysis.py --seasons winter summer --steps 15
"""

import argparse

import matplotlib
matplotlib.use("Agg")             # headless / file output
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from pyomo.environ import value, TerminationCondition

import config
from model.bilevel_mpec import BilevelMPEC
from plot import COLORS
from run import build_instance
from scenarios import SCENARIOS, SCENARIO_LABEL

# Columns of the result table, in order.
FIELDS = [
    ("factor",        "p / P_ref"),
    ("p",             "price [$/kWh]"),
    ("profit",        "leader profit [$]"),
    ("revenue",       "leader revenue [$]"),
    ("gen_cost",      "generation cost [$]"),
    ("follower_cost", "follower cost [$]"),
    ("qMGO",          "E[sum qMGO] [kWh]"),
    ("qPG",           "E[sum qPG] [kWh]"),
    ("qcri",          "E[sum qcri] [kWh]"),
    ("qsupp",         "sum qsupp [kWh]"),
    ("D",             "sum D [kWh]"),
    ("supp_share",    "qsupp / D [%]"),
    ("termination",   "termination"),
]


# --------------------------------------------------------------------------
def solve_at_price(season: str, p: float, opts: dict,
                   scenario: str | None = None) -> dict | None:
    """
    Solve the bilevel model with the leader price pinned to ``p``.

    With ``scenario`` given, that EDL outcome is treated as CERTAIN (its
    probability set to 1 and the others dropped), which is what makes a
    per-outcome optimum meaningful: in the stochastic solve the leader commits
    one price schedule before the scenario realizes, so there is no such thing
    as a per-scenario price.

    Returns one result record, or ``None`` if the solve did not reach an
    optimal/feasible point.
    """
    data = build_instance(season)

    if scenario is not None:
        data["S"] = [scenario]
        data["prob"] = {scenario: 1.0}
        data["QbarPG"] = {(t, scenario): data["QbarPG"][(t, scenario)]
                          for t in data["T"]}

    T, S, prob = data["T"], data["S"], data["prob"]

    # Pin the offer: a one-entry menu leaves the leader no price choice, and
    # the cap is moved to p so it cannot cut the sweep short.
    data["Pi"] = [p]
    data["Phat"] = {t: p for t in T}

    mpec = BilevelMPEC(data)
    m = mpec.build()
    solver_name, res = mpec.solve("gurobi", options=opts)
    if solver_name is None:
        raise RuntimeError(
            "Gurobi not available (check the license and that gurobipy is "
            "installed)."
        )

    tc = res.solver.termination_condition
    if tc not in (TerminationCondition.optimal, TerminationCondition.feasible):
        print(f"  p={p:.3f}: not solved ({tc})")
        return None

    def exp_sum(var):
        return sum(prob[s] * value(var[t, s]) for t in T for s in S)

    qMGO = exp_sum(m.qMGO)
    qPG  = exp_sum(m.qPG)
    qcri = exp_sum(m.qcri)
    qsupp = exp_sum(m.qsupp)
    sD = sum(data["D"][t] for t in T)

    revenue = p * qMGO
    gen_cost = sum(prob[s] * (data["cdiesel"][t] * value(m.qdiesel[t, s])
                              + data["cPV"][t] * value(m.qPV[t, s]))
                   for t in T for s in S)
    follower_cost = (revenue
                     + sum(prob[s] * data["pPG"][t] * value(m.qPG[t, s])
                           for t in T for s in S)
                     + data["pVOLL"] * qcri)

    return {
        "scenario": scenario or "stochastic",
        "factor": p / data["P"][T[0]],
        "p": p,
        "profit": value(m.profit),
        "revenue": revenue,
        "gen_cost": gen_cost,
        "follower_cost": follower_cost,
        "qMGO": qMGO, "qPG": qPG, "qcri": qcri, "qsupp": qsupp, "D": sD,
        "supp_share": 100.0 * qsupp / sD if sD else 0.0,
        "termination": str(tc),
    }


def _factors(steps: int) -> list[float]:
    """Price factors from 1.0 (the reference tariff) up to the configured cap."""
    f_max = config.CONFIG["phat_factor"]
    return [1.0 + (f_max - 1.0) * k / (steps - 1) for k in range(steps)]


def sweep(season: str, steps: int, opts: dict,
          scenario: str | None = None, quiet: bool = False) -> list[dict]:
    """Sweep the price from P_ref up to the configured cap for one season."""
    P_ref = config.CONFIG["P_ref"]

    if not quiet:
        print(f"[{season}] P_ref = {P_ref:.3f}, sweeping p to "
              f"{config.CONFIG['phat_factor']:.2f} x P_ref ({steps} steps) ...")
    rows = []
    for f in _factors(steps):
        rec = solve_at_price(season, f * P_ref, opts, scenario)
        if rec is None:
            continue
        rows.append(rec)
        if not quiet:
            print(f"  p={rec['p']:.3f} ({rec['factor']:.2f} x P_ref)  "
                  f"profit={rec['profit']:9.2f}  sales={rec['qMGO']:8.1f} kWh  "
                  f"suppressed={rec['supp_share']:5.1f}%")
    return rows


def sweep_scenarios(season: str, steps: int,
                    opts: dict) -> dict[str, list[dict]]:
    """
    Sweep the price once per EDL outcome, each treated as certain (weight 1).

    Returns ``{scenario_key: rows}``.
    """
    P_ref = config.CONFIG["P_ref"]
    print(f"[{season}] per-scenario sweeps (weight 1 each) ...")
    out: dict[str, list[dict]] = {}
    for sc in SCENARIOS:
        rows = sweep(season, steps, opts, scenario=sc.key, quiet=True)
        if not rows:
            print(f"  {sc.key:>12}: no price point solved")
            continue
        out[sc.key] = rows
        best = max(rows, key=lambda r: r["profit"])
        flat = (best["profit"] - min(r["profit"] for r in rows)) < 1e-6
        print(f"  {sc.key:>12}: optimum {100 * best['factor']:5.1f}% of P_ref "
              f"(p={best['p']:.3f})  profit={best['profit']:9.2f}  "
              f"sales={best['qMGO']:8.1f} kWh"
              f"{'   [flat curve - no optimum]' if flat else ''}")
    return out


# --------------------------------------------------------------------------
def _write_sheet(wb, title: str, rows: list[dict]) -> None:
    ws = wb.create_sheet(title)
    header = [label for _, label in FIELDS]
    ws.append(header)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for r in rows:
        ws.append([r[key] for key, _ in FIELDS])
    ws.freeze_panes = "A2"
    for i, label in enumerate(header, start=1):
        ws.column_dimensions[get_column_letter(i)].width = min(len(label) + 4, 24)


OPT_FIELDS = [
    ("season",        "season"),
    ("scenario",      "scenario"),
    ("label",         "outcome"),
    ("prob",          "prob (stochastic run)"),
    ("factor_pct",    "optimal p [% of P_ref]"),
    ("p",             "optimal p [$/kWh]"),
    ("profit",        "profit at optimum [$]"),
    ("qMGO",          "sales at optimum [kWh]"),
    ("supp_share",    "qsupp / D [%]"),
    ("note",          "note"),
]


def save_sweep(results: dict[str, list[dict]], out_path,
               per_scenario: dict[str, dict[str, list[dict]]] | None = None,
               steps: int = 0) -> None:
    """Write one sheet per season, a combined sheet, and -- when per-scenario
    sweeps were run -- their optima and full curves."""
    wb = Workbook()
    wb.remove(wb.active)
    for season, rows in results.items():
        _write_sheet(wb, season, rows)

    combined = wb.create_sheet("all")
    combined.append(["season"] + [label for _, label in FIELDS])
    for cell in combined[1]:
        cell.font = Font(bold=True)
    for season, rows in results.items():
        for r in rows:
            combined.append([season] + [r[key] for key, _ in FIELDS])
    combined.freeze_panes = "A2"

    if per_scenario:
        probs = {sc.key: sc.prob for sc in SCENARIOS}

        ws = wb.create_sheet("optimum_per_scenario")
        ws.append([label for _, label in OPT_FIELDS])
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for season, by_sc in per_scenario.items():
            for key, rows in by_sc.items():
                best = max(rows, key=lambda r: r["profit"])
                # If the profit curve is flat the argmax is just the first grid
                # point, not a real optimum -- happens when EDL covers the whole
                # load and the MGO sells nothing at any price.
                spread = best["profit"] - min(r["profit"] for r in rows)
                rec = {
                    "season": season, "scenario": key,
                    "label": SCENARIO_LABEL.get(key, key),
                    "prob": probs.get(key),
                    "factor_pct": 100.0 * best["factor"],
                    "p": best["p"], "profit": best["profit"],
                    "qMGO": best["qMGO"], "supp_share": best["supp_share"],
                    "note": ("flat profit curve - optimum not identified"
                             if spread < 1e-6 else ""),
                }
                ws.append([rec[k] for k, _ in OPT_FIELDS])
        ws.freeze_panes = "A2"
        for i, (_, label) in enumerate(OPT_FIELDS, start=1):
            ws.column_dimensions[get_column_letter(i)].width = len(label) + 4
        # The optimum can only be as precise as the sweep grid.
        ws.append([])
        ws.append([f"grid resolution: {steps} points over "
                   f"100..{100 * config.CONFIG['phat_factor']:.0f}% of P_ref"])

        # Full per-scenario curves, for plotting or further analysis.
        ws = wb.create_sheet("scenario_curves")
        ws.append(["season", "scenario"] + [label for _, label in FIELDS])
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for season, by_sc in per_scenario.items():
            for key, rows in by_sc.items():
                for r in rows:
                    ws.append([season, key] + [r[k] for k, _ in FIELDS])
        ws.freeze_panes = "A2"

    wb.save(out_path)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
def plot_sweep(results: dict[str, list[dict]], out_path) -> None:
    """Profit/cost and quantity curves, one column per season."""
    seasons = list(results)
    # The x axis is the price relative to that season's reference tariff, so
    # the seasons share one scale and can be read against each other directly.
    fig, axes = plt.subplots(2, len(seasons), figsize=(6.5 * len(seasons), 8),
                             sharex=True, squeeze=False)

    for j, season in enumerate(seasons):
        rows = results[season]
        p = [100.0 * r["factor"] for r in rows]        # % of P_ref
        P_ref = config.CONFIG["P_ref"]

        # ---- top: leader profit (+ the price that maximizes it) ----
        ax = axes[0][j]
        profit = [r["profit"] for r in rows]
        ax.plot(p, profit, color=COLORS["profit"], marker="o", markersize=3,
                linewidth=1.8, label="Leader profit")
        ax.plot(p, [r["follower_cost"] for r in rows], color=COLORS["supp"],
                marker="s", markersize=3, linewidth=1.5, label="Follower cost")
        ax.plot(p, [r["gen_cost"] for r in rows], color=COLORS["diesel"],
                linewidth=1.2, linestyle="--", label="Generation cost")

        best = max(rows, key=lambda r: r["profit"])
        ax.axvline(100.0 * best["factor"], color="#666666", linewidth=1.0,
                   linestyle=":")
        ax.annotate(f"max profit\n{100 * best['factor']:.0f}% of $P_{{ref}}$\n"
                    f"(p = {best['p']:.3f})",
                    xy=(100.0 * best["factor"], best["profit"]),
                    xytext=(6, -46), textcoords="offset points", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="#cccccc",
                              alpha=0.9))
        ax.axhline(0, color="#666666", linewidth=0.8)
        ax.set_ylabel("Daily total  [$]")
        ax.set_title(f"{season} — $P_{{ref}}$ = {P_ref:.3f} USD/kWh")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, framealpha=0.9)

        # ---- bottom: where the village gets its energy ----
        ax = axes[1][j]
        ax.plot(p, [r["qMGO"] for r in rows], color=COLORS["diesel"],
                marker="o", markersize=3, linewidth=1.8,
                label="Microgrid sales E[$\\Sigma q^{cov,MGO}$]")
        ax.plot(p, [r["qPG"] for r in rows], color=COLORS["grid"],
                marker="s", markersize=3, linewidth=1.5,
                label="Public grid E[$\\Sigma q^{cov,PG}$]")
        ax.plot(p, [r["qsupp"] for r in rows], color=COLORS["supp"],
                marker="^", markersize=3, linewidth=1.5,
                label="Suppressed demand $\\Sigma q^{supp}$")
        ax.plot(p, [r["qcri"] for r in rows], color=COLORS["unserved"],
                linewidth=1.2, label="Unserved E[$\\Sigma q^{cri}$]")
        ax.plot(p, [r["D"] for r in rows], color=COLORS["demand"],
                linewidth=1.0, linestyle="--", label="Demand $\\Sigma D_t$")

        ax.set_xlabel("Leader price $\\tilde p$  [% of $P_{ref}$]")
        ax.set_ylabel("Daily energy  [kWh]")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
def plot_scenario_optima(per_scenario: dict[str, dict[str, list[dict]]],
                         out_path) -> None:
    """Profit vs price for every EDL outcome taken as certain, one panel per
    season, with each outcome's optimum marked."""
    seasons = list(per_scenario)
    fig, axes = plt.subplots(1, len(seasons), figsize=(7.0 * len(seasons), 5.5),
                             sharex=True, squeeze=False)
    cmap = plt.get_cmap("tab10")
    order = [sc.key for sc in SCENARIOS]

    for j, season in enumerate(seasons):
        ax = axes[0][j]
        by_sc = per_scenario[season]
        for i, key in enumerate([k for k in order if k in by_sc]):
            rows = by_sc[key]
            x = [100.0 * r["factor"] for r in rows]
            y = [r["profit"] for r in rows]
            color = cmap(i % 10)
            ax.plot(x, y, color=color, marker="o", markersize=3, linewidth=1.6,
                    label=SCENARIO_LABEL.get(key, key))
            best = max(rows, key=lambda r: r["profit"])
            ax.plot(100.0 * best["factor"], best["profit"], marker="*",
                    markersize=14, color=color, markeredgecolor="#222222",
                    markeredgewidth=0.6, linestyle="none")

        ax.axhline(0, color="#666666", linewidth=0.8)
        ax.set_xlabel("Leader price $\\tilde p$  [% of $P_{ref}$]")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_ylabel("Leader profit  [$]")
        ax.set_title(f"{season} — optimum per EDL outcome (weight 1 each)\n"
                     f"$\\bigstar$ = profit-maximizing price")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
def main(seasons: list[str], steps: int) -> None:
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = {"MIPGap": config.CONFIG["MIPGap"],
            "TimeLimit": config.CONFIG["TimeLimit"]}

    results = {season: sweep(season, steps, opts) for season in seasons}
    results = {k: v for k, v in results.items() if v}
    if not results:
        print("No price point solved — nothing to write.")
        return

    # Optimum price per EDL outcome, each taken as certain.
    per_scenario = {season: sweep_scenarios(season, steps, opts)
                    for season in results}
    per_scenario = {k: v for k, v in per_scenario.items() if v}

    save_sweep(results, out_dir / "price_analysis.xlsx",
               per_scenario=per_scenario, steps=steps)
    plot_sweep(results, out_dir / "price_analysis.png")
    if per_scenario:
        plot_scenario_optima(per_scenario,
                             out_dir / "price_analysis_scenarios.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep the leader price upwards from the reference tariff.")
    parser.add_argument("--seasons", nargs="+", default=["winter", "summer"],
                        choices=list(config.SEASONS.keys()),
                        help="Seasons to analyse (default: winter summer).")
    parser.add_argument("--steps", type=int, default=13,
                        help="Number of price points from P_ref to the cap.")
    args = parser.parse_args()
    main(args.seasons, args.steps)
