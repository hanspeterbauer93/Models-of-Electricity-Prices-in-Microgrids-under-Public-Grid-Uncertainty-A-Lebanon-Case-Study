"""
24-hour dispatch under the synthetic EDL timing patterns, in one figure.

The hourly counterpart of :mod:`pattern_sweep`, which reports only daily price
and profit. EDL takes over the 8 highest-demand (peak) or the 8 lowest-demand
(offpeak) hours of the season completely -- its cap there is M_phys -- and
supplies nothing outside them. Every run is deterministic, prob = 1, so the
leader knows the pattern when it posts its price.

One figure, four panels:

                 peak                     offpeak
    DG only      400 kW DG                400 kW DG
    DG + PV      400 kW DG + 200 kWp PV   400 kW DG + 200 kWp PV

Each panel stacks, hour by hour, how the village load is met -- diesel and PV
in their own colours, then EDL, unserved and suppressed demand -- with the
demand curve and the shifted demand drawn over it, in the language of
:func:`plot.plot_timeline`. The leader's flat price and daily profit are
printed in the panel: the tariff has no hourly shape to plot.

The fleets are :mod:`pattern_sweep`'s at config's load shift, so each panel is
the same model as that module's prob = 1 run at the same shift, and its price
and profit should match that workbook.

Output, under :data:`config.OUTPUT_DIR`:

    timeseries_<season>.png

Usage:
    python timeseries.py                  # winter (default)
    python timeseries.py --season summer
"""

import argparse

import matplotlib
matplotlib.use("Agg")             # headless / file output
import matplotlib.pyplot as plt
import numpy as np

import config
from pattern_sweep import PATTERNS, PV_KWP
# The solve, the hourly extraction and the bar stacking are plot.py's own, so
# these panels and the per-scenario timelines are built the same way.
from plot import COLORS, _solve, _stacked_bars
from run import build_instance
from scenarios import uncapped_availability
from stochastic_commitment_analysis import Build, build_settings

# The rest of the fleet: the standard assets, read from config.
DG_KW: float = config.CONFIG["CAP"]["DG"]
SHIFT: float = config.CONFIG["shift_frac"]

# The two rows of the figure: (row name, microgrid PV [kWp]).
FLEETS: tuple[tuple[str, float], ...] = (("DG only", 0.0), ("DG + PV", PV_KWP))
WHERE = {"peak": "highest", "offpeak": "lowest"}

# Stack order bottom-up: the microgrid's own supply first, split by technology.
LEGEND_ORDER = ("Diesel", "PV", "Public grid (EDL)", "Unserved (critical)",
                "Suppressed demand", "Demand $D_t$",
                "Shifted demand $D_t+\\Delta d_t$")


def _fleet_label(pv_kwp: float) -> str:
    return (f"{DG_KW:.0f} kW DG" if pv_kwp <= 0.0
            else f"{DG_KW:.0f} kW DG + {pv_kwp:.0f} kWp PV")


# --------------------------------------------------------------------------
# Solve
# --------------------------------------------------------------------------
def solve_pattern(season: str, pattern: str, pv_kwp: float) -> dict | None:
    """
    One deterministic run: ``pattern``'s hours taken over by EDL, on the
    standard genset plus ``pv_kwp`` of PV. Returns :func:`plot._extract`'s
    hourly arrays, or ``None`` if the solve failed.
    """
    build = Build(f"pv{pv_kwp:g}", _fleet_label(pv_kwp), DG_KW, pv_kwp, SHIFT)
    # build_instance reads the fleet off config at call time; once built, the
    # data dict carries it and the solve no longer needs the patch.
    with build_settings(build):
        data = dict(build_instance(season))
    T = data["T"]
    qbar = uncapped_availability(season, pattern, data["D"])
    data["S"] = [pattern]
    data["prob"] = {pattern: 1.0}
    data["QbarPG"] = {(t, pattern): qbar[t] for t in T}

    res = _solve(data, pattern)
    if res is not None:
        res["covered"] = [t for t in T if qbar[t] > 0.0]
    return res


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def _panel(ax, res: dict, title: str) -> None:
    """One pattern on one fleet: the stacked dispatch and the demand curves."""
    T = res["T"]
    _stacked_bars(ax, T, [
        ("Diesel",              res["qdiesel"], COLORS["diesel"]),
        ("PV",                  res["qPV"],     COLORS["pv"]),
        ("Public grid (EDL)",   res["qPG"],     COLORS["grid"]),
        ("Unserved (critical)", res["qcri"],    COLORS["unserved"]),
        ("Suppressed demand",   res["qsupp"],   COLORS["supp"]),
    ])
    ax.plot(T, res["D"], color=COLORS["demand"], marker="o", markersize=3,
            linewidth=1.6, linestyle="--", label="Demand $D_t$")
    ax.plot(T, res["D"] + res["dd"], color=COLORS["shifted"], marker="s",
            markersize=3, linewidth=1.8,
            label="Shifted demand $D_t+\\Delta d_t$")

    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])
    ax.set_xlim(-0.6, 23.6)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_title(title, loc="left", fontsize=11)

    p, P_ref = res["p"], res["P_ref"]
    if np.allclose(p, p[0]):
        price = f"{p[0]:.3f} \\$/kWh  ({p[0] / P_ref[0]:.2f} × $P_{{ref}}$)"
    else:                        # hourly tariff (config flat_price = False)
        price = f"{p.min():.3f}–{p.max():.3f} \\$/kWh"
    ax.text(0.99, 0.97,
            f"price  {price}\nprofit  {res['profit'].sum():,.1f} \\$/day",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#cccccc",
                      alpha=0.95))


def plot(results: dict[tuple[str, str], dict], season: str, out_path) -> None:
    """Rows are the fleets, columns the patterns; one shared energy scale."""
    fig, axes = plt.subplots(len(FLEETS), len(PATTERNS), figsize=(16.0, 9.5),
                             sharex=True, sharey=True, squeeze=False)
    handles: dict[str, object] = {}

    for i, (fleet, pv) in enumerate(FLEETS):
        for j, pattern in enumerate(PATTERNS):
            ax = axes[i, j]
            title = (f"{_fleet_label(pv)}  ·  EDL on the {WHERE[pattern]}-"
                     f"demand hours ({pattern})")
            res = results.get((fleet, pattern))
            if res is None:
                ax.set_title(title, loc="left", fontsize=11)
                ax.text(0.5, 0.5, "not solved", transform=ax.transAxes,
                        ha="center", va="center", color="#999999")
                continue
            _panel(ax, res, title)
            for h, lbl in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(lbl, h)
            # Shared x hides the top row's hours; every panel keeps its scale.
            ax.tick_params(axis="x", labelbottom=True)
        axes[i, 0].set_ylabel("Energy per hour  [kWh]")
    for ax in axes[-1]:
        ax.set_xlabel("Hour of day")

    # Headroom above the tallest hour for the price / profit box.
    top = max(max((r["D"] + r["dd"]).max(),
                  (r["qdiesel"] + r["qPV"] + r["qPG"] + r["qcri"]
                   + r["qsupp"]).max())
              for r in results.values())
    axes[0, 0].set_ylim(0, top * 1.3)

    labels = [lbl for lbl in LEGEND_ORDER if lbl in handles]
    fig.legend([handles[lbl] for lbl in labels], labels, loc="lower center",
               ncol=len(labels), frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"24 h dispatch under the EDL timing patterns — "
                 f"{season.replace('_', '/')}, prob = 1, "
                 f"{100 * SHIFT:.0f}% load shift", fontsize=14)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
def main(season: str = "winter") -> None:
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[tuple[str, str], dict] = {}
    for fleet, pv in FLEETS:
        for pattern in PATTERNS:
            print(f"[{season}] {_fleet_label(pv)}, {pattern}")
            res = solve_pattern(season, pattern, pv)
            if res is None:
                continue
            results[(fleet, pattern)] = res
            print(f"  hours {','.join(map(str, res['covered']))}  "
                  f"p={res['p'][0]:.4f}  profit={res['profit'].sum():8.2f}  "
                  f"diesel={res['qdiesel'].sum():7.1f}  "
                  f"PV={res['qPV'].sum():6.1f}  EDL={res['qPG'].sum():7.1f}  "
                  f"supp={res['qsupp'].sum():6.1f}  "
                  f"unserved={res['qcri'].sum():5.1f} kWh")

    if not results:
        print("Nothing solved — no figure written.")
        return
    plot(results, season, out_dir / f"timeseries_{season}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="24 h dispatch when EDL covers the peak or the offpeak "
                    "hours, on a diesel-only and on a PV fleet, in one figure.")
    parser.add_argument("--season", default="winter",
                        choices=list(config.SEASONS.keys()),
                        help="Representative season to solve and plot "
                             "(default: winter).")
    args = parser.parse_args()
    main(args.season)
