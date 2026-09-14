"""
The EDL availability scenarios themselves, drawn as a figure per season.

Every other figure in the project shows what the model DOES with the six
public-grid outcomes of :mod:`scenarios`. This one shows what they ARE: the
assumed hourly supply cap QbarPG[t, s] against the village's own demand, so the
assumption can be read off a slide instead of off the source.

The EDL assumptions are season-independent (see :mod:`scenarios`), so the six
supply profiles are identical in every season's figure. What changes from one
season to the next is what they are set against: the village's demand, and when
the sun is up. Each panel therefore states its own daily energy as a share of
that season's demand, and the figure shows the season's PV capacity factor
phi_t twice, on the same hour axis as the windows:

    shading   every scenario panel is shaded behind its data wherever phi_t > 0,
              deeper where the sun is stronger, so each EDL window can be read
              against the daylight hours directly
    strip     a full-width panel under the six gives phi_t itself, in its own
              unit (kWh per kWp per hour), with its peak and daily yield
    note      each panel's box adds the share of the day's sun its window
              holds -- weighted by phi_t, not counted in hours, because phi_t
              is non-zero from 05 to 19 h in every season but the edge hours
              carry almost none of it

phi_t has its own unit and its own axis on purpose: on the demand axis it would
need a second y scale, and two scales on one plot invent a correspondence
between two numbers that have none. It is the same phi_t that already carved
the midday dip out of D_t -- the village demand is net of the 250 x 4 kWp of
rooftop PV -- and that scales the microgrid's own PV, QPV_t = CAP["PV"] * phi_t,
in any build that has some.

One panel per outcome, laid out 2 x 3 in the fixed order of
:data:`scenarios.SCENARIOS`, because six overlaid step profiles on one axis are
unreadable. Colour follows the outcome (:data:`plot.SCENARIO_COLOR`) so it
matches the analysis figures; the panel title carries the identity, so colour is
never the only thing separating two outcomes. PV wears the project's PV colour
(:data:`plot.COLORS` ``"pv"``) everywhere it appears.

Nothing here is solved -- the figure is drawn from the instance data alone, so
it costs no Gurobi time.

THE TIMING PATTERNS
-------------------
A second figure per season does the same for the two synthetic TIMING patterns
of :mod:`pattern_sweep`, ``peak`` and ``offpeak``. They are not EDL outcomes but
a knob: EDL takes over the 8 highest- or the 8 lowest-demand hours of the season
COMPLETELY -- its cap there is M_phys, so it covers whatever is in them -- and
supplies nothing outside. The figure is what makes that selection legible: which
hours the ranking picks, where the cut falls on the demand curve, how much
demand each set carries (the two are NOT equal, which is why each panel states
its own share), and how much of the day's sun it holds, since the whole price
effect in ``pattern_sweep`` runs through whether EDL takes the sunny hours or
the dark ones out of the microgrid's market.

Outputs, under :data:`config.OUTPUT_DIR`:

    scenario_assumptions_<season>.png    the six EDL outcomes
    pattern_assumptions.png              the peak / offpeak timing patterns,
                                         all seasons in one figure

Usage:
    python plot_scenarios.py                          # both, all three seasons
    python plot_scenarios.py --seasons winter summer
    python plot_scenarios.py --figures patterns
"""

import argparse
from typing import Iterable

import matplotlib
matplotlib.use("Agg")             # headless / file output
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import config
from data.input_data import load_data
from plot import COLORS, SCENARIO_COLOR
from run import build_instance
from scenarios import (BASELINE_KW, SCENARIOS, WINDOW_HOURS,
                       availability_shape)

# Presentation sizing: these are read from a projector, not from a page.
FONT = {"panel_title": 12.5, "tick": 10, "axis": 11, "note": 9.5,
        "suptitle": 15, "legend": 10.5}

NCOLS = 3

# PV keeps the project's PV hue for identity. That yellow is too light to read
# as a line on white, so the strip's outline is a darker step of the same hue
# and every PV value is also printed as text.
PV_FILL = COLORS["pv"]
PV_EDGE = "#A38F00"
PV_SHADE_MAX = 0.32     # shading opacity of the sunniest hour; others scale down
STRIP_RATIO = 0.45      # height of the phi_t strip, relative to a scenario row

# The two timing patterns, in the colours pattern_sweep gives them, so a pattern
# keeps its hue from this assumption figure through to the sweep results.
PATTERNS: tuple[str, ...] = ("peak", "offpeak")
PATTERN_COLOR = {"peak": "#D55E00", "offpeak": "#009E73"}
PATTERN_TITLE = {"peak": "Peak hours", "offpeak": "Off-peak hours"}
PATTERN_RULE = {"peak": "the {n} HIGHEST-demand hours",
                "offpeak": "the {n} LOWEST-demand hours"}


def _blocks(values: dict[int, float], hours: list[int]) -> tuple[list, list]:
    """
    Hourly values as a step outline covering the full day.

    The data is hour-BEGINNING: value ``t`` applies over [t, t+1). Drawing it
    with ``step="post"`` needs one extra point on the right, otherwise the last
    hour is a line rather than a block and the window looks an hour short.
    """
    x = [float(t) for t in hours] + [float(hours[-1] + 1)]
    y = [values[t] for t in hours] + [values[hours[-1]]]
    return x, y


def _hour_axis(ax) -> None:
    """The shared 0-24 h axis: ticks every 3 h, hour-beginning labels."""
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 25, 3)])
    ax.tick_params(labelsize=FONT["tick"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _hour_ranges(hours: Iterable[int]) -> str:
    """
    A set of hours as clock ranges, e.g. ``"h22-05, h12"``.

    The bounds are the hour-beginning indices themselves, both inclusive: the
    ranking picks hours, and an end-exclusive label would read "05-11, 12-13"
    for a set that skips h11. A block ending at 23 and one starting at 00 are
    the same night and are joined, which is how the summer peak actually reads.
    """
    hs = sorted(set(hours))
    if not hs:
        return "-"
    blocks: list[list[int]] = [[hs[0], hs[0]]]
    for h in hs[1:]:
        if h == blocks[-1][1] + 1:
            blocks[-1][1] = h
        else:
            blocks.append([h, h])
    if len(blocks) > 1 and blocks[0][0] == 0 and blocks[-1][1] == 23:
        blocks = [[blocks[-1][0], blocks[0][1]]] + blocks[1:-1]
    return ", ".join(f"h{a:02d}" if a == b else f"h{a:02d}-{b:02d}"
                     for a, b in blocks)


def _pv_strip(ax, phi: dict[int, float], T: list[int], season: str) -> None:
    """The phi_t strip: the season's PV capacity factor in its own unit."""
    xp, yp = _blocks(phi, T)
    ax.fill_between(xp, yp, step="post", color=PV_FILL, alpha=0.85, linewidth=0)
    ax.step(xp, yp, where="post", color=PV_EDGE, linewidth=1.6)
    _hour_axis(ax)
    ax.set_xticks(range(25), minor=True)
    ax.set_ylim(0, (max(phi.values()) or 1.0) * 1.32)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.set_xlabel("Hour of day", fontsize=FONT["axis"])
    ax.set_ylabel("$\\varphi_t$  [kWh/kWp]", fontsize=FONT["axis"])

    peak = max(T, key=lambda t: phi[t])
    ax.annotate(f"peak {phi[peak]:.2f}", (peak + 0.5, phi[peak]),
                xytext=(0, 4), textcoords="offset points", ha="center",
                va="bottom", fontsize=FONT["note"], color="#333333")
    ax.set_title(
        f"PV capacity factor $\\varphi_t$ — {season.replace('_', '/')}:  "
        f"{sum(phi.values()):.2f} kWh per kWp per day",
        fontsize=FONT["panel_title"], loc="left", pad=6)


def plot_season(season: str, out_path) -> dict:
    """Draw the six availability assumptions for one season, with its PV."""
    data = build_instance(season)
    T, QbarPG = data["T"], data["QbarPG"]
    D = data["D"]
    D_total = sum(D.values())

    # phi_t is not in the instance: build_instance only carries
    # QPV = CAP["PV"] * phi_t, which is all zeros while config has no PV.
    phi = load_data()["cap_factor"][season]          # {hour: kWh/kWp per h}
    phi_max = max(phi.values()) or 1.0
    yield_day = sum(phi.values())                    # kWh per kWp per day
    sun = [t for t in T if phi[t] > 0.0]

    nrows = -(-len(SCENARIOS) // NCOLS)
    fig = plt.figure(figsize=(13.5, 4.1 * nrows + 2.3))
    gs = fig.add_gridspec(nrows + 1, NCOLS,
                          height_ratios=[1.0] * nrows + [STRIP_RATIO])
    ax0 = fig.add_subplot(gs[0, 0])
    # One shared x and y scale across the six, so a panel's height is
    # comparable across outcomes -- the whole point of "partial" is that it is
    # half as tall as "announced".
    axes = [[ax0 if (r, c) == (0, 0)
             else fig.add_subplot(gs[r, c], sharex=ax0, sharey=ax0)
             for c in range(NCOLS)] for r in range(nrows)]

    y_max = max(max(QbarPG.values()), max(D.values())) * 1.18

    summary = {}
    for i, sc in enumerate(SCENARIOS):
        ax = axes[i // NCOLS][i % NCOLS]
        q = {t: QbarPG[(t, sc.key)] for t in T}
        energy = sum(q.values())
        share = 100.0 * energy / D_total if D_total else 0.0
        n_hours = sum(1 for t in T if q[t] > 0.0)
        # How much of the day's sun falls inside the window, weighted by phi_t
        # rather than counted in hours: phi_t is above zero from 05 to 19 h in
        # every season, but the edge hours carry 0-6% of the peak, so an hour
        # count would call a dusk hour as sunny as noon.
        sun_share = (100.0 * sum(phi[t] for t in T if q[t] > 0.0) / yield_day
                     if yield_day else 0.0)
        summary[sc.key] = {"energy": energy, "share": share, "hours": n_hours,
                           "sun_share": sun_share}

        # Daylight behind everything, deeper where phi_t is higher: the EDL
        # window is read against it, so it must never compete with the data.
        for t in sun:
            ax.axvspan(t, t + 1, color=PV_FILL, linewidth=0, zorder=0,
                       alpha=PV_SHADE_MAX * phi[t] / phi_max)

        # The village's own demand, recessive, in every panel: the window only
        # means something against the load it is supposed to cover.
        xd, yd = _blocks(D, T)
        ax.step(xd, yd, where="post", color=COLORS["demand"], linewidth=1.4,
                linestyle="--", alpha=0.75, label="Village demand $D_t$")

        color = SCENARIO_COLOR.get(sc.key)
        if energy > 0.0 and color is not None:
            xq, yq = _blocks(q, T)
            ax.fill_between(xq, yq, step="post", color=color, alpha=0.55,
                            linewidth=0, label="EDL availability $\\bar Q^{PG}_{t,s}$")
            ax.step(xq, yq, where="post", color=color, linewidth=2.0)
        else:
            # "No supply" draws nothing, so say so rather than leave it blank.
            ax.annotate("EDL off all day", (0.5, 0.55),
                        xycoords="axes fraction", ha="center", va="center",
                        fontsize=FONT["panel_title"], color="#888888",
                        style="italic")

        ax.set_title(f"{sc.label}   ($P$ = {sc.prob:.2f})",
                     fontsize=FONT["panel_title"], pad=7)
        note = (f"{energy:,.0f} kWh/day  =  {share:.0f}% of demand\n"
                f"{n_hours} h of supply"
                + (f"  at {sc.avail_fraction:.0%} capacity"
                   if sc.avail_fraction != 1.0 else "")
                + (f",  {sun_share:.0f}% of the day's sun" if n_hours else ""))
        ax.annotate(note, (0.5, 0.985), xycoords="axes fraction",
                    ha="center", va="top", fontsize=FONT["note"],
                    color="#333333",
                    bbox=dict(boxstyle="round,pad=0.32", fc="white",
                              ec="#cccccc", alpha=0.88))

        _hour_axis(ax)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        # The hour labels live on the bottom scenario row and on the strip;
        # repeating them on the top row only crowds the titles below it.
        if i // NCOLS < nrows - 1:
            ax.tick_params(labelbottom=False)

    for i in range(len(SCENARIOS), nrows * NCOLS):      # unused cells
        axes[i // NCOLS][i % NCOLS].set_visible(False)

    for r in range(nrows):
        axes[r][0].set_ylabel("Energy per hour  [kWh]", fontsize=FONT["axis"])

    # ---- the PV capacity factor itself, in its own unit ----------------------
    _pv_strip(fig.add_subplot(gs[nrows, :]), phi, T, season)
    peak = max(T, key=lambda t: phi[t])

    # The EDL assumptions are season-independent, so the capacity and window in
    # this subtitle are the same in every panel of every season; only the daily
    # demand and the daylight they are set against change.
    announced_h = len(WINDOW_HOURS["announced"])
    fig.suptitle(
        f"Assumed EDL supply scenarios — {season.replace('_', '/')}\n"
        f"baseline capacity {BASELINE_KW / 1000.0:.3f} MW  ·  announced window "
        f"{announced_h} h  ·  daily demand {D_total:,.0f} kWh  ·  "
        f"PV yield {yield_day:.2f} kWh/kWp",
        fontsize=FONT["suptitle"], y=0.985)

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(Patch(facecolor=PV_FILL, alpha=PV_SHADE_MAX + 0.1,
                         edgecolor="none"))
    labels.append("Daylight — shading deepens with $\\varphi_t$")
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=FONT["legend"], frameon=False,
               bbox_to_anchor=(0.5, -0.004))

    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    print(f"  wrote {out_path}")
    plt.close(fig)
    summary["_pv"] = {"yield": yield_day, "peak": phi[peak], "peak_hour": peak}
    return summary


def _pattern_row(axes, ax_pv, season: str, patterns: list[str], y_max: float,
                 *, top: bool, bottom: bool) -> dict:
    """
    One season's row of the pattern figure: a panel per pattern, then phi_t.

    In the hours a pattern names, EDL's cap is M_phys: it covers whatever is
    there, so the hour is taken over COMPLETELY and the cap itself is not a
    quantity worth drawing on a demand axis. What a panel shows instead is the
    demand EDL takes over -- the covered hours filled in under the demand curve
    -- and what it leaves the microgrid, which is the rest of the day. The two
    panels therefore do NOT carry equal energy, so each states its own share.
    """
    data = build_instance(season)
    T, D = data["T"], data["D"]
    D_total = sum(D.values())

    phi = load_data()["cap_factor"][season]
    phi_max = max(phi.values()) or 1.0
    yield_day = sum(phi.values())
    sun = [t for t in T if phi[t] > 0.0]

    # The hour selection only; the cap in those hours is unlimited by design.
    shapes = {p: availability_shape(season, p, D) for p in patterns}

    summary = {}
    for ax, pattern in zip(axes, patterns):
        on = [t for t in T if shapes[pattern][t] > 0.0]
        covered = sum(D[t] for t in on)          # what EDL takes over [kWh]
        share = covered / D_total if D_total else 0.0
        sun_share = (100.0 * sum(phi[t] for t in on) / yield_day
                     if yield_day else 0.0)
        summary[pattern] = {"energy": covered, "share": 100.0 * share,
                            "hours": len(on), "sun_share": sun_share,
                            "window": _hour_ranges(on)}

        for t in sun:
            ax.axvspan(t, t + 1, color=PV_FILL, linewidth=0, zorder=0,
                       alpha=PV_SHADE_MAX * phi[t] / phi_max)

        xd, yd = _blocks(D, T)
        ax.step(xd, yd, where="post", color=COLORS["demand"], linewidth=1.4,
                linestyle="--", alpha=0.75, label="Village demand $D_t$")

        # The cut the ranking makes, on the demand curve the ranking read: the
        # selected hours are exactly those above (peak) or below (offpeak) it.
        cut = min(D[t] for t in on) if pattern == "peak" else max(D[t] for t in on)
        ax.axhline(cut, color=COLORS["demand"], linewidth=0.9, linestyle=":",
                   alpha=0.6)
        ax.annotate(f"cut {cut:,.0f} kWh", (0.012, cut),
                    xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=FONT["note"] - 1.0, color="#555555")

        # The demand EDL takes over, filled to the demand curve and tinted full
        # height, since the cap above it is unlimited.
        color = PATTERN_COLOR[pattern]
        xq, yq = _blocks({t: (D[t] if t in on else 0.0) for t in T}, T)
        ax.fill_between(xq, yq, step="post", color=color, alpha=0.55,
                        linewidth=0,
                        label="Demand EDL takes over  ($\\bar Q_t = M$)")
        ax.step(xq, yq, where="post", color=color, linewidth=2.0)
        for t in on:
            ax.axvspan(t, t + 1, color=color, alpha=0.10, linewidth=0,
                       zorder=0.5)

        if top:
            ax.set_title(f"{PATTERN_TITLE[pattern]}   "
                         f"({PATTERN_RULE[pattern].format(n=len(on))})",
                         fontsize=FONT["panel_title"], pad=8)
        ax.annotate(f"{covered:,.0f} kWh/day = {100.0 * share:.0f}% of demand\n"
                    f"{summary[pattern]['window']},  "
                    f"{sun_share:.0f}% of the sun",
                    (0.5, 0.975), xycoords="axes fraction", ha="center",
                    va="top", fontsize=FONT["note"] - 0.5, color="#333333",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#cccccc", alpha=0.9))

        _hour_axis(ax)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.tick_params(labelbottom=bottom)
        if bottom:
            ax.set_xlabel("Hour of day", fontsize=FONT["axis"])

    axes[0].set_ylabel(f"{season.replace('_', '/').capitalize()}\n"
                       f"Energy per hour  [kWh]", fontsize=FONT["axis"])

    # phi_t in its own unit, beside the two panels on the same hour axis.
    _pv_strip(ax_pv, phi, T, season)
    ax_pv.set_title(("PV capacity factor $\\varphi_t$\n" if top else "")
                    + f"{yield_day:.2f} kWh/kWp per day",
                    fontsize=FONT["note"] + 0.5, loc="left", pad=6)
    ax_pv.tick_params(labelbottom=bottom)
    if not bottom:
        ax_pv.set_xlabel("")
    summary["_demand"] = D_total
    return summary


def plot_patterns(seasons: list[str], out_path,
                  patterns: Iterable[str] = PATTERNS) -> dict:
    """
    The synthetic timing patterns for every season, in one figure.

    A row per season, a column per pattern, and the season's PV capacity factor
    beside them. All demand panels share one scale, so a season's height can be
    read against the others -- summer's night-time peak is what makes its row
    tallest, and that is worth seeing rather than normalising away.
    """
    patterns = list(patterns)
    order = [s for s in config.SEASONS if s in seasons]
    y_max = max(max(build_instance(s)["D"].values()) for s in order) * 1.22

    nrows = len(order)
    fig = plt.figure(figsize=(16.0, 3.9 * nrows + 1.6))
    gs = fig.add_gridspec(nrows, len(patterns) + 1,
                          width_ratios=[1.0] * len(patterns) + [0.55])

    summaries = {}
    first = None
    for r, season in enumerate(order):
        row_axes = []
        for c in range(len(patterns)):
            kw = {} if first is None else {"sharex": first, "sharey": first}
            ax = fig.add_subplot(gs[r, c], **kw)
            first = first or ax
            row_axes.append(ax)
        ax_pv = fig.add_subplot(gs[r, len(patterns)])
        summaries[season] = _pattern_row(
            row_axes, ax_pv, season, patterns, y_max,
            top=(r == 0), bottom=(r == nrows - 1))

    fig.suptitle(
        "Synthetic EDL timing patterns: EDL covers its 8 hours completely "
        "($\\bar Q_t = M_{phys}$), so the two patterns differ in energy as "
        "well as timing", fontsize=FONT["suptitle"], y=0.992)

    handles, labels = first.get_legend_handles_labels()
    handles.append(Patch(facecolor=PV_FILL, alpha=PV_SHADE_MAX + 0.1,
                         edgecolor="none"))
    labels.append("Daylight — shading deepens with $\\varphi_t$")
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=FONT["legend"], frameon=False,
               bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(out_path, dpi=200)
    print(f"  wrote {out_path}")
    plt.close(fig)
    return summaries


def main(seasons: list[str], figures: list[str]) -> None:
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        if "scenarios" in figures:
            print(f"[{season}]")
            summary = plot_season(
                season, out_dir / f"scenario_assumptions_{season}.png")
            pv = summary.pop("_pv")
            print(f"  {'PV':>12}: {pv['yield']:.2f} kWh/kWp per day, peak "
                  f"{pv['peak']:.2f} at h{pv['peak_hour']:02d}")
            for key, s in summary.items():
                print(f"  {key:>12}: {s['hours']:2d} h  {s['energy']:7,.0f} kWh"
                      f"  {s['share']:5.1f}% of demand  "
                      f"{s['sun_share']:5.1f}% of the day's sun")


    # All seasons in one pattern figure, since the rows are read against each
    # other.
    if "patterns" in figures:
        for season, summary in plot_patterns(
                seasons, out_dir / "pattern_assumptions.png").items():
            print(f"[{season}] patterns, demand {summary.pop('_demand'):,.0f} "
                  f"kWh/day")
            for key, s in summary.items():
                print(f"  {key:>12}: {s['hours']:2d} h  {s['energy']:7,.0f} kWh"
                      f"  {s['share']:5.1f}% of demand  "
                      f"{s['sun_share']:5.1f}% of the day's sun  "
                      f"{s['window']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw the assumed EDL availability scenarios and the "
                    "synthetic peak / offpeak timing patterns against the "
                    "village demand and the season's PV capacity factor, one "
                    "figure per season.")
    parser.add_argument("--seasons", nargs="+",
                        default=list(config.SEASONS.keys()),
                        choices=list(config.SEASONS.keys()),
                        help="Seasons to draw (default: all three).")
    parser.add_argument("--figures", nargs="+",
                        default=["scenarios", "patterns"],
                        choices=["scenarios", "patterns"],
                        help="Which figures to draw (default: both).")
    args = parser.parse_args()
    main(args.seasons, args.figures)
