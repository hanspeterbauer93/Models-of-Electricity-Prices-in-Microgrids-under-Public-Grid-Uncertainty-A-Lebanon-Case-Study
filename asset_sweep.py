"""
How the size of the microgrid's PV fleet moves what EDL is worth to the leader.

:mod:`stochastic_commitment_analysis` compares two builds, with and without
PV. This module fills in the line between them: the genset is held at
:data:`stochastic_commitment_analysis.DG_KW` and the microgrid's own PV is swept
from 0 to :data:`PV_MAX` kWp in :data:`PV_STEP` kWp steps. At every size it
solves the two runs that module reads against each other:

    no supply    P(none) = 1: EDL is off all day, the microgrid faces the whole
                 village load
    commitment   the stochastic commitment: all six EDL outcomes at their real
                 probabilities, one price posted before the day starts

THE SPAN
    span  = commitment - no supply           [$/kWh, or $/day]
    span% = 100 * span / (no supply)         [% of the EDL-off day]

for the leader price and for the leader profit, drawn against the PV size as one
line per season. It is the same quantity, computed by the same functions, as
the fleet comparison's two rows -- the 0 and 400 kWp points here ARE those rows,
at the same menu and the same load shift.

WHAT TO LOOK FOR
PV enters on the COST side only: it is a cheaper way (c_OV["PV"] against the
diesel marginal cost) to serve kWh the microgrid would sell anyway, and only in
the hours the sun allows. The fleet comparison found it moving the profit span
far more than the price span. The sweep says whether that is a steady trend in
the PV size or a step somewhere between 0 and 400 kWp -- for instance once the
PV covers every midday hour the microgrid still serves, after which more panels
buy nothing.

READING THE PRICE LINES
The price is chosen from a discrete menu (config n_price_levels, evenly spaced
over [P_ref, Phat]), so the price span can only be a whole multiple of the menu
step. Its line is therefore a staircase, and a flat stretch means "less than
one step", not "no effect". The absolute price panel puts its y ticks on
multiples of that step so the staircase reads as one; --levels refines it, at
the cost of slower solves.

SHARED MACHINERY
Nothing below the solve is specific to PV: :func:`run_sweep` takes any list of
(swept value, build) points and a :class:`SweepAxis` describing what is swept.
:mod:`loadshift_sweep` uses it unchanged to sweep the load shift instead, so the
two sets of figures and workbooks are built the same way.

Outputs, under :data:`config.OUTPUT_DIR`:

    asset_sweep.xlsx          one row per (season, PV size) with both spans, and
                              every solved run in full -- the table view of the
                              two figures
    asset_sweep_price.png     price span against PV size: absolute | % of the
                              no-supply run
    asset_sweep_profit.png    profit span against PV size, the same two panels

Usage:
    python asset_sweep.py                          # 3 seasons x 9 sizes
    python asset_sweep.py --seasons winter
    python asset_sweep.py --pv-max 600 --pv-step 100
    python asset_sweep.py --levels 129
"""

import argparse
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")             # headless / file output
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

import config
from stochastic_analysis import FIELDS
# The builds, the solve and the span are the commitment module's own, so the
# endpoints here and its fleet comparison cannot drift apart.
from stochastic_commitment_analysis import (DG_KW, RUNS, TIME_LIMIT_S, Build,
                                            solve_build, summarize)

# The PV sizes swept [kWp].                                       [ASSUMPTION]
# 0 is the genset-only build and 400 the matched-nameplate build of the fleet
# comparison; 50 kWp steps put seven sizes between them.
PV_MAX: float = 400.0
PV_STEP: float = 50.0

# One colour AND one marker per season, keyed by season, never by plotting
# order. Okabe-Ito hues run through the palette validator: all adjacent pairs
# clear CVD dE 11 and every hue clears 3:1 against the surface. Warm for summer
# and cool for winter so the mapping is guessable; the marker carries identity
# where colour cannot (print, CVD).
SEASON_COLOR = {"fall_spring": "#009E73",   # green
                "summer":      "#D55E00",   # vermillion
                "winter":      "#0072B2"}   # blue
SEASON_MARKER = {"fall_spring": "o", "summer": "s", "winter": "D"}
SEASON_LABEL = {"fall_spring": "Fall / spring", "summer": "Summer",
                "winter": "Winter"}

# Text wears ink, never a series colour; grid and axes are recessive hairlines.
INK, MUTED, RULE, AXIS = "#1A1A1A", "#555555", "#E3E3E3", "#BBBBBB"


# --------------------------------------------------------------------------
# What a sweep varies
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SweepAxis:
    """
    The swept quantity, as the tables and figures need to name it.

    ``key`` is the field of a :func:`summarize` row that holds the swept value,
    and ``columns`` the build parameters written to the workbook, swept one
    first.
    """
    key: str
    name: str
    unit: str
    label: str
    driver: str
    columns: tuple[tuple[str, str], ...]

    @property
    def header(self) -> str:
        return f"{self.name} [{self.unit}]"

    def tick(self, v: float) -> str:
        """A value as an axis tick: 50, or 30%."""
        return f"{v:.0f}%" if self.unit == "%" else f"{v:.0f}"

    def show(self, v: float) -> str:
        """A value in running text: 50 kWp, or 30%."""
        return f"{v:.0f}%" if self.unit == "%" else f"{v:.0f} {self.unit}"


PV_AXIS = SweepAxis(
    key="pv_kwp", name="PV", unit="kWp",
    label="Microgrid PV capacity  [kWp]", driver="the PV fleet",
    columns=(("pv_kwp", "PV [kWp]"), ("dg_kw", "DG [kW]")))


# --------------------------------------------------------------------------
# Solving
# --------------------------------------------------------------------------
def pv_sizes(pv_max: float = PV_MAX, pv_step: float = PV_STEP) -> list[float]:
    """0, step, 2*step, ... up to and including ``pv_max``."""
    n = int(round(pv_max / pv_step))
    return [k * pv_step for k in range(n + 1)]


def build_at(pv_kwp: float, dg_kw: float = DG_KW) -> Build:
    """The genset plus ``pv_kwp`` of PV, at the configured load shift."""
    label = (f"{dg_kw:.0f} kW DG" if pv_kwp <= 0.0
             else f"{dg_kw:.0f} kW DG + {pv_kwp:.0f} kWp PV")
    return Build(f"pv{pv_kwp:g}", label, dg_kw, pv_kwp,
                 config.CONFIG["shift_frac"])


def sweep(season: str, points: list[tuple[float, Build]], opts: dict,
          levels: int | None) -> tuple[list[dict], dict, list[float]]:
    """
    Both runs at every point of a sweep, for one season.

    ``points`` pairs each swept value (a PV size here, a load shift in
    :mod:`loadshift_sweep`) with the build it stands for. Returns the summary
    rows, the solved runs keyed by that value, and the values that did not
    solve -- which are named, not dropped quietly, because a gap in a line
    reads as a finished result.
    """
    rows, blobs, missing = [], {}, []
    for x, build in points:
        blob = solve_build(season, build, opts, levels)
        if blob is None:
            missing.append(x)
            continue
        blobs[x] = blob
        rows.append(summarize(blob))
    return rows, blobs, missing


def menu_step(levels: int | None) -> float:
    """The price-menu step [$/kWh]: the resolution of every price span."""
    cfg = config.CONFIG
    n = levels or cfg["n_price_levels"]
    return (cfg["phat_factor"] - 1.0) * cfg["P_ref"] / (n - 1)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def _fmt(v, spec: str, width: int) -> str:
    """A number, or a dash of the same width where there is none."""
    return f"{'-':>{width}}" if v is None else f"{format(v, spec):>{width}}"


def print_summary(season: str, rows: list[dict], missing: list[float],
                  axis: SweepAxis = PV_AXIS) -> None:
    w = max(8, len(axis.header)) + 1
    print(f"\n[{season}] no supply -> stochastic commitment, by {axis.name}")
    print(f"  {axis.header:>{w}}{'p no sup':>10}{'p commit':>10}{'span':>9}"
          f"{'span %':>9}{'profit no sup':>15}{'profit commit':>15}"
          f"{'span':>11}{'span %':>9}")
    for r in rows:
        print(f"  {axis.tick(r[axis.key]):>{w}}"
              f"{_fmt(r['p_none'], '.4f', 10)}{_fmt(r['p_commit'], '.4f', 10)}"
              f"{_fmt(r['price_span'], '+.4f', 9)}"
              f"{_fmt(r['price_span_pct'], '+.1f', 8)}%"
              f"{_fmt(r['profit_none'], ',.2f', 15)}"
              f"{_fmt(r['profit_commit'], ',.2f', 15)}"
              f"{_fmt(r['profit_span'], '+,.2f', 11)}"
              f"{_fmt(r['profit_span_pct'], '+.1f', 8)}%")
    if missing:
        print(f"  !! no span at {axis.name} = "
              f"{', '.join(axis.show(x) for x in missing)} — those points are "
              f"missing from the figures and workbook")


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
        ws.column_dimensions[get_column_letter(i)].width = min(
            len(str(label)) + 3, 30)


def save_analysis(per_season: dict[str, list[dict]],
                  runs: dict[tuple[str, float], dict],
                  missing: dict[str, list[float]], step: float,
                  out_path, axis: SweepAxis = PV_AXIS) -> None:
    """Both spans at every point, and every run in full."""
    wb = Workbook()
    wb.remove(wb.active)

    ws = wb.create_sheet("summary")
    header = (["season"] + [h for _, h in axis.columns]
              + ["price, no supply [$/kWh]", "price, commitment [$/kWh]",
                 "price span [$/kWh]", "price span [% of no supply]",
                 "price span [menu steps]",
                 "profit, no supply [$/day]", "profit, commitment [$/day]",
                 "profit span [$/day]", "profit span [% of no supply]",
                 "E[qMGO], commitment [kWh]", "E[qPG], commitment [kWh]",
                 "E[qsupp], commitment [kWh]", "E[qcri], commitment [kWh]"])
    rows = []
    for season, srows in per_season.items():
        for r in srows:
            steps = (None if r["price_span"] is None
                     else round(r["price_span"] / step, 3))
            rows.append([season] + [r[k] for k, _ in axis.columns]
                        + [r["p_none"], r["p_commit"], r["price_span"],
                           r["price_span_pct"], steps,
                           r["profit_none"], r["profit_commit"],
                           r["profit_span"], r["profit_span_pct"],
                           r["qMGO_commit"], r["qPG_commit"],
                           r["qsupp_commit"], r["qcri_commit"]])
        rows.append([])
    _write_rows(ws, header, rows)
    ws.append([f"Span = stochastic commitment minus no supply; % is against "
               f"the no-supply run. Price-menu step {step:.5f} $/kWh: every "
               f"price span is a whole number of steps."])
    for season, miss in missing.items():
        if miss:
            ws.append([f"{season}: no span at {axis.name} = "
                       f"{', '.join(axis.show(x) for x in miss)} "
                       f"(a run did not solve)"])

    ws = wb.create_sheet("runs")
    detail = []
    for (season, x), blob in runs.items():
        for attr, run_label in RUNS:
            rec = blob[attr]
            detail.append([season, x, run_label]
                          + [rec[k] for k, _ in FIELDS])
    _write_rows(ws, ["season", axis.header, "run"]
                + [lbl for _, lbl in FIELDS], detail)

    wb.save(out_path)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def _style(ax) -> None:
    """Recessive chrome: solid hairline y grid, no top/right spines."""
    ax.grid(axis="y", color=RULE, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.axhline(0.0, color="#999999", linewidth=0.9, zorder=1)


def _end_labels(ax, ends: list[tuple[float, float, str]]) -> list:
    """
    Name each line at its right end, in ink, nudged apart where lines meet.

    ``ends`` is ``(x, y, text)`` at each line's last finite point. Labels closer
    than a minimum gap are spread symmetrically around their mean, so two
    seasons that land on the same value still get two readable names.
    """
    if not ends:
        return []
    y0, y1 = ax.get_ylim()
    gap = 0.075 * (y1 - y0)
    order = sorted(range(len(ends)), key=lambda i: ends[i][1])
    ys = [ends[i][1] for i in order]
    for _ in range(50):                      # relax until no pair is too close
        moved = False
        for k in range(1, len(ys)):
            if ys[k] - ys[k - 1] < gap:
                mid = 0.5 * (ys[k] + ys[k - 1])
                ys[k - 1], ys[k] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    lo, hi = min(ys + [y0]), max(ys + [y1])
    ax.set_ylim(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo))

    texts = []
    for k, i in enumerate(order):
        x, _, text = ends[i]
        texts.append(ax.annotate(text, (x, ys[k]), xytext=(9, 0),
                                 textcoords="offset points", ha="left",
                                 va="center", fontsize=9, color=INK))
    return texts


def _step_ticks(ax, step: float) -> bool:
    """
    Put the y ticks of the absolute price panel on multiples of the menu step,
    so the staircase reads as one. Skipped if that would mean a crowded axis.
    """
    y0, y1 = ax.get_ylim()
    k0, k1 = int(np.floor(y0 / step)), int(np.ceil(y1 / step))
    if k1 - k0 > 24:
        return False
    stride = max(1, int(np.ceil((k1 - k0) / 8)))
    ticks = [k * step for k in range(k0, k1 + 1) if k % stride == 0]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:+.4f}" if abs(t) > 1e-12 else "0" for t in ticks])
    ax.set_ylim(y0, y1)
    return True


def _layout_issues(fig, artists: list) -> list[str]:
    """
    The render check, done by geometry rather than by eye: any named text or
    legend box that overlaps another or runs off the figure.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [(getattr(a, "get_text", lambda: "legend")(), a.get_window_extent(r))
             for a in artists if a is not None and a.get_visible()]
    W, H = fig.bbox.width, fig.bbox.height
    out = []
    for i, (name, b) in enumerate(boxes):
        if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1:
            out.append(f"{name!r} runs off the figure")
        for other, c in boxes[i + 1:]:
            if b.overlaps(c):
                out.append(f"{name!r} overlaps {other!r}")
    return out


def plot_quantity(per_season: dict[str, list[dict]], out_path, *,
                  abs_key: str, pct_key: str, quantity: str, unit: str,
                  fmt: str, xs: list[float], axis: SweepAxis, subtitle: str,
                  step: float | None = None,
                  missing: dict[str, list[float]] | None = None) -> None:
    """
    One quantity's span against the swept value: absolute on the left, % of
    the no-supply run on the right -- two panels, never two y scales on one.
    """
    fig, (ax_a, ax_p) = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=True)
    ends_a, ends_p = [], []

    for season, rows in per_season.items():
        x = np.array([r[axis.key] for r in rows], dtype=float)
        style = dict(color=SEASON_COLOR.get(season, "#444444"), linewidth=1.8,
                     marker=SEASON_MARKER.get(season, "o"), markersize=6.5,
                     markeredgecolor="white", markeredgewidth=1.2, zorder=3,
                     label=SEASON_LABEL.get(season, season))
        for ax, key, ends in ((ax_a, abs_key, ends_a), (ax_p, pct_key, ends_p)):
            y = np.array([np.nan if r[key] is None else r[key] for r in rows],
                         dtype=float)
            ax.plot(x, y, **style)
            ok = np.isfinite(y)
            if ok.any():
                last = np.flatnonzero(ok)[-1]
                ends.append((x[last], y[last],
                             SEASON_LABEL.get(season, season)))

    # Room on the right for the end labels, measured on the swept range so a
    # sweep that does not start at zero gets the same proportions.
    span = (xs[-1] - xs[0]) or 1.0
    for ax in (ax_a, ax_p):
        _style(ax)
        ax.set_xticks(xs)
        ax.set_xticklabels([axis.tick(v) for v in xs])
        ax.set_xlim(xs[0] - 0.03 * span, xs[-1] + 0.17 * span)
        ax.set_xlabel(axis.label, color=INK, fontsize=10)
    ax_a.set_ylabel(f"{quantity} span  [{unit}]", color=INK, fontsize=10)
    ax_p.set_ylabel(f"{quantity} span  [% of the no-supply run]", color=INK,
                    fontsize=10)
    ax_a.set_title("absolute", color=INK, fontsize=10.5, loc="left")
    ax_p.set_title("relative to the EDL-off day", color=INK, fontsize=10.5,
                   loc="left")
    ax_p.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0%")
    if fmt:
        ax_a.yaxis.set_major_formatter(lambda v, _: format(v, fmt) if v else "0")

    stepped = step is not None and _step_ticks(ax_a, step)
    labels = _end_labels(ax_a, ends_a) + _end_labels(ax_p, ends_p)
    if stepped:                              # end labels may have moved ylim
        _step_ticks(ax_a, step)

    title = fig.suptitle(
        f"How {axis.driver} moves the {quantity.lower()} span — "
        f"no supply → stochastic commitment",
        x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    sub = fig.text(0.01, 0.905, subtitle, ha="left", fontsize=9.5,
                   color=MUTED)
    legend = fig.legend(*ax_a.get_legend_handles_labels(), loc="upper right",
                        bbox_to_anchor=(0.995, 0.985), ncol=3, frameon=False,
                        fontsize=9, handlelength=2.2)

    caption = ["Span = stochastic commitment minus the no-supply run "
               "(EDL off all day); the % panel divides by that run."]
    if stepped:
        caption.append(f"Left-panel ticks sit on multiples of the price-menu "
                       f"step ({step:.4f} $/kWh): the price span can take no "
                       f"other value, so flat stretches mean < 1 step.")
    gaps = [f"{s} @ {', '.join(axis.show(x) for x in m)}"
            for s, m in (missing or {}).items() if m]
    if gaps:
        caption.append("Not solved, so missing: " + "; ".join(gaps) + ".")
    cap = fig.text(0.01, 0.012, "\n".join(caption), fontsize=8, color=MUTED,
                   ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.018 + 0.028 * len(caption), 1, 0.89))
    issues = _layout_issues(
        fig, labels + [title, sub, cap, legend, ax_a.title, ax_p.title])
    for msg in issues:
        print(f"  layout: {msg}  [{out_path}]")

    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}"
          + ("" if issues else "  (layout check: no overlaps)"))


# --------------------------------------------------------------------------
# Running a sweep
# --------------------------------------------------------------------------
def run_sweep(seasons: list[str], points: list[tuple[float, Build]],
              axis: SweepAxis, subtitle: str, stem: str, levels: int | None,
              time_limit: float, fleet_note: str) -> None:
    """
    Solve every point in every season, then write ``<stem>_price.png``,
    ``<stem>_profit.png`` and ``<stem>.xlsx`` under :data:`config.OUTPUT_DIR`.
    """
    step = menu_step(levels)
    xs = [x for x, _ in points]
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = {"MIPGap": config.CONFIG["MIPGap"], "TimeLimit": time_limit}
    print(f"{len(seasons) * len(points) * len(RUNS)} solves: {len(seasons)} "
          f"season(s) x {len(points)} {axis.name} values "
          f"({axis.show(xs[0])}..{axis.show(xs[-1])}) x {len(RUNS)} runs "
          f"(no supply, stochastic commitment)")
    print(f"{fleet_note}; price-menu step {step:.4f} $/kWh; "
          f"TimeLimit {time_limit:g} s per solve")

    per_season: dict[str, list[dict]] = {}
    runs: dict[tuple[str, float], dict] = {}
    missing: dict[str, list[float]] = {}
    for season in seasons:
        rows, blobs, miss = sweep(season, points, opts, levels)
        missing[season] = miss
        runs.update({(season, x): b for x, b in blobs.items()})
        if not rows:
            print(f"[{season}] nothing solved.")
            continue
        per_season[season] = rows
        print_summary(season, rows, miss, axis)

    if not per_season:
        print("Nothing solved — no figures or workbook written.")
        return

    common = dict(xs=xs, axis=axis, subtitle=subtitle, missing=missing)
    plot_quantity(per_season, out_dir / f"{stem}_price.png",
                  abs_key="price_span", pct_key="price_span_pct",
                  quantity="Price", unit="$/kWh", fmt="+.4f", step=step,
                  **common)
    plot_quantity(per_season, out_dir / f"{stem}_profit.png",
                  abs_key="profit_span", pct_key="profit_span_pct",
                  quantity="Profit", unit="$/day", fmt="+,.0f", **common)
    save_analysis(per_season, runs, missing, step, out_dir / f"{stem}.xlsx",
                  axis)


def main(seasons: list[str] | None = None, levels: int | None = None,
         pv_max: float = PV_MAX, pv_step: float = PV_STEP,
         dg_kw: float = DG_KW, time_limit: float = TIME_LIMIT_S) -> None:
    seasons = seasons or list(config.SEASONS.keys())
    sizes = pv_sizes(pv_max, pv_step)
    shift = 100.0 * config.CONFIG["shift_frac"]
    subtitle = (f"{dg_kw:.0f} kW DG + {sizes[0]:.0f}–{sizes[-1]:.0f} kWp PV"
                + (f" in {sizes[1] - sizes[0]:.0f} kWp steps"
                   if len(sizes) > 1 else "")
                + f", {shift:.0f}% load shift; one line per season")
    run_sweep(seasons, [(pv, build_at(pv, dg_kw)) for pv in sizes], PV_AXIS,
              subtitle, "asset_sweep", levels, time_limit,
              f"fleet: {dg_kw:.0f} kW DG + swept PV, {shift:.0f}% load shift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep the microgrid's PV capacity and report the "
                    "no-supply-to-stochastic-commitment span of the leader's "
                    "price and profit, per season.")
    parser.add_argument("--seasons", nargs="+",
                        default=list(config.SEASONS.keys()),
                        choices=list(config.SEASONS.keys()),
                        help="Seasons to sweep (default: all three).")
    parser.add_argument("--pv-max", type=float, default=PV_MAX, metavar="KWP",
                        help=f"Largest PV size in kWp (default {PV_MAX:g}).")
    parser.add_argument("--pv-step", type=float, default=PV_STEP, metavar="KWP",
                        help=f"PV step in kWp (default {PV_STEP:g}).")
    parser.add_argument("--dg", type=float, default=DG_KW, metavar="KW",
                        help=f"Diesel capacity in kW (default {DG_KW:g}, the "
                             f"commitment module's genset).")
    parser.add_argument("--levels", type=int, default=None,
                        help="Re-space the price menu to this many levels; "
                             "default keeps config's. The step is the "
                             "resolution of every price span.")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_S,
                        metavar="S",
                        help=f"Solver time limit per solve in seconds "
                             f"(default {TIME_LIMIT_S:g}).")
    args = parser.parse_args()
    if args.pv_step <= 0.0 or args.pv_max < args.pv_step:
        parser.error("--pv-step must be positive and no larger than --pv-max.")
    if args.dg <= 0.0:
        parser.error("--dg must be positive: the microgrid needs a genset.")
    main(args.seasons, args.levels, args.pv_max, args.pv_step, args.dg,
         args.time_limit)
