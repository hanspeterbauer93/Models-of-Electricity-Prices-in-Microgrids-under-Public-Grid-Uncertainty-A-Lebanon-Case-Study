"""
How the size of the microgrid's PV fleet moves what EDL's TIMING does to the leader.

The PV companion of :mod:`pattern_sweep`. That module fixes the fleet and asks
what EDL does to the leader when it takes over the highest-demand (peak) or the
lowest-demand (offpeak) hours of the day; this one keeps the question and sweeps
the microgrid's own PV instead, from 0 to :data:`PV_MAX` kWp in :data:`PV_STEP`
kWp steps. At every size, per season, it solves three deterministic runs:

    no supply   EDL off all day -- the reference
    peak        EDL covers the 8 highest-demand hours completely, nothing else
    offpeak     EDL covers the 8 lowest-demand hours completely, nothing else

Every run is certain, P = 1, so the leader knows the pattern when it posts its
price. As in :mod:`pattern_sweep`, EDL's cap in a covered hour is ``M_phys``: it
takes the hour over whatever the demand in it, including load the village
shifts in. The hours are ranked on the village's net demand D_t, which the
microgrid's own PV does not touch, so each pattern names the SAME hours at every
PV size -- only the fleet behind the microgrid's supply changes.

THE FLEET
Config's genset (``CAP["DG"]``) and load shift (``shift_frac``); only
``CAP["PV"]`` is swept, patched onto the config for each solve by
:func:`stochastic_commitment_analysis.build_settings` and restored afterwards.
With config's 400 kW and 10%, the 0 kWp point is the diesel-only fleet of
:mod:`pattern_sweep` at 10% shift and the 200 kWp point its PV fleet: the same
models, so they should reproduce those numbers within the solver gap.

THE SPAN
    span  = pattern - no supply            [$/kWh, or $/day]
    span% = 100 * span / (no supply)       [% of the EDL-off day]

for the leader price and the leader profit, against the PV size, one line per
season. The no-supply run is re-solved at every size, so a span never carries
what the PV does to the leader on its own: that moves the reference, and is in
the workbook rather than in the span.

WHAT TO LOOK FOR
PV makes the hours differ in cost, which is the channel :mod:`pattern_sweep`
calls TECHNOLOGY. With a flat price, p* ~ 0.5 * (P_ref/alpha1 + cbar), and cbar
is the marginal cost of the hours the microgrid still serves:

    peak      takes dark, diesel-served hours out of the market and leaves the
              PV niche to the microgrid -> cbar falls as the PV grows
    offpeak   takes the sunny hours, and with them the PV output -> what the
              microgrid keeps is diesel -> cbar stays high

At 0 kWp every hour costs the same and the channel is zero; whatever span is
left there comes from the load the village shifts into the covered hours. The
sweep says whether the channel grows steadily with the PV size or saturates --
once the PV covers the midday hours the microgrid still serves, more panels buy
nothing.

READING THE PRICE LINES
The price comes from a discrete menu (config n_price_levels over [P_ref, Phat]),
so a price span is a whole number of menu steps and its line is a staircase: a
flat stretch means "less than one step", not "no effect". Where the range is
short enough, the absolute price panels put their ticks on multiples of the
step; --levels refines it, at the cost of slower solves.

SHARED MACHINERY
The styling, workbook and layout helpers here are shared with
:mod:`pattern_sweep` and :mod:`hour_sweep`. The COMMITMENT sweep --
:class:`SweepAxis` and :func:`run_sweep`, which span no supply against the
stochastic commitment -- stays here for :mod:`loadshift_sweep`, which uses it
unchanged; this module's own PV sweep no longer does.

Outputs, under :data:`config.OUTPUT_DIR`:

    asset_sweep.xlsx          every run -- no supply, peak, offpeak per (season,
                              PV size) -- with its span
    asset_sweep_price.png     2 x 2: peak on top, offpeak below; the price
                              level left, its change in % of
                              the DG-only (0 kWp) run right
    asset_sweep_profit.png    the same for the profit

Usage:
    python asset_sweep.py                          # 3 seasons x 5 sizes x 3 runs
    python asset_sweep.py --seasons winter
    python asset_sweep.py --pv-max 400 --pv-step 100
    python asset_sweep.py --replot                 # redraw, no solving
"""

import argparse
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")             # headless / file output
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

import config
from data.input_data import load_data
from run import build_instance
from scenarios import uncapped_availability
from stochastic_analysis import FIELDS
# The builds and the config patching are the commitment module's own, so a
# fleet here is built exactly as it is there.
from stochastic_commitment_analysis import (RUNS, TIME_LIMIT_S, Build,
                                            build_settings, solve_build,
                                            summarize)

# The PV sizes swept [kWp].                                       [ASSUMPTION]
# 0 is the diesel-only fleet and 200 the PV fleet of pattern_sweep and
# hour_sweep, so both ends are builds those modules already solve; 50 kWp steps
# put three sizes between them.
PV_MAX: float = 200.0
PV_STEP: float = 50.0

# The rest of the fleet: the standard assets, read from config rather than
# restated, so the sweep follows the case study if it is ever changed there.
DG_STD: float = config.CONFIG["CAP"]["DG"]
SHIFT_STD: float = config.CONFIG["shift_frac"]

PATTERNS: tuple[str, ...] = ("peak", "offpeak")
NONE = "no supply"
_TOL = 1e-6

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


# ==========================================================================
# Shared helpers (also used by pattern_sweep, hour_sweep, loadshift_sweep)
# ==========================================================================
def menu_step(levels: int | None) -> float:
    """The price-menu step [$/kWh]: the resolution of every price span."""
    cfg = config.CONFIG
    n = levels or cfg["n_price_levels"]
    return (cfg["phat_factor"] - 1.0) * cfg["P_ref"] / (n - 1)


def pv_sizes(pv_max: float = PV_MAX, pv_step: float = PV_STEP) -> list[float]:
    """0, step, 2*step, ... up to and including ``pv_max``."""
    n = int(round(pv_max / pv_step))
    return [k * pv_step for k in range(n + 1)]


def _fmt(v, spec: str, width: int) -> str:
    """A number, or a dash of the same width where there is none."""
    return f"{'-':>{width}}" if v is None else f"{format(v, spec):>{width}}"


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


def _style(ax, zero: bool = True) -> None:
    """Recessive chrome: solid hairline y grid, no top/right spines, and a zero
    line unless ``zero`` is off (on a level axis it would stretch it to 0)."""
    ax.grid(axis="y", color=RULE, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    if zero:
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
    Put the y ticks of an absolute price panel on multiples of the menu step,
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


# ==========================================================================
# The PV sweep under the timing patterns
# ==========================================================================
RUN_FIELDS = [
    ("season",       "season"),
    ("fleet",        "fleet"),
    ("dg_kw",        "DG [kW]"),
    ("pv_kwp",       "PV [kWp]"),
    ("shift",        "load shift [-]"),
    ("series",       "series"),
    ("hours",        "hours EDL covers [h]"),
    ("on_hours",     "covered hours (hour-beginning)"),
    ("share",        "demand in the covered hours / daily demand [-]"),
    ("sun_share",    "% of the day's sun in those hours"),
    ("p",            "leader price [$/kWh]"),
    ("p_ref",        "price, no supply [$/kWh]"),
    ("d_p",          "price span [$/kWh]"),
    ("d_p_pct",      "price span [% of no supply]"),
    ("d_p_steps",    "price span [menu steps]"),
    ("profit",       "leader profit [$/day]"),
    ("profit_ref",   "profit, no supply [$/day]"),
    ("d_profit",     "profit span [$/day]"),
    ("d_profit_pct", "profit span [% of no supply]"),
    ("qMGO",         "microgrid sales [kWh]"),
    ("qPG",          "EDL delivered [kWh]"),
    ("qsupp",        "suppressed demand [kWh]"),
    ("qcri",         "unserved [kWh]"),
    ("termination",  "termination"),
]


def _fleet(dg_kw: float, pv_kwp: float) -> str:
    return (f"{dg_kw:.0f} kW DG" if pv_kwp <= 0.0
            else f"{dg_kw:.0f} kW DG + {pv_kwp:.0f} kWp PV")


def _pct(v: float | None, base: float | None) -> float | None:
    """``v`` as a percentage of ``base``, or ``None`` where that is no number."""
    if v is None or base is None or abs(base) <= _TOL:
        return None
    return 100.0 * v / base


def _record(rec: dict, ref: dict, step: float, **extra) -> dict:
    """A solved run with its span against the no-supply run of the same season,
    fleet and load shift."""
    out = dict(rec) | extra
    out["p_ref"], out["profit_ref"] = ref["p"], ref["profit"]
    d_p = None if out["p"] is None or ref["p"] is None else out["p"] - ref["p"]
    # Two runs on the same menu level differ by float noise only, which would
    # otherwise print as a signed "-0.0%".
    if d_p is not None and abs(d_p) <= _TOL:
        d_p = 0.0
    out["d_p"] = d_p
    out["d_p_pct"] = _pct(d_p, ref["p"])
    out["d_p_steps"] = None if d_p is None else d_p / step
    out["d_profit"] = out["profit"] - ref["profit"]
    out["d_profit_pct"] = _pct(out["d_profit"], ref["profit"])
    return out


def pv_pattern_sweep(seasons: list[str], sizes: list[float], *, dg_kw: float,
                     shift: float, opts: dict, levels: int | None,
                     step: float) -> tuple[list[dict], dict[str, list[str]]]:
    """
    No supply, peak and offpeak at every PV size, per season.

    Returns every solved run with its span, and per season the runs that did
    not solve -- named, not dropped quietly, because a gap in a line reads as a
    finished result. A size whose no-supply run fails has no spans at all.
    """
    # Imported here, not at module level: pattern_sweep imports this module's
    # helpers, so a top-level import would be circular.
    from pattern_sweep import solve_at_belief

    bundle = load_data()
    rows: list[dict] = []
    missing: dict[str, list[str]] = {}
    for season in seasons:
        D = build_instance(season)["D"]
        T = sorted(D)
        total = sum(D.values())
        phi = bundle["cap_factor"][season]
        yield_day = sum(phi.values())
        # The covered hours depend on D_t only, so they are fixed per season.
        qbar = {p: uncapped_availability(season, p, D) for p in PATTERNS}
        on = {p: [t for t in T if qbar[p][t] > 0.0] for p in PATTERNS}
        missing[season] = []
        print(f"\n[{season}] "
              + "; ".join(f"{p}: h{','.join(map(str, on[p]))} = "
                          f"{100 * sum(D[t] for t in on[p]) / total:.0f}% of "
                          f"demand" for p in PATTERNS))

        for pv in sizes:
            build = Build(f"pv{pv:g}", _fleet(dg_kw, pv), dg_kw, pv, shift)
            common = {"season": season, "fleet": build.label, "dg_kw": dg_kw,
                      "pv_kwp": pv, "shift": shift}

            with build_settings(build):
                ref = solve_at_belief(season, {t: 0.0 for t in T}, 1.0, opts,
                                      levels)
            if ref is None or ref["p"] is None:
                print(f"  {pv:5.0f} kWp  no-supply run did not solve — no "
                      f"spans at this size")
                missing[season].append(f"all @ {pv:.0f} kWp")
                continue
            runs = [_record(ref, ref, step, **common, series=NONE, hours=0,
                            on_hours="", share=0.0, sun_share=0.0)]

            for pattern in PATTERNS:
                with build_settings(build):
                    rec = solve_at_belief(season, qbar[pattern], 1.0, opts,
                                          levels)
                if rec is None:
                    print(f"  {pv:5.0f} kWp  {pattern} did not solve — "
                          f"missing from the figures")
                    missing[season].append(f"{pattern} @ {pv:.0f} kWp")
                    continue
                hs = on[pattern]
                runs.append(_record(
                    rec, ref, step, **common, series=pattern, hours=len(hs),
                    on_hours=",".join(map(str, hs)),
                    share=sum(D[t] for t in hs) / total,
                    sun_share=(100.0 * sum(phi[t] for t in hs) / yield_day
                               if yield_day else 0.0)))

            for r in runs:
                dp = ("      -      " if r["d_p"] is None
                      else f"{r['d_p']:+.4f} ({r['d_p_pct']:+5.1f}%)")
                print(f"  {pv:5.0f} kWp {r['series']:>10}  p={_fmt(r['p'], '.4f', 6)}"
                      f"  Δp={dp}  profit={r['profit']:8.2f}  "
                      f"Δprofit={r['d_profit']:+8.2f}  qMGO={r['qMGO']:7.1f}"
                      f"  qPG={r['qPG']:7.1f}")
            rows.extend(runs)
    return rows, missing


def report(rows: list[dict], seasons: list[str]) -> None:
    """Both spans of both patterns against the PV size, a table per season."""
    for season in [s for s in config.SEASONS if s in seasons]:
        sel = {(r["pv_kwp"], r["series"]): r for r in rows
               if r["season"] == season}
        sizes = sorted({pv for pv, _ in sel})
        if not sizes:
            continue
        print(f"\n[{season}] span against no supply, same fleet and shift")
        print(f"  {'PV kWp':>7}{'p none':>9}{'Δp peak':>10}{'Δp offpk':>10}"
              f"{'profit none':>13}{'Δ peak':>10}{'Δ offpk':>10}")
        for pv in sizes:
            ref = sel.get((pv, NONE), {})
            pk, off = sel.get((pv, "peak"), {}), sel.get((pv, "offpeak"), {})
            print(f"  {pv:7.0f}{_fmt(ref.get('p'), '.4f', 9)}"
                  f"{_fmt(pk.get('d_p'), '+.4f', 10)}"
                  f"{_fmt(off.get('d_p'), '+.4f', 10)}"
                  f"{_fmt(ref.get('profit'), ',.2f', 13)}"
                  f"{_fmt(pk.get('d_profit'), '+,.2f', 10)}"
                  f"{_fmt(off.get('d_profit'), '+,.2f', 10)}")


def save_runs(rows: list[dict], out_path, step: float) -> None:
    """Every run of the PV sweep, with its span, on one sheet."""
    wb = Workbook()
    ws = wb.active
    ws.title = "runs"
    _write_rows(ws, [lbl for _, lbl in RUN_FIELDS],
                ([r.get(k) for k, _ in RUN_FIELDS] for r in rows))
    ws.append([])
    ws.append([f"Deterministic runs, P = 1. EDL's cap is M_phys in the covered "
               f"hours and 0 elsewhere. Spans are against the no-supply row of "
               f"the same season, PV size and load shift. Price-menu step "
               f"{step:.5f} $/kWh."])
    wb.save(out_path)
    print(f"  wrote {out_path}")


def load_runs(path) -> list[dict]:
    """Every run of a previous PV sweep, read back from its workbook."""
    ws = load_workbook(path, data_only=True)["runs"]
    table = list(ws.iter_rows(values_only=True))
    key_of = {label: key for key, label in RUN_FIELDS}
    header = [key_of.get(cell) for cell in table[0]]
    recs = [{k: v for k, v in zip(header, row) if k is not None}
            for row in table[1:]]
    # The blank line and the caption carry no series.
    return [r for r in recs if r.get("series") is not None]


ROW_TITLE = {"peak": "EDL on the highest-demand hours (peak)",
             "offpeak": "EDL on the lowest-demand hours (offpeak)"}


def plot_pattern_span(rows: list[dict], out_path, *, key: str, quantity: str,
                      unit: str, fmt: str, sizes: list[float],
                      seasons: list[str], subtitle: str,
                      step: float | None = None,
                      missing: dict[str, list[str]] | None = None) -> None:
    """
    Four panels: peak on top, offpeak below. Left, the LEVEL of the pattern
    run; right, its change in % of the DG-only run (the smallest PV size, 0
    kWp) of the same season and pattern. Never two y scales on one panel. Each
    column shares its y axis, so the two patterns are read on one scale.
    """
    order = [s for s in config.SEASONS if s in seasons]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8), sharex=True,
                             sharey="col")

    def points(season, pattern, relative):
        pts = sorted((r["pv_kwp"], r[key]) for r in rows
                     if r["season"] == season and r["series"] == pattern
                     and r.get(key) is not None)
        if not relative or not pts:
            return pts
        # The base is the DG-only run, the first PV size; without it the
        # season has no relative line rather than one against another base.
        base_pv, base = pts[0]
        if base_pv > _TOL or not base:
            return []
        return [(x, 100.0 * (v - base) / abs(base)) for x, v in pts]

    for i, pattern in enumerate(PATTERNS):
        for j in (0, 1):
            ax = axes[i, j]
            # The level axis must not be stretched down to a zero line.
            _style(ax, zero=(j == 1))
            for season in order:
                pts = points(season, pattern, relative=(j == 1))
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=SEASON_COLOR.get(season, "#444444"),
                        linewidth=2.0, marker=SEASON_MARKER.get(season, "o"),
                        markersize=6.5, markeredgecolor="white",
                        markeredgewidth=1.2, zorder=3)
            ax.set_title(ROW_TITLE[pattern] + ("" if j == 0
                                               else "  ·  change vs DG only"),
                         loc="left", fontsize=10.5, color=INK)
            ax.set_ylabel(f"Leader {quantity.lower()}  [{unit}]" if j == 0
                          else f"{quantity} change  [% of DG only]",
                          fontsize=9.5, color=INK)

    for ax in axes[-1]:
        ax.set_xlabel("Microgrid PV capacity  [kWp]", fontsize=10, color=INK)
    # Shared x hides the top row's tick labels; each panel keeps its scale.
    for ax in axes[0]:
        ax.tick_params(axis="x", labelbottom=True)
    width = (sizes[-1] - sizes[0]) or 1.0
    axes[0, 0].set_xticks(sizes)
    axes[0, 0].set_xticklabels([f"{v:.0f}" for v in sizes])
    axes[0, 0].set_xlim(sizes[0] - 0.05 * width, sizes[-1] + 0.05 * width)
    axes[0, 1].yaxis.set_major_formatter(
        lambda v, _: f"{v:+.0f}%" if abs(v) > 1e-12 else "0%")
    axes[0, 0].yaxis.set_major_formatter(lambda v, _: format(v, fmt))

    handles = [plt.Line2D([], [], color=SEASON_COLOR.get(s, "#444444"),
                          linewidth=2.0, marker=SEASON_MARKER.get(s, "o"),
                          markersize=6.5, markeredgecolor="white",
                          label=SEASON_LABEL.get(s, s)) for s in order]
    ttl = fig.suptitle(
        f"How the PV fleet moves the leader's {quantity.lower()} under EDL's "
        f"timing patterns",
        x=0.01, y=0.985, ha="left", va="top", fontsize=13, color=INK,
        fontweight="bold")
    sub = fig.text(0.01, 0.950, subtitle, ha="left", va="top", fontsize=9.5,
                   color=MUTED)
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.925), ncol=len(handles),
                     frameon=False, fontsize=9)

    caption = ["Left: the leader's committed value in the pattern run. Right: "
               "its change against the DG-only run (0 kWp PV) of the same "
               "season and pattern, in %."]
    if step is not None:
        caption.append(f"Prices come from a menu in steps of {step:.4f} $/kWh, "
                       f"so price lines move in whole steps.")
    gaps = [f"{SEASON_LABEL.get(s, s)}: {', '.join(m)}"
            for s, m in (missing or {}).items() if m]
    if gaps:
        caption.append("Not solved, so missing: " + "; ".join(gaps) + ".")
    cap = fig.text(0.01, 0.012, "\n".join(caption), fontsize=8, color=MUTED,
                   ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.018 + 0.022 * len(caption), 1, 0.885))
    issues = _layout_issues(fig, [ttl, sub, leg, cap]
                            + [ax.title for ax in axes.flat])
    for msg in issues:
        print(f"  layout: {msg}  [{out_path}]")
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}"
          + ("" if issues else "  (layout check: no overlaps)"))


def main(seasons: list[str] | None = None, levels: int | None = None,
         pv_max: float = PV_MAX, pv_step: float = PV_STEP,
         dg_kw: float = DG_STD, shift: float = SHIFT_STD,
         time_limit: float = TIME_LIMIT_S, replot: bool = False) -> None:
    seasons = seasons or list(config.SEASONS.keys())
    sizes = pv_sizes(pv_max, pv_step)
    step = menu_step(levels)
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    book = out_dir / "asset_sweep.xlsx"
    missing: dict[str, list[str]] = {}

    # Redrawing must not need the solver: the workbook is a complete record.
    if replot:
        rows = load_runs(book)
        print(f"replot from {book}: {len(rows)} runs")
        if not rows:
            return
        sizes = sorted({r["pv_kwp"] for r in rows})
        dg_kw, shift = rows[0]["dg_kw"], rows[0]["shift"]
    else:
        opts = {"MIPGap": config.CONFIG["MIPGap"], "TimeLimit": time_limit}
        print(f"{len(seasons) * len(sizes) * (1 + len(PATTERNS))} solves: "
              f"{len(seasons)} season(s) x {len(sizes)} PV sizes "
              f"({sizes[0]:.0f}..{sizes[-1]:.0f} kWp) x no supply / "
              f"{' / '.join(PATTERNS)}; {_fleet(dg_kw, 0.0)}, "
              f"{100 * shift:.0f}% load shift; price-menu step {step:.4f} "
              f"$/kWh; TimeLimit {time_limit:g} s per solve")
        rows, missing = pv_pattern_sweep(seasons, sizes, dg_kw=dg_kw,
                                         shift=shift, opts=opts,
                                         levels=levels, step=step)
        if not rows:
            print("Nothing solved — no figures or workbook written.")
            return
        save_runs(rows, book, step)

    report(rows, seasons)
    std = ("standard assets" if (dg_kw, shift) == (DG_STD, SHIFT_STD)
           else "non-standard genset / shift")
    subtitle = (f"{dg_kw:.0f} kW DG + {sizes[0]:.0f}–{sizes[-1]:.0f} kWp PV"
                + (f" in {sizes[1] - sizes[0]:.0f} kWp steps"
                   if len(sizes) > 1 else "")
                + f", {100 * shift:.0f}% load shift ({std}); EDL covers its "
                  f"hours completely and nothing else, prob = 1; one line per "
                  f"season")
    common = dict(sizes=sizes, seasons=seasons, subtitle=subtitle,
                  missing=missing)
    plot_pattern_span(rows, out_dir / "asset_sweep_price.png", key="p",
                      quantity="Price", unit="$/kWh", fmt=".3f", step=step,
                      **common)
    plot_pattern_span(rows, out_dir / "asset_sweep_profit.png", key="profit",
                      quantity="Profit", unit="$/day", fmt=",.0f", **common)


# ==========================================================================
# The commitment sweep: no supply -> stochastic commitment (loadshift_sweep)
# ==========================================================================
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


def commitment_sweep(season: str, points: list[tuple[float, Build]],
                     opts: dict, levels: int | None
                     ) -> tuple[list[dict], dict, list[float]]:
    """
    Both runs at every point of a sweep, for one season.

    ``points`` pairs each swept value (a load shift in :mod:`loadshift_sweep`)
    with the build it stands for. Returns the summary rows, the solved runs
    keyed by that value, and the values that did not solve -- which are named,
    not dropped quietly, because a gap in a line reads as a finished result.
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


def print_summary(season: str, rows: list[dict], missing: list[float],
                  axis: SweepAxis) -> None:
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


def save_analysis(per_season: dict[str, list[dict]],
                  runs: dict[tuple[str, float], dict],
                  missing: dict[str, list[float]], step: float,
                  out_path, axis: SweepAxis) -> None:
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
        rows, blobs, miss = commitment_sweep(season, points, opts, levels)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep the microgrid's PV capacity and report the span of "
                    "the leader's price and profit when EDL covers the peak or "
                    "the off-peak hours, against no supply, per season.")
    parser.add_argument("--seasons", nargs="+",
                        default=list(config.SEASONS.keys()),
                        choices=list(config.SEASONS.keys()),
                        help="Seasons to sweep (default: all three).")
    parser.add_argument("--pv-max", type=float, default=PV_MAX, metavar="KWP",
                        help=f"Largest PV size in kWp (default {PV_MAX:g}).")
    parser.add_argument("--pv-step", type=float, default=PV_STEP, metavar="KWP",
                        help=f"PV step in kWp (default {PV_STEP:g}).")
    parser.add_argument("--dg", type=float, default=DG_STD, metavar="KW",
                        help=f"Diesel capacity in kW (default {DG_STD:g}, "
                             f"config's standard assets).")
    parser.add_argument("--shift", type=float, default=SHIFT_STD,
                        metavar="FRAC",
                        help=f"Load shift as a fraction of each hour's demand "
                             f"(default {SHIFT_STD:g}, config's).")
    parser.add_argument("--levels", type=int, default=None,
                        help="Re-space the price menu to this many levels; "
                             "default keeps config's. The step is the "
                             "resolution of every price span.")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_S,
                        metavar="S",
                        help=f"Solver time limit per solve in seconds "
                             f"(default {TIME_LIMIT_S:g}).")
    parser.add_argument("--replot", action="store_true",
                        help="Redraw the figures from the existing workbook "
                             "without solving anything.")
    args = parser.parse_args()
    if args.pv_step <= 0.0 or args.pv_max < args.pv_step:
        parser.error("--pv-step must be positive and no larger than --pv-max.")
    if args.dg <= 0.0:
        parser.error("--dg must be positive: the microgrid needs a genset.")
    if not 0.0 <= args.shift <= 1.0:
        parser.error("--shift must lie in [0, 1]: it is a fraction of D_t.")
    main(args.seasons, args.levels, args.pv_max, args.pv_step, args.dg,
         args.shift, args.time_limit, args.replot)
