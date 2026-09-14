"""
Slide graphics for the bilevel microgrid model: parameter tables and the
scenario weights of the stochastic commitment.

Nothing here solves the model. Everything is read from :mod:`config` and
:mod:`scenarios`, so the slides cannot drift from the instance that IS solved.
Derived quantities (the diesel marginal cost, the price cap, the implied
elasticity, the expected EDL availability) are recomputed here exactly as
:mod:`run` and :mod:`model.bilevel_mpec` build them, and are marked as derived.

Three figures are written to ``output/graphs``:

    table1_parameters.png      village, generation, costs, prices, demand response
    table2_edl_numerics.png    EDL availability scenarios, modelling and solver
    stochastic_commitment_weights.png
                               the probabilities the stochastic commitment is
                               made against, and the expected hourly EDL
                               availability they imply

Usage:
    python graphs.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import scenarios as sc

CFG = config.CONFIG

# Figures live in their own subfolder, so the slide graphics stay separate from
# the analysis output (workbooks, result plots) in output/.
GRAPH_DIR = config.OUTPUT_DIR / "graphs"

# --------------------------------------------------------------------------
# Style: light grey section bands, hairline row rules, no outer box -- reads
# cleanly when dropped on a white slide and stays legible when scaled down.
# --------------------------------------------------------------------------
BAND = "#EAEFF3"
RULE = "#C8CDD2"
HEAD = "#0072B2"          # same blue as the public-grid series in plot.py
TEXT = "#1A1A1A"
MUTED = "#555555"

FS_HEAD = 10.5            # column headers
FS_SECT = 10.5            # section labels
FS_BODY = 10.0
FS_NOTE = 8.8

# Column left edges (axes fraction) and horizontal alignment.
COLS = [
    (0.010, "left"),      # symbol
    (0.105, "left"),      # parameter
    (0.505, "right"),     # value
    (0.520, "left"),      # unit
    (0.640, "left"),      # source / note
]
HEADERS = ["Symbol", "Parameter", "Value", "Unit", "Source / remark"]

ROW_H = 0.30              # row height [in]


def _fmt(x, nd=2):
    """Trim trailing zeros so 0.10 -> 0.1 but 0.4031 keeps its digits."""
    s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return s if s else "0"


# --------------------------------------------------------------------------
# Derived quantities -- recomputed from config, exactly as run.py does.
# --------------------------------------------------------------------------
C_DIESEL = CFG["c_OV"]["DG"] + CFG["alpha_fuel"] * CFG["C_fuel_L"]
PHAT = CFG["phat_factor"] * CFG["P_ref"]
ELAST = -CFG["alpha_1"] / (1.0 - CFG["alpha_1"])
CHOKE = 1.0 / CFG["alpha_1"]
BIGM_DUAL = CFG["dual_bigM_factor"] * CFG["C_pen"]

_win = sc.WINDOW_HOURS["announced"]
_blocks = f"{_win[0]:02d}-{_win[2]+1:02d} h, {_win[3]:02d}-{_win[5]+1:02d} h"


# --------------------------------------------------------------------------
# Table 1: the physical and economic instance
# --------------------------------------------------------------------------
TABLE1 = [
    ("§", "Village and demand"),
    (r"$N^{\mathrm{nonPV}}$", "Households without rooftop PV",
     f"{CFG['n_Households_nonPV']}", "–", "Bou Gebrael et al. (2026), DKN case study"),
    (r"$N^{\mathrm{PV}}$", "Households with rooftop PV",
     f"{CFG['n_Households_PV']}", "–", "Bou Gebrael et al. (2026)"),
    (r"$\hat{c}^{\mathrm{PV}}$", "Rooftop PV capacity per PV household",
     f"{CFG['cap_PV_per_HH']}", "kWp", "Bou Gebrael et al. (2026); self-consumed only"),
    (r"$D_t$", "Baseline hourly village net demand",
     "24 h profile", "kWh/h", "Logged Aug 2024 – Apr 2025; 3 representative days"),
    (r"$\varphi_t$", "Rooftop PV capacity factor",
     "24 h profile", "kWh/kWp", "Per representative day (summer / fall-spring / winter)"),

    ("§", "Microgrid generation"),
    (r"$\bar{Q}^{\mathrm{diesel}}$", "Diesel genset capacity",
     f"{CFG['CAP']['DG']}", "kW", "Existing asset, Bou Gebrael et al. (2026)"),
    (r"$\bar{Q}^{\mathrm{PV}}$", "MGO-owned PV capacity",
     f"{CFG['CAP']['PV']}", "kW", "Base case: no utility-scale PV in the microgrid"),
    (r"$c^{\mathrm{OV}}_{\mathrm{diesel}}$", "Genset O&M cost",
     _fmt(CFG["c_OV"]["DG"]), "$/kWh", "Assumption"),
    (r"$c^{\mathrm{OV}}_{\mathrm{PV}}$", "PV O&M cost",
     _fmt(CFG["c_OV"]["PV"]), "$/kWh", "Assumption"),
    (r"$C^{\mathrm{fuel}}$", "Diesel fuel price",
     _fmt(CFG["C_fuel_L"]), "$/L", "Assumption"),
    (r"$\alpha^{\mathrm{fuel}}$", "Specific fuel consumption",
     _fmt(CFG["alpha_fuel"], 1), "L/kWh", "Assumption"),
    (r"$c^{\mathrm{diesel}}$", "Diesel marginal cost",
     _fmt(C_DIESEL), "$/kWh", r"Derived: $c^{\mathrm{OV}}+\alpha^{\mathrm{fuel}}C^{\mathrm{fuel}}$"),

    ("§", "Prices and demand response"),
    (r"$P^{\mathrm{ref}}$", "Reference tariff (elasticity anchor)",
     _fmt(CFG["P_ref"], 4), "$/kWh", "Zwickl-Bernhard et al. (2025), mean Jun 2023 – May 2024"),
    (r"$P^{\mathrm{EDL}}$", "Public-grid (EDL) tariff",
     _fmt(CFG["P_EDL"]), "$/kWh", "Regulated utility tariff"),
    (r"$\alpha_1$", r"Demand suppression share at $P^{\mathrm{ref}}$",
     _fmt(CFG["alpha_1"], 4), "–", "Zwickl-Bernhard et al. (2025), Lebanon elasticity study"),
    (r"$\varepsilon$", r"Implied own-price elasticity at $P^{\mathrm{ref}}$",
     _fmt(ELAST), "–", r"Derived: $-\alpha_1/(1-\alpha_1)$"),
    (r"$C^{\mathrm{pen}}$", "Value of lost load (critical demand)",
     _fmt(CFG["C_pen"], 1), "$/kWh", "Assumption"),
    (r"$\sigma$", "Max hourly load shift, share of $D_t$",
     _fmt(CFG["shift_frac"]), "–", r"Assumption; $\sum_t \Delta d_t = 0$"),
]


# --------------------------------------------------------------------------
# Table 2: EDL availability scenarios, modelling and solver
# --------------------------------------------------------------------------
_SC_NOTE = {
    "announced": "Supply exactly as EDL announced",
    "partial": f"Announced window at {_fmt(sc.SCENARIOS[1].avail_fraction)} × capacity",
    "delayed": "Announced window, 2 h late",
    "bonus": "Announced window, +1 h on each side",
    "full_supply": "Uninterrupted 24 h supply",
    "none": "EDL off for the whole day",
}

TABLE2 = [
    ("§", "Public-grid (EDL) availability  —  season-independent"),
    (r"$\bar{Q}^{\mathrm{PG}}$", "EDL capacity while supplying",
     _fmt(sc.BASELINE_KW, 1), "kW", "Assumption; midpoint of the 250–300 kW range"),
    ("", "Announced supply window",
     _blocks, "–", "Abi Ghanem (2018) blocks; L'Orient Today (2025) 6 h/day"),
] + [
    (r"$\pi_s$" if i == 0 else "", f"Scenario “{s.label}”",
     _fmt(s.prob), "–", _SC_NOTE[s.key])
    for i, s in enumerate(sc.SCENARIOS)
] + [
    ("", "Sum of scenario probabilities",
     _fmt(sum(s.prob for s in sc.SCENARIOS), 2), "–", "Six discrete outcomes, shared across seasons"),

    ("§", "Bilevel formulation"),
    ("", "Demand-response form",
     f"{CFG['demand_response']}", "–",
     r"$q^{\mathrm{supp}}_{t,s}=\alpha_1\,(p/P^{\mathrm{ref}})\,\bar{R}_{t,s}$"),
    (r"$\hat{P}$", "Leader price cap",
     _fmt(PHAT, 3), "$/kWh", rf"Derived: ${_fmt(CFG['phat_factor'],1)}\times P^{{\mathrm{{ref}}}}$; "
     rf"choke at {_fmt(CHOKE)}×"),
    (r"$K$", r"Discrete price levels in the menu $\Pi$",
     f"{CFG['n_price_levels']}", "–",
     rf"Even over $[P^{{\mathrm{{ref}}}},\hat{{P}}]$; step ${_fmt(1/(CFG['n_price_levels']-1),3)}\times P^{{\mathrm{{ref}}}}$"),
    ("", "Flat daily tariff",
     "yes" if CFG["flat_price"] else "no", "–", r"$p_t=p\ \forall t$ (one posted price per day)"),

    ("§", "Numerics and solver"),
    (r"$M^{\mathrm{dual}}$", "Big-M for reduced-cost complementarity",
     _fmt(BIGM_DUAL, 1), "$/kWh", rf"Derived: ${_fmt(CFG['dual_bigM_factor'],1)}\times C^{{\mathrm{{pen}}}}$"),
    (r"$M^{\mathrm{phys}}$", "Big-M for quantity complementarity",
     r"$10^{4}$", "kWh", "Above any physically feasible quantity"),
    ("", "Relative MIP gap / time limit",
     r"$10^{-4}$ / 300", "– / s", "Gurobi via Pyomo"),
]


# --------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------
def render(rows, title, subtitle, out_name, width=13.33):
    """Draw one parameter table as a standalone figure."""
    n = len(rows)
    height = 0.95 + ROW_H * (n + 1)          # + header row
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, height)
    ax.axis("off")

    y_top = height - 0.62
    fig.text(0.010, height - 0.30, title, fontsize=14.5, fontweight="bold",
             color=TEXT, va="center")
    if subtitle:
        fig.text(0.010, height - 0.54, subtitle, fontsize=9.5, color=MUTED,
                 va="center")

    # Column header row.
    y = y_top - 0.34
    for (x, ha), h in zip(COLS, HEADERS):
        if h == "Unit":                       # value/unit share one visual block
            ax.text(x + 0.006, y, h, fontsize=FS_HEAD, color=HEAD,
                    ha="left", va="center")
        else:
            ax.text(x, y, h, fontsize=FS_HEAD, color=HEAD, ha=ha, va="center")
    ax.plot([0.005, 0.995], [y - 0.14, y - 0.14], color=HEAD, lw=1.3,
            clip_on=False)

    y -= 0.16
    for sym, *rest in rows:
        if sym == "§":                        # section band
            label = rest[0]
            ax.add_patch(plt.Rectangle((0.005, y - ROW_H), 0.99, ROW_H,
                                       facecolor=BAND, edgecolor="none"))
            ax.text(0.010, y - ROW_H / 2, label, fontsize=FS_SECT,
                    fontweight="bold", color=TEXT, va="center")
            y -= ROW_H
            continue

        cy = y - ROW_H / 2
        cells = [sym, *rest]                  # symbol, name, value, unit, note
        for i, ((x, ha), txt) in enumerate(zip(COLS, cells)):
            if not txt:
                continue
            is_unit, is_note = i == 3, i == 4
            ax.text(x + 0.006 if is_unit else x, cy, txt,
                    fontsize=FS_NOTE if is_note else FS_BODY,
                    color=MUTED if (is_unit or is_note) else TEXT,
                    ha=ha, va="center")
        ax.plot([0.005, 0.995], [y - ROW_H, y - ROW_H], color=RULE, lw=0.6,
                clip_on=False)
        y -= ROW_H

    _save(fig, out_name, f"{width:.2f} x {height:.2f} in, {n} rows")


def _save(fig, out_name, note):
    out = GRAPH_DIR / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}  ({note})")


# --------------------------------------------------------------------------
# Stochastic commitment: scenario weights
#
# The stochastic commitment is the run the leader actually makes: it commits
# ONE price before the day starts, against all six EDL outcomes at their real
# probabilities. Those probabilities are the weights drawn here.
#
# The left panel is the belief itself. The right panel is what the belief means
# physically -- each hour's expected EDL supply
#
#     E[Qbar^PG_t] = sum_s pi_s * Qbar^PG_{t,s}
#
# split into the per-scenario contributions pi_s * Qbar^PG_{t,s} that make it
# up. This is the quantity the leader's residual demand Rbar = [D - Qbar]^+ is
# measured against scenario by scenario, so the stack shows WHICH outcome
# supplies which hour of the day and with what weight -- the announced window
# carries the morning and evening blocks, "delayed" fills the two hours after
# each of them, and "full supply" is the thin floor under the whole day.
# "none" contributes nothing by construction and so appears in the bar panel
# only.
#
# Colours mirror plot.py's SCENARIO_COLOR, so an outcome keeps its colour
# across every figure in the slides; "none" is absent there (it draws nothing
# in the timelines) and gets the neutral grey used for unserved load.
# --------------------------------------------------------------------------
SCENARIO_COLOR: dict[str, str] = {
    "announced":   "#0072B2",   # blue
    "partial":     "#D55E00",   # vermillion
    "delayed":     "#009E73",   # green
    "bonus":       "#CC79A7",   # magenta
    "full_supply": "#E69F00",   # orange
    "none":        "#999999",   # grey
}


def plot_commitment_weights(season: str = "winter",
                            out_name: str = "stochastic_commitment_weights.png"):
    """Draw the stochastic commitment's scenario weights and the availability
    they imply."""
    S, prob, QbarPG = sc.grid_scenarios(season)
    T = list(range(24))
    keys = [s.key for s in sc.SCENARIOS]                  # fixed, documented order
    labels = [s.label for s in sc.SCENARIOS]

    fig, (axw, axq) = plt.subplots(
        1, 2, figsize=(13.33, 5.4), gridspec_kw={"width_ratios": [1.0, 2.1]})
    fig.suptitle("Stochastic commitment: scenario weights", x=0.008, y=0.975,
                 ha="left", fontsize=14.5, fontweight="bold", color=TEXT)
    fig.text(0.008, 0.915,
             "One price committed before the day starts, against all six EDL "
             "outcomes at their probabilities — EDL assumptions are "
             "season-independent, so these weights hold for every season",
             ha="left", fontsize=9.5, color=MUTED)

    # ---- left: the weights themselves -------------------------------------
    y = np.arange(len(keys))[::-1]                        # first scenario on top
    w = [prob[k] for k in keys]
    axw.barh(y, w, height=0.62,
             color=[SCENARIO_COLOR[k] for k in keys],
             edgecolor="white", linewidth=0.6)
    for yi, wi in zip(y, w):
        axw.text(wi + 0.008, yi, f"{wi:.0%}", va="center", ha="left",
                 fontsize=10, color=TEXT)
    axw.set_yticks(y, labels, fontsize=10)
    axw.set_xlim(0, max(w) * 1.28)
    axw.set_xlabel(r"probability $\pi_s$  [–]", fontsize=10)
    axw.set_title(r"Leader's belief   ($\sum_s \pi_s = 1$)", fontsize=11,
                  color=TEXT, loc="left", pad=8)
    axw.spines[["top", "right"]].set_visible(False)
    axw.tick_params(axis="both", labelsize=10)

    # ---- right: the expected availability the weights imply ---------------
    bottom = np.zeros(len(T))
    for k, lab in zip(keys, labels):
        contrib = np.array([prob[k] * QbarPG[(t, k)] for t in T])
        if not contrib.any():                             # "none" supplies nothing
            continue
        axq.bar(T, contrib, bottom=bottom, width=0.82, label=lab,
                color=SCENARIO_COLOR[k], edgecolor="white", linewidth=0.4)
        bottom += contrib

    axq.plot(T, bottom, color=TEXT, lw=1.4, marker="o", markersize=3.5,
             label=r"$E[\bar{Q}^{\mathrm{PG}}_t]$")
    axq.axhline(sc.BASELINE_KW, color=MUTED, lw=1.0, ls=":")
    axq.text(23.4, sc.BASELINE_KW, f" capacity while supplying "
             f"({sc.BASELINE_KW:.0f} kW)", va="bottom", ha="right",
             fontsize=8.5, color=MUTED)
    axq.set_xticks(T, [f"{t:02d}" for t in T], fontsize=8.5)
    axq.set_xlim(-0.7, 23.7)
    axq.set_ylim(0, sc.BASELINE_KW * 1.12)
    axq.set_xlabel("hour of day", fontsize=10)
    axq.set_ylabel(r"expected EDL supply  [kWh/h]", fontsize=10)
    axq.set_title(r"Probability-weighted availability   "
                  r"$\pi_s \cdot \bar{Q}^{\mathrm{PG}}_{t,s}$", fontsize=11,
                  color=TEXT, loc="left", pad=8)
    axq.spines[["top", "right"]].set_visible(False)
    axq.grid(axis="y", color=RULE, lw=0.5, alpha=0.7)
    axq.set_axisbelow(True)
    axq.legend(fontsize=9, ncol=3, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    expected = bottom.sum()
    _save(fig, out_name,
          f"E[EDL] = {expected:,.1f} kWh/day, peak "
          f"{bottom.max():.1f} kW at h{int(bottom.argmax()):02d}")
    return bottom


def main():
    render(
        TABLE1,
        "Model parameters — physical and economic instance",
        "Deir Kanoun al Naher (DKN) case study; profiles per representative day "
        "(summer / fall-spring / winter)",
        "table1_parameters.png",
    )
    render(
        TABLE2,
        "Model parameters — EDL availability, formulation and solver",
        "Six discrete public-grid availability scenarios; leader price menu and "
        "complementarity constants",
        "table2_edl_numerics.png",
    )
    plot_commitment_weights()


if __name__ == "__main__":
    main()
