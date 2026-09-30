"""
A Lebanese summer day: four EDL availability scenarios, the leader's stochastic
commitment against them, and what it would have done knowing the outcome.

A self-contained scenario set for the summer representative day, separate from
the six season-independent outcomes of :mod:`scenarios`. Each scenario is a
single contiguous supply window; inside it EDL's cap is ``M_phys``, so it covers
whatever the village draws in those hours, and outside it EDL supplies nothing:

    as announced          06-18 h    prob 0.3
    reduced               06-10 h    prob 0.4
    reduced and shifted   20-24 h    prob 0.2
    no supply             --         prob 0.1

Windows are clock intervals [start, end) on the hour-beginning index, so
"06-18" is hours 6..17, twelve hours of supply. The fleet is config's genset
(:data:`config.CONFIG` ``CAP["DG"]``) plus :data:`pattern_sweep.PV_KWP` of
microgrid PV, the standard build for this case study.

THE SOLVES
----------
    stochastic commitment   one solve over all four outcomes at their
                            probabilities: ONE price, committed before the day,
                            and a recourse dispatch per outcome
    perfect information     one deterministic solve per outcome (prob = 1): the
                            price and profit the leader would pick knowing it

The commitment's realised profit in outcome s is its own objective evaluated in
that scenario alone,

    profit_s = sum_t p_t * qMGO[t,s] - cdiesel_t * qdiesel[t,s] - cPV_t * qPV[t,s]

so its probability-weighted mean is the solve's expected profit.

VILLAGE INFORMATION
-------------------
The leader always prices against the four outcomes at their probabilities.
What changes is what the village plans its load shift against:

    stochastic      the commitment above: the shift is scheduled against the
                    probabilities, jointly with the price
    low-informed    the village takes "as announced" as certain: its shift is
                    the one it chooses in the certain "announced" run
    non-informed    the village plans for no supply, where moving load gains
                    it nothing: its shift is the certain "none" run's -- none

In the two held settings the shift is FIXED in the four-outcome model and the
village's shift-optimality conditions are switched off; its dispatch in each
outcome is still its own optimal response to that shift, and the leader
re-optimizes its one price against the true outcomes. The village's belief-run
shift does not depend on the price: every level on the menu exceeds the EDL
tariff by far more than the shift discomfort, so the gain per kWh moved into a
supplied hour is positive at any of them. Which hours give up load is a tie for
the village; the certain run's leader breaks it. Holding the stochastic run's
own shift reproduces that run, which checks the construction (printed).

The leader's EVPI in each setting holds the village's shift at that setting's
schedule and lets the leader price per outcome (``leaderInformed``):

    EVPI = E[profit | price per outcome, shift held] - E[profit | committed]

Holding the shift is what makes it the value of information to the LEADER
alone; with the shift free, the village re-plans against the tariff too and
the difference is no longer sign-definite.

The VSS holds the shift the same way:

    EV     the certain run on the average day: EDL's cap in hour t is
           pi_t * (D_t + dd_t), pi_t the probability that a window covers t
    EEV    the four outcomes with the leader held to EV's price
    VSS    = E[profit | committed] - EEV

Outputs, under :data:`config.OUTPUT_DIR`:

    lebanon_summerday_scenarios.png      the four scenarios against summer demand
    lebanon_summerday_boxes.png          committed vs certain, a price and a
                                         profit boxplot side by side
    lebanon_summerday_shift.png          the commitment's load-shift schedule
                                         against its limit, EDL windows behind
    lebanon_summerday_information.png    committed price and expected profit
                                         per village-information setting, bars
    lebanon_summerday_evpi.png           the leader's EVPI (left) and
                                         maximum regret (right) per setting,
                                         shift held
    lebanon_summerday_regret.png         table: the leader's profit per EDL
                                         window at the stochastic price with
                                         the village planning for no supply,
                                         and the difference (and %) on a
                                         perfect-information or stochastic shift
    lebanon_summerday_village.png        the village's committed shift against
                                         the outcomes, price held: hit rate
                                         and village regret (VPI)
    lebanon_summerday.xlsx               the solved numbers behind the figures

Usage:
    python lebanon_summerday.py                        # everything
    python lebanon_summerday.py --figures scenarios    # no solve
"""

import argparse
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")             # headless / file output
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from openpyxl import Workbook
from openpyxl.styles import Font
from pyomo.environ import TerminationCondition, value

import config
from model.bilevel_mpec import BilevelMPEC
from plot import COLORS, SCENARIO_COLOR, STOCHASTIC_LABEL
from plot_scenarios import FONT, _blocks, _hour_axis, _hour_ranges
from pattern_sweep import PV_KWP
from run import build_instance
from stochastic_commitment_analysis import Build, build_settings

SEASON = "summer"

_OK = (TerminationCondition.optimal, TerminationCondition.feasible)


@dataclass(frozen=True)
class SummerScenario:
    key: str
    label: str
    prob: float
    start: int | None = None          # first supplied hour; None -> no supply
    end: int | None = None            # first hour after the window

    @property
    def hours(self) -> tuple[int, ...]:
        return () if self.start is None else tuple(range(self.start, self.end))


SCENARIOS: tuple[SummerScenario, ...] = (
    SummerScenario("announced",       "As announced",        0.3, 6, 18),
    SummerScenario("reduced",         "Reduced",             0.4, 6, 10),
    SummerScenario("reduced_shifted", "Reduced and shifted", 0.2, 20, 24),
    SummerScenario("none",            "No supply",           0.1),
)

assert abs(sum(s.prob for s in SCENARIOS) - 1.0) < 1e-9

# The same four hues as the validated set before: "reduced and shifted" wears
# "delayed"'s green, the window moved; "reduced" takes "bonus"'s magenta. "none"
# needs a dot in the boxplots, and a grey one fails the validator's chroma
# floor, so it takes the unused full_supply orange. Validated: CVD deltaE 7.6
# worst pair, normal-vision floor 18.7 -- legal because every dot is also named
# in the legend.
SCENARIO_COLOR_SUMMER: dict[str, str] = {
    "announced":       SCENARIO_COLOR["announced"],
    "reduced":         SCENARIO_COLOR["bonus"],
    "reduced_shifted": SCENARIO_COLOR["delayed"],
    "none":            SCENARIO_COLOR["full_supply"],
}

PERFECT_LABEL = "Perfect information"


def grid_scenarios(T) -> tuple[list[str], dict[str, float],
                                dict[tuple[int, str], float]]:
    """``S, prob, QbarPG`` in the shape :func:`run.build_instance` carries."""
    cap = config.CONFIG["M_phys"]
    S = [s.key for s in SCENARIOS]
    prob = {s.key: s.prob for s in SCENARIOS}
    QbarPG = {(t, s.key): (cap if t in s.hours else 0.0)
              for s in SCENARIOS for t in T}
    return S, prob, QbarPG


# --------------------------------------------------------------------------
# Solve
# --------------------------------------------------------------------------
def _instance(keys: list[str] | None = None) -> dict:
    """
    The summer instance on this module's scenarios.

    ``keys`` restricts the scenario set to those outcomes, renormalised; a
    single key is that outcome with prob = 1. Outcomes left out are dropped
    rather than carried at probability zero, which would leave their recourse
    variables outside the objective.
    """
    data = dict(build_instance(SEASON))
    T = data["T"]
    S, prob, QbarPG = grid_scenarios(T)
    if keys is not None:
        total = sum(prob[s] for s in keys)
        S, prob = list(keys), {s: prob[s] / total for s in keys}
    data["S"] = S
    data["prob"] = prob
    data["QbarPG"] = {(t, s): QbarPG[(t, s)] for t in T for s in S}
    return data


def _solve(data: dict, name: str, prepare=None):
    """
    Build and solve with config's Gurobi settings; ``None`` if unsolved.

    ``prepare(m)``, if given, edits the built model before the solve.
    """
    mpec = BilevelMPEC(data)
    m = mpec.build()
    if prepare is not None:
        prepare(m)
    opts = {"MIPGap": config.CONFIG["MIPGap"],
            "TimeLimit": config.CONFIG["TimeLimit"]}
    solver_name, res = mpec.solve("gurobi", options=opts)
    if solver_name is None:
        raise RuntimeError("Gurobi not available (check the license and that "
                           "gurobipy is installed).")
    tc = res.solver.termination_condition
    if tc not in _OK:
        print(f"  {name}: not solved ({tc})")
        return None
    m.termination = str(tc)
    return m


def _per_scenario(m, data: dict) -> dict[str, dict]:
    """Price, profit and volumes of a solved model, one entry per scenario."""
    T = data["T"]
    p = [value(m.p[t]) for t in T]
    # The village's shift is decided before the outcome, so it is one schedule
    # shared by every scenario of the solve; reported as kWh moved.
    shift = sum(max(value(m.dd[t]), 0.0) for t in T)
    out = {}
    for s in data["S"]:
        revenue = sum(value(m.p[t]) * value(m.qMGO[t, s]) for t in T)
        cost = sum(data["cdiesel"][t] * value(m.qdiesel[t, s])
                   + data["cPV"][t] * value(m.qPV[t, s]) for t in T)
        out[s] = {
            "p": sum(p) / len(p),          # == p_t itself under a flat tariff
            "p_min": min(p), "p_max": max(p),
            "profit": revenue - cost, "revenue": revenue, "cost": cost,
            "qMGO": sum(value(m.qMGO[t, s]) for t in T),
            "qPG": sum(value(m.qPG[t, s]) for t in T),
            "qsupp": sum(value(m.qsupp[t, s]) for t in T),
            "qcri": sum(value(m.qcri[t, s]) for t in T),
            "shift": shift,
            "dd": [value(m.dd[t]) for t in T],
            "termination": m.termination,
        }
    return out


def solve_all() -> dict[str, dict[str, dict]]:
    """
    ``{"stochastic": {scenario: row}, "perfect": {scenario: row}}``.

    Outcomes whose solve failed are absent from their side.
    """
    results: dict[str, dict[str, dict]] = {"stochastic": {}, "perfect": {}}

    print(f"[{SEASON}] stochastic commitment over "
          f"{', '.join(s.key for s in SCENARIOS)} ...")
    data = _instance()
    m = _solve(data, "stochastic")
    if m is not None:
        results["stochastic"] = _per_scenario(m, data)
        print(f"  p = {value(m.p[data['T'][0]]):.4f}  "
              f"E[profit] = {value(m.profit):,.2f}  shift = "
              f"{next(iter(results['stochastic'].values()))['shift']:,.1f} kWh"
              f"  ({m.termination})")

    for sc in SCENARIOS:
        print(f"[{SEASON}] perfect information: {sc.key} (prob = 1) ...")
        data = _instance([sc.key])
        m = _solve(data, sc.key)
        if m is None:
            continue
        results["perfect"][sc.key] = _per_scenario(m, data)[sc.key]
        r = results["perfect"][sc.key]
        print(f"  p = {r['p']:.4f}  profit = {r['profit']:,.2f}  "
              f"shift = {r['shift']:,.1f} kWh  ({r['termination']})")
    return results


# --------------------------------------------------------------------------
# Solve: village information settings
# --------------------------------------------------------------------------
# (key, label, outcome the village plans its shift for). None: the stochastic
# commitment, where it plans against the probabilities.
INFO_SETTINGS: tuple[tuple[str, str, str | None], ...] = (
    ("stochastic", "Stochastic",   None),
    ("low",        "Low-informed", "announced"),
    ("non",        "Non-informed", "none"),
)

# Everything in the model that makes Δd the village's OPTIMAL shift -- its
# bounds, balance and split, stationarity in Δd and the complementarities on
# the shift duals and halves. With Δd fixed these are switched off; the
# per-outcome dispatch KKT stay, so the village still responds optimally to
# the held schedule in every outcome.
_SHIFT_BLOCK = ("shift_balance", "shift_split", "stat_shift", "shift_up",
                "shift_lo", "cc_gp_a", "cc_gp_b", "cc_gm_a", "cc_gm_b",
                "rc_ddp_geq", "rc_ddm_geq", "cc_ddp_a", "cc_ddp_b",
                "cc_ddm_a", "cc_ddm_b")


def _hold_shift(m, T, dd: list[float]) -> None:
    """Fix the village's shift to ``dd`` and drop its shift optimality."""
    for t, v in zip(T, dd):
        m.dd[t].fix(v)
        m.ddp[t].fix(max(v, 0.0))
        m.ddm[t].fix(max(-v, 0.0))
    for name in _SHIFT_BLOCK:
        getattr(m, name).deactivate()


def solve_held_shift(dd: list[float], name: str) -> dict | None:
    """The four-outcome commitment with the village's shift held at ``dd``."""
    data = _instance()
    m = _solve(data, name, prepare=lambda mm: _hold_shift(mm, data["T"], dd))
    if m is None:
        return None
    rows = _per_scenario(m, data)
    first = next(iter(rows.values()))
    return {"p": first["p"], "profit": value(m.profit),
            "shift": first["shift"],
            "by_scenario": {s: r["profit"] for s, r in rows.items()},
            "termination": m.termination}


def solve_information(results: dict[str, dict[str, dict]]) -> list[dict]:
    """One row per :data:`INFO_SETTINGS` entry that solved."""
    sto, per = results["stochastic"], results["perfect"]
    info = []

    if sto:
        first = next(iter(sto.values()))
        print(f"[{SEASON}] check: stochastic shift held ...")
        chk = solve_held_shift(first["dd"], "check")
        if chk is not None:
            print(f"  p = {chk['p']:.4f} (stochastic {first['p']:.4f})  "
                  f"E[profit] = {chk['profit']:,.2f} "
                  f"(stochastic {_expected(sto, 'profit'):,.2f})")

    for key, label, plans_for in INFO_SETTINGS:
        if plans_for is None:
            if len(sto) != len(SCENARIOS):
                continue
            first = next(iter(sto.values()))
            rec = {"p": first["p"], "profit": _expected(sto, "profit"),
                   "shift": first["shift"],
                   "by_scenario": {s: r["profit"] for s, r in sto.items()},
                   "termination": first["termination"]}
        else:
            belief = per.get(plans_for)
            if belief is None:
                print(f"  {label}: no certain '{plans_for}' run to take the "
                      f"shift from — skipped")
                continue
            print(f"[{SEASON}] {label}: village plans for '{plans_for}', "
                  f"leader prices against all outcomes ...")
            rec = solve_held_shift(belief["dd"], key)
            if rec is None:
                continue
        rec |= {"key": key, "label": label, "plans_for": plans_for,
                "dd": first["dd"] if plans_for is None else belief["dd"]}
        # Risk: the lowest profit any outcome leaves at the committed price.
        worst = min(rec["by_scenario"], key=rec["by_scenario"].get)
        rec |= {"worst": rec["by_scenario"][worst], "worst_outcome": worst}
        info.append(rec)
        print(f"  {label:>12}: shift {rec['shift']:6.1f} kWh  "
              f"p = {rec['p']:.4f}  E[profit] = {rec['profit']:,.2f}  ("
              + "  ".join(f"{s} {v:,.1f}" for s, v in rec["by_scenario"].items())
              + ")")
    return info


def solve_leader_informed(dd: list[float], name: str) -> dict | None:
    """
    The four outcomes with the village's shift held at ``dd`` and the leader
    told the outcome before it prices: one price per outcome
    (``leaderInformed``). With the shift fixed the outcomes decouple, so this
    is each outcome's best price against that schedule.
    """
    data = _instance()
    data["leaderInformed"] = True
    m = _solve(data, name, prepare=lambda mm: _hold_shift(mm, data["T"], dd))
    if m is None:
        return None
    T, t0 = data["T"], data["T"][0]
    by_scenario = {
        s: sum(value(m.p[t, s]) * value(m.qMGO[t, s])
               - data["cdiesel"][t] * value(m.qdiesel[t, s])
               - data["cPV"][t] * value(m.qPV[t, s]) for t in T)
        for s in data["S"]}
    return {"profit": value(m.profit),
            "p_by_scenario": {s: value(m.p[t0, s]) for s in data["S"]},
            "by_scenario": by_scenario,
            "termination": m.termination}


def solve_evpi(info: list[dict]) -> None:
    """
    Add the leader's EVPI to every information row, in place: what learning the
    outcome before pricing is worth to it, the village's shift held as in that
    setting. The committed price is available in every outcome of the informed
    run, so EVPI >= 0 up to the solver gap; anything below is printed.
    """
    for rec in info:
        print(f"[{SEASON}] {rec['label']}: leader informed, shift held ...")
        inf = solve_leader_informed(rec["dd"], f"{rec['key']}_informed")
        if inf is None:
            continue
        rec["profit_informed"] = inf["profit"]
        rec["p_informed"] = inf["p_by_scenario"]
        rec["evpi"] = inf["profit"] - rec["profit"]
        # Regret per outcome: that outcome's best price against the committed
        # one, shift held. Its probability-weighted mean is the EVPI above.
        rec["regret"] = {s: inf["by_scenario"][s] - rec["by_scenario"][s]
                         for s in inf["by_scenario"]}
        top = max(rec["regret"], key=rec["regret"].get)
        rec["max_regret"], rec["max_regret_outcome"] = rec["regret"][top], top
        print("  regret  " + "  ".join(f"{s} {v:+.2f}"
                                       for s, v in rec["regret"].items())
              + f"   max {rec['max_regret']:.2f} ({top})")
        flag = ("   <- negative beyond the gap" if rec["evpi"] <
                -config.CONFIG["MIPGap"] * abs(inf["profit"]) else "")
        print(f"  E[profit] informed = {inf['profit']:,.2f}  committed = "
              f"{rec['profit']:,.2f}  EVPI = {rec['evpi']:+,.2f}  prices "
              + "  ".join(f"{s} {p:.4f}" for s, p in inf["p_by_scenario"].items())
              + flag)


MEAN = "mean"                 # the scenario key of the average day


def solve_ev(dd: list[float], name: str) -> dict | None:
    """
    The certain run on the average day, the village's shift held at ``dd``.

    EDL's cap in hour t is the probability that some window covers t times
    the load there, pi_t * (D_t + dd_t): a covered hour is covered completely,
    so this is EDL's expected delivery in that hour. The literal mean cap,
    pi_t * M_phys, would still be unlimited wherever pi_t > 0. With the shift
    held the hour's load is known, so (1 + shift_frac) * D_t -- the bound
    :mod:`stochastic_sweep` uses for a free shift -- is not needed.
    """
    data = dict(build_instance(SEASON))
    T, D = data["T"], data["D"]
    pi = {t: sum(sc.prob for sc in SCENARIOS if t in sc.hours) for t in T}
    data["S"], data["prob"] = [MEAN], {MEAN: 1.0}
    data["QbarPG"] = {(t, MEAN): pi[t] * (D[t] + d) for t, d in zip(T, dd)}
    m = _solve(data, name, prepare=lambda mm: _hold_shift(mm, T, dd))
    if m is None:
        return None
    return {"p": value(m.p[T[0]]), "profit": value(m.profit),
            "termination": m.termination}


def solve_eev(dd: list[float], price: float, name: str) -> dict | None:
    """The four outcomes, shift held at ``dd``, the leader held to ``price``
    (the menu is that one level)."""
    data = _instance()
    data["Pi"] = [price]
    m = _solve(data, name, prepare=lambda mm: _hold_shift(mm, data["T"], dd))
    if m is None:
        return None
    return {"profit": value(m.profit), "termination": m.termination}


def solve_vss(info: list[dict]) -> None:
    """
    Add the VSS to every information row, in place: what committing against
    the four outcomes is worth over committing against their average day, the
    village's shift held as in that setting. The EV price is on the committed
    run's menu, so VSS >= 0 up to the solver gap; anything below is printed.
    Where the two prices coincide no EEV solve is needed.
    """
    for rec in info:
        print(f"[{SEASON}] {rec['label']}: average day, shift held ...")
        ev = solve_ev(rec["dd"], f"{rec['key']}_ev")
        if ev is None:
            continue
        rec["p_ev"] = ev["p"]
        if abs(ev["p"] - rec["p"]) < 1e-9:
            rec["eev"] = rec["profit"]
        else:
            eev = solve_eev(rec["dd"], ev["p"], f"{rec['key']}_eev")
            if eev is None:
                continue
            rec["eev"] = eev["profit"]
        rec["vss"] = rec["profit"] - rec["eev"]
        flag = ("   <- negative beyond the gap" if rec["vss"] <
                -config.CONFIG["MIPGap"] * abs(rec["profit"]) else "")
        print(f"  p_EV = {ev['p']:.4f} (committed {rec['p']:.4f})  "
              f"EEV = {rec['eev']:,.2f}  RP = {rec['profit']:,.2f}  "
              f"VSS = {rec['vss']:+,.2f}{flag}")


# --------------------------------------------------------------------------
# Solve: leader regret from the village's shift, stochastic price held
# --------------------------------------------------------------------------
# (key, label) of the village's shift, most informed first. "perfect": it
# knows the outcome and shifts as in that outcome's certain run. "non" is the
# reference the others are measured against.
REGRET_SETTINGS: tuple[tuple[str, str], ...] = (
    ("perfect",    "Perfect information"),
    ("stochastic", "Stochastic shift commitment"),
    ("non",        "Plans for no supply"),
)
REGRET_BASE = "non"


def _held_profits(dd: list[float], price: float, name: str,
                  keys: list[str] | None = None) -> dict[str, float] | None:
    """Leader profit per outcome at ``price`` (the menu is that one level),
    the village's shift held at ``dd``."""
    data = _instance(keys)
    data["Pi"] = [price]
    m = _solve(data, name, prepare=lambda mm: _hold_shift(mm, data["T"], dd))
    if m is None:
        return None
    return {s: r["profit"] for s, r in _per_scenario(m, data).items()}


def solve_regret(results: dict[str, dict[str, dict]]) -> dict:
    """
    The leader's profit per outcome at the stochastic commitment's price, for
    each :data:`REGRET_SETTINGS` village shift, and its difference to the
    village planning for no supply:

        diff_s = profit_s(p_commit, shift) - profit_s(p_commit, no-supply shift)

    Only the village's shift changes; its dispatch in each outcome is still
    its optimal response to that shift and the price. ``{"p": price,
    setting: {outcome: row}}``.
    """
    sto, per = results["stochastic"], results["perfect"]
    if len(sto) != len(SCENARIOS) or "none" not in per:
        return {}
    first = next(iter(sto.values()))
    price = first["p"]
    out: dict = {"p": price}

    for key, label in REGRET_SETTINGS:
        print(f"[{SEASON}] price held at {price:.4f}: village shift "
              f"'{label.lower()}' ...")
        rows = {}
        if key == "stochastic":
            rows = {s: {"shift": r["shift"], "profit": r["profit"]}
                    for s, r in sto.items()}
        elif key == "perfect":
            # One village schedule per outcome, so one solve per outcome.
            for sc in SCENARIOS:
                own = per.get(sc.key)
                held = (_held_profits(own["dd"], price, f"perfect_{sc.key}",
                                      [sc.key]) if own else None)
                if held is not None:
                    rows[sc.key] = {"shift": own["shift"],
                                    "profit": held[sc.key]}
        else:
            none = per["none"]
            held = _held_profits(none["dd"], price, key)
            if held is not None:
                rows = {s: {"shift": none["shift"], "profit": v}
                        for s, v in held.items()}
        out[key] = rows

    base = out.get(REGRET_BASE)
    if not base:
        return {}
    for key, _ in REGRET_SETTINGS:
        for s, r in out[key].items():
            r["diff"] = r["profit"] - base[s]["profit"]
        print(f"  {key:>10}: " + "  ".join(
            f"{s} {r['profit']:,.2f} ({r['diff']:+.2f})"
            for s, r in out[key].items()))
    return out


# --------------------------------------------------------------------------
# Solve: the village's committed shift against the outcomes, price held
# --------------------------------------------------------------------------
def _bill(row: dict, price: float) -> float:
    """The village's cost in one outcome: microgrid purchases at ``price``,
    EDL at its tariff, unserved critical load at VOLL, and the shift
    discomfort on sum_t |dd_t| = 2 * (kWh moved)."""
    cfg = config.CONFIG
    return (price * row["qMGO"] + cfg["P_EDL"] * row["qPG"]
            + cfg["C_pen"] * row["qcri"] + cfg["eps_shift"] * 2.0 * row["shift"])


def solve_village(results: dict[str, dict[str, dict]]) -> dict:
    """
    Two KPIs of the village's committed load shift, per outcome, the leader's
    price held at the stochastic commitment so that only the shift differs:

        hit rate   eta_s = sum_{t in W_s} dd_t / sum_t dd_t^+
                   the net share of the planned shift that lands in hours EDL
                   supplies (W_s); negative if the schedule moves load OUT of
                   them
        regret     bill_s(committed shift) - bill_s(best shift for s)  [$/day]
                   its probability-weighted mean is the village's value of
                   perfect information (VPI)

    The best shift for s is the village's own choice in the certain outcome
    with the menu reduced to the committed price. ``{"p": price, "kwh": planned
    kWh, "rows": {outcome: row}, "vpi": ..., "eta": ...}``.
    """
    sto = results["stochastic"]
    if len(sto) != len(SCENARIOS):
        return {}
    first = next(iter(sto.values()))
    price, dd = first["p"], first["dd"]
    planned = sum(max(v, 0.0) for v in dd)

    rows = {}
    for sc in SCENARIOS:
        print(f"[{SEASON}] village informed of '{sc.key}', price held at "
              f"{price:.4f} ...")
        data = _instance([sc.key])
        data["Pi"] = [price]
        m = _solve(data, f"village_{sc.key}")
        if m is None:
            continue
        best = _per_scenario(m, data)[sc.key]
        eta = (sum(dd[t] for t in sc.hours) / planned) if planned > 1e-9 else None
        best_planned = sum(max(v, 0.0) for v in best["dd"])
        rows[sc.key] = {
            "eta": eta,
            "eta_best": (sum(best["dd"][t] for t in sc.hours) / best_planned
                         if best_planned > 1e-9 else None),
            "bill": _bill(sto[sc.key], price),
            "bill_best": _bill(best, price),
            "shift_best": best["shift"],
        }
        r = rows[sc.key]
        r["regret"] = r["bill"] - r["bill_best"]
        print(f"  hit rate {100 * (eta or 0):5.1f}%  regret "
              f"{r['regret']:+6.2f} $/day  (best shift {best['shift']:.1f} kWh)")

    out = {"p": price, "kwh": planned, "rows": rows}
    if len(rows) == len(SCENARIOS):
        prob = {sc.key: sc.prob for sc in SCENARIOS}
        out["eta"] = sum(prob[s] * (r["eta"] or 0.0) for s, r in rows.items())
        out["vpi"] = sum(prob[s] * r["regret"] for s, r in rows.items())
        print(f"  expected: hit rate {100 * out['eta']:.1f}%  village VPI "
              f"{out['vpi']:.2f} $/day")
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
SIDES = (("stochastic", STOCHASTIC_LABEL), ("perfect", PERFECT_LABEL))
FIELDS = (("p", "price [$/kWh]"), ("profit", "profit [$/day]"),
          ("revenue", "revenue [$/day]"), ("cost", "generation cost [$/day]"),
          ("qMGO", "microgrid sales [kWh]"), ("qPG", "EDL delivered [kWh]"),
          ("qsupp", "suppressed demand [kWh]"), ("qcri", "unserved [kWh]"),
          ("shift", "load shifted [kWh]"), ("termination", "termination"))


def _expected(rows: dict[str, dict], field: str) -> float | None:
    """Probability-weighted mean over the outcomes, if all four solved."""
    if len(rows) != len(SCENARIOS) or field == "termination":
        return None
    return sum(sc.prob * rows[sc.key][field] for sc in SCENARIOS)


def print_summary(results: dict[str, dict[str, dict]]) -> None:
    sto, per = results["stochastic"], results["perfect"]
    print(f"\n{'outcome':>15} {'prob':>5} | {'p commit':>9} {'p perfect':>9} | "
          f"{'profit commit':>13} {'profit perfect':>14} {'regret':>9}")
    for sc in SCENARIOS:
        a, b = sto.get(sc.key), per.get(sc.key)
        f = lambda r, k, spec: format(r[k], spec) if r else "-"
        regret = (f"{b['profit'] - a['profit']:9,.2f}" if a and b else "-")
        print(f"{sc.key:>15} {sc.prob:5.2f} | {f(a, 'p', '9.4f'):>9} "
              f"{f(b, 'p', '9.4f'):>9} | {f(a, 'profit', '13,.2f'):>13} "
              f"{f(b, 'profit', '14,.2f'):>14} {regret:>9}")
    ea, eb = _expected(sto, "profit"), _expected(per, "profit")
    if ea is not None and eb is not None:
        print(f"{'expected':>15} {'':>5} | {'':>9} "
              f"{_expected(per, 'p'):9.4f} | {ea:13,.2f} {eb:14,.2f} "
              f"{eb - ea:9,.2f}   (EVPI)")


def save_workbook(results: dict[str, dict[str, dict]], out_path,
                  info: list[dict] | None = None,
                  regret: dict | None = None,
                  village: dict | None = None) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "results"
    ws.append(["side", "scenario", "outcome", "prob"]
              + [h for _, h in FIELDS])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for side, side_label in SIDES:
        for sc in SCENARIOS:
            r = results[side].get(sc.key)
            if r is None:
                continue
            ws.append([side_label, sc.key, sc.label, sc.prob]
                      + [r[k] for k, _ in FIELDS])
        ws.append([side_label, "expected", "probability-weighted", 1.0]
                  + [_expected(results[side], k) for k, _ in FIELDS])
    ws.append([])
    ws.append([f"Summer day, {config.CONFIG['CAP']['DG']:g} kW DG, "
               f"{config.CONFIG['CAP']['PV']:g} kWp PV, shift "
               f"{config.CONFIG['shift_frac']:.0%}. EDL cap M_phys inside each "
               f"window, 0 outside. Commitment profit per outcome is the "
               f"objective evaluated in that scenario alone."])
    if info:
        wi = wb.create_sheet("information")
        wi.append(["setting", "village plans for", "load shifted [kWh]",
                   "committed price [$/kWh]", "expected profit [$/day]"]
                  + [f"profit | {sc.key} [$/day]" for sc in SCENARIOS]
                  + ["E[profit], leader informed [$/day]",
                     "EVPI, shift held [$/day]",
                     "EV price, average day [$/kWh]", "EEV [$/day]",
                     "VSS, shift held [$/day]",
                     "worst-outcome profit, committed price [$/day]",
                     "worst outcome", "max regret [$/day]",
                     "max-regret outcome"]
                  + [f"regret | {sc.key} [$/day]" for sc in SCENARIOS]
                  + [f"informed price | {sc.key} [$/kWh]" for sc in SCENARIOS]
                  + ["termination"])
        for cell in wi[1]:
            cell.font = Font(bold=True)
        for r in info:
            wi.append([r["label"], r["plans_for"] or "probabilities",
                       r["shift"], r["p"], r["profit"]]
                      + [r["by_scenario"].get(sc.key) for sc in SCENARIOS]
                      + [r.get("profit_informed"), r.get("evpi"),
                         r.get("p_ev"), r.get("eev"), r.get("vss"),
                         r.get("worst"), r.get("worst_outcome"),
                         r.get("max_regret"), r.get("max_regret_outcome")]
                      + [r.get("regret", {}).get(sc.key) for sc in SCENARIOS]
                      + [r.get("p_informed", {}).get(sc.key)
                         for sc in SCENARIOS]
                      + [r["termination"]])
    if regret:
        wr = wb.create_sheet("regret")
        wr.append(["village shift", "outcome", "load shifted [kWh]",
                   "committed price [$/kWh]", "leader profit [$/day]",
                   "difference to 'plans for no supply' [$/day]"])
        for cell in wr[1]:
            cell.font = Font(bold=True)
        for key, label in REGRET_SETTINGS:
            for sc in SCENARIOS:
                r = regret.get(key, {}).get(sc.key)
                if r is not None:
                    wr.append([label, sc.label, r["shift"], regret["p"],
                               r["profit"], r["diff"]])
    if village and village.get("rows"):
        wv = wb.create_sheet("village")
        wv.append(["outcome", "prob", "committed price [$/kWh]",
                   "planned shift [kWh]", "hit rate [-]",
                   "hit rate, best shift [-]", "village bill, committed [$/day]",
                   "village bill, best shift [$/day]", "best shift [kWh]",
                   "village regret [$/day]"])
        for cell in wv[1]:
            cell.font = Font(bold=True)
        for sc in SCENARIOS:
            r = village["rows"].get(sc.key)
            if r is not None:
                wv.append([sc.label, sc.prob, village["p"], village["kwh"],
                           r["eta"], r["eta_best"], r["bill"], r["bill_best"],
                           r["shift_best"], r["regret"]])
        if "vpi" in village:
            wv.append(["expected", 1.0, village["p"], village["kwh"],
                       village["eta"], None, None, None, None, village["vpi"]])
        wv.append([])
        wv.append(["Price held at the stochastic commitment. Hit rate = net "
                   "shift into EDL hours / planned shift. Regret = bill with "
                   "the committed shift - bill with the best shift for that "
                   "outcome; its expectation is the village's VPI."])
    wb.save(out_path)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def plot_scenarios(out_path) -> dict:
    """The four scenarios against the summer demand, a 2 x 2 grid."""
    data = build_instance(SEASON)
    T, D = data["T"], data["D"]
    D_total = sum(D.values())
    _, _, QbarPG = grid_scenarios(T)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True,
                             sharey=True)
    y_max = max(D.values()) * 1.3

    summary = {}
    for ax, sc in zip(axes.flat, SCENARIOS):
        on = [t for t in T if QbarPG[(t, sc.key)] > 0.0]
        covered = sum(D[t] for t in on)
        share = 100.0 * covered / D_total if D_total else 0.0
        summary[sc.key] = {"hours": len(on), "covered": covered,
                           "share": share, "window": _hour_ranges(on)}

        xd, yd = _blocks(D, T)
        ax.step(xd, yd, where="post", color=COLORS["demand"], linewidth=1.4,
                linestyle="--", alpha=0.75, label="Village demand $D_t$")

        color = SCENARIO_COLOR_SUMMER[sc.key]
        if on:
            for t in on:
                ax.axvspan(t, t + 1, color=color, alpha=0.10, linewidth=0,
                           zorder=0.5)
            xq, yq = _blocks({t: (D[t] if t in on else 0.0) for t in T}, T)
            ax.fill_between(xq, yq, step="post", color=color, alpha=0.55,
                            linewidth=0,
                            label="Demand EDL takes over  "
                                  "($\\bar Q^{PG}_t = M_{phys}$)")
            ax.step(xq, yq, where="post", color=color, linewidth=2.0)
            note = (f"EDL on {sc.start:02d}-{sc.end:02d} h  ({len(on)} h)\n"
                    f"{covered:,.0f} kWh/day = {share:.0f}% of demand")
        else:
            ax.annotate("EDL off all day", (0.5, 0.5),
                        xycoords="axes fraction", ha="center", va="center",
                        fontsize=FONT["panel_title"], color="#888888",
                        style="italic")
            note = "0 h of supply\nthe microgrid serves the whole day"

        ax.annotate(note, (0.5, 0.975), xycoords="axes fraction", ha="center",
                    va="top", fontsize=FONT["note"], color="#333333",
                    bbox=dict(boxstyle="round,pad=0.32", fc="white",
                              ec="#cccccc", alpha=0.9))
        ax.set_title(f"{sc.label}   (prob = {sc.prob:.1f})",
                     fontsize=FONT["panel_title"], pad=7)

        _hour_axis(ax)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", linestyle=":", alpha=0.45)

    for ax in axes[:, 0]:
        ax.set_ylabel("Energy per hour  [kWh]", fontsize=FONT["axis"])
    for ax in axes[-1]:
        ax.set_xlabel("Hour of day", fontsize=FONT["axis"])

    fig.suptitle(f"EDL availability scenarios — Lebanese summer day  "
                 f"(daily demand {D_total:,.0f} kWh)",
                 fontsize=FONT["suptitle"], y=0.985)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=FONT["legend"], frameon=False,
               bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(out_path, dpi=200)
    print(f"  wrote {out_path}")
    plt.close(fig)
    return summary


# Horizontal offset of each outcome's dot inside the box, so the four never
# sit on top of each other when two outcomes land on the same value.
_DOT_OFFSET = {sc.key: -0.15 + 0.1 * i for i, sc in enumerate(SCENARIOS)}

COMMIT_INK = "#222222"
BOX_X, BOX_W = 1.0, 0.5


def _box_panel(ax, results: dict[str, dict[str, dict]], field: str, *,
               ylabel: str, title: str, fmt: str) -> None:
    """
    One box over the four CERTAIN runs, with the commitment drawn across it.

    The commitment is a single number -- one price, one expected profit (its
    solve's objective) -- so it is a line, not a box. The box is the spread of
    what the leader would pick, or earn, knowing the outcome; each outcome is
    also a dot in its scenario colour, since four values make a thin box.

    The certain runs' probability-weighted mean is deliberately NOT drawn: in
    those runs the village re-plans its shift too, so the mean is not the
    wait-and-see profit and its gap to the committed line is not the EVPI. The
    EVPI with the shift held is in :func:`plot_evpi`.
    """
    certain = results["perfect"]
    committed = (next(iter(results["stochastic"].values()))["p"]
                 if field == "p" else _expected(results["stochastic"], field))

    if certain:
        ax.boxplot([[r[field] for r in certain.values()]],
                   positions=[BOX_X], widths=BOX_W,
                   tick_labels=[""],
                   patch_artist=True, showfliers=False, whis=(0, 100),
                   boxprops=dict(facecolor="#EEEEEE", edgecolor="#555555",
                                 linewidth=1.2),
                   medianprops=dict(color="#555555", linewidth=1.4),
                   whiskerprops=dict(color="#555555", linewidth=1.2),
                   capprops=dict(color="#555555", linewidth=1.2), zorder=1)

    for sc in SCENARIOS:
        r = certain.get(sc.key)
        if r is None:
            continue
        ax.plot(BOX_X + _DOT_OFFSET[sc.key], r[field], marker="o",
                markersize=9, color=SCENARIO_COLOR_SUMMER[sc.key],
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)

    if committed is not None:
        ax.hlines(committed, BOX_X - 0.42, BOX_X + 0.42, color=COMMIT_INK,
                  linewidth=2.4, linestyle="--", zorder=4)
        name = "committed" if field == "p" else "committed, E"
        ax.annotate(f"{name} = {committed:{fmt}}", (BOX_X - 0.42, committed),
                    textcoords="offset points", xytext=(-6, 0), ha="right",
                    va="center", fontsize=FONT["note"], color=COMMIT_INK,
                    fontweight="bold")

    if field == "p":
        P_ref = config.CONFIG["P_ref"]
        ax.axhline(P_ref, color="#666666", linewidth=1.0, linestyle=":",
                   zorder=0)
        ax.annotate(f"$P_{{ref}}$ = {P_ref:.3f}", (1.0, P_ref),
                    xycoords=("axes fraction", "data"), xytext=(-4, 4),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=FONT["note"], color="#666666")

    ax.set_xlim(BOX_X - 0.95, BOX_X + 0.85)
    ax.set_xticks([])
    ax.margins(y=0.12)
    ax.set_ylabel(ylabel, fontsize=FONT["axis"])
    ax.tick_params(labelsize=FONT["tick"])
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title(title, fontsize=FONT["panel_title"] + 0.5, pad=10)


def plot_boxes(results: dict[str, dict[str, dict]], out_dir) -> None:
    """Price and profit side by side in one figure, one shared legend."""
    fleet = (f"{config.CONFIG['CAP']['DG']:g} kW DG"
             + (f" + {config.CONFIG['CAP']['PV']:g} kWp PV"
                if config.CONFIG["CAP"]["PV"] else ""))
    fig, (ax_p, ax_pi) = plt.subplots(1, 2, figsize=(12.0, 6.8))
    _box_panel(ax_p, results, "p", ylabel="Leader price  [\\$/kWh]",
               fmt=".3f", title="Price")
    _box_panel(ax_pi, results, "profit", ylabel="Leader profit  [\\$/day]",
               fmt=",.0f", title="Profit")

    handles = [Line2D([], [], linestyle="none", marker="o", markersize=9,
                      color=SCENARIO_COLOR_SUMMER[sc.key],
                      markeredgecolor="white",
                      label=f"{sc.label}  (prob {sc.prob:.1f})")
               for sc in SCENARIOS]
    handles.append(Line2D([], [], color=COMMIT_INK, linewidth=2.4,
                          linestyle="--",
                          label="Stochastic commitment (outcome unknown)"))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=FONT["legend"] - 0.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Price and profit over the EDL outcomes — Lebanese summer "
                 f"day, {fleet}",
                 fontsize=FONT["suptitle"], y=0.985)

    out_path = out_dir / "lebanon_summerday_boxes.png"
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    print(f"  wrote {out_path}")
    plt.close(fig)


def plot_shift(results: dict[str, dict[str, dict]], out_dir) -> None:
    """
    The village's load-shift schedule under the stochastic commitment, against
    its limit and the EDL windows.

    The shift is decided before the outcome, so the commitment has ONE schedule
    for all four scenarios. Dark bars are Δd_t: up is load moved into hour t,
    down is load moved out of it. Behind each is a light bar for the hour's
    limit, shift_frac * D_t, on the same side and the same kWh axis -- so the
    share of the light bar a dark one fills IS its share of the limit, and the
    % of D_t reads off the figure without a second scale. Behind both, each
    scenario's window is a coloured lane of its own, labelled in the right
    margin: the windows overlap (06-18, 06-10), and translucent spans
    stacked on each other would mix into colours no scenario has.
    """
    sto = results["stochastic"]
    if not sto:
        return
    dd = next(iter(sto.values()))["dd"]
    T = list(range(len(dd)))
    fleet = (f"{config.CONFIG['CAP']['DG']:g} kW DG"
             + (f" + {config.CONFIG['CAP']['PV']:g} kWp PV"
                if config.CONFIG["CAP"]["PV"] else ""))
    windowed = [sc for sc in SCENARIOS if sc.start is not None]
    n = len(windowed)

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    for i, sc in enumerate(windowed):
        lo, hi = 1.0 - (i + 1) / n, 1.0 - i / n
        ax.axvspan(sc.start, sc.end, ymin=lo, ymax=hi, linewidth=0,
                   color=SCENARIO_COLOR_SUMMER[sc.key], alpha=0.25, zorder=0)
        ax.text(1.01, 0.5 * (lo + hi),
                f"{sc.label}\n{sc.start:02d}-{sc.end:02d} h  "
                f"(prob {sc.prob:.1f})",
                transform=ax.transAxes, ha="left", va="center",
                fontsize=FONT["note"], color="#333333")

    D = build_instance(SEASON)["D"]
    frac = config.CONFIG["shift_frac"]
    limit = [frac * D[t] for t in T]
    x = [t + 0.5 for t in T]
    # The limit on the side the hour shifts to; an unshifted hour shows it
    # upward, as room left unused.
    ax.bar(x, [c if v >= 0.0 else -c for c, v in zip(limit, dd)], width=0.8,
           color="#CFCFCF", zorder=1.5)
    ax.bar(x, dd, width=0.5, color="#333333", zorder=2)
    ax.axhline(0, color="#666666", linewidth=0.8, zorder=2)

    lim = (max(max(limit), max(abs(v) for v in dd)) or 1.0) * 1.3
    _hour_axis(ax)
    ax.set_xticks(range(25), minor=True)
    ax.set_ylim(-lim, lim)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.set_xlabel("Hour of day", fontsize=FONT["axis"])
    ax.set_ylabel("Load shift $\\Delta d_t$  [kWh]", fontsize=FONT["axis"])

    moved = sum(max(v, 0.0) for v in dd)
    at_limit = sum(1 for v, c in zip(dd, limit) if abs(v) >= c - 0.05)
    ax.annotate(f"{moved:,.0f} kWh moved = "
                f"{100.0 * moved / sum(D.values()):.1f}% of daily demand\n"
                f"at the {frac:.0%} limit in {at_limit} of {len(T)} h",
                (0.01, 0.97),
                xycoords="axes fraction", ha="left", va="top",
                fontsize=FONT["note"], color="#333333",
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#cccccc",
                          alpha=0.9))

    none = [sc for sc in SCENARIOS if sc.start is None]
    fig.text(0.5, 0.01,
             "Up: load moved into the hour  ·  down: load moved out of it  ·  "
             "one schedule for every outcome"
             + "".join(f"  ·  {sc.label} (prob {sc.prob:.1f}) has no window"
                       for sc in none),
             ha="center", va="bottom", fontsize=FONT["note"], color="#555555")
    fig.legend(handles=[
        Patch(facecolor="#333333", label="Load shift $\\Delta d_t$"),
        Patch(facecolor="#CFCFCF",
              label=f"Shift limit  {frac:.0%} of $D_t$")],
        loc="lower center", ncol=2, frameon=False, fontsize=FONT["legend"],
        bbox_to_anchor=(0.5, 0.035))
    fig.suptitle(f"Village load-shift schedule under the stochastic commitment"
                 f" — Lebanese summer day, {fleet}",
                 fontsize=FONT["suptitle"] - 1.0, y=0.985)

    out_path = out_dir / "lebanon_summerday_shift.png"
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    print(f"  wrote {out_path}")
    plt.close(fig)


INFO_BAR = "#5B6770"          # one series per panel: a single neutral hue


def _info_ticks(info: list[dict]) -> list[str]:
    """Tick labels for the information settings: name and what the village
    plans for."""
    label = {sc.key: sc.label for sc in SCENARIOS}
    return [f"{r['label']}\nvillage plans for\n"
            + ("the probabilities" if r["plans_for"] is None
               else f"'{label[r['plans_for']].lower()}'")
            for r in info]


def plot_information(info: list[dict], out_dir) -> None:
    """
    Committed price and expected profit per village-information setting, a
    bar panel each. Bars start at zero, so small price differences stay small;
    every bar carries its value.
    """
    if not info:
        return
    fleet = (f"{config.CONFIG['CAP']['DG']:g} kW DG"
             + (f" + {config.CONFIG['CAP']['PV']:g} kWp PV"
                if config.CONFIG["CAP"]["PV"] else ""))
    ticks = _info_ticks(info)
    x = list(range(len(info)))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0))
    panels = ((axes[0], "p", "Committed price  [\\$/kWh]", ".3f",
               "Committed price"),
              (axes[1], "profit", "Expected profit  [\\$/day]", ",.0f",
               "Expected profit"))
    for ax, key, ylabel, fmt, title in panels:
        vals = [r[key] for r in info]
        ax.bar(x, vals, width=0.6, color=INFO_BAR, zorder=2)
        for xi, v in zip(x, vals):
            ax.annotate(format(v, fmt), (xi, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=FONT["note"] + 0.5, color="#222222")
        if key == "p":
            P_ref = config.CONFIG["P_ref"]
            ax.axhline(P_ref, color="#666666", linewidth=1.0, linestyle=":",
                       zorder=3)
            ax.annotate(f"$P_{{ref}}$ = {P_ref:.3f}", (1.0, P_ref),
                        xycoords=("axes fraction", "data"), xytext=(-4, 4),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=FONT["note"], color="#666666")
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=FONT["tick"])
        ax.set_ylim(0, max(vals) * 1.15)
        ax.set_ylabel(ylabel, fontsize=FONT["axis"])
        ax.tick_params(axis="y", labelsize=FONT["tick"])
        ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title(title, fontsize=FONT["panel_title"] + 0.5, pad=10)

    fig.suptitle(f"Village information — Lebanese summer day, {fleet}",
                 fontsize=FONT["suptitle"], y=0.985)
    out_path = out_dir / "lebanon_summerday_information.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    print(f"  wrote {out_path}")
    plt.close(fig)


# EVPI, VSS and regret: a validated triple (worst adjacent normal-vision
# deltaE 21.8, CVD 21.0); the amber sits below 3:1 against white, so every bar
# carries its value.
EVPI_COLOR, VSS_COLOR, REGRET_COLOR = "#3A6EA5", "#D08A2E", "#A0527E"


def plot_evpi(info: list[dict], out_dir) -> None:
    """
    The leader's EVPI and risk per village-information setting, the village's
    shift held as in that setting, in two panels:

        left    EVPI   pricing per outcome instead of committing
        right   max regret   the largest profit any single outcome loses to
                             the committed price, against that outcome's
                             own best price

    Regret per outcome averages to the EVPI, so the right panel shows its
    maximum -- the minimax-regret figure -- rather than repeat the mean. EVPI
    bars carry their value and their share of the committed expected profit;
    the regret bars carry their value and the outcome.

    The VSS is computed and written to the workbook but NOT drawn: its
    mean-value problem averages an all-or-nothing supply window into a day
    that never occurs, and both benchmarks here land on the same price, so the
    bars would be zero. EVPI and regret compare against the real outcomes.
    """
    rows = [r for r in info
            if any(r.get(k) is not None for k in ("evpi", "max_regret"))]
    if not rows:
        return
    fleet = (f"{config.CONFIG['CAP']['DG']:g} kW DG"
             + (f" + {config.CONFIG['CAP']['PV']:g} kWp PV"
                if config.CONFIG["CAP"]["PV"] else ""))
    outcome = {sc.key: sc.label for sc in SCENARIOS}
    x = list(range(len(rows)))
    ticks = _info_ticks(rows)

    fig, (ax_v, ax_r) = plt.subplots(
        1, 2, figsize=(14.0, 6.6), gridspec_kw={"width_ratios": [1.35, 1.0]})

    def _chrome(ax, vals, ylabel, title):
        ax.axhline(0, color="#666666", linewidth=0.8, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=FONT["tick"])
        ax.set_ylim(min(0.0, min(vals) * 1.2), max(max(vals), 1e-9) * 1.4)
        ax.set_ylabel(ylabel, fontsize=FONT["axis"])
        ax.tick_params(axis="y", labelsize=FONT["tick"])
        ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title(title, fontsize=FONT["panel_title"] + 0.5, pad=10)

    # ---- left: EVPI (the VSS is in the workbook only, see the docstring) -----
    series = (("evpi", "EVPI  — price per outcome vs. committed", EVPI_COLOR),)
    width = 0.55
    vals_v = []
    for i, (key, label, color) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * width
        pts = [(xi + off, r) for xi, r in zip(x, rows)
               if r.get(key) is not None]
        if not pts:
            continue
        ax_v.bar([px for px, _ in pts], [r[key] for _, r in pts],
                 width=width * 0.92, color=color, label=label, zorder=2)
        for px, r in pts:
            share = 100.0 * r[key] / r["profit"] if r["profit"] else 0.0
            ax_v.annotate(f"{r[key]:,.2f}\n{share:.1f}%",
                          (px, max(r[key], 0.0)), xytext=(0, 4),
                          textcoords="offset points", ha="center",
                          va="bottom", fontsize=FONT["note"], color="#222222")
        vals_v += [r[key] for _, r in pts]
    if vals_v:
        _chrome(ax_v, vals_v, "Value to the leader  [\\$/day]",
                "EVPI  ·  labels: \\$/day and share of committed E[profit]")
        ax_v.legend(loc="upper left", frameon=False, fontsize=FONT["legend"])

    # ---- right: maximum regret -------------------------------------------------
    pts = [(xi, r) for xi, r in zip(x, rows)
           if r.get("max_regret") is not None]
    if pts:
        ax_r.bar([px for px, _ in pts], [r["max_regret"] for _, r in pts],
                 width=0.55, color=REGRET_COLOR, zorder=2)
        for px, r in pts:
            ax_r.annotate(f"{r['max_regret']:,.2f}\n"
                          f"{outcome[r['max_regret_outcome']].lower()}",
                          (px, max(r["max_regret"], 0.0)), xytext=(0, 4),
                          textcoords="offset points", ha="center",
                          va="bottom", fontsize=FONT["note"], color="#222222")
        _chrome(ax_r, [r["max_regret"] for _, r in pts],
                "Maximum regret  [\\$/day]",
                "Regret  ·  optimal vs. committed price, worst outcome")

    fig.suptitle(f"Leader's EVPI and regret — Lebanese summer day, {fleet}"
                 f"  ·  load shift held at each setting's schedule",
                 fontsize=FONT["suptitle"] - 1.0, y=0.985)
    out_path = out_dir / "lebanon_summerday_evpi.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    print(f"  wrote {out_path}")
    plt.close(fig)


def _signed(v: float, fmt: str) -> str:
    """``v`` with an explicit sign and a true minus."""
    return ("+" if v >= 0 else "\u2212") + format(abs(v), fmt)


def plot_regret(regret: dict, out_dir) -> None:
    """
    What the village's shift costs the leader at the stochastic price, as a
    table: one column per outcome with an EDL window, the first row the
    leader's profit with the village planning for no supply, the others the
    difference to it on a perfect-information or the stochastic shift, with
    the % change in brackets. No supply is left out: without a window the
    shift changes nothing the village plans for.
    """
    rows = [(k, lbl) for k, lbl in REGRET_SETTINGS
            if k != REGRET_BASE and regret.get(k)]
    base = regret.get(REGRET_BASE)
    if not rows or not base:
        return
    fleet = (f"{config.CONFIG['CAP']['DG']:g} kW DG"
             + (f" + {config.CONFIG['CAP']['PV']:g} kWp PV"
                if config.CONFIG["CAP"]["PV"] else ""))
    outcomes = [sc for sc in SCENARIOS if sc.start is not None]
    base_label = dict(REGRET_SETTINGS)[REGRET_BASE]

    header = [f"{sc.label}\n{sc.start:02d}-{sc.end:02d} h  (prob {sc.prob:.1f})"
              for sc in outcomes]
    body = [(f"Village {base_label.lower()}\nleader profit  [\\$/day]",
             [f"{base[sc.key]['profit']:,.2f}" for sc in outcomes], True)]
    for key, label in rows:
        cells = []
        for sc in outcomes:
            r, b0 = regret[key].get(sc.key), base[sc.key]["profit"]
            cells.append("-" if r is None else
                         f"{_signed(r['diff'], ',.2f')}  "
                         f"({_signed(100.0 * r['diff'] / b0, '.1f')}%)"
                         if b0 else _signed(r["diff"], ",.2f"))
        body.append((f"{label}\nvs. {base_label.lower()}  [\\$/day]", cells,
                     False))

    # Column x positions in axes fraction: the row labels, then one per outcome.
    col_x = [0.0] + [0.40 + i * 0.215 for i in range(len(outcomes))]
    n = len(body) + 1
    row_h = 1.0 / n
    ink, muted, rule = "#222222", "#555555", "#888888"

    fig, ax = plt.subplots(figsize=(12.0, 1.3 + 0.95 * n))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def _y(i):                       # centre of row i, 0 = header
        return 1.0 - (i + 0.5) * row_h

    for x, text in zip(col_x[1:], header):
        ax.text(x + 0.1075, _y(0), text, ha="center", va="center",
                fontsize=FONT["axis"], fontweight="bold", color=ink)
    for i, (label, cells, is_base) in enumerate(body, start=1):
        ax.text(col_x[0], _y(i), label, ha="left", va="center",
                fontsize=FONT["axis"], color=ink)
        for x, text in zip(col_x[1:], cells):
            ax.text(x + 0.1075, _y(i), text, ha="center", va="center",
                    fontsize=FONT["axis"] + 1, color=muted if is_base else ink,
                    fontweight="normal" if is_base else "bold")
    # Booktabs rules: top, under the header, under the reference row, bottom.
    for y, lw in ((1.0, 1.4), (1.0 - row_h, 0.9), (1.0 - 2 * row_h, 0.6),
                  (0.0, 1.4)):
        ax.axhline(y, color=rule, linewidth=lw, clip_on=False)

    fig.suptitle(f"What the village's load shift is worth to the leader — "
                 f"Lebanese summer day, {fleet}",
                 fontsize=FONT["suptitle"] - 1.0, y=0.98)
    fig.text(0.5, 0.015,
             f"Leader price held at the stochastic commitment, "
             f"p = {regret['p']:.4f} \\$/kWh; only the village's shift "
             f"changes. In brackets: % change against the village "
             f"{base_label.lower()}.",
             ha="center", va="bottom", fontsize=FONT["note"], color=muted)

    out_path = out_dir / "lebanon_summerday_regret.png"
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.9))
    fig.savefig(out_path, dpi=200)
    print(f"  wrote {out_path}")
    plt.close(fig)


def plot_village(village: dict, out_dir, info: list[dict] | None = None) -> None:
    """
    The village's committed shift against the four outcomes, price held, in
    two panels -- one bar per outcome in its scenario colour, the expected
    value as a dashed line:

        left    hit rate      net share of the planned shift landing in EDL
                              hours; hollow marker: the best shift for that
                              outcome
        right   regret        bill with the committed shift minus bill with
                              the best shift; expected = the village's VPI,
                              the leader's EVPI alongside for scale
    """
    rows = village.get("rows", {})
    if not rows:
        return
    fleet = (f"{config.CONFIG['CAP']['DG']:g} kW DG"
             + (f" + {config.CONFIG['CAP']['PV']:g} kWp PV"
                if config.CONFIG["CAP"]["PV"] else ""))
    outcomes = [sc for sc in SCENARIOS if sc.key in rows]
    x = list(range(len(outcomes)))
    ticks = [f"{sc.label}\n" + (f"{sc.start:02d}-{sc.end:02d} h"
                                if sc.start is not None else "no window")
             + f"\nprob {sc.prob:.1f}" for sc in outcomes]
    colors = [SCENARIO_COLOR_SUMMER[sc.key] for sc in outcomes]
    ink = "#222222"

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2))

    def _chrome(ax, vals, ylabel, title, pct=False):
        ax.axhline(0, color="#666666", linewidth=0.8, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=FONT["tick"] - 0.5)
        lo, hi = min(0.0, min(vals)), max(max(vals), 1e-9)
        span = hi - lo
        ax.set_ylim(lo - (0.12 * span if lo < 0 else 0.0), hi + 0.28 * span)
        ax.set_ylabel(ylabel, fontsize=FONT["axis"])
        ax.tick_params(axis="y", labelsize=FONT["tick"])
        if pct:
            ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
        ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title(title, fontsize=FONT["panel_title"], pad=10, loc="left")

    def _expected_line(ax, v, text):
        ax.axhline(v, color=ink, linewidth=1.1, linestyle=(0, (5, 3)), zorder=4)
        ax.annotate(text, (1.0, v), xycoords=("axes fraction", "data"),
                    xytext=(-2, 4), textcoords="offset points", ha="right",
                    va="bottom", fontsize=FONT["note"], color=ink,
                    fontweight="bold")

    def _labels(ax, vals, fmt):
        for xi, v in zip(x, vals):
            if v is None:
                ax.annotate("n/a", (xi, 0.0), xytext=(0, 4),
                            textcoords="offset points", ha="center",
                            va="bottom", fontsize=FONT["note"], color="#555555")
                continue
            ax.annotate(fmt(v), (xi, v), xytext=(0, 4 if v >= 0 else -4),
                        textcoords="offset points", ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=FONT["note"], color=ink)

    # ---- 1: hit rate ------------------------------------------------------
    ax = axes[0]
    eta = [rows[sc.key]["eta"] for sc in outcomes]
    ax.bar(x, [v or 0.0 for v in eta], width=0.6, color=colors, zorder=2)
    best = [rows[sc.key]["eta_best"] for sc in outcomes]
    bx = [xi for xi, v in zip(x, best) if v is not None]
    ax.scatter(bx, [v for v in best if v is not None], s=70, facecolors="none",
               edgecolors=ink, linewidths=1.4, zorder=5,
               label="best shift for that outcome")
    _labels(ax, eta, lambda v: f"{100 * v:.0f}%")
    _chrome(ax, [v or 0.0 for v in eta] + [v for v in best if v is not None]
            + [1.0], "Share reaching EDL hours", "① Hit rate", pct=True)
    if "eta" in village:
        _expected_line(ax, village["eta"], f"expected {100 * village['eta']:.0f}%")
    ax.legend(loc="upper left", frameon=False, fontsize=FONT["note"])

    # ---- 2: village regret -------------------------------------------------
    ax = axes[1]
    rg = [rows[sc.key]["regret"] for sc in outcomes]
    ax.bar(x, rg, width=0.6, color=colors, zorder=2)
    _labels(ax, rg, lambda v: f"{v:,.2f} \\$")
    _chrome(ax, rg, "Extra bill vs. best shift  [\\$/day]",
            "② Village regret")
    if "vpi" in village:
        _expected_line(ax, village["vpi"],
                       f"village VPI {village['vpi']:,.2f} \\$/day")
    sto = next((r for r in (info or []) if r.get("key") == "stochastic"), None)
    if sto is not None and sto.get("evpi") is not None:
        ax.text(0.02, 0.97, f"leader's EVPI: {sto['evpi']:,.2f} \\$/day",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=FONT["note"], color="#555555")

    fig.suptitle(f"The village's committed load shift  ·  {village['kwh']:,.0f}"
                 f" kWh planned, price {village['p']:.3f} \\$/kWh",
                 fontsize=FONT["suptitle"] - 1.5, y=0.985)
    out_path = out_dir / "lebanon_summerday_village.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    print(f"  wrote {out_path}")
    plt.close(fig)

# --------------------------------------------------------------------------
def main(figures: list[str]) -> None:
    cfg = config.CONFIG
    build = Build(f"pv{PV_KWP:g}", f"{cfg['CAP']['DG']:g} kW DG + "
                  f"{PV_KWP:g} kWp PV", cfg["CAP"]["DG"], PV_KWP,
                  cfg["shift_frac"])
    # build_instance reads the fleet off config at call time, and the titles
    # and workbook note read it too, so the whole run sits inside the patch.
    with build_settings(build):
        _main(figures)


def _main(figures: list[str]) -> None:
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if "scenarios" in figures:
        print(f"[{SEASON}] scenarios")
        for key, s in plot_scenarios(
                out_dir / "lebanon_summerday_scenarios.png").items():
            print(f"  {key:>15}: {s['hours']:2d} h  {s['window']:>9}  "
                  f"{s['covered']:7,.0f} kWh  {s['share']:5.1f}% of demand")

    if {"boxes", "shift", "information", "evpi", "regret",
            "village"} & set(figures):
        results = solve_all()
        print_summary(results)
        if not results["stochastic"] and not results["perfect"]:
            print("Nothing solved — no figures written.")
            return
        info = (solve_information(results)
                if {"information", "evpi"} & set(figures) else None)
        if info and "evpi" in figures:
            solve_evpi(info)
            solve_vss(info)
        regret = solve_regret(results) if "regret" in figures else None
        village = solve_village(results) if "village" in figures else None
        save_workbook(results, out_dir / "lebanon_summerday.xlsx", info,
                      regret, village)
        if village:
            plot_village(village, out_dir, info)
        if info and "information" in figures:
            plot_information(info, out_dir)
        if info and "evpi" in figures:
            plot_evpi(info, out_dir)
        if regret:
            plot_regret(regret, out_dir)
        if "boxes" in figures:
            plot_boxes(results, out_dir)
        if "shift" in figures:
            plot_shift(results, out_dir)
            dd = next(iter(results["stochastic"].values()), {}).get("dd", [])
            print("  shift [kWh]: " + "  ".join(
                f"h{t:02d} {v:+.1f}" for t, v in enumerate(dd)
                if abs(v) > 1e-6))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Four EDL scenarios on a Lebanese summer day: the "
                    "scenario figure, and the stochastic commitment against "
                    "perfect information as price and profit boxplots.")
    parser.add_argument("--figures", nargs="+",
                        default=["scenarios", "boxes", "shift",
                                 "information", "evpi", "regret", "village"],
                        choices=["scenarios", "boxes", "shift",
                                 "information", "evpi", "regret", "village"],
                        help="What to produce (default: both; 'boxes' "
                             "solves).")
    args = parser.parse_args()
    main(args.figures)
