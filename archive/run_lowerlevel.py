"""
Run script for the lower level (follower) of the bilevel microgrid model.

Assembles a follower instance from the central config (:mod:`config`), the
hourly YAML inputs (:func:`data.input_data.load_data`) and the EDL scenarios
(:func:`scenarios.grid_scenarios`) at a FIXED leader price, then builds and
solves it.

Usage:
    python run_lowerlevel.py                          # winter, price = P_ref
    python run_lowerlevel.py --season summer --price 0.4
"""

import argparse

from pyomo.environ import TerminationCondition

import config
from data.input_data import load_data, get_demand
from model.lower_level import LowerLevel
from output import save_results
from scenarios import grid_scenarios


def build_instance(season: str, price=None) -> dict:
    """
    Map config + hourly YAML data into the dict expected by LowerLevel.

    Parameters
    ----------
    season : str
        One of the seasons in the YAML bundle.
    price : float or dict or None
        The FIXED leader price offer \\tilde p_t [$/kWh]. A scalar is broadcast
        over all hours; ``None`` uses that season's reference price ``P_ref``.

    Returns
    -------
    dict
        Keys: ``T, S, prob, p, D, P, alpha1, pPG, pVOLL, QbarPG, shiftFrac, BigM``.
    """
    cfg = config.CONFIG
    bundle = load_data()

    if season not in bundle["seasons"]:
        raise ValueError(
            f"Unknown season {season!r}. Expected one of {bundle['seasons']}."
        )

    # Hourly village net demand (households, PV self-consumption clamped >= 0).
    D = get_demand(bundle, cfg, season)          # {hour: kWh}
    T = sorted(D.keys())                         # 0..23

    # Reference price for this season (scalar -> constant over hours).
    P_ref = cfg[f"P_ref_{season}"]
    P = {t: P_ref for t in T}

    # Fixed leader price offer. Unlike in the bilevel, p is a PARAMETER here.
    if price is None:
        p = dict(P)
    elif isinstance(price, dict):
        p = {t: float(price[t]) for t in T}
    else:
        p = {t: float(price) for t in T}

    # Public-grid (EDL) availability scenarios for this season.
    S, prob, QbarPG = grid_scenarios(season, T)

    M_phys = cfg["M_phys"]
    dual_bigM = cfg["dual_bigM"]
    BigM = {
        "qMGO": M_phys, "qPG": M_phys, "qsupp": M_phys, "qcri": M_phys,
        "slackPG": M_phys, "slackShift": M_phys,
        "beta": dual_bigM, "gamma": dual_bigM, "rc": dual_bigM,
    }

    return {
        "T": T,
        "S": S,
        "prob": prob,
        "p": p,                                  # fixed leader price offer
        "D": D,
        "P": P,
        "alpha1": cfg["alpha_1"],                # price elasticity of demand
        "pPG": {t: cfg["P_EDL"] for t in T},     # public-grid supply price
        "pVOLL": cfg["C_pen"],                   # value of lost load ~ penalty cost
        "QbarPG": QbarPG,
        "shiftFrac": cfg["shift_frac"],          # |Δd_t| <= shift_frac * D_t
        "BigM": BigM,
    }


def solver_options() -> dict:
    return {
        "MIPGap": config.CONFIG["MIPGap"],
        "TimeLimit": config.CONFIG["TimeLimit"],
    }


def main(season: str = "winter", price=None, solver: str = "gurobi") -> None:
    data = build_instance(season, price)

    llp = LowerLevel(data)
    m = llp.build()

    from pyomo.environ import Constraint, Var, value
    n_con = sum(1 for _ in m.component_data_objects(Constraint))
    n_var = sum(1 for _ in m.component_data_objects(Var))
    print(f"[{season}] Lower-level LP built: {n_var} vars, {n_con} constraints "
          f"(price = {data['p'][data['T'][0]]:.3f} $/kWh).")

    solver_name, res = llp.solve(solver, options=solver_options())
    if solver_name is None:
        print(f"{solver} not available (check the license / installation).")
        return

    tc = res.solver.termination_condition
    if tc not in (TerminationCondition.optimal, TerminationCondition.feasible):
        print(f"{solver_name}: termination = {tc}")
        save_results(m, f"lowerlevel_{season}",
                     meta={"model": "LowerLevel", "season": season,
                           "solver": solver_name, "termination": tc})
        return

    print(f"Solved with {solver_name}: expected supply cost = "
          f"{llp.objective_value():.4f}")
    # qMGO/qPG/qcri are recourse (scenario-dependent) -> report expectations.
    for t in data["T"]:
        def exp(var):
            return sum(data["prob"][s] * value(var[t, s]) for s in data["S"])
        print(f"  t={t:>2}: D={data['D'][t]:>8.2f}  E[qMGO]={exp(m.qMGO):>8.2f}  "
              f"E[qPG]={exp(m.qPG):>8.2f}  qsupp={value(m.qsupp[t]):>7.2f}  "
              f"E[qcri]={exp(m.qcri):>8.2f}  dd={value(m.dd[t]):>7.2f}")

    save_results(m, f"lowerlevel_{season}",
                 meta={"model": "LowerLevel", "season": season,
                       "solver": solver_name, "termination": tc,
                       "price": data["p"][data["T"][0]]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the lower level of the bilevel microgrid model.")
    parser.add_argument("--season", default="winter",
                        choices=list(config.SEASONS.keys()),
                        help="Representative season to build the instance for.")
    parser.add_argument("--price", type=float, default=None,
                        help="Fixed leader price [$/kWh] (default: the season's P_ref).")
    parser.add_argument("--solver", default="gurobi",
                        help="Pyomo solver name (default: gurobi).")
    args = parser.parse_args()

    main(args.season, args.price, args.solver)
