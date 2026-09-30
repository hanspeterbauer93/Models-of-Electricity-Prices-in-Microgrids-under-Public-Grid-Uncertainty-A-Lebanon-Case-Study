"""
Run script for the bilevel MPEC microgrid model.

Assembles a model instance from the central config (:mod:`config`) and the
hourly YAML inputs (:func:`data.input_data.load_data`), then builds and solves
it with Gurobi.

After the solve it also produces the per-scenario dispatch/profit plots
(:mod:`plot`) for that season, which can be switched off.

Usage:
    python run.py                       # default season (winter), solve + plots
    python run.py --season summer
    python run.py --skip-plots          # solve only
"""

import argparse

from pyomo.environ import value, TerminationCondition

import config
from data.input_data import load_data, get_demand
from model.bilevel_mpec import BilevelMPEC
from output import save_results
from scenarios import grid_scenarios


def build_instance(season: str) -> dict:
    """Map config + hourly YAML data into the dict expected by BilevelMPEC."""
    cfg = config.CONFIG
    bundle = load_data()

    if season not in bundle["seasons"]:
        raise ValueError(
            f"Unknown season {season!r}. Expected one of {bundle['seasons']}."
        )

    # Hourly village net demand (households, PV self-consumption clamped >= 0).
    D = get_demand(bundle, cfg, season)          # {hour: kWh}
    T = sorted(D.keys())                         # 0..23

    cap_factor = bundle["cap_factor"][season]    # {hour: kWh/kWp/h}

    # Leader price-menu assumptions (centralized in config).
    phat_factor = cfg["phat_factor"]
    n_levels = cfg["n_price_levels"]

    # Reference price for this season (scalar -> constant over hours).
    P_ref = cfg["P_ref"]
    P = {t: P_ref for t in T}
    Phat = {t: phat_factor * P_ref for t in T}   # price cap

    # Discrete price menu: N levels evenly spaced in [P_ref, Phat].
    # The menu starts at the reference tariff, not at 0. Under the "deviation"
    # response levels below P_ref are outright infeasible (q^supp >= 0 forces
    # p >= P_ref); under "level" they are merely never chosen, since the leader
    # loses money below its marginal cost of 1.25 * P_ref. Either way they would
    # only waste the menu's resolution, which with a flat tariff IS the
    # precision of the reported price.
    Pi = [P_ref * (1.0 + (phat_factor - 1.0) * k / (n_levels - 1))
          for k in range(n_levels)]

    # Public-grid (EDL) availability scenarios for this season.
    S, prob, QbarPG = grid_scenarios(season, T)

    # Marginal costs.
    #   diesel = fuel ($/L * L/kWh) + variable O&M         [ASSUMPTION]
    #   PV     = variable O&M
    c_diesel = cfg["C_fuel_L"] * cfg["alpha_fuel"] + cfg["c_OV"]["DG"]
    cdiesel = {t: c_diesel for t in T}
    # Optional two-step heat rate (Svolba 2025): marginal cost below / above the
    # load threshold; the model then ignores cdiesel for the diesel cost.
    diesel_curve = None
    if cfg.get("heat_rate", "constant") == "two_step":
        hr = cfg["heat_rate_two_step"]
        diesel_curve = {
            "threshold": hr["threshold"],
            "c_low": cfg["C_fuel_L"] * hr["L_per_kWh"][0] + cfg["c_OV"]["DG"],
            "c_high": cfg["C_fuel_L"] * hr["L_per_kWh"][1] + cfg["c_OV"]["DG"],
        }
    cPV = {t: cfg["c_OV"]["PV"] for t in T}

    # Capacities.
    Qdiesel = cfg["CAP"]["DG"]                    # [kW] diesel genset cap
    QPV = {t: cfg["CAP"]["PV"] * cap_factor[t] for t in T}   # PV cap = kWp * phi_t

    M_phys = cfg["M_phys"]
    # Duals live on the scale of the follower's prices, not on the physical one:
    # C_pen is the largest of them, so it sets the ceiling for every reduced cost.
    dual_bigM = cfg["dual_bigM_factor"] * cfg["C_pen"]
    BigM = {
        "qMGO": M_phys, "qPG": M_phys, "qsupp": M_phys, "qcri": M_phys,
        "slackPG": M_phys, "slackShift": M_phys,
        "beta": dual_bigM, "gamma": dual_bigM, "rc": dual_bigM,
    }

    return {
        "T": T,
        "S": S,
        "prob": prob,
        "D": D,
        "P": P,
        "alpha1": cfg["alpha_1"],                # price elasticity of demand
        "Phat": Phat,
        "pPG": {t: cfg["P_EDL"] for t in T},     # public-grid supply price
        "pVOLL": cfg["C_pen"],                   # value of lost load ~ penalty cost
        "QbarPG": QbarPG,
        "cdiesel": cdiesel,
        "dieselCurve": diesel_curve,
        "cPV": cPV,
        "Qdiesel": Qdiesel,
        "QPV": QPV,
        "Pi": Pi,
        "shiftFrac": cfg["shift_frac"],          # |Δd_t| <= shift_frac * D_t
        "epsShift": cfg["eps_shift"],            # follower's cost of shifting
        "flatPrice": cfg["flat_price"],          # one tariff for the whole day
        "demandResponse": cfg["demand_response"],  # "level" | "deviation"
        "BigM": BigM,
    }


def solve_and_save(season: str) -> bool:
    """
    Solve the stochastic model once for ``season`` and write the workbook.

    Returns ``False`` if Gurobi is unavailable (the downstream stages solve the
    same model and would only fail again).
    """
    data = build_instance(season)

    mpec = BilevelMPEC(data)
    m = mpec.build()

    from pyomo.environ import Constraint, Var
    n_con = sum(1 for _ in m.component_data_objects(Constraint))
    n_var = sum(1 for _ in m.component_data_objects(Var))
    n_bin = sum(1 for v in m.component_data_objects(Var) if v.is_binary())
    print(f"[{season}] Model built: {n_var} vars ({n_bin} binary), "
          f"{n_con} constraints.")

    gurobi_opts = {
        "MIPGap": config.CONFIG["MIPGap"],
        "TimeLimit": config.CONFIG["TimeLimit"],
    }
    solver_name, res = mpec.solve("gurobi", options=gurobi_opts)
    if solver_name is None:
        print("Gurobi not available (check the license and that gurobipy is installed).")
        return False

    tc = res.solver.termination_condition
    if tc in (TerminationCondition.optimal, TerminationCondition.feasible):
        print(f"Solved with {solver_name}: expected profit = {value(m.profit):.4f}")
        # qMGO/qdiesel/qPV are recourse (scenario-dependent) -> report expectations.
        for t in data["T"]:
            def exp(var):
                return sum(data["prob"][s] * value(var[t, s]) for s in data["S"])
            print(f"  t={t:>2}: p={value(m.p[t]):.3f}  E[qMGO]={exp(m.qMGO):.2f}  "
                  f"E[qdiesel]={exp(m.qdiesel):.2f}  E[qPV]={exp(m.qPV):.2f}")
    else:
        print(f"{solver_name}: termination = {tc}")

    save_results(m, f"bilevel_{season}",
                 meta={"model": "BilevelMPEC", "season": season,
                       "solver": solver_name, "termination": tc})
    return True


def main(season: str = "winter", plots: bool = True) -> None:
    """Solve the model, then produce the plots."""
    if not solve_and_save(season):
        return

    # The stage module is imported here, not at module level: it imports
    # build_instance from this module, so a top-level import would be circular.
    # (The price and availability sweeps and the scenario-mismatch analysis are
    # archived; see archive/README.md.)
    if plots:
        print(f"\n=== Plots [{season}] ===")
        import plot
        plot.main(season)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the bilevel MPEC microgrid model.")
    parser.add_argument("--season", default="winter",
                        choices=list(config.SEASONS.keys()),
                        help="Representative season to build the instance for.")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Do not produce the per-scenario plots.")
    args = parser.parse_args()
    main(args.season, plots=not args.skip_plots)
