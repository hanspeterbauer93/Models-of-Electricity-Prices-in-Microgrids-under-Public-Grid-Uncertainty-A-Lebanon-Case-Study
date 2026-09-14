"""
Central configuration: parameters, solver settings, and run scope.
"""

from pathlib import Path

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

# Mapping: internal season key -> Excel sheet name in the
# 20260513_deir_qanoun_microgrid.xlsx workbook.
SEASONS = {
    "fall_spring":  "Fall_Spring_Day",
    "summer":  "Summer_Day",
    "winter":  "Winter_Day",
}

CONFIG = {
    # Generation
    "CAP": {"PV": 0, "DG": 400}, # [kW] generation capacity limits for PV and diesel genset (e.g., based on existing assets or budget constraints)

    # Costs
    "c_OV": {"PV": 0.01, "DG": 0.05},  # [$/(kWh)] operation and maintenance costs
    "C_fuel_L": 1.50,  # [$/(L)] fuel cost
    "alpha_fuel": 0.3,  # [L/kWh] fuel consumption rate for diesel generator (e.g., liters per kWh generated)
    "C_pen": 5.0,  # [$/(kWh)] penalty cost for load shedding (value of lost load)

    # Prices
    # -----------------------------------------------------------------------
    # P_ref is the ELASTICITY ANCHOR, not a price the MGO charges -- the MGO's
    # price p is an OUTCOME. It does three jobs in :mod:`run`: it anchors the
    # demand response (q^supp = alpha_1 * p/P_ref * Rbar), it sets the price cap
    # Phat = phat_factor * P_ref, and it spans the price menu Pi. It is
    # therefore inseparable from alpha_1, which IS the suppression share AT
    # P_ref -- so a single alpha_1 admits only a single P_ref.
    #
    # Source: Zwickl-Bernhard, S., Neumann, A. F., Olleik, M., & Dbouk, H.
    # (2025), "Market Organization in Low-Income Countries' Microgrids: Insights
    # From Electricity Demand Elasticity and Game-Theory Optimization. Case
    # Study: Lebanon", IEEE Trans. Energy Markets, Policy and Regulation, with
    # the repository https://github.com/sebastianzwickl/lebanese-microgrid.
    # Its run_stackelberg_game.py anchors on the OBSERVED selling price of each
    # month ("Selling Price [$/kWh]"; 41 months, Jan 2021 - May 2024) while
    # holding the elasticity at -0.4788 = alpha_1 throughout -- one elasticity,
    # anchored to whatever price prevailed at the time.
    #
    # 0.4031 is the mean of the last twelve observed months (Jun 2023 - May
    # 2024): the closest stretch to the Aug 2024 - Apr 2025 period the DKN load
    # loggers ran in (Bou Gebrael et al. 2026, arXiv:2603.16893), and after the
    # currency had stabilised. The pairing is the point -- the linearised
    # response needs the demand measured AT P_ref, and D here is the 2024/25
    # logged demand. The source's own full-sample pair is (0.3053 $/kWh,
    # 141566 kWh), the mean price and mean consumption over all 41 months, but
    # its price is dominated by collapse-era months up to 8x cheaper and so does
    # not match this D.
    #
    # NOT seasonal. The series rises 9.4x across the sample (currency collapse);
    # its between-year sd (0.135) is more than twice its detrended within-year
    # sd (0.058); and the apparent month-of-year effects run MONOTONICALLY from
    # -0.069 in January to +0.077 in September, which is a rising trend leaking
    # through the detrending, not a season (a real seasonal effect would hump,
    # not ramp). Earlier versions of this file carried 0.400 / 0.364 / 0.395 for
    # winter / fall-spring / summer; no aggregation of the source data
    # reproduces those, and every aggregation makes winter the CHEAPEST season
    # rather than the dearest. See also the season-independence argument in
    # :mod:`scenarios`, which is the same argument applied to EDL availability.
    # -----------------------------------------------------------------------
    "P_ref": 0.4031,  # [$/(kWh)] elasticity anchor; season-independent
    "P_EDL": 0.1,  # [$/(kWh)] price for grid-supplied electricity (e.g., utility tariff)

    # Demand
    # Which demand response the lower level uses. Suppression applies to the
    # load EDL leaves uncovered, R_t,s = D_t + Δd_t - q^cov,PG_t,s, NOT to gross
    # demand: only that part of the load is exposed to the microgrid price.
    # Switching the form changes the choke price, so phat_factor must follow it:
    #   "level"     q^supp = alpha_1 * p/P_ref * R        choke at 1/alpha_1
    #                                                     (= 2.09) -> cap 2.0
    #   "deviation" q^supp = alpha_1 * (p-P_ref)/P_ref * R  choke at 1 + 1/alpha_1
    #                                                     (= 3.09) -> cap 3.0
    # The choke prices are unchanged from a response tied to D: the factor
    # multiplying the load is the same, only the load it acts on has shrunk.
    "demand_response": "level",

    "alpha_1": 0.4788,    # [-] demand response to the price LEVEL:
                          #     q^supp_t,s = alpha_1 * p_t/P_ref * R_t,s
                          #     Note alpha_1 * R is ALREADY suppressed at the
                          #     reference tariff itself (47.9% of the uncovered
                          #     load), and that load is choked off entirely at
                          #     p = P_ref/alpha_1
                          #     (= 2.09 x P_ref). Implies an own-price elasticity
                          #     of -alpha_1/(1-alpha_1) = -0.92 at the reference
                          #     tariff -- near unit-elastic, hence a large markup.
    "n_Households_nonPV": 400, # [-] number of households without PV (for demand scaling)
    "n_Households_PV": 250,     # [-] number of households with PV (for demand scaling)
    "cap_PV_per_HH": 4,        # [kWp] PV capacity per household (for demand scaling)

    # Modeling / bilevel (leader price menu, load shift, complementarity Big-M)
    "shift_frac": 0.10,     # [-] max hourly load shift |Δd_t| as a fraction of that hour's demand D_t

    # -----------------------------------------------------------------------
    # Shift discomfort [$/kWh]: the village's own cost of moving a kWh of load
    # to another hour, eps * sum_t |Δd_t| in the FOLLOWER's objective.
    #
    # It exists to break a DEGENERACY, not to model comfort in any detail. With
    # a flat tariff the village's bill is p * sum_t q^{cov,MGO}, and since
    # sum_t Δd_t = 0 and Δd does not enter Rbar, reallocating load leaves that
    # bill unchanged: every feasible shift is optimal for the follower. This is
    # an optimistic bilevel program, so the leader then picks the shift IT
    # prefers and uses the village's flexibility as free demand response --
    # with 200 kWp of PV and no EDL at all, the winter run moved 206 kWh into
    # the daylight hours and cut diesel cost by 76 $/day, which nothing in the
    # village's problem wanted. eps > 0 sends ties to not shifting.
    #
    # The level only has to separate "no reason to shift" from the reasons that
    # are real, so it sits orders of magnitude below both: 0.001 against a
    # price-menu step of 0.0063 $/kWh, and against the p - P_EDL ~ 0.5 $/kWh the
    # village saves by moving load into an hour where EDL has capacity to
    # spare. Set it to 0 to recover the earlier, degenerate behaviour.
    # -----------------------------------------------------------------------
    "eps_shift": 0.001,
    "phat_factor": 2.0,     # [-] leader price cap: Phat_t = phat_factor * P_ref
                            #     Must stay below 1/alpha_1 (= 2.09): there the suppressed
                            #     demand equals the uncovered load and the follower turns
                            #     infeasible. The unconstrained optimum is
                            #     0.5 * (c/P_ref + 1/alpha_1) = 1.67 x P_ref, less
                            #     sigma/(2*alpha_1) for whatever load the village shifts
                            #     out of the served hours (1.63 x P_ref on the stochastic
                            #     winter run), so this cap does NOT bind and the price
                            #     stays a model outcome.
    "n_price_levels": 65,   # [-] number of discrete leader price levels in the menu Pi,
                            #     evenly spaced over [P_ref, Phat] (below P_ref the demand
                            #     response is infeasible). With a flat tariff the whole
                            #     decision is this one scalar, so the step size IS the
                            #     precision of the reported price -- and of the suppressed
                            #     demand, which is alpha_1 * (p/P_ref - 1) and therefore
                            #     inherits it directly. 65 levels = 0.031 x P_ref: at 9
                            #     levels the menu step was 0.25 x P_ref, which moved the
                            #     reported suppression by 4-8 percentage points while
                            #     barely changing profit (the optimum is a flat parabola).
                            #     Costs ~8 s per solve instead of ~1 s.
    "flat_price": True,     # [-] one tariff for the whole day: p_t = p for all t.
                            #     The MGO posts a single price the village can plan
                            #     against, rather than an hourly schedule. Also removes
                            #     the degeneracy of hourly pricing (in hours where the
                            #     MGO sells nothing its price is arbitrary) and shrinks
                            #     the binary search space. Set False for hourly prices.
    "dual_bigM_factor": 2.0,  # [-] Big-M for the dual / reduced-cost complementarities,
                            #     as a multiple of C_pen (the largest price in the
                            #     follower's objective, so the scale every dual lives on).
                            #     Every reduced cost is bounded by it:
                            #       rc_mgo = prob*p   - lam        <= prob*Phat
                            #       rc_pg  = prob*pPG - lam + beta <= prob*(pPG + C_pen)
                            #       rc_cri = prob*C_pen - lam      <= prob*C_pen
                            #       rc_supp = mu - lam             == 0 is always available,
                            #         since mu appears in no other condition and may simply
                            #         be set to lam.
                            #     A flat 1e2 used to sit ~140x above the largest dual the
                            #     model actually produces (0.70 on the winter instances),
                            #     and the relaxation spent that slack: it let z_PG go to
                            #     ~0 while rc_pg stayed large, i.e. it could decline EDL
                            #     for free. That held the root bound at 148 against a true
                            #     98.6. Raise the factor if a run turns infeasible.

    # Solver / numerical
    "M_phys": 1e4, # [kWh] large constant for big-M constraints, should be larger than any physically feasible quantity in the model
    "MIPGap": 1e-4, # [-] relative MIP gap for solver termination
    "TimeLimit": 300, # [s] time limit for solver runs (e.g., for sensitivity analyses)
}

# The public-grid (EDL) availability scenarios live in :mod:`scenarios`
# (GridScenario / SCENARIOS), since their supply windows are season-dependent.