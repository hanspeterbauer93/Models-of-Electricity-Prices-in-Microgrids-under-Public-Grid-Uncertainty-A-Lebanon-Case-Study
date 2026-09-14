r"""
Single-level reformulation of the stochastic bilevel microgrid model
(supervisor draft). The lower-level LP is replaced by its KKT system; the
complementarity conditions are linearized with disjunctive Big-M constraints.

Mapping to the LaTeX:
    p        = \tilde p_t            (UL price offer, first stage)
    qMGO     = q^{cov,MGO}_{t,s}
    qPG      = q^{cov,PG}_{t,s}
    Rbar     = [D_t - \bar Q^{PG}_{t,s}]^+   (parameter: uncovered baseline load)
    qsupp    = q^{supp}_{t,s} = alpha_1 * p_t/P_t * Rbar_{t,s}   (price LEVEL)
    qcri     = q^{supp,cri}_{t,s}
    dd       = \Delta d_t            (free, first stage)
    qdiesel  = q^{diesel}_{t,s}
    qPV      = q^{PV}_{t,s}
    lam      = \lambda_{t,s} (free)  beta = \beta_{t,s} >= 0
    mu       = \mu_{t,s} (free)      gp/gm = \gamma^{+}/\gamma^{-}_t >= 0
    rho      = \rho (free, scalar)

RECOURSE STRUCTURE:
The leader commits ONE price schedule p[t] for the whole day, before the EDL
availability scenario realizes -- p and the binaries b[t,k] are first stage.
Generation and the resulting sale, q^{diesel}_{t,s} / q^{PV}_{t,s} / qMGO[t,s],
are RECOURSE: the MGO observes the realized public-grid availability and only
then dispatches, so cheap EDL supply can displace microgrid supply scenario by
scenario. Revenue and generation cost are therefore expectations over s.

DISCRETE PRICE MENU (exact linearization):
The leader offers prices from a finite menu Pi = {pi_1, ..., pi_K}. One level is
chosen per hour via binaries b[t,k] (sum_k b[t,k] == 1, p[t] == sum_k pi_k b[t,k]).
The bilinear revenue p[t]*qMGO[t,s] then becomes binary*continuous and is linearized
EXACTLY with auxiliary w[t,k,s] = b[t,k]*qMGO[t,s]. The whole model is now a pure MILP
(no quadratic terms), solved with Gurobi. p[t] enters the follower KKT
(stationarity, demand response) only linearly, so the KKT stay linear too.
This is a restriction of the leader's strategy space, not a relaxation: the result
is exact over the chosen menu, and menu granularity is a modeling decision.

RESIDUAL-BASED DEMAND RESPONSE:
Suppression is proportional to the load the PUBLIC GRID DOES NOT COVER, not to
gross demand: only that part of the load is exposed to the microgrid's price,
since what EDL delivers is billed at the EDL tariff and is not price-responsive.
The uncovered load is measured against EDL's AVAILABLE CAPACITY and the BASELINE
demand profile,

    Rbar_{t,s} = [ D_t - \bar Q^{PG}_{t,s} ]^+ ,
    q^{supp}_{t,s} = alpha_1 * ratio(p_t) * Rbar_{t,s} ,

so it is a parameter, not a follower decision. Two consequences:

  * q^{supp} inherits the SCENARIO index -- what EDL can cover is scenario
    dependent, so the suppressed quantity is recourse too, and its dual mu
    likewise becomes mu_{t,s} (per scenario, no longer summed over s).
  * The response stays LINEAR in p, exactly as a response tied to D_t was, so
    the KKT need no new products and the model does not grow.

Measuring against capacity rather than against the delivered q^{cov,PG} is what
keeps this well posed. Tied to the DECISION q^{cov,PG}, a kWh from EDL would
cancel alpha_1*ratio kWh of suppression, so it would save only
(1 - alpha_1*ratio)*p of microgrid purchase; above a threshold price
(~1.80 * P_ref with the winter data) that falls below the EDL tariff and the
follower refuses available EDL energy outright in order to suppress instead --
suppression being costless to it. The leader then prices deliberately to push
the village off the public grid. Anchoring on \bar Q removes that lever, and
also keeps [.]^+ out of the lower level, which would otherwise make the
follower's feasible set nonconvex and invalidate the KKT reformulation.

Note that Delta d does NOT enter Rbar: the elasticity is calibrated on baseline
consumption, and load shifting is a within-day reallocation on top of it rather
than something that changes how much load is priced out. This matches how a
response tied to D_t behaved.

FLAT TARIFF (data["flatPrice"]):
With flatPrice set, the leader posts ONE price for the whole day (p_t = p for all
t) instead of an hourly schedule -- the village gets a single tariff it can plan
against. It also removes a degeneracy of hourly pricing: in an hour where the MGO
sells nothing (EDL covers the load) its price is arbitrary, so the solver returns
an unmeaningful p_t there.

SHIFT DISCOMFORT (data["epsShift"]):
The flat tariff leaves \Delta d DEGENERATE whenever EDL is absent. The village's
bill is p * sum_t q^{cov,MGO}, and since sum_t \Delta d_t = 0, \Delta d does not
enter Rbar, and unserved load stays at zero, reallocating load over the day
leaves that bill EXACTLY unchanged -- the follower is indifferent between every
feasible shift. This is an optimistic bilevel program, so of the follower's many
optima the leader gets the one it likes best, and it uses the village's
flexibility as free demand response: with 200 kWp of PV and no EDL, the winter
run moves 206 kWh out of the dark hours into every daylight hour, lifting PV
output from 303 to 362 kWh and cutting diesel cost by 76 $/day. Nothing in the
village's problem wants that; on a diesel-only fleet the same run still moves
204 kWh and every reported quantity is unchanged to the last digit, which is the
signature of a tie broken by the leader rather than a decision by the follower.

    epsShift = eps > 0 puts a discomfort cost eps * sum_t |\Delta d_t| in the
    FOLLOWER's objective, written with \Delta d = \Delta d^{+} - \Delta d^{-}
    and both parts nonnegative, so |\Delta d| is linear.

Ties then go to NOT shifting, and the village shifts only where it actually
saves money -- above all into hours where EDL has capacity to spare, worth
p - p^{PG} ~ 0.5 $/kWh, four hundred times eps. The leader's objective is
untouched: the discomfort is the village's disutility, not a cash flow, so it
belongs in the lower level only.

Two complementarity families come with it, one per sign, and stationarity in
\Delta d picks up the multiplier nu_t of the splitting identity:

    eps + nu_t >= 0   _|_   \Delta d^{+}_t >= 0
    eps - nu_t >= 0   _|_   \Delta d^{-}_t >= 0
    sum_s lambda_{t,s} + gamma^{+}_t - gamma^{-}_t + rho - nu_t == 0

The split is ALWAYS in the model, so there is one model to reason about rather
than two. It is inert at eps = 0: the two rows above then read nu_t >= 0 and
-nu_t >= 0, which force nu_t = 0, stationarity in \Delta d collapses to exactly
the equation it had before, and both halves carry a zero reduced cost -- the old
degenerate behaviour, reproduced rather than special-cased.
"""

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, Objective,
    NonNegativeReals, Reals, Binary, maximize, value,
    SolverFactory, TerminationCondition,
)


class BilevelMPEC:
    """
    Single-level (KKT + Big-M) reformulation of the stochastic bilevel
    microgrid model, as a Pyomo ``ConcreteModel``.

    Typical use:

        mpec = BilevelMPEC(data)
        mpec.build()
        mpec.solve()

    Parameters
    ----------
    data : dict
        Instance data. Expected keys: ``T, S, prob, D, P, alpha1, Phat,
        pPG, pVOLL, QbarPG, cdiesel, cPV, Qdiesel, QPV, Pi, shiftFrac, BigM``.
        Optional: ``flatPrice`` (bool) -- one tariff for the whole day.
    """

    def __init__(self, data):
        self.data = data
        self.model = None

    # ------------------------------------------------------------------
    def build(self):
        """Construct the Pyomo model, store it on ``self.model``, return it."""
        data = self.data
        m = ConcreteModel()

        # ---- sets ----
        m.T = Set(initialize=data["T"], ordered=True)
        m.S = Set(initialize=data["S"], ordered=True)
        m.K = Set(initialize=list(range(len(data["Pi"]))), ordered=True)  # price levels

        # ---- parameters ----
        m.prob   = Param(m.S, initialize=data["prob"])
        m.D      = Param(m.T, initialize=data["D"])
        m.P      = Param(m.T, initialize=data["P"])          # reference price P_t
        m.alpha1 = Param(initialize=data["alpha1"])
        m.Phat   = Param(m.T, initialize=data["Phat"])       # price cap
        m.pPG    = Param(m.T, initialize=data["pPG"])
        m.pVOLL  = Param(initialize=data["pVOLL"])
        m.QbarPG = Param(m.T, m.S, initialize=data["QbarPG"])
        # Uncovered baseline load [D_t - Qbar_{t,s}]^+ -- the part of the load
        # exposed to the microgrid price, and the base of the demand response.
        # Clamped at zero here, where D and Qbar are both data; inside the
        # lower level a [.]^+ would break the LP structure the KKT rely on.
        m.Rbar = Param(m.T, m.S, initialize={
            (t, s): max(0.0, data["D"][t] - data["QbarPG"][(t, s)])
            for t in data["T"] for s in data["S"]})
        m.cdiesel = Param(m.T, initialize=data["cdiesel"])
        m.cPV     = Param(m.T, initialize=data["cPV"])
        m.Qdiesel = Param(initialize=data["Qdiesel"])        # diesel capacity
        m.QPV     = Param(m.T, initialize=data["QPV"])       # PV cap (time dep.)
        m.Pi      = Param(m.K, initialize={k: v for k, v in enumerate(data["Pi"])})  # menu
        m.shiftFrac = Param(initialize=data["shiftFrac"])    # |Δd_t| <= shiftFrac * D_t
        # Follower's shift discomfort [$/kWh]; 0 disables it (see the docstring).
        eps = float(data.get("epsShift", 0.0) or 0.0)
        m.epsShift = Param(initialize=eps)

        # q^{cov,MGO}_t upper bound (used as the exact-product big-M):  Qdiesel + QPV_t

        # Big-M values (per complementarity family)
        M = data["BigM"]

        # ------------------------------------------------------------------
        # Tight per-index bounds for the QUANTITY complementarities.
        #
        # A single physical constant (data["BigM"]["qMGO"] etc., 1e4 by default)
        # has to clear every hour and scenario at once, so it sits ~25x above
        # any quantity the model can actually reach. Big-M constraints of the
        # form q <= M*z relax to q <= M whenever z is fractional, so that slack
        # is exactly what the LP relaxation exploits: the root bound came in at
        # 148 against a true optimum of 98.6 for a two-scenario instance, and
        # closing that gap by cuts alone dominated the solve time.
        #
        # Each bound below is implied by a constraint already in the model, so
        # substituting it CANNOT cut off a feasible point -- it only stops the
        # relaxation from pretending a quantity could be larger than its own
        # constraint allows:
        #
        #   qMGO[t,s]  <= Qdiesel + QPV[t]        mgo_gen_balance + the caps
        #   qPG[t,s]   <= QbarPG[t,s]             pg_cap
        #   qcri[t,s]  <= D[t]*(1+shiftFrac)      demand_balance, all terms >= 0
        #   qsupp[t,s] <= alpha1*ratio(Phat)*Rbar demand_response at the cap
        #   QbarPG-qPG <= QbarPG[t,s]             qPG >= 0
        #   shiftFrac*D -/+ dd <= 2*shiftFrac*D   shift_up / shift_lo
        #
        # The configured scalar is kept as a ceiling via min(), so lowering it
        # still bites and nothing ever comes out looser than before.
        _deviation = data.get("demandResponse", "level") == "deviation"
        _D, _Qbar = data["D"], data["QbarPG"]
        _sf, _a1 = data["shiftFrac"], data["alpha1"]

        def _ratio_cap(t):
            """ratio(p) at the price cap -- the largest suppression on offer."""
            return ((data["Phat"][t] - data["P"][t]) / data["P"][t] if _deviation
                    else data["Phat"][t] / data["P"][t])

        M_qMGO = {(t, s): min(M["qMGO"], data["Qdiesel"] + data["QPV"][t])
                  for t in data["T"] for s in data["S"]}
        M_qPG = {(t, s): min(M["qPG"], _Qbar[(t, s)])
                 for t in data["T"] for s in data["S"]}
        M_qcri = {(t, s): min(M["qcri"], _D[t] * (1.0 + _sf))
                  for t in data["T"] for s in data["S"]}
        M_qsupp = {(t, s): min(M["qsupp"],
                               _a1 * _ratio_cap(t) * max(0.0, _D[t] - _Qbar[(t, s)]))
                   for t in data["T"] for s in data["S"]}
        M_slackPG = {(t, s): min(M["slackPG"], _Qbar[(t, s)])
                     for t in data["T"] for s in data["S"]}
        M_slackShift = {t: min(M["slackShift"], 2.0 * _sf * _D[t])
                        for t in data["T"]}
        # Each half of the split shift. At the follower's optimum only one half
        # is ever positive -- carrying both costs eps on each and buys nothing --
        # so |Δd_t| <= shiftFrac * D_t bounds each half there.
        M_ddhalf = {t: min(M["slackShift"], _sf * _D[t]) for t in data["T"]}
        # The shift reduced costs are eps +/- nu with |nu| <= eps, so they live
        # in [0, 2*eps] -- five microcents wide, against the ~5 $/kWh ceiling
        # M["rc"] carries for the price-scale duals. Left at M["rc"], the
        # relaxation satisfies these two complementarities for free at any
        # fractional z and the bound goes slack; this is the same substitution
        # of an implied bound as the quantity big-Ms above.
        M_rc_dd = 2.0 * eps if eps > 0.0 else M["rc"]

        # ---- upper-level variables ----
        m.p       = Var(m.T, domain=NonNegativeReals, bounds=lambda mm, t: (0, mm.Phat[t]))
        m.qdiesel = Var(m.T, m.S, domain=NonNegativeReals)   # recourse
        m.qPV     = Var(m.T, m.S, domain=NonNegativeReals)   # recourse

        # ---- discrete price menu ----
        m.b = Var(m.T, m.K, domain=Binary)                   # b[t,k]=1 -> price level k chosen at t
        m.w = Var(m.T, m.K, m.S, domain=NonNegativeReals)    # w[t,k,s] = b[t,k] * qMGO[t,s]  (exact)

        # ---- lower-level primal variables ----
        m.qMGO  = Var(m.T, m.S,   domain=NonNegativeReals)
        m.qPG   = Var(m.T, m.S,   domain=NonNegativeReals)
        m.qsupp = Var(m.T, m.S,   domain=NonNegativeReals)   # recourse: follows q^{cov,PG}
        m.qcri  = Var(m.T, m.S,   domain=NonNegativeReals)
        m.dd    = Var(m.T,        domain=Reals)              # free
        # Δd split by sign, so the discomfort eps * |Δd| stays linear.
        m.ddp = Var(m.T, domain=NonNegativeReals)
        m.ddm = Var(m.T, domain=NonNegativeReals)

        # ---- lower-level dual variables ----
        # mu is the dual of the demand response, now one per (t,s) because the
        # response is scenario dependent through Rbar.
        m.lam = Var(m.T, m.S, domain=Reals)                 # free
        m.mu  = Var(m.T, m.S, domain=Reals)                 # free
        m.rho = Var(          domain=Reals)                 # free scalar
        m.beta = Var(m.T, m.S, domain=NonNegativeReals)
        m.gp   = Var(m.T,      domain=NonNegativeReals)
        m.gm   = Var(m.T,      domain=NonNegativeReals)
        # Dual of Δd = Δd+ - Δd-. Free in the KKT, but its own two reduced-cost
        # rows (eps + nu >= 0 and eps - nu >= 0) pin it to [-eps, eps], so the
        # bound is implied and only stops the relaxation from roaming.
        m.nu = Var(m.T, domain=Reals, bounds=(-eps, eps))

        # ---- complementarity binaries ----
        m.zMGO  = Var(m.T, m.S, domain=Binary)
        m.zPG   = Var(m.T, m.S, domain=Binary)
        m.zsupp = Var(m.T, m.S, domain=Binary)
        m.zcri  = Var(m.T, m.S, domain=Binary)
        m.zbeta = Var(m.T, m.S, domain=Binary)
        m.zgp   = Var(m.T,      domain=Binary)
        m.zgm   = Var(m.T,      domain=Binary)
        m.zddp = Var(m.T, domain=Binary)
        m.zddm = Var(m.T, domain=Binary)

        # =====================================================================
        # Upper-level constraints
        # =====================================================================
        m.mgo_gen_balance = Constraint(
            m.T, m.S, rule=lambda mm, t, s: mm.qMGO[t, s] == mm.qdiesel[t, s] + mm.qPV[t, s])
        m.diesel_cap = Constraint(
            m.T, m.S, rule=lambda mm, t, s: mm.qdiesel[t, s] <= mm.Qdiesel)
        m.pv_cap = Constraint(
            m.T, m.S, rule=lambda mm, t, s: mm.qPV[t, s] <= mm.QPV[t])

        # =====================================================================
        # Discrete price menu + exact linearization of revenue p[t]*qMGO[t]
        # =====================================================================
        # exactly one price level per hour
        m.price_select = Constraint(
            m.T, rule=lambda mm, t: sum(mm.b[t, k] for k in mm.K) == 1)
        # p[t] = sum_k pi_k b[t,k]  (its bound [0, Phat] forbids menu levels above the cap)
        m.price_link = Constraint(
            m.T, rule=lambda mm, t: mm.p[t] == sum(mm.Pi[k] * mm.b[t, k] for k in mm.K))

        # One tariff for the whole day: tie every hour to the first hour's level.
        # Linking the BINARIES rather than p[t] directly is the same restriction
        # (p is a function of b through price_link) but a tighter formulation:
        # Gurobi fixes all 24 hours the moment hour 0 is branched on.
        if data.get("flatPrice", False):
            t0 = m.T.first()
            m.price_flat = Constraint(
                m.T, m.K,
                rule=lambda mm, t, k: (Constraint.Skip if t == t0
                                       else mm.b[t, k] == mm.b[t0, k]))

        # w[t,k,s] = b[t,k] * qMGO[t,s]  (binary * continuous, exact):
        m.w_le_b = Constraint(
            m.T, m.K, m.S, rule=lambda mm, t, k, s: mm.w[t, k, s] <= (mm.Qdiesel + mm.QPV[t]) * mm.b[t, k])
        m.w_le_q = Constraint(
            m.T, m.K, m.S, rule=lambda mm, t, k, s: mm.w[t, k, s] <= mm.qMGO[t, s])
        m.w_ge = Constraint(
            m.T, m.K, m.S, rule=lambda mm, t, k, s:
                mm.w[t, k, s] >= mm.qMGO[t, s] - (mm.Qdiesel + mm.QPV[t]) * (1 - mm.b[t, k]))
        # valid equality (implied by the above + price_select): sum_k w[t,k,s] = qMGO[t,s]
        m.w_sum = Constraint(
            m.T, m.S, rule=lambda mm, t, s: sum(mm.w[t, k, s] for k in mm.K) == mm.qMGO[t, s])

        # =====================================================================
        # Lower-level primal feasibility
        # =====================================================================
        m.demand_balance = Constraint(
            m.T, m.S,
            rule=lambda mm, t, s:
                mm.D[t] + mm.dd[t]
                == mm.qMGO[t, s] + mm.qPG[t, s] + mm.qsupp[t, s] + mm.qcri[t, s])

        # Demand response on the uncovered baseline load Rbar_{t,s}, in one of
        # two forms (data["demandResponse"]):
        #
        #   "level"      q^supp = alpha1 * p/P_ref * Rbar
        #       Suppression is proportional to the price LEVEL, so alpha1 * Rbar
        #       is already suppressed at p = P_ref itself (47.9% of the
        #       uncovered load with alpha1 = 0.4788) and that load is choked off
        #       entirely at p = P_ref/alpha1 = 2.09 * P_ref.
        #
        #   "deviation"  q^supp = alpha1 * (p - P_ref)/P_ref * Rbar
        #       Only a price ABOVE the reference tariff suppresses demand, so
        #       p = P_ref reproduces the reference demand and the choke moves to
        #       (1 + 1/alpha1) * P_ref. Since q^supp >= 0 this also floors the
        #       price at P_ref (undercutting would mean INDUCED demand, which a
        #       one-sided response cannot express).
        #
        # Above the choke price the follower is INFEASIBLE, which is what bounds
        # phat_factor -- the two forms therefore need different caps. Basing the
        # response on Rbar rather than on D leaves both choke prices where they
        # were: the factor multiplying the load is unchanged, only the load it
        # acts on has shrunk.
        def demand_response_rule(mm, t, s):
            ratio = ((mm.p[t] - mm.P[t]) / mm.P[t] if _deviation
                     else mm.p[t] / mm.P[t])
            return mm.qsupp[t, s] == ratio * mm.alpha1 * mm.Rbar[t, s]

        m.demand_response = Constraint(m.T, m.S, rule=demand_response_rule)

        m.shift_balance = Constraint(
            rule=lambda mm: sum(mm.dd[t] for t in mm.T) == 0)

        # Δd_t = Δd+_t - Δd-_t, the identity the discomfort is written on. Its
        # dual nu_t is what enters stationarity in Δd below.
        m.shift_split = Constraint(
            m.T, rule=lambda mm, t: mm.dd[t] == mm.ddp[t] - mm.ddm[t])

        # =====================================================================
        # Lower-level stationarity
        # =====================================================================
        # The demand response contributes mu[t,s] * (q^supp - alpha1*ratio*Rbar)
        # to the Lagrangian. Rbar is a parameter, so mu appears in the
        # stationarity of q^supp and nowhere else -- unlike a response tied to
        # q^{cov,PG}, which would put a mu term into rc_pg and stat_shift.

        # Free variable dd -> equality (eq. stat_shift). dd also appears in the
        # splitting identity, whose dual nu enters here; at eps = 0 the two rows
        # below force nu = 0 and this is the equation it always was.
        m.stat_shift = Constraint(
            m.T,
            rule=lambda mm, t:
                sum(mm.lam[t, s] for s in mm.S) + mm.gp[t] - mm.gm[t] + mm.rho
                - mm.nu[t] == 0)

        # Reduced costs (>= 0 side of the complementarities)
        def rc_mgo(mm, t, s):
            return mm.prob[s] * mm.p[t] - mm.lam[t, s]

        def rc_pg(mm, t, s):
            return mm.prob[s] * mm.pPG[t] - mm.lam[t, s] + mm.beta[t, s]

        def rc_supp(mm, t, s):
            return mm.mu[t, s] - mm.lam[t, s]

        def rc_cri(mm, t, s):
            return mm.prob[s] * mm.pVOLL - mm.lam[t, s]

        m.rc_mgo_geq  = Constraint(m.T, m.S, rule=lambda mm, t, s: rc_mgo(mm, t, s) >= 0)
        m.rc_pg_geq   = Constraint(m.T, m.S, rule=lambda mm, t, s: rc_pg(mm, t, s) >= 0)
        m.rc_supp_geq = Constraint(m.T, m.S, rule=lambda mm, t, s: rc_supp(mm, t, s) >= 0)
        m.rc_cri_geq  = Constraint(m.T, m.S, rule=lambda mm, t, s: rc_cri(mm, t, s) >= 0)

        # =====================================================================
        # Big-M complementarity   0 <= a  _|_  b >= 0
        #   a <= M_a * z ,  b <= M_b * (1 - z)
        # =====================================================================
        # 1) qMGO _|_ rc_mgo
        m.cc_mgo_a = Constraint(m.T, m.S, rule=lambda mm, t, s: mm.qMGO[t, s] <= M_qMGO[t, s] * mm.zMGO[t, s])
        m.cc_mgo_b = Constraint(m.T, m.S, rule=lambda mm, t, s: rc_mgo(mm, t, s) <= M["rc"] * (1 - mm.zMGO[t, s]))

        # 2) qPG _|_ rc_pg
        m.cc_pg_a = Constraint(m.T, m.S, rule=lambda mm, t, s: mm.qPG[t, s] <= M_qPG[t, s] * mm.zPG[t, s])
        m.cc_pg_b = Constraint(m.T, m.S, rule=lambda mm, t, s: rc_pg(mm, t, s) <= M["rc"] * (1 - mm.zPG[t, s]))

        # 3) qsupp _|_ rc_supp
        m.cc_supp_a = Constraint(m.T, m.S, rule=lambda mm, t, s:
                                 mm.qsupp[t, s] <= M_qsupp[t, s] * mm.zsupp[t, s])
        m.cc_supp_b = Constraint(m.T, m.S, rule=lambda mm, t, s:
                                 rc_supp(mm, t, s) <= M["rc"] * (1 - mm.zsupp[t, s]))

        # 4) qcri _|_ rc_cri
        m.cc_cri_a = Constraint(m.T, m.S, rule=lambda mm, t, s: mm.qcri[t, s] <= M_qcri[t, s] * mm.zcri[t, s])
        m.cc_cri_b = Constraint(m.T, m.S, rule=lambda mm, t, s: rc_cri(mm, t, s) <= M["rc"] * (1 - mm.zcri[t, s]))

        # 5) beta _|_ (QbarPG - qPG)     (PG capacity slack)
        m.cc_beta_a = Constraint(m.T, m.S, rule=lambda mm, t, s: mm.beta[t, s] <= M["beta"] * mm.zbeta[t, s])
        m.cc_beta_b = Constraint(m.T, m.S, rule=lambda mm, t, s:
                                 mm.QbarPG[t, s] - mm.qPG[t, s] <= M_slackPG[t, s] * (1 - mm.zbeta[t, s]))
        m.pg_cap = Constraint(m.T, m.S, rule=lambda mm, t, s: mm.qPG[t, s] <= mm.QbarPG[t, s])

        # 6) gp _|_ (shiftFrac*D - dd)   (shift upper slack)
        m.cc_gp_a = Constraint(m.T, rule=lambda mm, t: mm.gp[t] <= M["gamma"] * mm.zgp[t])
        m.cc_gp_b = Constraint(m.T, rule=lambda mm, t:
                               mm.shiftFrac * mm.D[t] - mm.dd[t] <= M_slackShift[t] * (1 - mm.zgp[t]))
        m.shift_up = Constraint(m.T, rule=lambda mm, t: mm.dd[t] <= mm.shiftFrac * mm.D[t])

        # 7) gm _|_ (shiftFrac*D + dd)   (shift lower slack)
        m.cc_gm_a = Constraint(m.T, rule=lambda mm, t: mm.gm[t] <= M["gamma"] * mm.zgm[t])
        m.cc_gm_b = Constraint(m.T, rule=lambda mm, t:
                               mm.shiftFrac * mm.D[t] + mm.dd[t] <= M_slackShift[t] * (1 - mm.zgm[t]))
        m.shift_lo = Constraint(m.T, rule=lambda mm, t: mm.dd[t] >= -mm.shiftFrac * mm.D[t])

        # 8) / 9) the two halves of the shift against their reduced costs.
        # eps > 0 on both sides is what breaks the tie: with nothing to gain,
        # each half prices itself out and the village leaves its load alone.
        def rc_ddp(mm, t):
            return mm.epsShift + mm.nu[t]

        def rc_ddm(mm, t):
            return mm.epsShift - mm.nu[t]

        m.rc_ddp_geq = Constraint(m.T, rule=lambda mm, t: rc_ddp(mm, t) >= 0)
        m.rc_ddm_geq = Constraint(m.T, rule=lambda mm, t: rc_ddm(mm, t) >= 0)
        m.cc_ddp_a = Constraint(m.T, rule=lambda mm, t:
                                mm.ddp[t] <= M_ddhalf[t] * mm.zddp[t])
        m.cc_ddp_b = Constraint(m.T, rule=lambda mm, t:
                                rc_ddp(mm, t) <= M_rc_dd * (1 - mm.zddp[t]))
        m.cc_ddm_a = Constraint(m.T, rule=lambda mm, t:
                                mm.ddm[t] <= M_ddhalf[t] * mm.zddm[t])
        m.cc_ddm_b = Constraint(m.T, rule=lambda mm, t:
                                rc_ddm(mm, t) <= M_rc_dd * (1 - mm.zddm[t]))

        # =====================================================================
        # Objective: EXPECTED MGO revenue - diesel - PV  (eq. mpec_obj)
        # =====================================================================
        def obj_rule(mm):
            # revenue = E_s[ sum_t p[t]*qMGO[t,s] ]
            #         = sum_{t,k,s} prob_s * pi_k * w[t,k,s]   (exact, linear)
            revenue = sum(mm.prob[s] * mm.Pi[k] * mm.w[t, k, s]
                          for t in mm.T for k in mm.K for s in mm.S)
            c_diesel = sum(mm.prob[s] * mm.cdiesel[t] * mm.qdiesel[t, s]
                           for t in mm.T for s in mm.S)
            c_pv = sum(mm.prob[s] * mm.cPV[t] * mm.qPV[t, s]
                       for t in mm.T for s in mm.S)
            return revenue - c_diesel - c_pv

        m.profit = Objective(rule=obj_rule, sense=maximize)

        self.model = m
        return m

    # ------------------------------------------------------------------
    def solve(self, solver_name="gurobi", options=None, tee=False):
        """
        Solve the model (building it first if needed) with Gurobi.

        Parameters
        ----------
        solver_name : str
            Pyomo solver name (default ``"gurobi"``).
        options : dict, optional
            Solver options passed through to Gurobi (e.g.
            ``{"MIPGap": 1e-4, "TimeLimit": 300}``).
        tee : bool
            Stream the solver log to stdout.

        Returns
        -------
        tuple
            ``(solver_name, results)`` on success, or ``(None, None)`` if the
            solver is not available.
        """
        if self.model is None:
            self.build()

        solver = SolverFactory(solver_name)
        if not solver.available():
            return None, None

        if options:
            solver.options.update(options)

        res = solver.solve(self.model, tee=tee)
        return solver_name, res
