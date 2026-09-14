r"""
Lower level (follower) of the stochastic bilevel microgrid model.

The follower is the village: at a leader price offer \tilde p_t fixed by the
MGO, it minimizes its expected supply cost by choosing how much to buy from the
microgrid, how much to draw from the intermittent public grid (EDL), and how
much demand to shed or shift.

Mapping to the LaTeX (same symbols as ``model.bilevel_mpec``):
    p        = \tilde p_t            (leader price, a PARAMETER here)
    qMGO     = q^{cov,MGO}_{t,s}
    qPG      = q^{cov,PG}_{t,s}
    qsupp    = q^{supp}_t
    qcri     = q^{supp,cri}_{t,s}
    dd       = \Delta d_t            (free)

This module holds the follower's primal LP only. Its KKT + Big-M reformulation
-- the block that ``model.bilevel_mpec.BilevelMPEC`` embeds -- lives in
``model.lower_level_kkt``, which subclasses :class:`LowerLevel` and reuses the
primal part built here. Since the LP is convex, both must attain the same cost
at any fixed price; ``run_lowerlevel.py --validate`` checks exactly that.

NOTE ON THE SCENARIO-DEPENDENT q^{cov,MGO}: qMGO is indexed by (t, s), matching
``model.bilevel_mpec``. The microgrid purchase is a RECOURSE decision: the
village observes the realized EDL availability and only then settles how much
it buys from the MGO, so cheap public-grid supply can displace microgrid supply
scenario by scenario. Its reduced cost therefore carries the probability
weight, ``prob_s p_t - lam[t,s]``, exactly like qPG and qcri. Only the leader's
price schedule \tilde p_t and the load shift \Delta d_t are first-stage
(one schedule for the whole day, chosen before the scenario realizes).
"""

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, Objective,
    NonNegativeReals, Reals, minimize, value,
    SolverFactory,
)


class LowerLevel:
    """
    The follower's primal LP at a FIXED leader price, as a Pyomo
    ``ConcreteModel``.

    Typical use:

        llp = LowerLevel(data)
        llp.build()
        llp.solve()
        cost = llp.objective_value()

    Parameters
    ----------
    data : dict
        Instance data. Expected keys: ``T, S, prob, p, D, P, alpha1, pPG,
        pVOLL, QbarPG, shiftFrac``. ``p`` is the fixed leader price
        ``{t: price}``; the remaining keys carry the same meaning as in
        :class:`model.bilevel_mpec.BilevelMPEC`.
    """

    def __init__(self, data):
        self.data = data
        self.model = None

    # ------------------------------------------------------------------
    def _base(self):
        """Sets, parameters, primal variables and primal feasibility.

        Kept separate from :meth:`build` because ``model.lower_level_kkt``
        reuses this primal part and appends the duals to it."""
        data = self.data
        m = ConcreteModel()

        # ---- sets ----
        m.T = Set(initialize=data["T"], ordered=True)
        m.S = Set(initialize=data["S"], ordered=True)

        # ---- parameters ----
        m.p      = Param(m.T, initialize=data["p"])          # leader price offer (FIXED)
        m.prob   = Param(m.S, initialize=data["prob"])
        m.D      = Param(m.T, initialize=data["D"])
        m.P      = Param(m.T, initialize=data["P"])          # reference price P_t
        m.alpha1 = Param(initialize=data["alpha1"])
        m.pPG    = Param(m.T, initialize=data["pPG"])
        m.pVOLL  = Param(initialize=data["pVOLL"])
        m.QbarPG = Param(m.T, m.S, initialize=data["QbarPG"])
        m.shiftFrac = Param(initialize=data["shiftFrac"])    # |Δd_t| <= shiftFrac * D_t

        # ---- primal variables ----
        m.qMGO  = Var(m.T, m.S, domain=NonNegativeReals)     # recourse
        m.qPG   = Var(m.T, m.S, domain=NonNegativeReals)
        m.qsupp = Var(m.T,      domain=NonNegativeReals)
        m.qcri  = Var(m.T, m.S, domain=NonNegativeReals)
        m.dd    = Var(m.T,      domain=Reals)                # free

        # ---- primal feasibility ----
        m.demand_balance = Constraint(          # dual: lam[t,s] (free)
            m.T, m.S,
            rule=lambda mm, t, s:
                mm.D[t] + mm.dd[t]
                == mm.qMGO[t, s] + mm.qPG[t, s] + mm.qsupp[t] + mm.qcri[t, s])

        m.demand_response = Constraint(         # dual: mu[t] (free)
            m.T,
            rule=lambda mm, t:
                mm.qsupp[t] == (mm.p[t] / mm.P[t]) * mm.alpha1 * mm.D[t])

        m.shift_balance = Constraint(           # dual: rho (free, scalar)
            rule=lambda mm: sum(mm.dd[t] for t in mm.T) == 0)

        m.pg_cap = Constraint(                  # dual: beta[t,s] >= 0
            m.T, m.S, rule=lambda mm, t, s: mm.qPG[t, s] <= mm.QbarPG[t, s])

        m.shift_up = Constraint(                # dual: gp[t] >= 0
            m.T, rule=lambda mm, t: mm.dd[t] <= mm.shiftFrac * mm.D[t])

        m.shift_lo = Constraint(                # dual: gm[t] >= 0
            m.T, rule=lambda mm, t: mm.dd[t] >= -mm.shiftFrac * mm.D[t])

        return m

    # ------------------------------------------------------------------
    @staticmethod
    def _cost_rule(mm):
        r"""Expected supply cost of the village.

        Probability-weighted recourse cost of the microgrid purchase at the
        leader price, of the public-grid draw, and of the unserved (critical)
        load. ``qsupp`` and ``dd`` are costless -- they are pinned by the
        demand-response and shift constraints.
        """
        return sum(
            mm.prob[s] * (mm.p[t] * mm.qMGO[t, s]
                          + mm.pPG[t] * mm.qPG[t, s]
                          + mm.pVOLL * mm.qcri[t, s])
            for t in mm.T for s in mm.S
        )

    # ------------------------------------------------------------------
    def build(self):
        """Construct the primal LP, store it on ``self.model``, return it."""
        m = self._base()
        m.cost = Objective(rule=self._cost_rule, sense=minimize)
        self.model = m
        return m

    # ------------------------------------------------------------------
    def solve(self, solver_name="gurobi", options=None, tee=False):
        """
        Solve the model (building it first if needed).

        Parameters
        ----------
        solver_name : str
            Pyomo solver name (default ``"gurobi"``).
        options : dict, optional
            Solver options passed through to the solver.
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

    # ------------------------------------------------------------------
    def objective_value(self):
        """Expected supply cost of the solved model."""
        return value(self.model.cost)
