r"""
KKT + Big-M reformulation of the lower level.

This is the standalone counterpart of the follower block that
:class:`model.bilevel_mpec.BilevelMPEC` embeds: the KKT system of the LP in
:mod:`model.lower_level`, with the complementarity conditions linearized by
disjunctive Big-M constraints.

Because the follower LP is convex, its KKT system is necessary AND sufficient
for optimality, so at any fixed leader price the two models must attain the
SAME cost. That identity is the check run by ``run_lowerlevel.py --validate``:
a mismatch (or an infeasible KKT model) means the KKT derivation or a Big-M is
wrong. The check is self-policing -- a too-small Big-M makes the KKT model
infeasible or suboptimal, so it fails rather than passing silently.

Dual variables (same symbols as ``model.bilevel_mpec``):
    lam  = \lambda_{t,s} (free)  beta = \beta_{t,s} >= 0
    mu   = \mu_t (free)          gp/gm = \gamma^{+}/\gamma^{-}_t >= 0
    rho  = \rho (free, scalar)
"""

from pyomo.environ import (
    Var, Constraint, Objective,
    NonNegativeReals, Reals, Binary, minimize,
)

from model.lower_level import LowerLevel


class LowerLevelKKT(LowerLevel):
    r"""
    KKT + Big-M reformulation of :class:`model.lower_level.LowerLevel`.

    Sets, parameters, primal variables and primal feasibility are inherited
    unchanged; this class adds the duals, the stationarity conditions and the
    linearized complementarities. Complementarity ``0 <= a _|_ b >= 0`` becomes
    ``a <= M_a z`` and ``b <= M_b (1 - z)`` with ``z`` binary:

      stationarity (complementarity form, 0 <= x _|_ reduced cost >= 0):
        qMGO[t,s]  _|_  prob_s p_t      - lam[t,s]
        qPG[t,s]   _|_  prob_s pPG_t    - lam[t,s] + beta[t,s]
        qsupp[t]   _|_  mu[t]           - sum_s lam[t,s]
        qcri[t,s]  _|_  prob_s pVOLL    - lam[t,s]
        dd[t] free -> equality:  sum_s lam[t,s] + gp[t] - gm[t] + rho = 0

      complementarity of the inequality constraints:
        beta[t,s]  _|_  QbarPG[t,s] - qPG[t,s]
        gp[t]      _|_  shiftFrac D_t - dd[t]
        gm[t]      _|_  shiftFrac D_t + dd[t]

    Needs the ``BigM`` key in ``data`` on top of what :class:`LowerLevel`
    requires.

    The objective is the primal cost -- redundant given KKT feasibility (the
    feasible set is already the LP's optimal set), but it keeps the model
    well-posed and makes the cost directly comparable to :class:`LowerLevel`.
    """

    # ---- reduced costs (the >= 0 sides of the complementarities) ----
    @staticmethod
    def _rc_mgo(mm, t, s):
        return mm.prob[s] * mm.p[t] - mm.lam[t, s]

    @staticmethod
    def _rc_pg(mm, t, s):
        return mm.prob[s] * mm.pPG[t] - mm.lam[t, s] + mm.beta[t, s]

    @staticmethod
    def _rc_supp(mm, t):
        return mm.mu[t] - sum(mm.lam[t, s] for s in mm.S)

    @staticmethod
    def _rc_cri(mm, t, s):
        return mm.prob[s] * mm.pVOLL - mm.lam[t, s]

    # ------------------------------------------------------------------
    def build(self):
        """Construct the KKT model, store it on ``self.model``, return it."""
        m = self._base()
        M = self.data["BigM"]
        rc_mgo, rc_pg = self._rc_mgo, self._rc_pg
        rc_supp, rc_cri = self._rc_supp, self._rc_cri

        # ---- dual variables ----
        m.lam  = Var(m.T, m.S, domain=Reals)              # demand balance (free)
        m.mu   = Var(m.T,      domain=Reals)              # demand response (free)
        m.rho  = Var(          domain=Reals)              # shift balance (free)
        m.beta = Var(m.T, m.S, domain=NonNegativeReals)   # PG capacity
        m.gp   = Var(m.T,      domain=NonNegativeReals)   # shift upper
        m.gm   = Var(m.T,      domain=NonNegativeReals)   # shift lower

        # ---- complementarity binaries ----
        m.zMGO  = Var(m.T, m.S, domain=Binary)
        m.zPG   = Var(m.T, m.S, domain=Binary)
        m.zsupp = Var(m.T,      domain=Binary)
        m.zcri  = Var(m.T, m.S, domain=Binary)
        m.zbeta = Var(m.T, m.S, domain=Binary)
        m.zgp   = Var(m.T,      domain=Binary)
        m.zgm   = Var(m.T,      domain=Binary)

        # ---- stationarity for the free variable dd ----
        m.stat_shift = Constraint(
            m.T, rule=lambda mm, t:
                sum(mm.lam[t, s] for s in mm.S) + mm.gp[t] - mm.gm[t] + mm.rho == 0)

        # ---- dual feasibility: reduced costs >= 0 ----
        m.rc_mgo_geq  = Constraint(m.T, m.S, rule=lambda mm, t, s: rc_mgo(mm, t, s) >= 0)
        m.rc_pg_geq   = Constraint(m.T, m.S, rule=lambda mm, t, s: rc_pg(mm, t, s) >= 0)
        m.rc_supp_geq = Constraint(m.T,      rule=lambda mm, t: rc_supp(mm, t) >= 0)
        m.rc_cri_geq  = Constraint(m.T, m.S, rule=lambda mm, t, s: rc_cri(mm, t, s) >= 0)

        # =====================================================================
        # Big-M complementarity:  a <= M_a z ,  b <= M_b (1 - z)
        # =====================================================================
        # 1) qMGO _|_ rc_mgo
        m.cc_mgo_a = Constraint(m.T, m.S, rule=lambda mm, t, s: mm.qMGO[t, s] <= M["qMGO"] * mm.zMGO[t, s])
        m.cc_mgo_b = Constraint(m.T, m.S, rule=lambda mm, t, s: rc_mgo(mm, t, s) <= M["rc"] * (1 - mm.zMGO[t, s]))

        # 2) qPG _|_ rc_pg
        m.cc_pg_a = Constraint(m.T, m.S, rule=lambda mm, t, s: mm.qPG[t, s] <= M["qPG"] * mm.zPG[t, s])
        m.cc_pg_b = Constraint(m.T, m.S, rule=lambda mm, t, s: rc_pg(mm, t, s) <= M["rc"] * (1 - mm.zPG[t, s]))

        # 3) qsupp _|_ rc_supp
        m.cc_supp_a = Constraint(m.T, rule=lambda mm, t: mm.qsupp[t] <= M["qsupp"] * mm.zsupp[t])
        m.cc_supp_b = Constraint(m.T, rule=lambda mm, t: rc_supp(mm, t) <= M["rc"] * (1 - mm.zsupp[t]))

        # 4) qcri _|_ rc_cri
        m.cc_cri_a = Constraint(m.T, m.S, rule=lambda mm, t, s: mm.qcri[t, s] <= M["qcri"] * mm.zcri[t, s])
        m.cc_cri_b = Constraint(m.T, m.S, rule=lambda mm, t, s: rc_cri(mm, t, s) <= M["rc"] * (1 - mm.zcri[t, s]))

        # 5) beta _|_ (QbarPG - qPG)     (PG capacity slack; pg_cap is in _base)
        m.cc_beta_a = Constraint(m.T, m.S, rule=lambda mm, t, s: mm.beta[t, s] <= M["beta"] * mm.zbeta[t, s])
        m.cc_beta_b = Constraint(m.T, m.S, rule=lambda mm, t, s:
                                 mm.QbarPG[t, s] - mm.qPG[t, s] <= M["slackPG"] * (1 - mm.zbeta[t, s]))

        # 6) gp _|_ (shiftFrac*D - dd)   (shift upper slack; shift_up is in _base)
        m.cc_gp_a = Constraint(m.T, rule=lambda mm, t: mm.gp[t] <= M["gamma"] * mm.zgp[t])
        m.cc_gp_b = Constraint(m.T, rule=lambda mm, t:
                               mm.shiftFrac * mm.D[t] - mm.dd[t] <= M["slackShift"] * (1 - mm.zgp[t]))

        # 7) gm _|_ (shiftFrac*D + dd)   (shift lower slack; shift_lo is in _base)
        m.cc_gm_a = Constraint(m.T, rule=lambda mm, t: mm.gm[t] <= M["gamma"] * mm.zgm[t])
        m.cc_gm_b = Constraint(m.T, rule=lambda mm, t:
                               mm.shiftFrac * mm.D[t] + mm.dd[t] <= M["slackShift"] * (1 - mm.zgm[t]))

        m.cost = Objective(rule=self._cost_rule, sense=minimize)

        self.model = m
        return m
