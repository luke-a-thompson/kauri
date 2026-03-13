"""Runge-Kutta method representation (no runtime stepping/integration)."""

import copy
from functools import cached_property

from kauri.hopf_algebras.bck import counit
from kauri.hopf_algebras.maps import Map, exact_weights, sign
from kauri.numerics.methods.tableau import ButcherTableau
from kauri.trees.gentrees import trees_of_order

_EXACT_SUBSTITUTION_LOG_MAP = exact_weights.log()


class RK:
    def __init__(self, tableau: ButcherTableau, name: str | None = None):
        self.name = name
        self.tableau = tableau
        self.deriv_dict = {}
        for i in range(self.s):
            self.deriv_dict[(i, repr(None))] = 1
            self.deriv_dict[(i, repr([]))] = 1

    @cached_property
    def s(self) -> int:
        return self.tableau.s

    @cached_property
    def explicit(self) -> bool:
        return self.tableau.explicit

    def __repr__(self):
        return f"RK(tableau={self.tableau!r}, name={self.name!r})"

    def _inverse(self):
        b_inv = [-self.tableau.b[i] for i in range(self.s)]
        a_inv = [
            [self.tableau.a[i][j] - self.tableau.b[j] for j in range(self.s)]
            for i in range(self.s)
        ]
        return RK(ButcherTableau(a=a_inv, b=b_inv))

    def reverse(self) -> "RK":
        return RK(
            ButcherTableau(
                a=[[-self.tableau.a[i][j] for j in range(self.s)] for i in range(self.s)],
                b=[-self.tableau.b[i] for i in range(self.s)],
            )
        )

    def adjoint(self) -> "RK":
        b_adj = [self.tableau.b[self.s - 1 - j] for j in range(self.s)]
        a_adj = [
            [
                self.tableau.b[self.s - 1 - j]
                - self.tableau.a[self.s - 1 - i][self.s - j - 1]
                for j in range(self.s)
            ]
            for i in range(self.s)
        ]
        return RK(ButcherTableau(a=a_adj, b=b_adj))

    def __mul__(self, other: "RK") -> "RK":
        s1 = other.s
        a1 = other.tableau.a
        b1 = other.tableau.b
        s2 = self.s
        a2 = self.tableau.a
        b2 = self.tableau.b
        a = [[a1[i][j] for j in range(s1)] + [0 for _ in range(s2)] for i in range(s1)]
        a += [[b1[j] for j in range(s1)] + [a2[i][j] for j in range(s2)] for i in range(s2)]
        return RK(ButcherTableau(a=a, b=list(b1) + list(b2)))

    def __pow__(self, exponent: int) -> "RK":
        if exponent == 0:
            return RK(ButcherTableau(a=[[0]], b=[0]))
        expn_ = exponent
        out = self._inverse() if exponent < 0 else copy.deepcopy(self)
        if exponent < 0:
            expn_ = -exponent
        for _ in range(expn_ - 1):
            out = out * self
        return out

    def _internal_weights(self, i, t_rep):
        return sum(
            self.tableau.a[i][j] * self._derivative_weights(j, t_rep) for j in range(self.s)
        )

    def _derivative_weights(self, i, t_rep):
        if (i, repr(t_rep)) in self.deriv_dict:
            return self.deriv_dict[(i, repr(t_rep))]
        out = 1
        for subtree in t_rep[:-1]:
            out *= self._internal_weights(i, subtree)
        self.deriv_dict[(i, repr(t_rep))] = out
        return out

    def _elementary_weights(self, t_rep):
        if t_rep is None:
            return 1
        return sum(self.tableau.b[i] * self._derivative_weights(i, t_rep) for i in range(self.s))

    def elementary_weights_map(self) -> Map:
        return Map(lambda x: self._elementary_weights(x.list_repr))

    def modified_equation_map(self) -> Map:
        return self.elementary_weights_map().modified_equation()

    def order(self, tol: float = 1e-10, limit: int = 10) -> int:
        theta = self.elementary_weights_map().log()
        n = 0
        while True:
            for t in trees_of_order(n):
                if abs(theta(t) - _EXACT_SUBSTITUTION_LOG_MAP(t)) > tol:
                    return n - 1
            if n >= limit:
                raise RuntimeError("Order equals or exceeds limit of " + str(limit))
            n += 1

    def antisymmetric_order(self, tol: float = 1e-10, limit: int = 10) -> int:
        ew = self.elementary_weights_map()
        m = (ew & sign) * ew
        n = 0
        while True:
            for t in trees_of_order(n):
                if abs(m(t) - counit(t)) > tol:
                    return n - 1
            if n >= limit:
                raise RuntimeError("Order equals or exceeds limit of " + str(limit))
            n += 1
