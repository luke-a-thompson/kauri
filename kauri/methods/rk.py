"""Runge-Kutta method representation (no runtime stepping/integration)."""

import copy
from dataclasses import dataclass
from functools import cached_property

import sympy

from kauri.hopf_algebras.bck import counit
from kauri.hopf_algebras.maps import Map, exact_weights, sign
from kauri.trees.gentrees import trees_of_order

_EXACT_SUBSTITUTION_LOG_MAP = exact_weights.log()
ExprLike = sympy.Expr | int | float


@dataclass(frozen=True)
class ButcherTableau:
    a: list[list[ExprLike]]
    b: list[ExprLike]

    def __post_init__(self) -> None:
        stages = len(self.b)
        if stages == 0:
            raise ValueError("Parameter 'b' must be a non-empty vector")
        if len(self.a) != stages or any(len(row) != stages for row in self.a):
            raise ValueError(
                "Parameter 'a' must be a square s x s matrix and b a vector of length s"
            )

    @cached_property
    def stages(self) -> int:
        return len(self.b)

    @cached_property
    def c(self) -> list[ExprLike]:
        return [sum(self.a[i][j] for j in range(self.stages)) for i in range(self.stages)]

    @cached_property
    def explicit(self) -> bool:
        for i in range(self.stages):
            for j in range(i, self.stages):
                if self.a[i][j]:
                    return False
        return True

    @cached_property
    def ssal(self) -> bool:
        """ """
        last_stage = self.stages - 1
        if self.a[last_stage][last_stage]:
            return False
        for j in range(self.stages):
            if self.a[last_stage][j] != self.b[j]:
                return False
        return True

    @cached_property
    def fsal(self) -> bool:
        if any(self.a[0][j] for j in range(self.stages)):
            return False
        return self.ssal

    def __str__(self) -> str:
        from kauri.rk_builder.rk_maker_format import format_tableau_text

        return format_tableau_text(self)


class RK:
    def __init__(self, tableau: ButcherTableau, name: str | None = None):
        self.name = name
        self.tableau = tableau
        self.deriv_dict: dict[tuple[int, str], ExprLike] = {}
        for i in range(self.s):
            self.deriv_dict[(i, repr(None))] = 1
            self.deriv_dict[(i, repr([]))] = 1

    @cached_property
    def s(self) -> int:
        return self.tableau.stages

    @cached_property
    def explicit(self) -> bool:
        return self.tableau.explicit

    def __repr__(self):
        return f"RK(tableau={self.tableau!r}, name={self.name!r})"

    def _inverse(self):
        b_inv = [-self.tableau.b[i] for i in range(self.s)]
        a_inv = [
            [self.tableau.a[i][j] - self.tableau.b[j] for j in range(self.s)] for i in range(self.s)
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
                self.tableau.b[self.s - 1 - j] - self.tableau.a[self.s - 1 - i][self.s - j - 1]
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

    def _internal_weights(self, i, t_rep) -> ExprLike:
        return sum(
            (self.tableau.a[i][j] * self._derivative_weights(j, t_rep) for j in range(self.s)), 0
        )

    def _derivative_weights(self, i, t_rep) -> ExprLike:
        if (i, repr(t_rep)) in self.deriv_dict:
            return self.deriv_dict[(i, repr(t_rep))]
        out: ExprLike = 1
        for subtree in t_rep[:-1]:
            out *= self._internal_weights(i, subtree)
        self.deriv_dict[(i, repr(t_rep))] = out
        return out

    def _elementary_weights(self, t_rep) -> ExprLike:
        if t_rep is None:
            return 1
        return sum(
            (self.tableau.b[i] * self._derivative_weights(i, t_rep) for i in range(self.s)), 0
        )

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
