"""Runge-Kutta method representation (no runtime stepping/integration)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import sympy

from kauri.hopf_algebras.bck import counit
from kauri.hopf_algebras.maps import Map, exact_weights, sign
from kauri.trees.gentrees import trees_of_order

if TYPE_CHECKING:
    from kauri.methods.williamson import WilliamsonRK
_EXACT_SUBSTITUTION_LOG_MAP = exact_weights.log()
ExprLike = sympy.Expr | int | float


@dataclass(frozen=True)
class ButcherTableau:
    a: list[list[ExprLike]]
    b: list[ExprLike]
    b_hat: list[ExprLike] | None = None

    def __post_init__(self) -> None:
        stages = len(self.b)
        if stages == 0:
            raise ValueError("Parameter 'b' must be a non-empty vector")
        if len(self.a) != stages or any(len(row) != stages for row in self.a):
            raise ValueError(
                "Parameter 'a' must be a square s x s matrix and b a vector of length s"
            )
        if self.b_hat is not None and len(self.b_hat) != stages:
            raise ValueError("Parameter 'b_hat' must be a vector of length s")

    @cached_property
    def stages(self) -> int:
        return len(self.b)

    @cached_property
    def c(self) -> list[ExprLike]:
        return [sum(self.a[i][j] for j in range(self.stages)) for i in range(self.stages)]

    @cached_property
    def embedded(self) -> bool:
        return self.b_hat is not None

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

    def to_latex(self) -> str:
        from kauri.rk_builder.rk_maker_format import format_tableau_latex

        return format_tableau_latex(self)


class RK:
    def __init__(self, tableau: ButcherTableau, name: str | None = None):
        self.name = name
        self.tableau = tableau
        self.deriv_dict: dict[tuple[int, str], ExprLike] = {}
        for i in range(self.stages):
            self.deriv_dict[(i, repr(None))] = 1
            self.deriv_dict[(i, repr([]))] = 1

    @cached_property
    def stages(self) -> int:
        return self.tableau.stages

    @cached_property
    def explicit(self) -> bool:
        return self.tableau.explicit

    def __str__(self):
        return f"RK method: {self.name} with tableau:\n{self.tableau}"

    def to_williamson(self) -> WilliamsonRK:
        from kauri.methods.williamson import WilliamsonRecursion, WilliamsonRK
        from kauri.rk_builder.williamson import verify_williamson_relations

        if not self.explicit:
            raise ValueError("Only explicit RK methods can be Williamson 2N in this converter.")
        a: list[list[sympy.Expr]] = [
            [sympy.nsimplify(self.tableau.a[i][j], rational=True) for j in range(self.stages)]
            for i in range(self.stages)
        ]
        b: list[sympy.Expr] = [sympy.nsimplify(value, rational=True) for value in self.tableau.b]
        B: list[sympy.Expr] = [sympy.simplify(a[idx + 1][idx]) for idx in range(self.stages - 1)]
        B.append(sympy.simplify(b[-1]))

        A_params: list[sympy.Expr] = [sympy.Integer(0)]
        for i_idx in range(1, self.stages):
            b_i = sympy.simplify(b[i_idx])
            if b_i != 0:
                A_params.append(sympy.simplify((b[i_idx - 1] - B[i_idx - 1]) / b_i))
                continue
            B_i = sympy.simplify(B[i_idx])
            if B_i == 0:
                raise ValueError("Cannot infer Williamson A_i when both b_i and B_i are zero.")
            a_next_prev = sympy.simplify(
                b[i_idx - 1] if i_idx == self.stages - 1 else a[i_idx + 1][i_idx - 1]
            )
            A_params.append(
                sympy.simplify(
                    (a_next_prev - sum((a[i_idx][j] for j in range(self.stages)), sympy.Integer(0)))
                    / B_i
                )
            )

        verify_williamson_relations(a=a, b=b, A_params=A_params, B=B)
        recursion_a: list[list[sympy.Expr]] = [
            [sympy.Integer(0)] * self.stages for _ in range(self.stages)
        ]
        for i in range(1, self.stages):
            recursion_a[i][i - 1] = A_params[i]
        method_name = self.name if self.name is not None else "unnamed_rk"
        return WilliamsonRK(
            recursion=WilliamsonRecursion(A=recursion_a, B=B),
            name=f"{method_name}_williamson2n",
        )

    def _inverse(self):
        b_inv = [-self.tableau.b[i] for i in range(self.stages)]
        a_inv = [
            [self.tableau.a[i][j] - self.tableau.b[j] for j in range(self.stages)]
            for i in range(self.stages)
        ]
        return RK(ButcherTableau(a=a_inv, b=b_inv))

    def reverse(self) -> RK:
        return RK(
            ButcherTableau(
                a=[[-self.tableau.a[i][j] for j in range(self.stages)] for i in range(self.stages)],
                b=[-self.tableau.b[i] for i in range(self.stages)],
            )
        )

    def adjoint(self) -> RK:
        b_adj = [self.tableau.b[self.stages - 1 - j] for j in range(self.stages)]
        a_adj = [
            [
                self.tableau.b[self.stages - 1 - j]
                - self.tableau.a[self.stages - 1 - i][self.stages - j - 1]
                for j in range(self.stages)
            ]
            for i in range(self.stages)
        ]
        return RK(ButcherTableau(a=a_adj, b=b_adj))

    def __mul__(self, other: RK) -> RK:
        s1 = other.stages
        a1 = other.tableau.a
        b1 = other.tableau.b
        s2 = self.stages
        a2 = self.tableau.a
        b2 = self.tableau.b
        a = [[a1[i][j] for j in range(s1)] + [0 for _ in range(s2)] for i in range(s1)]
        a += [[b1[j] for j in range(s1)] + [a2[i][j] for j in range(s2)] for i in range(s2)]
        return RK(ButcherTableau(a=a, b=list(b1) + list(b2)))

    def __pow__(self, exponent: int) -> RK:
        if exponent == 0:
            return RK(ButcherTableau(a=[[0]], b=[0]))
        expn_ = exponent
        out = (
            self._inverse()
            if exponent < 0
            else RK(
                ButcherTableau(
                    a=copy.deepcopy(self.tableau.a),
                    b=copy.deepcopy(self.tableau.b),
                ),
                name=self.name,
            )
        )
        if exponent < 0:
            expn_ = -exponent
        for _ in range(expn_ - 1):
            out = out * self
        return out

    def _internal_weights(self, i, t_rep) -> ExprLike:
        return sum(
            (self.tableau.a[i][j] * self._derivative_weights(j, t_rep) for j in range(self.stages)),
            0,
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
            (self.tableau.b[i] * self._derivative_weights(i, t_rep) for i in range(self.stages)), 0
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
