"""
Williamson 2N representations and commutator-free lift helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import sympy

from kauri.methods.rk import RK, ButcherTableau
from kauri.rk_builder.williamson import (
    williamson_tableau_expressions,
)


@dataclass(frozen=True)
class WilliamsonRecursion:
    A: list[list[sympy.Expr]]
    B: list[sympy.Expr]

    def __post_init__(self) -> None:
        stages = len(self.B)
        if stages == 0:
            raise ValueError("Parameter 'b' must be a non-empty vector")
        if len(self.A) != stages or any(len(row) != stages for row in self.A):
            raise ValueError(
                "Parameter 'a' must be a square stages x stages matrix and b a vector of length stages"
            )

    @cached_property
    def stages(self) -> int:
        return len(self.B)

    def __str__(self) -> str:
        from kauri.rk_builder.rk_maker_format import format_williamson_recursion_text

        return format_williamson_recursion_text(self)

    def to_latex(self) -> str:
        from kauri.rk_builder.rk_maker_format import format_williamson_recursion_latex

        return format_williamson_recursion_latex(self)


@dataclass(frozen=True)
class WilliamsonRK:
    recursion: WilliamsonRecursion
    name: str
    embedded_from_penultimate: bool = False

    def __post_init__(self) -> None:
        if any(sympy.simplify(x) != 0 for x in self.recursion.A[0]):
            raise ValueError("Williamson methods require A0 = 0")

    @cached_property
    def stages(self) -> int:
        return self.recursion.stages

    @cached_property
    def tableau(self) -> ButcherTableau:
        a_expr, b_expr = williamson_tableau_expressions(
            stages=self.recursion.stages,
            A_symbols=[sympy.Integer(0)]
            + [self.recursion.A[i][i - 1] for i in range(1, self.recursion.stages)],
            B_symbols=self.recursion.B,
        )
        a = [
            [sympy.simplify(a_expr[i_idx][j_idx]) for j_idx in range(self.recursion.stages)]
            for i_idx in range(self.recursion.stages)
        ]
        b = [sympy.simplify(value) for value in b_expr]
        return ButcherTableau(a=a, b=b)

    def to_cf(self) -> WilliamsonCF:
        return WilliamsonCF(
            base=self,
            name=f"{self.name}_cf",
        )


@dataclass(frozen=True)
class WilliamsonCF:
    base: WilliamsonRK
    name: str

    @cached_property
    def stage_nodes(self) -> list[sympy.Expr]:
        return [sympy.simplify(value) for value in self.base.tableau.c]

    @cached_property
    def storage_a(self) -> list[sympy.Expr]:
        return [
            sympy.simplify(
                self.base.recursion.A[i_idx][i_idx - 1] if i_idx > 0 else sympy.Integer(0)
            )
            for i_idx in range(self.base.stages)
        ]

    @cached_property
    def exp_coeffs(self) -> list[sympy.Expr]:
        return [sympy.simplify(value) for value in self.base.recursion.B]

    @property
    def exponentials_per_update(self) -> int:
        return self.base.stages

    def to_williamson_rk(self) -> WilliamsonRK:
        return self.base

    def to_text(self) -> str:
        from kauri.rk_builder.rk_maker_format import format_williamson_recursion_text

        return format_williamson_recursion_text(
            self.base.recursion,
            name=self.name,
            title="=== Williamson Commutator-Free Method ===",
            mode="exp",
            exponentials_per_timestep=self.exponentials_per_update,
        )

    def to_latex(self) -> str:
        from kauri.rk_builder.rk_maker_format import format_williamson_recursion_latex

        return format_williamson_recursion_latex(
            self.base.recursion,
            name=self.name,
            heading="Williamson Commutator-Free Method",
            mode="exp",
            exponentials_per_timestep=self.exponentials_per_update,
        )

    def elementary_weights_map(self):
        from kauri.planar_trees.mkw_truncated import MKWMap

        rk_weights = RK(tableau=self.base.tableau, name=self.base.name).elementary_weights_map()
        return MKWMap(lambda tree: rk_weights(tree.to_nonplanar_tree()))

    def verify_antisymmetric_order(self, order: int) -> bool:
        from kauri.planar_trees.mkw_truncated import verify_mkw_ees

        return verify_mkw_ees(self.elementary_weights_map(), order)
