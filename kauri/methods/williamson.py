"""
Williamson 2N representations and commutator-free lift helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import sympy

from kauri.hopf_algebras.utils import _as_expr
from kauri.methods.rk import RK, ButcherTableau
from kauri.rk_builder.williamson import (
    verify_williamson_relations,
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
            raise ValueError("Parameter 'a' must be a square stages x stages matrix and b a vector of length stages")

    @cached_property
    def stages(self) -> int:
        return len(self.B)


@dataclass(frozen=True)
class WilliamsonRK:
    recursion: WilliamsonRecursion
    name: str

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
            A_symbols=[sympy.Integer(0)] + [self.recursion.A[i][i - 1] for i in range(1, self.recursion.stages)],
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
            stage_nodes=[sympy.simplify(value) for value in self.tableau.c],
            storage_a=[sympy.simplify(self.recursion.A[i][i - 1] if i > 0 else sympy.Integer(0)) for i in range(self.recursion.stages)],
            exp_coeffs=[sympy.simplify(value) for value in self.recursion.B],
            exponentials_per_update=self.recursion.stages,
        )


@dataclass(frozen=True)
class WilliamsonCF:
    base: WilliamsonRK
    name: str
    stage_nodes: list[sympy.core.basic.Basic]
    storage_a: list[sympy.core.basic.Basic]
    exp_coeffs: list[sympy.core.basic.Basic]
    exponentials_per_update: int

    def to_williamson_rk(self) -> WilliamsonRK:
        return self.base

    def to_text(self) -> str:
        lines: list[str] = [
            "=== Williamson Commutator-Free Method ===",
            f"name: {self.name}",
            f"stages: {self.base.stages}",
            f"exponentials per timestep: {self.exponentials_per_update}",
            "",
            "equations:",
            "  Y_0 = Y_t",
            "  ΔY_0 = 0",
        ]
        for stage_idx in range(self.base.stages):
            stage_num = stage_idx + 1
            lines.append(
                "  "
                f"K_{stage_num} = F(t + ({sympy.sstr(self.stage_nodes[stage_idx])})*h, "
                f"Y_{stage_num - 1})"
            )
            lines.append(
                "  "
                f"ΔY_{stage_num} = ({sympy.sstr(self.storage_a[stage_idx])})"
                f"*ΔY_{stage_num - 1} + h*K_{stage_num}"
            )
            lines.append(
                "  "
                f"Y_{stage_num} = exp(({sympy.sstr(self.exp_coeffs[stage_idx])})"
                f"*ΔY_{stage_num}) * Y_{stage_num - 1}"
            )
        lines.append(f"  Y_(t+h) = Y_{self.base.stages}")
        return "\n".join(lines)

    def elementary_weights_map(self):
        from kauri.planar_trees.mkw_truncated import MKWMap

        rk_weights = RK(tableau=self.base.tableau, name=self.base.name).elementary_weights_map()
        return MKWMap(lambda tree: rk_weights(tree.to_nonplanar_tree()))


