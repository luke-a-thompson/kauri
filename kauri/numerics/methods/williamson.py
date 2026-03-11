"""
Williamson 2N representations and commutator-free lift helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy

from kauri.hopf_algebras.utils import _as_expr
from kauri.numerics.ansatze.williamson import verify_williamson_relations
from kauri.numerics.methods.rk import RK


@dataclass(frozen=True)
class WilliamsonRK:
    stages: int
    a: list[list[sympy.core.basic.Basic]]
    b: list[sympy.core.basic.Basic]
    c: list[sympy.core.basic.Basic]
    A: list[sympy.core.basic.Basic]
    B: list[sympy.core.basic.Basic]
    name: str

    def to_cf(self) -> WilliamsonCF:
        return WilliamsonCF(
            base=self,
            name=f"{self.name}_cf",
            stage_nodes=[sympy.simplify(value) for value in self.c],
            storage_a=[sympy.simplify(value) for value in self.A],
            exp_coeffs=[sympy.simplify(value) for value in self.B],
            exponentials_per_update=self.stages,
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
                f"  K_{stage_num} = F(t + ({sympy.sstr(self.stage_nodes[stage_idx])})*h, Y_{stage_num - 1})"
            )
            lines.append(
                f"  ΔY_{stage_num} = ({sympy.sstr(self.storage_a[stage_idx])})*ΔY_{stage_num - 1} + h*K_{stage_num}"
            )
            lines.append(
                f"  Y_{stage_num} = exp(({sympy.sstr(self.exp_coeffs[stage_idx])})*ΔY_{stage_num}) * Y_{stage_num - 1}"
            )
        lines.append(f"  Y_(t+h) = Y_{self.base.stages}")
        return "\n".join(lines)

    def elementary_weights_map(self):
        from kauri.numerics.planar_trees.mkw_truncated import MKWMap
        from kauri.numerics.rk.rk import rk_symbolic_weight_for_tableau

        def eval_tree(tree):
            return rk_symbolic_weight_for_tableau(
                t=tree.to_nonplanar_tree(),
                a_tableau=self.base.a,
                b_weights=self.base.b,
                explicit=True,
            )

        return MKWMap(eval_tree)


def rk_to_williamson_2n(rk: RK) -> WilliamsonRK:
    if not rk.explicit:
        raise ValueError("Only explicit RK methods can be Williamson 2N in this converter.")
    stages: int = rk.s
    a: list[list[sympy.core.basic.Basic]] = [
        [sympy.nsimplify(rk.a[i][j], rational=True) for j in range(stages)] for i in range(stages)
    ]
    b: list[sympy.core.basic.Basic] = [sympy.nsimplify(value, rational=True) for value in rk.b]
    c: list[sympy.core.basic.Basic] = [
        sympy.simplify(sum((_as_expr(a[i][j]) for j in range(stages)), _as_expr(0)))
        for i in range(stages)
    ]
    B: list[sympy.core.basic.Basic] = [sympy.simplify(a[idx + 1][idx]) for idx in range(stages - 1)]
    B.append(sympy.simplify(b[-1]))

    A_params: list[sympy.core.basic.Basic] = [sympy.Integer(0)]
    for i_idx in range(1, stages):
        denominator = sympy.simplify(b[i_idx])
        if sympy.simplify(denominator) != 0:
            A_params.append(
                sympy.simplify(
                    (_as_expr(b[i_idx - 1]) - _as_expr(B[i_idx - 1])) / _as_expr(denominator)
                )
            )
            continue
        b_i = sympy.simplify(B[i_idx])
        if sympy.simplify(b_i) == 0:
            raise ValueError("Cannot infer Williamson A_i when both b_i and B_i are zero.")
        a_next_prev = sympy.simplify(
            b[i_idx - 1] if i_idx == stages - 1 else a[i_idx + 1][i_idx - 1]
        )
        A_params.append(
            sympy.simplify((_as_expr(a_next_prev) - _as_expr(c[i_idx])) / _as_expr(b_i))
        )

    verify_williamson_relations(a=a, b=b, A_params=A_params, B=B)
    method_name: str = rk.name if rk.name is not None else "unnamed_rk"
    return WilliamsonRK(
        stages=stages, a=a, b=b, c=c, A=A_params, B=B, name=f"{method_name}_williamson2n"
    )
