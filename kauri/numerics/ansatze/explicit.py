"""
Explicit RK tableau ansatz and equation helpers.
"""

import sympy

from kauri.hopf_algebras.bck import counit
from kauri.hopf_algebras.maps import sign
from kauri.numerics.ansatze.base import BaseAnsatz
from kauri.numerics.rk.rk import _rk_symbolic_weights_map, rk_order_cond
from kauri.trees.gentrees import trees_up_to_order
from kauri.trees.trees import Tree


def explicit_unknown_symbols(stages: int) -> tuple[list[sympy.Symbol], list[sympy.Symbol]]:
    if stages <= 0:
        raise ValueError("stages must be positive")
    a_symbols = [sympy.symbols(f"a{i}{j}") for i in range(stages) for j in range(i)]
    b_symbols = [sympy.symbols(f"b{i}") for i in range(stages)]
    return a_symbols, b_symbols


def generate_explicit_order_equations(
    order: int, stages: int, rationalise: bool = True
) -> tuple[list[sympy.core.basic.Basic], list[Tree]]:
    if order <= 0:
        raise ValueError("order must be positive")
    trees = [t for t in trees_up_to_order(order) if t != Tree(None)]
    equations = [
        sympy.expand(rk_order_cond(t, stages, explicit=True, rationalise=rationalise))
        for t in trees
    ]
    return equations, trees


def generate_explicit_antisymmetric_equations(
    antisymmetric_order: int, stages: int, rationalise: bool = True
) -> tuple[list[sympy.core.basic.Basic], list[Tree]]:
    if antisymmetric_order <= 0:
        raise ValueError("antisymmetric_order must be positive")
    phi = _rk_symbolic_weights_map(stages, explicit=True)
    m = (phi & sign) * phi
    trees: list[Tree] = [t for t in trees_up_to_order(antisymmetric_order) if t != Tree(None)]
    equations: list[sympy.core.basic.Basic] = [
        sympy.expand(sympy.sympify(m(t) - counit(t))) for t in trees
    ]
    if rationalise:
        equations = [
            sympy.sympify(sympy.nsimplify(equation, tolerance=1e-10, rational=True))
            for equation in equations
        ]
    return equations, trees


def construct_explicit_tableau(
    stages: int, symbol_values: dict[str, sympy.core.basic.Basic]
) -> tuple[list[list[float]], list[float]]:
    a_matrix: list[list[float]] = [[0.0 for _ in range(stages)] for _ in range(stages)]
    b_vector: list[float] = [0.0 for _ in range(stages)]
    for i in range(stages):
        for j in range(i):
            a_matrix[i][j] = float(sympy.N(symbol_values.get(f"a{i}{j}", sympy.Integer(0)), 20))
        b_vector[i] = float(sympy.N(symbol_values.get(f"b{i}", sympy.Integer(0)), 20))
    return a_matrix, b_vector


class ExplicitAnsatz(BaseAnsatz):
    def unknown_symbols(self, stages: int) -> list[sympy.Symbol]:
        a_symbols, b_symbols = explicit_unknown_symbols(stages)
        return a_symbols + b_symbols

    def base_equations(
        self,
        order: int,
        stages: int,
        antisymmetric_order: int | None = None,
    ) -> tuple[list[sympy.core.basic.Basic], list[Tree]]:
        equations, trees = generate_explicit_order_equations(order, stages, rationalise=True)
        if antisymmetric_order is not None:
            antisymmetric_equations, antisymmetric_trees = (
                generate_explicit_antisymmetric_equations(
                    antisymmetric_order, stages, rationalise=True
                )
            )
            equations = equations + antisymmetric_equations
            trees = trees + antisymmetric_trees
        return equations, trees

    def build_tableau(
        self,
        stages: int,
        named_solution: dict[str, sympy.core.basic.Basic],
    ) -> tuple[list[list[float]], list[float]]:
        return construct_explicit_tableau(stages, named_solution)

        