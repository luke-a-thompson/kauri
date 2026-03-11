"""
Williamson/2N ansatz and shared parameterisation helpers.
"""

from dataclasses import dataclass

import sympy

from kauri.hopf_algebras.utils import _as_expr
from kauri.numerics.ansatze.explicit import ExplicitAnsatz


def generate_2n_polynomial_constraints(stages: int) -> list[sympy.core.basic.Basic]:
    equations: list[sympy.core.basic.Basic] = []
    for i_idx in range(2, stages):
        for j_idx in range(1, i_idx):
            a_ij = sympy.symbols(f"a{i_idx}{j_idx}")
            a_i_jm1 = sympy.symbols(f"a{i_idx}{j_idx - 1}")
            a_j_jm1 = sympy.symbols(f"a{j_idx}{j_idx - 1}")
            b_jm1 = sympy.symbols(f"b{j_idx - 1}")
            b_j = sympy.symbols(f"b{j_idx}")
            equations.append(sympy.expand(a_ij * (b_jm1 - a_j_jm1) - (a_i_jm1 - a_j_jm1) * b_j))
    return equations


def is_2n_tableau(
    a_matrix: list[list[sympy.core.basic.Basic | int | float]],
    b_vector: list[sympy.core.basic.Basic | int | float],
) -> bool:
    stages = len(b_vector)
    if len(a_matrix) != stages or any(len(row) != stages for row in a_matrix):
        raise ValueError("a_matrix must be a square stages x stages matrix")
    substitutions: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
    for i_idx in range(stages):
        substitutions[sympy.symbols(f"b{i_idx}")] = sympy.sympify(b_vector[i_idx])
        for j_idx in range(stages):
            substitutions[sympy.symbols(f"a{i_idx}{j_idx}")] = sympy.sympify(a_matrix[i_idx][j_idx])
    return all(
        sympy.simplify(sympy.expand(eq.subs(list(substitutions.items())))) == 0
        for eq in generate_2n_polynomial_constraints(stages)
    )


def williamson_tableau_expressions(
    stages: int,
    A_symbols: list[sympy.Symbol] | None = None,
    B_symbols: list[sympy.Symbol] | None = None,
) -> tuple[list[list[sympy.core.basic.Basic]], list[sympy.core.basic.Basic]]:
    if stages <= 0:
        raise ValueError("stages must be positive")
    A_vals = (
        [sympy.Integer(0)] + [sympy.symbols(f"A{i}") for i in range(1, stages)]
        if A_symbols is None
        else A_symbols
    )
    B_vals = [sympy.symbols(f"B{i}") for i in range(stages)] if B_symbols is None else B_symbols
    if len(A_vals) != stages or len(B_vals) != stages:
        raise ValueError("A_symbols and B_symbols must both have length equal to stages")

    a_expr: list[list[sympy.core.basic.Basic]] = [
        [sympy.Integer(0) for _ in range(stages)] for _ in range(stages)
    ]
    for i_idx in range(stages):
        for j_idx in range(i_idx - 1, -1, -1):
            a_expr[i_idx][j_idx] = (
                sympy.simplify(B_vals[j_idx])
                if j_idx == i_idx - 1
                else sympy.simplify(
                    _as_expr(A_vals[j_idx + 1]) * _as_expr(a_expr[i_idx][j_idx + 1])
                    + _as_expr(B_vals[j_idx])
                )
            )

    b_expr: list[sympy.core.basic.Basic] = [sympy.Integer(0) for _ in range(stages)]
    b_expr[stages - 1] = sympy.simplify(B_vals[stages - 1])
    for i_idx in range(stages - 2, -1, -1):
        b_expr[i_idx] = sympy.simplify(
            _as_expr(A_vals[i_idx + 1]) * _as_expr(b_expr[i_idx + 1]) + _as_expr(B_vals[i_idx])
        )
    return a_expr, b_expr


def verify_williamson_relations(
    a: list[list[sympy.core.basic.Basic]],
    b: list[sympy.core.basic.Basic],
    A_params: list[sympy.core.basic.Basic],
    B: list[sympy.core.basic.Basic],
) -> None:
    stages = len(b)
    for i_idx in range(stages):
        for j_idx in range(i_idx):
            expected = (
                B[j_idx]
                if j_idx == i_idx - 1
                else sympy.simplify(
                    _as_expr(A_params[j_idx + 1]) * _as_expr(a[i_idx][j_idx + 1])
                    + _as_expr(B[j_idx])
                )
            )
            if sympy.simplify(_as_expr(a[i_idx][j_idx]) - _as_expr(expected)) != 0:
                raise ValueError("RK tableau does not satisfy Williamson recursion for a_ij.")
    for i_idx in range(stages - 1):
        expected_b = sympy.simplify(
            _as_expr(A_params[i_idx + 1]) * _as_expr(b[i_idx + 1]) + _as_expr(B[i_idx])
        )
        if sympy.simplify(_as_expr(b[i_idx]) - _as_expr(expected_b)) != 0:
            raise ValueError("RK tableau does not satisfy Williamson recursion for b_i.")


@dataclass
class WilliamsonAnsatz(ExplicitAnsatz):
    validate_2n_polynomials: bool = True

    def unknown_symbols(self, stages: int) -> list[sympy.Symbol]:
        return [sympy.symbols(f"B{i_idx}") for i_idx in range(stages)] + [
            sympy.symbols(f"A{i_idx}") for i_idx in range(1, stages)
        ]

    def tableau_substitutions(self, stages: int) -> dict[sympy.Symbol, sympy.core.basic.Basic]:
        substitutions: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
        a_expr, b_expr = williamson_tableau_expressions(stages=stages)
        for i_idx in range(stages):
            substitutions[sympy.symbols(f"b{i_idx}")] = sympy.simplify(b_expr[i_idx])
            for j_idx in range(stages):
                substitutions[sympy.symbols(f"a{i_idx}{j_idx}")] = sympy.simplify(
                    a_expr[i_idx][j_idx]
                )
        return substitutions

    def post_validate(
        self,
        stages: int,
        named_solution: dict[str, sympy.core.basic.Basic],
    ) -> bool:
        if not self.validate_2n_polynomials:
            return True
        subs = {sympy.symbols(k): sympy.sympify(v) for k, v in named_solution.items()}
        return all(
            sympy.simplify(sympy.expand(eq.subs(list(subs.items())))) == 0
            for eq in generate_2n_polynomial_constraints(stages)
        )
