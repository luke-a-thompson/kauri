"""
Williamson/2N symbolic helpers for RK construction.
"""

import sympy

from kauri.hopf_algebras.utils import _as_expr


def verify_williamson_relations(
    a: list[list[sympy.Expr]],
    b: list[sympy.Expr],
    A_params: list[sympy.Expr],
    B: list[sympy.Expr],
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


def williamson_unknown_symbols(stages: int) -> list[sympy.Symbol]:
    if stages <= 0:
        raise ValueError("stages must be positive")
    return [sympy.symbols(f"B{i_idx}") for i_idx in range(stages)] + [
        sympy.symbols(f"A{i_idx}") for i_idx in range(1, stages)
    ]


def generate_2n_polynomial_constraints(stages: int) -> list[sympy.Expr]:
    equations: list[sympy.Expr] = []
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
        for j_idx in range(i_idx):
            substitutions[sympy.symbols(f"a{i_idx}{j_idx}")] = sympy.sympify(a_matrix[i_idx][j_idx])
    return all(
        sympy.simplify(sympy.expand(eq.subs(list(substitutions.items())))) == 0
        for eq in generate_2n_polynomial_constraints(stages)
    )


def williamson_tableau_expressions(
    stages: int,
    A_symbols: list[sympy.Expr] | None = None,
    B_symbols: list[sympy.Expr] | None = None,
) -> tuple[list[list[sympy.Expr]], list[sympy.Expr]]:
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

    a_expr: list[list[sympy.Expr]] = [
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

    b_expr: list[sympy.Expr] = [sympy.Integer(0) for _ in range(stages)]
    b_expr[stages - 1] = sympy.simplify(B_vals[stages - 1])
    for i_idx in range(stages - 2, -1, -1):
        b_expr[i_idx] = sympy.simplify(
            _as_expr(A_vals[i_idx + 1]) * _as_expr(b_expr[i_idx + 1]) + _as_expr(B_vals[i_idx])
        )
    return a_expr, b_expr
