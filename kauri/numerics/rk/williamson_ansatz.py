"""
Williamson/2N ansatz for explicit RK construction.
"""

from dataclasses import dataclass

import sympy

from kauri.numerics.rk.rk_ansatz import Ansatz


def generate_2n_polynomial_constraints(stages: int) -> list[sympy.core.basic.Basic]:
    """
    Generate quadratic 2N constraints in tableau coefficients.
    """
    equations: list[sympy.core.basic.Basic] = []
    for i_idx in range(2, stages):
        for j_idx in range(1, i_idx):
            a_ij = sympy.symbols(f"a{i_idx}{j_idx}")
            a_i_jm1 = sympy.symbols(f"a{i_idx}{j_idx - 1}")
            a_j_jm1 = sympy.symbols(f"a{j_idx}{j_idx - 1}")
            b_jm1 = sympy.symbols(f"b{j_idx - 1}")
            b_j = sympy.symbols(f"b{j_idx}")
            equation = sympy.expand(a_ij * (b_jm1 - a_j_jm1) - (a_i_jm1 - a_j_jm1) * b_j)
            equations.append(equation)
    return equations


def is_2n_tableau(
    a_matrix: list[list[sympy.core.basic.Basic | int | float]],
    b_vector: list[sympy.core.basic.Basic | int | float],
    tol: float = 1e-10,
) -> bool:
    """
    Check whether an explicit RK tableau satisfies the 2N polynomial constraints.
    """
    stages = len(b_vector)
    if len(a_matrix) != stages or any(len(row) != stages for row in a_matrix):
        raise ValueError("a_matrix must be a square stages x stages matrix")
    if tol < 0:
        raise ValueError("tol must be non-negative")

    substitutions: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
    for i_idx in range(stages):
        substitutions[sympy.symbols(f"b{i_idx}")] = sympy.sympify(b_vector[i_idx])
        for j_idx in range(stages):
            value = a_matrix[i_idx][j_idx]
            substitutions[sympy.symbols(f"a{i_idx}{j_idx}")] = sympy.sympify(value)

    for equation in generate_2n_polynomial_constraints(stages):
        if sympy.simplify(sympy.expand(equation.subs(list(substitutions.items())))) != 0:
            return False
    return True


def _a_expr_from_ab(stages: int) -> list[list[sympy.core.basic.Basic]]:
    a_expr: list[list[sympy.core.basic.Basic]] = [
        [sympy.Integer(0) for _ in range(stages)] for _ in range(stages)
    ]
    for i_idx in range(stages):
        for j_idx in range(i_idx - 1, -1, -1):
            if j_idx == i_idx - 1:
                a_expr[i_idx][j_idx] = sympy.symbols(f"B{j_idx}")
            else:
                a_expr[i_idx][j_idx] = sympy.simplify(
                    sympy.symbols(f"A{j_idx + 1}") * a_expr[i_idx][j_idx + 1]
                    + sympy.symbols(f"B{j_idx}")
                )
    return a_expr


def _b_expr_from_ab(stages: int) -> list[sympy.core.basic.Basic]:
    b_expr: list[sympy.core.basic.Basic] = [sympy.Integer(0) for _ in range(stages)]
    b_expr[stages - 1] = sympy.symbols(f"B{stages - 1}")
    for i_idx in range(stages - 2, -1, -1):
        b_expr[i_idx] = sympy.simplify(
            sympy.symbols(f"A{i_idx + 1}") * b_expr[i_idx + 1]
            + sympy.symbols(f"B{i_idx}")
        )
    return b_expr


@dataclass
class WilliamsonAnsatz(Ansatz):
    """
    Low-storage 2N ansatz with Williamson coefficients as primary unknowns.
    """

    name: str = "williamson_2n"
    validate_2n_polynomials: bool = True

    def extra_equations(self, stages: int) -> list[sympy.core.basic.Basic]:
        return []

    def extra_substitutions(self, stages: int) -> dict[sympy.Symbol, sympy.core.basic.Basic]:
        substitutions: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
        a_expr = _a_expr_from_ab(stages)
        b_expr = _b_expr_from_ab(stages)
        for i_idx in range(stages):
            substitutions[sympy.symbols(f"b{i_idx}")] = sympy.simplify(b_expr[i_idx])
            for j_idx in range(stages):
                substitutions[sympy.symbols(f"a{i_idx}{j_idx}")] = sympy.simplify(a_expr[i_idx][j_idx])
        return substitutions

    def solve_symbols(self, stages: int) -> list[sympy.Symbol] | None:
        return [sympy.symbols(f"B{i_idx}") for i_idx in range(stages)] + [
            sympy.symbols(f"A{i_idx}") for i_idx in range(1, stages)
        ]

    def post_validate(
        self,
        stages: int,
        named_solution: dict[str, sympy.core.basic.Basic],
        solver: str,
        tol: float,
    ) -> bool:
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if not self.validate_2n_polynomials:
            return True
        subs = {sympy.symbols(k): sympy.sympify(v) for k, v in named_solution.items()}
        return all(
            sympy.simplify(sympy.expand(eq.subs(list(subs.items())))) == 0
            for eq in generate_2n_polynomial_constraints(stages)
        )
