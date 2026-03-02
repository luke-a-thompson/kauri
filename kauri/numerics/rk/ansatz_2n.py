"""
2N-storage ansatz utilities (Bazavov 2025).
"""

from dataclasses import dataclass

import sympy

from kauri.hopf_algebras.utils import _as_expr, _is_zero_like_expr
from kauri.numerics.rk.rk_ansatz import Ansatz


def generate_2n_aform_constraints(stages: int) -> list[sympy.core.basic.Basic]:
    """
    Generate 2N-storage constraints in a-form (Theorem 2, form II).
    """
    if not isinstance(stages, int):
        raise TypeError("stages must be int, not " + str(type(stages)))
    if stages <= 0:
        raise ValueError("stages must be positive")

    equations: list[sympy.core.basic.Basic] = []
    for i_idx in range(2, stages):
        for j_idx in range(1, i_idx):
            a_ij = _as_expr(sympy.symbols(f"a{i_idx}{j_idx}"))
            a_i_jm1 = _as_expr(sympy.symbols(f"a{i_idx}{j_idx - 1}"))
            a_j_jm1 = _as_expr(sympy.symbols(f"a{j_idx}{j_idx - 1}"))
            b_jm1 = _as_expr(sympy.symbols(f"b{j_idx - 1}"))
            b_j = _as_expr(sympy.symbols(f"b{j_idx}"))
            equation = sympy.expand(a_ij * (b_jm1 - a_j_jm1) - (a_i_jm1 - a_j_jm1) * b_j)
            equations.append(equation)
    return equations


def _a_from_named(
    named_solution: dict[str, sympy.core.basic.Basic], i_idx: int, j_idx: int
) -> sympy.core.basic.Basic:
    if i_idx <= j_idx:
        return sympy.Integer(0)
    return sympy.sympify(named_solution.get(f"a{i_idx}{j_idx}", sympy.Integer(0)))


def _b_from_named(
    named_solution: dict[str, sympy.core.basic.Basic], i_idx: int
) -> sympy.core.basic.Basic:
    return sympy.sympify(named_solution.get(f"b{i_idx}", sympy.Integer(0)))


def _alpha_beta_from_named(
    stages: int, named_solution: dict[str, sympy.core.basic.Basic]
) -> tuple[dict[tuple[int, int], sympy.core.basic.Basic], dict[int, sympy.core.basic.Basic]]:
    alpha: dict[tuple[int, int], sympy.core.basic.Basic] = {}
    beta: dict[int, sympy.core.basic.Basic] = {}

    for i_idx in range(1, stages):
        for j_idx in range(i_idx):
            alpha[(i_idx, j_idx)] = sympy.simplify(
                _as_expr(_a_from_named(named_solution, i_idx, j_idx))
                - _as_expr(_a_from_named(named_solution, i_idx - 1, j_idx))
            )

    for j_idx in range(stages):
        beta[j_idx] = sympy.simplify(
            _as_expr(_b_from_named(named_solution, j_idx))
            - _as_expr(_a_from_named(named_solution, stages - 1, j_idx))
        )
    return alpha, beta


@dataclass
class TwoNStorageAnsatz(Ansatz):
    """
    2N-storage ansatz using a-form quadratic constraints.
    """

    name: str = "2n_storage"
    validate_alpha: bool = True

    def extra_equations(self, stages: int) -> list[sympy.core.basic.Basic]:
        return generate_2n_aform_constraints(stages)

    def extra_substitutions(self, stages: int) -> dict[sympy.Symbol, sympy.core.basic.Basic]:
        if stages <= 0:
            raise ValueError("stages must be positive")
        return {}

    def post_validate(
        self,
        stages: int,
        named_solution: dict[str, sympy.core.basic.Basic],
        solver: str,
        tol: float,
    ) -> bool:
        if stages <= 0:
            raise ValueError("stages must be positive")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        _ = solver
        if not self.validate_alpha:
            return True

        alpha, beta = _alpha_beta_from_named(stages=stages, named_solution=named_solution)
        for j_idx in range(0, stages - 2):
            for i_idx in range(j_idx + 2, stages):
                expression = sympy.expand(
                    _as_expr(beta[j_idx + 1]) * _as_expr(alpha[(i_idx, j_idx)])
                    - _as_expr(beta[j_idx]) * _as_expr(alpha[(i_idx, j_idx + 1)])
                )
                if not _is_zero_like_expr(value=expression, tol=tol):
                    return False
        return True
