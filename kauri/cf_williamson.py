"""
Williamson 2N representation and commutator-free lift helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import sympy

from kauri.rk import RK


def _as_expr(value: sympy.core.basic.Basic | int | float) -> sympy.Expr:
    return cast(sympy.Expr, sympy.sympify(value))


@dataclass(frozen=True)
class Williamson2N:
    """
    Williamson 2N parameters coupled with equivalent RK tableau.
    """

    stages: int
    a: list[list[sympy.core.basic.Basic]]
    b: list[sympy.core.basic.Basic]
    c: list[sympy.core.basic.Basic]
    A: list[sympy.core.basic.Basic]
    B: list[sympy.core.basic.Basic]
    name: str


@dataclass(frozen=True)
class Williamson2NCF:
    """
    Commutator-free lift of a Williamson 2N method.
    """

    base: Williamson2N
    name: str


def rk_to_williamson_2n(rk: RK, tol: float = 1e-12) -> Williamson2N:
    """
    Convert an explicit RK method into Williamson 2N coefficients.
    """
    if not isinstance(rk, RK):
        raise TypeError(f"rk must be RK, not {type(rk)}")
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

    B: list[sympy.core.basic.Basic] = []
    for idx in range(stages - 1):
        B.append(sympy.simplify(a[idx + 1][idx]))
    B.append(sympy.simplify(b[-1]))

    A_params: list[sympy.core.basic.Basic] = [sympy.Integer(0)]
    for i_idx in range(1, stages):
        denominator: sympy.core.basic.Basic = sympy.simplify(b[i_idx])
        if sympy.simplify(denominator) != 0:
            numerator = _as_expr(b[i_idx - 1]) - _as_expr(B[i_idx - 1])
            value = sympy.simplify(numerator / _as_expr(denominator))
            A_params.append(value)
            continue
        b_i: sympy.core.basic.Basic = sympy.simplify(B[i_idx])
        if sympy.simplify(b_i) == 0:
            raise ValueError("Cannot infer Williamson A_i when both b_i and B_i are zero.")
        if i_idx == stages - 1:
            a_next_prev = sympy.simplify(b[i_idx - 1])
        else:
            a_next_prev = sympy.simplify(a[i_idx + 1][i_idx - 1])
        value = sympy.simplify((_as_expr(a_next_prev) - _as_expr(c[i_idx])) / _as_expr(b_i))
        A_params.append(value)

    _verify_williamson_relations(a=a, b=b, A_params=A_params, B=B, tol=tol)
    method_name: str = rk.name if rk.name is not None else "unnamed_rk"
    return Williamson2N(
        stages=stages,
        a=a,
        b=b,
        c=c,
        A=A_params,
        B=B,
        name=f"{method_name}_williamson2n",
    )


def lift_to_cf(method: Williamson2N) -> Williamson2NCF:
    """
    Lift a Williamson 2N method to its commutator-free Lie-group form.
    """
    if not isinstance(method, Williamson2N):
        raise TypeError(f"method must be Williamson2N, not {type(method)}")
    return Williamson2NCF(base=method, name=f"{method.name}_cf")


def cf_to_rk_tableau(
    method: Williamson2NCF,
) -> tuple[list[list[sympy.core.basic.Basic]], list[sympy.core.basic.Basic]]:
    """
    Extract RK tableau carried by the lifted CF method.
    """
    if not isinstance(method, Williamson2NCF):
        raise TypeError(f"method must be Williamson2NCF, not {type(method)}")
    return method.base.a, method.base.b


def _verify_williamson_relations(
    a: list[list[sympy.core.basic.Basic]],
    b: list[sympy.core.basic.Basic],
    A_params: list[sympy.core.basic.Basic],
    B: list[sympy.core.basic.Basic],
    tol: float,
) -> None:
    stages: int = len(b)
    for i_idx in range(stages):
        for j_idx in range(i_idx):
            if j_idx == i_idx - 1:
                expected = B[j_idx]
            else:
                expected = sympy.simplify(
                    _as_expr(A_params[j_idx + 1]) * _as_expr(a[i_idx][j_idx + 1])
                    + _as_expr(B[j_idx])
                )
            residual = sympy.simplify(_as_expr(a[i_idx][j_idx]) - _as_expr(expected))
            if not _is_zero_like(residual, tol=tol):
                raise ValueError("RK tableau does not satisfy Williamson recursion for a_ij.")
    for i_idx in range(stages - 1):
        expected_b = sympy.simplify(
            _as_expr(A_params[i_idx + 1]) * _as_expr(b[i_idx + 1]) + _as_expr(B[i_idx])
        )
        residual_b = sympy.simplify(_as_expr(b[i_idx]) - _as_expr(expected_b))
        if not _is_zero_like(residual_b, tol=tol):
            raise ValueError("RK tableau does not satisfy Williamson recursion for b_i.")


def _is_zero_like(expr: sympy.core.basic.Basic, tol: float) -> bool:
    simplified = sympy.simplify(sympy.expand(expr))
    if simplified == 0:
        return True
    if len(simplified.free_symbols) > 0:
        return False
    return abs(float(sympy.N(simplified, 30))) <= tol
