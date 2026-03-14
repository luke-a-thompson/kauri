"""
Generator-layer symbolic tools for RK order conditions and elementary weights.
"""

import sympy

from kauri.hopf_algebras.maps import Map, exact_weights
from kauri.methods.rk import RK, ButcherTableau, ExprLike
from kauri.trees.trees import Forest, ForestSum, Tree


def _rk_symbolic_weights_map(
    s: int,
    explicit: bool = False,
    a_mask: list | None = None,
    b_mask: list | None = None,
    weight_symbols: list[ExprLike] | None = None,
) -> Map:
    if a_mask is None:
        a_mask = [[1] * s for _ in range(s)]
    if b_mask is None:
        b_mask = [1] * s
    if weight_symbols is not None and len(weight_symbols) != s:
        raise ValueError("weight_symbols must have length s")
    a: list[list[ExprLike]] = [
        [
            sympy.symbols(f"a{i}{j}")
            if (a_mask[i][j] and (not explicit or j < i))
            else sympy.Integer(0)
            for j in range(s)
        ]
        for i in range(s)
    ]
    b: list[ExprLike] = [
        (
            (weight_symbols[j] if weight_symbols is not None else sympy.symbols(f"b{j}"))
            if b_mask[j]
            else sympy.Integer(0)
        )
        for j in range(s)
    ]
    return RK(ButcherTableau(a=a, b=b)).elementary_weights_map()


def _normalize_tree_like_input(
    t: Tree | Forest | ForestSum | int | float,
) -> Tree | Forest | ForestSum:
    if isinstance(t, (int, float)):
        return t * Tree(None).as_forest_sum()
    return t


def _finalize_symbolic_output(
    expression: sympy.core.basic.Basic,
    rationalise: bool,
    mathematica_code: bool,
) -> sympy.core.basic.Basic | str:
    out: sympy.core.basic.Basic | str = expression
    if rationalise:
        out = sympy.nsimplify(out, tolerance=1e-10, rational=True)
    if mathematica_code:
        out = sympy.mathematica_code(out)
    return out


def _rk_symbolic_expression(
    t: Tree | Forest | ForestSum | int | float,
    s: int,
    explicit: bool = False,
    a_mask: list | None = None,
    b_mask: list | None = None,
    subtract_exact: bool = False,
    mathematica_code: bool = False,
    rationalise: bool = True,
    weight_symbols: list[ExprLike] | None = None,
) -> sympy.core.basic.Basic | str:
    normalized_t = _normalize_tree_like_input(t)
    weights_map = _rk_symbolic_weights_map(
        s=s,
        explicit=explicit,
        a_mask=a_mask,
        b_mask=b_mask,
        weight_symbols=weight_symbols,
    )
    expression = sympy.sympify(
        (weights_map - exact_weights)(normalized_t) if subtract_exact else weights_map(normalized_t)
    )
    return _finalize_symbolic_output(expression, rationalise, mathematica_code)


def rk_symbolic_weight(
    t: Tree | Forest | ForestSum,
    s: int,
    explicit: bool = False,
    a_mask: list | None = None,
    b_mask: list | None = None,
    mathematica_code: bool = False,
    rationalise: bool = True,
    weight_symbols: list[ExprLike] | None = None,
) -> sympy.core.basic.Basic | str | tuple:
    return _rk_symbolic_expression(
        t=t,
        s=s,
        explicit=explicit,
        a_mask=a_mask,
        b_mask=b_mask,
        subtract_exact=False,
        mathematica_code=mathematica_code,
        rationalise=rationalise,
        weight_symbols=weight_symbols,
    )


def rk_order_cond(
    t: Tree | Forest | ForestSum,
    s: int,
    explicit: bool = False,
    a_mask: list | None = None,
    b_mask: list | None = None,
    mathematica_code: bool = False,
    rationalise: bool = True,
    weight_symbols: list[ExprLike] | None = None,
) -> sympy.core.basic.Basic | str | tuple:
    return _rk_symbolic_expression(
        t=t,
        s=s,
        explicit=explicit,
        a_mask=a_mask,
        b_mask=b_mask,
        subtract_exact=True,
        mathematica_code=mathematica_code,
        rationalise=rationalise,
        weight_symbols=weight_symbols,
    )
