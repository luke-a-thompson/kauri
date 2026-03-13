"""
Generator-layer symbolic tools for RK order conditions and elementary weights.
"""

import sympy

from kauri.hopf_algebras.maps import Map, exact_weights
from kauri.trees.trees import Forest, ForestSum, Tree


def _internal_symbolic(i, t_rep, a, b, s):
    return sum(a[i, j] * _derivative_symbolic(j, t_rep, a, b, s) for j in range(s))


def _derivative_symbolic(i, t_rep, a, b, s):
    if t_rep in (None, []):
        return 1
    out = 1
    for subtree in t_rep[:-1]:
        out *= _internal_symbolic(i, subtree, a, b, s)
    return out


def _elementary_symbolic(t_rep, a, b, s):
    if t_rep is None:
        return 1
    if len(t_rep) == 1:
        return sum(b)
    return sum(b[i] * _derivative_symbolic(i, t_rep, a, b, s) for i in range(s))


def _rk_symbolic_weight(t, s, explicit=False, a_mask=None, b_mask=None):
    if a_mask is None:
        a_mask = [[1 for _ in range(s)] for _ in range(s)]
    if b_mask is None:
        b_mask = [1 for _ in range(s)]
    if explicit:
        for i in range(s):
            for j in range(i, s):
                a_mask[i][j] = 0

    a = sympy.Matrix(s, s, lambda i, j: sympy.symbols(f"a{i}{j}"))
    b = sympy.Matrix(1, s, lambda i, j: sympy.symbols(f"b{j}"))
    for i in range(s):
        for j in range(s):
            if not a_mask[i][j]:
                a[i, j] = 0
    for i in range(s):
        if not b_mask[i]:
            b[i] = 0
    return _elementary_symbolic(t.list_repr, a, b, s)


def _rk_symbolic_weights_map(
    s: int, explicit: bool = False, a_mask: list | None = None, b_mask: list | None = None
) -> Map:
    return Map(lambda x: _rk_symbolic_weight(x, s, explicit, a_mask, b_mask))


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
) -> sympy.core.basic.Basic | str:
    normalized_t = _normalize_tree_like_input(t)
    weights_map = _rk_symbolic_weights_map(s, explicit, a_mask, b_mask)
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
    )


def rk_order_cond(
    t: Tree | Forest | ForestSum,
    s: int,
    explicit: bool = False,
    a_mask: list | None = None,
    b_mask: list | None = None,
    mathematica_code: bool = False,
    rationalise: bool = True,
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
    )
