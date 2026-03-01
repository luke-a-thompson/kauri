"""
Build truncated ordered-tree characters for lifted commutator-free methods.
"""

from __future__ import annotations

import sympy

from kauri.cf_williamson import Williamson2NCF, cf_to_rk_tableau
from kauri.mkw_truncated import MKWMap
from kauri.planar_basis import PlanarTree
from kauri.rk import _rk_symbolic_weight


def cf_character_map(method: Williamson2NCF) -> MKWMap:
    """
    Character of the lifted method on planar trees, evaluated through RK tableau projection.

    This provides a consistent symbolic map for truncated verification while reusing existing
    RK elementary-weight infrastructure.
    """
    if not isinstance(method, Williamson2NCF):
        raise TypeError(f"method must be Williamson2NCF, not {type(method)}")
    a_tableau, b_weights = cf_to_rk_tableau(method)
    stages: int = len(b_weights)
    substitutions: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
    for i_idx in range(stages):
        for j_idx in range(stages):
            symbol = sympy.symbols(f"a{i_idx}{j_idx}")
            substitutions[symbol] = sympy.sympify(a_tableau[i_idx][j_idx])
    for i_idx in range(stages):
        substitutions[sympy.symbols(f"b{i_idx}")] = sympy.sympify(b_weights[i_idx])

    def eval_tree(tree: PlanarTree) -> sympy.core.basic.Basic:
        nonplanar = tree.to_nonplanar_tree()
        symbolic = sympy.sympify(_rk_symbolic_weight(nonplanar, stages, explicit=True))
        return sympy.simplify(sympy.expand(symbolic.subs(list(substitutions.items()))))

    return MKWMap(eval_tree)
