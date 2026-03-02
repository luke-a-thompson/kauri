"""
Build truncated ordered-tree characters for lifted commutator-free methods.
"""

from __future__ import annotations

import sympy

from kauri.numerics.cf.cf_williamson import Williamson2NCF, cf_to_rk_tableau
from kauri.numerics.planar_trees.mkw_truncated import MKWMap
from kauri.numerics.planar_trees.planar_basis import PlanarTree
from kauri.numerics.rk.rk import rk_symbolic_weight_for_tableau


def cf_character_map(method: Williamson2NCF) -> MKWMap:
    """
    Character of the lifted method on planar trees, evaluated through RK tableau projection.

    This provides a consistent symbolic map for truncated verification while reusing existing
    RK elementary-weight infrastructure.
    """
    a_tableau, b_weights = cf_to_rk_tableau(method)

    def eval_tree(tree: PlanarTree) -> sympy.core.basic.Basic:
        nonplanar = tree.to_nonplanar_tree()
        return rk_symbolic_weight_for_tableau(
            t=nonplanar,
            a_tableau=a_tableau,
            b_weights=b_weights,
            explicit=True,
        )

    return MKWMap(eval_tree)
