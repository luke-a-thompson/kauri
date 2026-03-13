"""
Truncated ordered-tree Hopf-algebra utilities for symbolic verification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

import sympy

from kauri.hopf_algebras.utils import _as_expr, _simplify_expanded
from kauri.planar_trees.mkw_ees_spec import counit_planar, sign_for_tree
from kauri.planar_trees.planar_basis import (
    EMPTY_ORDERED_FOREST,
    EMPTY_PLANAR_TREE,
    OrderedForest,
    OrderedForestSum,
    PlanarTree,
)


@dataclass(frozen=True)
class CoproductTerm:
    coeff: sympy.core.basic.Basic
    left: OrderedForest
    right: PlanarTree


def coproduct_terms(tree: PlanarTree) -> tuple[CoproductTerm, ...]:
    """
    Ordered-tree BCK-style coproduct terms, preserving sibling order.
    """
    return tuple(
        CoproductTerm(coeff=sympy.Integer(1), left=left, right=right)
        for left, right in _coproduct_helper(tree)
    )


def _coproduct_helper(tree: PlanarTree) -> tuple[tuple[OrderedForest, PlanarTree], ...]:
    if tree.list_repr is None:
        return ((EMPTY_ORDERED_FOREST, EMPTY_PLANAR_TREE),)
    if len(tree.list_repr) == 1:
        return ((EMPTY_ORDERED_FOREST, tree), (tree.as_ordered_forest(), EMPTY_PLANAR_TREE))

    children: list[PlanarTree] = [PlanarTree(rep) for rep in tree.list_repr[:-1]]
    child_coproducts: list[tuple[tuple[OrderedForest, PlanarTree], ...]] = [
        _coproduct_helper(child) for child in children
    ]
    out_terms: list[tuple[OrderedForest, PlanarTree]] = [
        (OrderedForest((tree,)), EMPTY_PLANAR_TREE)
    ]
    for picks in product(*child_coproducts):
        right_repr_children: list[tuple] = []
        left_trees: list[PlanarTree] = []
        for left_forest, right_tree in picks:
            if right_tree.list_repr is not None:
                right_repr_children.append(right_tree.list_repr)
            left_trees.extend(left_forest.tree_list)
        right_repr_children.append(tree.list_repr[-1])
        out_terms.append(
            (
                OrderedForest(tuple(left_trees)).simplify(),
                PlanarTree(tuple(right_repr_children)),
            )
        )
    return tuple(out_terms)


class MKWMap:
    """
    Minimal multiplicative linear map on ordered trees/forests.
    """

    def __init__(self, func: Callable[[PlanarTree], sympy.core.basic.Basic]) -> None:
        self._func = func
        self._cache: dict[PlanarTree, sympy.core.basic.Basic] = {}

    def _call_tree(self, tree: PlanarTree) -> sympy.core.basic.Basic:
        if tree not in self._cache:
            self._cache[tree] = sympy.sympify(self._func(tree))
        return self._cache[tree]

    def __call__(
        self, value: PlanarTree | OrderedForest | OrderedForestSum
    ) -> sympy.core.basic.Basic:
        if isinstance(value, PlanarTree):
            return self._call_tree(value)
        if isinstance(value, OrderedForest):
            out: sympy.core.basic.Basic = sympy.Integer(1)
            for tree in value.tree_list:
                out = sympy.expand(_as_expr(out) * _as_expr(self._call_tree(tree)))
            return _simplify_expanded(out)
        if isinstance(value, OrderedForestSum):
            out_sum: sympy.core.basic.Basic = sympy.Integer(0)
            for coeff, forest in value.term_list:
                out_sum = sympy.expand(_as_expr(out_sum) + _as_expr(coeff) * _as_expr(self(forest)))
            return _simplify_expanded(out_sum)
        raise TypeError(f"Unsupported value type for MKWMap: {type(value)}")

    def convolution(self, other: MKWMap) -> MKWMap:
        def conv(tree: PlanarTree) -> sympy.core.basic.Basic:
            out: sympy.core.basic.Basic = sympy.Integer(0)
            for term in coproduct_terms(tree):
                coeff_expr = _as_expr(term.coeff)
                left_expr = _as_expr(self(term.left))
                right_expr = _as_expr(other(term.right))
                out = sympy.expand(_as_expr(out) + coeff_expr * left_expr * right_expr)
            return _simplify_expanded(out)

        return MKWMap(conv)

    def sign_twisted(self) -> MKWMap:
        def twisted(tree: PlanarTree) -> sympy.core.basic.Basic:
            return sympy.expand(_as_expr(sign_for_tree(tree)) * _as_expr(self(tree)))

        return MKWMap(twisted)


def counit_map() -> MKWMap:
    return MKWMap(counit_planar)
