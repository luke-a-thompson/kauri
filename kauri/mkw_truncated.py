"""
Truncated ordered-tree Hopf-algebra utilities for symbolic verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import cast

import sympy

from kauri.mkw_ees_spec import counit_planar, sign_for_tree
from kauri.planar_basis import (
    EMPTY_ORDERED_FOREST,
    EMPTY_PLANAR_TREE,
    OrderedForest,
    OrderedForestSum,
    PlanarTree,
)


def _as_expr(value: sympy.core.basic.Basic | int | float) -> sympy.Expr:
    return cast(sympy.Expr, sympy.sympify(value))


@dataclass(frozen=True)
class CoproductTerm:
    coeff: sympy.core.basic.Basic
    left: OrderedForest
    right: PlanarTree


def coproduct_terms(tree: PlanarTree) -> tuple[CoproductTerm, ...]:
    """
    Ordered-tree BCK-style coproduct terms, preserving sibling order.
    """
    if not isinstance(tree, PlanarTree):
        raise TypeError(f"tree must be PlanarTree, not {type(tree)}")
    raw: tuple[tuple[OrderedForest, PlanarTree], ...] = _coproduct_helper(tree)
    return tuple(
        CoproductTerm(coeff=sympy.Integer(1), left=left, right=right) for left, right in raw
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
                right_repr_children.append(cast(tuple, right_tree.list_repr))
            for left_tree in left_forest.tree_list:
                left_trees.append(left_tree)
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

    def __init__(self, func) -> None:
        if not callable(func):
            raise TypeError("func must be callable")
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
            return sympy.simplify(out)
        if isinstance(value, OrderedForestSum):
            out_sum: sympy.core.basic.Basic = sympy.Integer(0)
            for coeff, forest in value.term_list:
                out_sum = sympy.expand(
                    _as_expr(out_sum) + _as_expr(coeff) * _as_expr(self(forest))
                )
            return sympy.simplify(out_sum)
        raise TypeError(f"Unsupported value type for MKWMap: {type(value)}")

    def convolution(self, other: MKWMap) -> MKWMap:
        if not isinstance(other, MKWMap):
            raise TypeError(f"other must be MKWMap, not {type(other)}")

        def conv(tree: PlanarTree) -> sympy.core.basic.Basic:
            out: sympy.core.basic.Basic = sympy.Integer(0)
            for term in coproduct_terms(tree):
                coeff_expr = _as_expr(term.coeff)
                left_expr = _as_expr(self(term.left))
                right_expr = _as_expr(other(term.right))
                out = sympy.expand(
                    _as_expr(out) + coeff_expr * left_expr * right_expr
                )
            return sympy.simplify(out)

        return MKWMap(conv)

    def sign_twisted(self) -> MKWMap:
        def twisted(tree: PlanarTree) -> sympy.core.basic.Basic:
            return sympy.expand(_as_expr(sign_for_tree(tree)) * _as_expr(self(tree)))

        return MKWMap(twisted)


def counit_map() -> MKWMap:
    return MKWMap(lambda tree: counit_planar(tree))
