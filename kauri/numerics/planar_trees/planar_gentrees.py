"""
Generation utilities for ordered (planar) rooted trees.
"""

from __future__ import annotations

from collections.abc import Generator
from functools import cache
from itertools import product

from kauri.numerics.planar_trees.planar_basis import EMPTY_PLANAR_TREE, PlanarTree, validate_order


def planar_trees_up_to_order(order: int) -> Generator[PlanarTree, None, None]:
    """
    Yield planar trees up to and including `order` nodes.
    """
    validate_order(order, allow_zero=True)
    for n_nodes in range(0, order + 1):
        yield from planar_trees_of_order(n_nodes)


def planar_trees_of_order(order: int) -> Generator[PlanarTree, None, None]:
    """
    Yield planar trees with exactly `order` nodes.
    """
    validate_order(order, allow_zero=True)
    if order == 0:
        yield EMPTY_PLANAR_TREE
        return
    yield from _trees_with_nodes(order)


@cache
def _trees_with_nodes(n_nodes: int) -> tuple[PlanarTree, ...]:
    if n_nodes <= 0:
        return tuple()
    if n_nodes == 1:
        return (PlanarTree([]),)

    generated: list[PlanarTree] = []
    for children_sizes in _ordered_compositions(n_nodes - 1):
        child_choices: list[tuple[PlanarTree, ...]] = [
            _trees_with_nodes(size) for size in children_sizes
        ]
        for combo in product(*child_choices):
            child_reprs: tuple = tuple(child.list_repr for child in combo)
            rep: tuple = child_reprs + (0,)
            generated.append(PlanarTree(rep))
    return tuple(generated)


@cache
def _ordered_compositions(n_value: int) -> tuple[tuple[int, ...], ...]:
    """
    Ordered compositions of n_value into positive integers.
    """
    if n_value <= 0:
        return tuple()
    if n_value == 1:
        return ((1,),)
    compositions: list[tuple[int, ...]] = [(n_value,)]
    for first in range(1, n_value):
        for tail in _ordered_compositions(n_value - first):
            compositions.append((first,) + tail)
    return tuple(compositions)
