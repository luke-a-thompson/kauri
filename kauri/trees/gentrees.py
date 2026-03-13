# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""
Functions for generating rooted trees in lexicographic order, based on the algorithms of :cite:`beyer1980constant`.
"""

from collections.abc import Generator
from functools import cache
from itertools import product

from kauri.hopf_algebras.utils import _level_sequence_to_list_repr
from kauri.trees.trees import Tree


def trees_up_to_order(order: int) -> Generator[Tree, None, None]:
    """
    Yields the trees up to and including order :math:`n`, ordered by the lexicographic order.

    :param order: Maximum order
    :type order: int
    :yields: The next tree in lexicographic order, as long as the
        order of the tree does not exceed :math:`n`.
    :rtype: Tree

    Example usage::

            import kauri as kr
            import kauri.hopf_algebras.bck as bck

            for t in kr.trees_up_to_order(4):
                display(bck.antipode(t))
    """
    if not isinstance(order, int):
        raise TypeError("order must be an int, not " + str(type(order)))
    if order < 0:
        raise ValueError("order must be positive")

    t = Tree(None)
    while t.nodes() <= order:
        yield t
        t = next(t)


def trees_of_order(order: int) -> Generator[Tree, None, None]:
    """
    Yields the trees of order :math:`n`, ordered by the lexicographic order.

    :param order: Order
    :type order: int
    :yields: The next tree in lexicographic order, as long as the order of the tree is :math:`n`.
    :rtype: Tree

    Example usage::

            import kauri as kr
            import kauri.hopf_algebras.bck as bck

            for t in kr.trees_of_order(4):
                display(bck.antipode(t))
    """
    if not isinstance(order, int):
        raise TypeError("order must be an int, not " + str(type(order)))
    if order < 0:
        raise ValueError("order must be positive")

    t = Tree(_level_sequence_to_list_repr(list(range(order))))
    while t.nodes() == order:
        yield t
        t = next(t)


def planar_trees_up_to_order(order: int) -> Generator["PlanarTree", None, None]:
    """
    Yield planar trees up to and including `order` nodes.
    """
    from kauri.planar_trees.planar_basis import validate_order

    validate_order(order, allow_zero=True)
    for n_nodes in range(0, order + 1):
        yield from planar_trees_of_order(n_nodes)


def planar_trees_of_order(order: int) -> Generator["PlanarTree", None, None]:
    """
    Yield planar trees with exactly `order` nodes.
    """
    from kauri.planar_trees.planar_basis import EMPTY_PLANAR_TREE, validate_order

    validate_order(order, allow_zero=True)
    if order == 0:
        yield EMPTY_PLANAR_TREE
        return
    yield from _planar_trees_with_nodes(order)


@cache
def _planar_trees_with_nodes(n_nodes: int) -> tuple["PlanarTree", ...]:
    from kauri.planar_trees.planar_basis import PlanarTree

    if n_nodes <= 0:
        return tuple()
    if n_nodes == 1:
        return (PlanarTree([]),)

    generated: list[PlanarTree] = []
    for children_sizes in _ordered_compositions(n_nodes - 1):
        child_choices: list[tuple[PlanarTree, ...]] = [
            _planar_trees_with_nodes(size) for size in children_sizes
        ]
        for combo in product(*child_choices):
            child_reprs: tuple = tuple(child.list_repr for child in combo)
            rep: tuple = child_reprs + (0,)
            generated.append(PlanarTree(rep))
    return tuple(generated)


@cache
def _ordered_compositions(n_value: int) -> tuple[tuple[int, ...], ...]:
    if n_value <= 0:
        return tuple()
    if n_value == 1:
        return ((1,),)
    compositions: list[tuple[int, ...]] = [(n_value,)]
    for first in range(1, n_value):
        for tail in _ordered_compositions(n_value - first):
            compositions.append((first,) + tail)
    return tuple(compositions)
