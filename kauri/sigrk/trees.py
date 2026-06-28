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

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from functools import cache
from itertools import product
from typing import Protocol


@dataclass(frozen=True)
class LabeledTree:
    """A non-planar rooted tree with a driver label at each vertex."""

    root: str
    children: tuple[LabeledTree, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "root", str(self.root))
        object.__setattr__(self, "children", tuple(sorted(self.children, key=str)))

    @property
    def order(self) -> int:
        return 1 + sum(child.order for child in self.children)

    def __str__(self) -> str:
        if not self.children:
            return f"b{self.root}"
        return f"[{','.join(str(child) for child in self.children)}]_{self.root}"


class TreeBackend(Protocol):
    def trees(self, order: int, labels: tuple[str, ...]) -> tuple[LabeledTree, ...]:
        """Return all labelled non-planar rooted trees of exact ``order``."""

    def root_label(self, tree: LabeledTree) -> str:
        """Return the root label."""

    def children(self, tree: LabeledTree) -> tuple[LabeledTree, ...]:
        """Return the children."""

    def symmetry(self, tree: LabeledTree) -> int:
        """Return the non-planar rooted-tree symmetry factor."""


class PurePythonTreeBackend:
    """Small labelled-tree backend for SigRK discovery mode."""

    def trees(self, order: int, labels: tuple[str, ...]) -> tuple[LabeledTree, ...]:
        labels = tuple(str(label) for label in labels)
        return _trees(order, labels)

    def root_label(self, tree: LabeledTree) -> str:
        return tree.root

    def children(self, tree: LabeledTree) -> tuple[LabeledTree, ...]:
        return tree.children

    def symmetry(self, tree: LabeledTree) -> int:
        counts = Counter(tree.children)
        child_factor = 1
        for child in tree.children:
            child_factor *= self.symmetry(child)
        multiplicity_factor = 1
        for count in counts.values():
            multiplicity_factor *= math.factorial(count)
        return child_factor * multiplicity_factor


class KauriTreeBackend(PurePythonTreeBackend):
    """
    Adapter placeholder for Kauri-backed tree operations.

    Kauri is intentionally kept behind the same interface.  The first SigRK
    milestone only needs labelled-tree generation and symmetry, which the pure
    Python backend already provides.
    """


@cache
def _trees(order: int, labels: tuple[str, ...]) -> tuple[LabeledTree, ...]:
    if order < 1:
        raise ValueError("Tree order must be positive.")
    if not labels:
        raise ValueError("At least one driver label is required.")
    if order == 1:
        return tuple(LabeledTree(label) for label in labels)

    previous = tuple(tree for n in range(1, order) for tree in _trees(n, labels))
    child_forests = _child_multisets(order - 1, previous)
    out = [LabeledTree(root, children) for root, children in product(labels, child_forests)]
    return tuple(sorted(out, key=str))


def _child_multisets(total_order: int, candidates: tuple[LabeledTree, ...]):
    by_key = tuple(sorted(candidates, key=str))

    @cache
    def rec(remaining: int, start: int) -> tuple[tuple[LabeledTree, ...], ...]:
        if remaining == 0:
            return ((),)

        forests = []
        for idx in range(start, len(by_key)):
            tree = by_key[idx]
            if tree.order > remaining:
                continue
            for tail in rec(remaining - tree.order, idx):
                forests.append((tree,) + tail)
        return tuple(forests)

    return rec(total_order, 0)
