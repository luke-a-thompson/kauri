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

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .templates import SigRKTemplate
from .trees import LabeledTree, PurePythonTreeBackend, TreeBackend
from .weights import SchemeWeights
from .words import Word, exact_weight


@dataclass(frozen=True)
class OrderCondition:
    tree: LabeledTree
    power: Fraction
    word: Word
    expression: Any


def _active(power: Fraction, word: Word, order: int, alpha: Fraction) -> bool:
    return power + len(word) * alpha < (order + 1) * alpha


def order_conditions(
    template: SigRKTemplate,
    order: int,
    d: int,
    alpha: Fraction | int = Fraction(1, 3),
    backend: TreeBackend | None = None,
    active_only: bool = True,
) -> tuple[OrderCondition, ...]:
    """
    Generate fixed-dimension discovery order conditions.

    Coefficients are shared by the indexed template, but all trees and stages
    are expanded at the concrete dimension ``d``.
    """

    if order < 1:
        raise ValueError("Order must be positive.")
    if d < 1:
        raise ValueError("Driver dimension must be positive.")

    alpha = Fraction(alpha)
    backend = backend or PurePythonTreeBackend()
    labels = tuple(str(i) for i in range(1, d + 1))
    weights = SchemeWeights(template, d)
    conditions = []

    for tree_order in range(1, order + 1):
        for tree in backend.trees(tree_order, labels):
            difference = (weights.update_weight(tree) - exact_weight(tree)).simplify()
            for (power, word), expression in difference.terms:
                if active_only and not _active(power, word, order, alpha):
                    continue
                if expression != 0:
                    conditions.append(OrderCondition(tree, power, word, expression))

    return tuple(conditions)


def _simplify(expression):
    try:
        import sympy as sp
    except ImportError:
        return expression
    return sp.simplify(expression)


def consistency_conditions(template: SigRKTemplate, d: int) -> tuple[Any, ...]:
    """
    Generate the fixed-dimension consistency equations from the spec.

    These are
    ``sum_j a_{ij}^k(0, empty)=0`` for each concrete stage ``i`` and label
    ``k``, and ``sum_i b_i^k(0, empty)=0`` for each label ``k``.
    """

    weights = SchemeWeights(template, d)
    equations = []

    for target in weights.stages:
        for label in weights.labels:
            expr = 0
            for source in weights.stages:
                expr += weights.a_coeff(target, source, label).coefficient(0, ())
            expr = _simplify(expr)
            if expr != 0:
                equations.append(expr)

    for label in weights.labels:
        expr = 0
        for stage in weights.stages:
            expr += weights.b_coeff(stage, label).coefficient(0, ())
        expr = _simplify(expr)
        if expr != 0:
            equations.append(expr)

    return tuple(equations)
