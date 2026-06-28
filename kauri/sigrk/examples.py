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

from .equations import OrderCondition, order_conditions
from .solve import solve_equations
from .supports import Guard, SupportTerm
from .templates import EdgeRule, SigRKTemplate, StageFamily, UpdateRule

EMPTY_GUARD = Guard()


def _support(power, word=(), coeff=1, guard=None):
    return SupportTerm(Fraction(power), tuple(word), coeff, guard or EMPTY_GUARD)


def known_order2_template(symbolic: bool = False):
    """
    Return the indexed order-two template.

    If ``symbolic`` is true, the shared scalar coefficients are unknowns
    ``A, B, C_word, C_empty``.  Otherwise the recovered coefficients are used:

    ``a_{A_k,K_*}^l = e_{lk}``,
    ``b_{A_{k'}}^k = delta_{kk'} e_empty``,
    ``b_{K_*}^k = e_k - e_empty``.
    """

    if symbolic:
        try:
            import sympy as sp
        except ImportError as exc:
            raise ImportError("Symbolic order-two recovery requires sympy.") from exc
        a, b, c_word, c_empty = sp.symbols("A B C_word C_empty")
        variables = (a, b, c_word, c_empty)
    else:
        a, b, c_word, c_empty = 1, 1, 1, -1
        variables = ()

    template = SigRKTemplate(
        families=(
            StageFamily("K", (), layer=0),
            StageFamily("A", ("kp",), layer=1),
        ),
        edge_rules=(
            EdgeRule(
                "K",
                "A",
                "l",
                (_support(0, ("l", "kp"), a),),
                name="A_k uses f_l(K)",
            ),
        ),
        update_rules=(
            UpdateRule(
                "K",
                "k",
                (
                    _support(0, ("k",), c_word),
                    _support(0, (), c_empty),
                ),
                name="base update",
            ),
            UpdateRule(
                "A",
                "k",
                (_support(0, (), b, Guard((("k", "kp"),))),),
                name="diagonal A update",
            ),
        ),
        name="known-order-2-symbolic" if symbolic else "known-order-2",
    )
    return template, variables


@dataclass(frozen=True)
class RecoveryProblem:
    template: SigRKTemplate
    variables: tuple
    conditions: tuple[OrderCondition, ...]
    equations: tuple
    solutions: tuple[dict, ...]


def known_order2_recovery_problem(d: int = 2) -> RecoveryProblem:
    """
    Build and solve the milestone order-two recovery problem.

    The equations determine the displayed solution after fixing the harmless
    scaling gauge ``B=1`` for the diagonal update stage.
    """

    template, variables = known_order2_template(symbolic=True)
    conditions = order_conditions(template, order=2, d=d, alpha=Fraction(1, 2))
    equations = tuple(condition.expression for condition in conditions)

    try:
        import sympy  # noqa: F401
    except ImportError as exc:
        raise ImportError("Symbolic order-two recovery requires sympy.") from exc

    _a, b, _c_word, _c_empty = variables
    gauge_fixed_equations = equations + (b - 1,)
    solutions = tuple(solve_equations(gauge_fixed_equations, variables))
    return RecoveryProblem(template, variables, conditions, gauge_fixed_equations, solutions)


def known_order3_template():
    """Return the indexed order-three template from the displayed SigRK scheme."""

    half = Fraction(1, 2)
    return SigRKTemplate(
        families=(
            StageFamily("K", (), layer=0),
            StageFamily("Q", ("l", "k"), layer=1),
            StageFamily("Dplus", ("l",), layer=1),
            StageFamily("Dminus", ("l",), layer=1),
            StageFamily("Pplus", ("k",), layer=2),
            StageFamily("Pminus", ("k",), layer=2),
            StageFamily("Eplus", ("k", "l"), layer=1),
            StageFamily("Eminus", ("k", "l"), layer=1),
        ),
        edge_rules=(
            EdgeRule(
                "K",
                "Q",
                "m",
                (_support(Fraction(-1, 2), ("m", "l", "k")),),
                name="Q_{lk}",
            ),
            EdgeRule(
                "K",
                "Dplus",
                "l",
                (_support(Fraction(1, 2)),),
                name="D_{l,+}",
            ),
            EdgeRule(
                "K",
                "Dminus",
                "l",
                (_support(Fraction(1, 2), coeff=-1),),
                name="D_{l,-}",
            ),
            EdgeRule(
                "K",
                "Pplus",
                "l",
                (
                    _support(0, ("l", "k")),
                    _support(Fraction(1, 2), coeff=-1),
                ),
                name="P_{k,+} base",
            ),
            EdgeRule(
                "Q",
                "Pplus",
                "l",
                (_support(Fraction(1, 2)),),
                name="P_{k,+} Q",
            ),
            EdgeRule(
                "K",
                "Pminus",
                "l",
                (
                    _support(0, ("l", "k"), coeff=-1),
                    _support(Fraction(1, 2)),
                ),
                name="P_{k,-} base",
            ),
            EdgeRule(
                "Q",
                "Pminus",
                "l",
                (_support(Fraction(1, 2), coeff=-1),),
                name="P_{k,-} Q",
            ),
            EdgeRule(
                "K",
                "Eplus",
                "l",
                (_support(Fraction(1, 2)),),
                name="E_{kl,+} D part",
            ),
            EdgeRule(
                "K",
                "Eplus",
                "m",
                (_support(Fraction(-1, 2), ("m", "l", "k")),),
                name="E_{kl,+} Q part",
            ),
            EdgeRule(
                "K",
                "Eminus",
                "l",
                (_support(Fraction(1, 2), coeff=-1),),
                name="E_{kl,-} D part",
            ),
            EdgeRule(
                "K",
                "Eminus",
                "m",
                (_support(Fraction(-1, 2), ("m", "l", "k")),),
                name="E_{kl,-} Q part",
            ),
        ),
        update_rules=(
            UpdateRule("K", "k", (_support(0, ("k",)),), name="level 1"),
            UpdateRule("Pplus", "k", (_support(0, coeff=half),), name="P plus"),
            UpdateRule("Pminus", "k", (_support(0, coeff=-half),), name="P minus"),
            UpdateRule("Eplus", "k", (_support(0, coeff=half),), name="E plus"),
            UpdateRule("Dplus", "k", (_support(0, coeff=-half),), name="D plus correction"),
            UpdateRule("Eminus", "k", (_support(0, coeff=-half),), name="E minus"),
            UpdateRule("Dminus", "k", (_support(0, coeff=half),), name="D minus correction"),
        ),
        name="known-order-3",
    )
