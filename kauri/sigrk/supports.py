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

from .words import LinearFunctional


@dataclass(frozen=True)
class Guard:
    """A product of Kronecker-delta equality constraints in discovery mode."""

    equalities: tuple[tuple[str, str], ...] = ()

    def holds(self, env: dict[str, str]) -> bool:
        for left, right in self.equalities:
            if env[left] != env[right]:
                return False
        return True


@dataclass(frozen=True)
class SupportTerm:
    """
    A support element ``(q, w, gamma)`` with a scalar coefficient.

    ``word`` is written in template variables.  During fixed-dimension
    expansion each token is replaced using the environment built from source
    indices, target indices, and the vector-field label.
    """

    power: Fraction
    word: tuple[str, ...] = ()
    coeff: Any = 1
    guard: Guard = Guard()

    def instantiate(self, env: dict[str, str]) -> LinearFunctional:
        if not self.guard.holds(env):
            return LinearFunctional.zero()
        try:
            word = tuple(env[token] for token in self.word)
        except KeyError as exc:
            raise KeyError(f"Unknown support word token {exc.args[0]!r}.") from exc
        return LinearFunctional.basis(self.power, word, self.coeff)


def instantiate_support(terms: tuple[SupportTerm, ...], env: dict[str, str]) -> LinearFunctional:
    out = LinearFunctional.zero()
    for term in terms:
        out = out + term.instantiate(env)
    return out.simplify()


def fixed_power_admissible(
    term: SupportTerm,
    alpha_min: Fraction,
    alpha_max: Fraction,
) -> bool:
    """Check the fixed-power admissibility inequality for one support term."""

    values = (len(term.word) * alpha_min + term.power, len(term.word) * alpha_max + term.power)
    if min(values) < 0:
        return False
    if values[0] == 0 and values[1] == 0:
        return term.power == 0 and term.word == ()
    return True


def filter_admissible_fixed_power(
    terms: tuple[SupportTerm, ...],
    alpha_min: Fraction,
    alpha_max: Fraction,
) -> tuple[SupportTerm, ...]:
    return tuple(term for term in terms if fixed_power_admissible(term, alpha_min, alpha_max))
