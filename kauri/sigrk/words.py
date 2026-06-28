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

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import cache, reduce
from typing import Any

Word = tuple[str, ...]
PowerWord = tuple[Fraction, Word]


def _as_fraction(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(value)


def _is_zero(value: Any) -> bool:
    return value == 0


@cache
def _shuffle_two(left: Word, right: Word) -> tuple[tuple[Word, int], ...]:
    if not left:
        return ((right, 1),)
    if not right:
        return ((left, 1),)

    out: Counter[Word] = Counter()
    for word, coeff in _shuffle_two(left[1:], right):
        out[(left[0],) + word] += coeff
    for word, coeff in _shuffle_two(left, right[1:]):
        out[(right[0],) + word] += coeff
    return tuple(sorted(out.items()))


def shuffle_counts(words: tuple[Word, ...]) -> dict[Word, int]:
    """
    Return coefficients of the shuffle product of ``words``.

    The implementation folds the binary shuffle.  Target orders in the SigRK
    search are small, so this direct dynamic programme is enough for discovery
    mode and keeps the word algebra transparent.
    """

    def combine(left_counts: dict[Word, int], right: Word) -> dict[Word, int]:
        out: Counter[Word] = Counter()
        for left_word, left_coeff in left_counts.items():
            for word, coeff in _shuffle_two(left_word, right):
                out[word] += left_coeff * coeff
        return dict(out)

    return reduce(combine, words, {(): 1})


@dataclass(frozen=True)
class LinearFunctional:
    """A sparse functional ``(q, word) -> coefficient``."""

    terms: tuple[tuple[PowerWord, Any], ...] = ()

    def __post_init__(self):
        out = {}
        for (power, word), coeff in self.terms:
            power = _as_fraction(power)
            word = tuple(str(label) for label in word)
            if _is_zero(coeff):
                continue
            key = (power, word)
            out[key] = out.get(key, 0) + coeff
        object.__setattr__(
            self,
            "terms",
            tuple(
                (key, coeff)
                for key, coeff in sorted(out.items(), key=lambda item: (item[0][0], item[0][1]))
                if not _is_zero(coeff)
            ),
        )

    @classmethod
    def zero(cls) -> LinearFunctional:
        return cls()

    @classmethod
    def one(cls) -> LinearFunctional:
        return cls((((Fraction(0), ()), 1),))

    @classmethod
    def basis(
        cls,
        power: Fraction | int = Fraction(0),
        word: Word = (),
        coeff: Any = 1,
    ) -> LinearFunctional:
        return cls((((Fraction(power), tuple(word)), coeff),))

    @property
    def is_zero(self) -> bool:
        return len(self.terms) == 0

    def as_dict(self) -> dict[PowerWord, Any]:
        return dict(self.terms)

    def keys(self) -> set[PowerWord]:
        return {key for key, _ in self.terms}

    def coefficient(self, power: Fraction | int, word: Word) -> Any:
        return self.as_dict().get((Fraction(power), tuple(word)), 0)

    def scale(self, coeff: Any) -> LinearFunctional:
        if _is_zero(coeff):
            return LinearFunctional.zero()
        return LinearFunctional((key, coeff * value) for key, value in self.terms)

    def append(self, label: str) -> LinearFunctional:
        return LinearFunctional(
            ((power, word + (str(label),)), coeff)
            for (power, word), coeff in self.terms
        )

    def simplify(self) -> LinearFunctional:
        try:
            import sympy as sp
        except ImportError:
            return self
        return LinearFunctional((key, sp.simplify(coeff)) for key, coeff in self.terms)

    def shuffle(self, other: LinearFunctional) -> LinearFunctional:
        if self.is_zero or other.is_zero:
            return LinearFunctional.zero()
        out = []
        for (left_power, left_word), left_coeff in self.terms:
            for (right_power, right_word), right_coeff in other.terms:
                for word, shuffle_coeff in shuffle_counts((left_word, right_word)).items():
                    out.append(
                        (
                            (left_power + right_power, word),
                            left_coeff * right_coeff * shuffle_coeff,
                        )
                    )
        return LinearFunctional(out)

    def __add__(self, other: LinearFunctional) -> LinearFunctional:
        return LinearFunctional(self.terms + other.terms)

    def __neg__(self) -> LinearFunctional:
        return LinearFunctional((key, -coeff) for key, coeff in self.terms)

    def __sub__(self, other: LinearFunctional) -> LinearFunctional:
        return self + (-other)

    def __bool__(self) -> bool:
        return not self.is_zero

    def __str__(self) -> str:
        if self.is_zero:
            return "0"
        pieces = []
        for (power, word), coeff in self.terms:
            suffix = "e_" + ("".join(word) if word else "empty")
            if power:
                suffix = f"theta^{power} {suffix}"
            pieces.append(f"{coeff}*{suffix}")
        return " + ".join(pieces)


def shuffle_functionals(functionals: tuple[LinearFunctional, ...]) -> LinearFunctional:
    out = LinearFunctional.one()
    for functional in functionals:
        out = out.shuffle(functional)
        if out.is_zero:
            break
    return out


def exact_weight(tree) -> LinearFunctional:
    """Compute the exact word weight ``iota`` for a labelled rooted tree."""

    child_weights = tuple(exact_weight(child) for child in tree.children)
    return shuffle_functionals(child_weights).append(tree.root)
