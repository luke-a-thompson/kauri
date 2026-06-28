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

from typing import Any


def _require_sympy():
    try:
        import sympy as sp
    except ImportError as exc:
        raise ImportError("SigRK polynomial solving requires sympy.") from exc
    return sp


def groebner_basis(equations: tuple[Any, ...], variables: tuple[Any, ...], order: str = "lex"):
    sp = _require_sympy()
    return sp.groebner([sp.expand(eq) for eq in equations], *variables, order=order)


def groebner_is_impossible(equations: tuple[Any, ...], variables: tuple[Any, ...]) -> bool:
    basis = groebner_basis(equations, variables)
    return any(poly.as_expr() == 1 for poly in basis.polys)


def solve_equations(equations: tuple[Any, ...], variables: tuple[Any, ...]):
    sp = _require_sympy()
    return sp.solve([sp.expand(eq) for eq in equations], variables, dict=True)
