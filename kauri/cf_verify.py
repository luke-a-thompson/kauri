"""
Verification utilities for lifted commutator-free methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import sympy

from kauri.cf_mkw_character import cf_character_map
from kauri.cf_williamson import Williamson2NCF
from kauri.mkw_ees_spec import MKWEESSpecification, counit_planar
from kauri.planar_gentrees import planar_trees_up_to_order


def _as_expr(value: sympy.core.basic.Basic | int | float) -> sympy.Expr:
    return cast(sympy.Expr, sympy.sympify(value))


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    order: int
    checked_elements: int
    first_failure: str | None
    residual: sympy.core.basic.Basic | None


def verify_cf_ees(method: Williamson2NCF, order: int) -> VerificationResult:
    """
    Verify truncated EES residual (phi_sign ⋆ phi - counit) on planar trees up to `order`.
    """
    if not isinstance(order, int):
        raise TypeError(f"order must be int, not {type(order)}")
    if order <= 0:
        raise ValueError("order must be positive")
    _spec = MKWEESSpecification(truncation_order=order)
    phi = cf_character_map(method)
    residual_map = phi.sign_twisted().convolution(phi)
    checked: int = 0
    for tree in planar_trees_up_to_order(order):
        checked += 1
        residual: sympy.core.basic.Basic = sympy.simplify(
            sympy.expand(_as_expr(residual_map(tree)) - _as_expr(counit_planar(tree)))
        )
        if residual != 0:
            return VerificationResult(
                passed=False,
                order=order,
                checked_elements=checked,
                first_failure=repr(tree),
                residual=residual,
            )
    return VerificationResult(
        passed=True,
        order=order,
        checked_elements=checked,
        first_failure=None,
        residual=None,
    )
