"""
Verification utilities for lifted commutator-free methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy

from kauri.hopf_algebras.utils import _as_expr, _simplify_expanded
from kauri.numerics.cf.cf_williamson import WilliamsonCF
from kauri.numerics.planar_trees.mkw_ees_spec import counit_planar
from kauri.numerics.planar_trees.planar_basis import validate_order
from kauri.numerics.planar_trees.planar_gentrees import planar_trees_up_to_order


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    order: int
    checked_elements: int
    first_failure: str | None
    residual: sympy.core.basic.Basic | None


def verify_cf_ees(method: WilliamsonCF, order: int) -> VerificationResult:
    """
    Verify truncated EES residual (phi_sign ⋆ phi - counit) on planar trees up to `order`.
    """
    validate_order(order, allow_zero=False)
    phi = method.elementary_weights_map()
    residual_map = phi.sign_twisted().convolution(phi)
    for checked, tree in enumerate(planar_trees_up_to_order(order), start=1):
        residual: sympy.core.basic.Basic = _simplify_expanded(
            _as_expr(residual_map(tree)) - _as_expr(counit_planar(tree))
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
