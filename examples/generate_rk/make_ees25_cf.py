"""
Derive the commutator-free EES(2,5) scheme.

Starting from the EES(2,5) order conditions (explicit order 2, antisymmetric order 5,
Williamson 2N parameterization, b0 = 1/4 fixed), the candidate RK method is lifted to its
commutator-free Lie-group form and verified up to a chosen truncation order.

Run with:
    uv run python examples/generate_rk/make_ees25_cf.py
"""

import sympy
from kauri.methods.cf import verify_cf_ees
from kauri.rk_builder.rk_constraints import SetConstraint
from kauri.rk_builder.rk_maker import build_williamson_rk


def main() -> int:
    methods, solve_result = build_williamson_rk(
        order=2,
        stages=3,
        antisymmetric_order=5,
        constraints=[SetConstraint(sympy.Symbol("b0"), sympy.Rational(1, 4))],
        max_solutions=1,
    )
    print(solve_result)

    for method in methods:
        cf = method.to_cf()
        print(cf.to_text())

        verification = verify_cf_ees(cf, order=4)
        print(
            f"verification: passed={verification.passed}, "
            f"checked_elements={verification.checked_elements}, "
            f"first_failure={verification.first_failure}, "
            f"residual={verification.residual}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
