"""
Derive the commutator-free EES(2,5) scheme.

Starting from the EES(2,5) order conditions (explicit order 2, antisymmetric order 5,
Williamson 2N ansatz, b0 = 1/4 fixed), the candidate RK method is lifted to its
commutator-free Lie-group form and verified up to a chosen truncation order.

Run with:
    uv run python examples/generate_rk/make_ees25_cf.py
"""

import sympy
from kauri.numerics.ansatze.williamson import WilliamsonAnsatz
from kauri.numerics.methods.cf import verify_cf_ees
from kauri.numerics.methods.williamson import rk_to_williamson_2n
from kauri.numerics.rk.rk_maker import make_explicit_rk_methods


def main() -> int:
    result = make_explicit_rk_methods(
        order=2,
        stages=3,
        antisymmetric_order=5,
        ansatz=WilliamsonAnsatz(),
        fixed_values={"b0": sympy.Rational(1, 4)},
        max_solutions=1,
    )
    print(result)

    for method in result.methods:
        cf = rk_to_williamson_2n(method).to_cf()
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
