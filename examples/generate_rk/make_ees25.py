"""
Derive the EES(2,5) scheme: explicit order 2, antisymmetric order 5, 2N low-storage structure.

The method is found by solving the rooted-tree order conditions together with the
antisymmetric-order conditions and the Williamson 2N ansatz, with b0 = 1/4 fixed as
a free parameter.

Run with:
    uv run python examples/generate_rk/make_ees25.py
"""

import sympy
from kauri.numerics.rk import RKMakerResult, make_explicit_rk_methods
from kauri.numerics.rk.williamson_ansatz import WilliamsonAnsatz


def main() -> int:
    result: RKMakerResult = make_explicit_rk_methods(
        order=2,
        stages=3,
        antisymmetric_order=5,
        ansatz=WilliamsonAnsatz(),
        fixed_values={"b0": sympy.Rational(1, 4)},
        max_solutions=1,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
