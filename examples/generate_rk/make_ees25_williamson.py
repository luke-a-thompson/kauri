"""
Derive the EES(2,5) scheme: explicit order 2, antisymmetric order 5, 2N low-storage structure.

The method is found by solving the rooted-tree order conditions together with the
antisymmetric-order conditions and the Williamson 2N parameterization, with b0 = 1/4
fixed as a free parameter.

Run with:
    uv run python examples/generate_rk/make_ees25.py
"""

import sympy
from kauri.rk_builder.rk_constraints import SetConstraint
from kauri.rk_builder.williamson_rk_maker import build_williamson_rk


def main():
    constraints = [
        SetConstraint(sympy.Symbol("b0"), sympy.Rational(1, 4)),
    ]

    methods, solve_result = build_williamson_rk(
        order=2,
        stages=3,
        antisymmetric_order=5,
        constraints=constraints,
        embedded=False,
    )
    print(solve_result)
    print(f"methods found: {len(methods)}")
    if len(methods) != 0:
        print(methods[0].recursion.to_latex())


if __name__ == "__main__":
    raise SystemExit(main())
