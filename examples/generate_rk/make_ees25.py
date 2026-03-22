"""
Derive the EES(2,5) scheme: explicit order 2, antisymmetric order 5, 2N low-storage structure.

The method is found by solving the rooted-tree order conditions together with the
antisymmetric-order conditions and the Williamson 2N parameterization, with b0 = 1/4
fixed as a free parameter.

Run with:
    uv run python examples/generate_rk/make_ees25.py
    uv run python examples/generate_rk/make_ees25.py --max-stages 6
"""

import sympy
from kauri.rk_builder.explicit_rk_maker import build_explicit_rk
from kauri.rk_builder.rk_constraints import FSALConstraint, SetConstraint
from kauri.rk_builder.rk_objectives import LeadingErrorObjective


def main():
    constraints = []
    methods, solve_result = build_explicit_rk(
        order=2,
        stages=3,
        antisymmetric_order=5,
        constraints=constraints,
        embedded=True,
        objectives=[LeadingErrorObjective(order=2, antisymmetric_order=5)],
    )
    print(solve_result)
    print(f"methods found: {len(methods)}")
    if methods:
        print(methods[0])
        # print(methods[0].to_williamson().recursion.to_latex())


if __name__ == "__main__":
    main()
