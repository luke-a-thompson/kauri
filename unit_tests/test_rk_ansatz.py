import unittest

import sympy

from kauri import make_explicit_rk_methods
from kauri.numerics.rk.rk_constraints import Constraint, compile_constraints
from kauri.numerics.rk.williamson_ansatz import (
    WilliamsonAnsatz,
    generate_2n_polynomial_constraints,
    is_2n_tableau,
)


class RKAnsatzTests(unittest.TestCase):
    def test_constraints_compile_tie_and_set(self):
        compiled = compile_constraints(
            [
                Constraint.tie("a21", "a10"),
                Constraint.set("a10", sympy.Rational(1, 2)),
                Constraint.equation("b0 + b1 + b2", 1),
            ]
        )
        self.assertEqual(sympy.Rational(1, 2), compiled.substitutions[sympy.symbols("a10")])
        self.assertEqual(sympy.symbols("a10"), compiled.substitutions[sympy.symbols("a21")])
        self.assertEqual(1, len(compiled.equations))

    def test_2n_constraints_hold_for_known_tableau(self):
        equations = generate_2n_polynomial_constraints(stages=4)
        substitutions = {
            sympy.symbols("a10"): sympy.Rational(1, 2),
            sympy.symbols("a20"): sympy.Rational(2, 9),
            sympy.symbols("a21"): sympy.Rational(1, 3),
            sympy.symbols("a30"): sympy.Rational(3, 176),
            sympy.symbols("a31"): sympy.Rational(51, 88),
            sympy.symbols("a32"): sympy.Rational(27, 176),
            sympy.symbols("b0"): sympy.Rational(2, 9),
            sympy.symbols("b1"): sympy.Rational(1, 3),
            sympy.symbols("b2"): sympy.Integer(0),
            sympy.symbols("b3"): sympy.Rational(4, 9),
        }
        for equation in equations:
            residual = sympy.simplify(sympy.expand(equation.subs(substitutions)))
            self.assertEqual(sympy.Integer(0), residual)

    def test_rk_maker_williamson_ansatz_smoke(self):
        result = make_explicit_rk_methods(
            order=1,
            stages=3,
            ansatz=WilliamsonAnsatz(),
            fixed_values={
                "A1": 0,
                "A2": 0,
                "B0": 0,
                "B1": 0,
                "B2": 1,
            },
            max_solutions=1,
            solver="grobner",
        )
        self.assertEqual("williamson_2n", result.ansatz)
        self.assertGreaterEqual(len(result.methods), 1)

    def test_recover_ees25_via_williamson_ansatz(self):
        result = make_explicit_rk_methods(
            order=2,
            stages=3,
            antisymmetric_order=5,
            ansatz=WilliamsonAnsatz(),
            fixed_values={"b0": sympy.Rational(1, 4)},
            max_solutions=1,
            solver="grobner",
        )
        self.assertEqual(1, len(result.methods))
        method = result.methods[0]
        self.assertEqual(2, method.order())
        self.assertEqual(5, method.antisymmetric_order())

        self.assertAlmostEqual(0.25, method.b[0])
        self.assertAlmostEqual(0.5, method.b[1])
        self.assertAlmostEqual(0.25, method.b[2])
        self.assertAlmostEqual(0.5, method.a[1][0])
        self.assertAlmostEqual(0.0, method.a[2][0])
        self.assertAlmostEqual(1.0, method.a[2][1])

        self.assertTrue(is_2n_tableau(method.a, method.b))
