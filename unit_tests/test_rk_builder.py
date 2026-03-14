import unittest

import sympy
from kauri import build_williamson_rk
from kauri.methods.rk import RK
from kauri.methods.williamson import WilliamsonRK
from kauri.rk_builder.rk_constraints import (
    EquationConstraint,
    SetConstraint,
    TieConstraint,
    compile_constraints,
)
from kauri.rk_builder.williamson import generate_2n_polynomial_constraints, is_2n_tableau


class RKBuilderTests(unittest.TestCase):
    def test_constraints_compile_tie_and_set(self):
        compiled = compile_constraints(
            [
                TieConstraint(sympy.Symbol("a21"), sympy.Symbol("a10")),
                SetConstraint(sympy.Symbol("a10"), sympy.Rational(1, 2)),
                EquationConstraint(sympy.sympify("b0 + b1 + b2"), sympy.Integer(1)),
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

    def test_build_williamson_rk_smoke(self):
        methods, solve_result = build_williamson_rk(
            order=1,
            stages=3,
            constraints=[
                SetConstraint(sympy.Symbol("A1"), sympy.Integer(0)),
                SetConstraint(sympy.Symbol("A2"), sympy.Integer(0)),
                SetConstraint(sympy.Symbol("B0"), sympy.Integer(0)),
                SetConstraint(sympy.Symbol("B1"), sympy.Integer(0)),
                SetConstraint(sympy.Symbol("B2"), sympy.Integer(1)),
            ],
            max_solutions=1,
        )
        self.assertEqual("williamson_2n", solve_result.parameterization)
        self.assertGreaterEqual(len(methods), 1)
        self.assertTrue(all(isinstance(method, WilliamsonRK) for method in methods))

    def test_recover_ees25_via_williamson_builder(self):
        methods, _ = build_williamson_rk(
            order=2,
            stages=3,
            antisymmetric_order=5,
            constraints=[SetConstraint(sympy.Symbol("b0"), sympy.Rational(1, 4))],
            max_solutions=1,
        )
        self.assertEqual(1, len(methods))
        method = methods[0]
        self.assertIsInstance(method, WilliamsonRK)
        rk_method = RK(method.tableau, name=method.name)
        self.assertEqual(2, rk_method.order())
        self.assertEqual(5, rk_method.antisymmetric_order())

        self.assertAlmostEqual(0.25, method.tableau.b[0])
        self.assertAlmostEqual(0.5, method.tableau.b[1])
        self.assertAlmostEqual(0.25, method.tableau.b[2])
        self.assertAlmostEqual(0.5, method.tableau.a[1][0])
        self.assertAlmostEqual(0.0, method.tableau.a[2][0])
        self.assertAlmostEqual(1.0, method.tableau.a[2][1])

        self.assertTrue(is_2n_tableau(method.tableau.a, method.tableau.b))
