import unittest

import sympy
from kauri import build_williamson_rk
from kauri.methods.rk import RK, ButcherTableau
from kauri.methods.williamson import WilliamsonRK, WilliamsonRecursion
from kauri.rk_builder.rk_constraints import (
    EquationConstraint,
    SetConstraint,
    TieConstraint,
    compile_constraints,
)
from kauri.rk_builder.rk_maker_format import (
    format_williamson_recursion_latex,
    format_williamson_recursion_text,
)
from kauri.rk_builder.williamson import (
    generate_2n_polynomial_constraints,
    is_2n_tableau,
    williamson_tableau_expressions,
)


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
        )
        self.assertEqual("williamson_2n", solve_result.parameterization)
        self.assertGreaterEqual(len(methods), 1)
        self.assertTrue(all(isinstance(method, WilliamsonRK) for method in methods))
        self.assertTrue(all(not method.embedded_from_penultimate for method in methods))

    def test_build_williamson_rk_embedded_requires_order_at_least_two(self):
        with self.assertRaises(ValueError):
            build_williamson_rk(order=1, stages=2, embedded=True)

    def test_build_williamson_rk_embedded_from_penultimate(self):
        methods, solve_result = build_williamson_rk(
            order=2,
            stages=3,
            constraints=[
                SetConstraint(sympy.Symbol("A1"), sympy.Integer(0)),
                SetConstraint(sympy.Symbol("A2"), -sympy.Integer(1)),
                SetConstraint(sympy.Symbol("B0"), sympy.Rational(1, 2)),
                SetConstraint(sympy.Symbol("B1"), sympy.Rational(1, 2)),
                SetConstraint(sympy.Symbol("B2"), sympy.Rational(1, 2)),
            ],
            embedded=True,
        )

        self.assertEqual("williamson_2n_embedded", solve_result.parameterization)
        self.assertEqual(1, len(methods))
        method = methods[0]
        self.assertTrue(method.embedded_from_penultimate)
        self.assertEqual(2, RK(method.tableau, name=method.name).order())

        truncated_a, truncated_b = williamson_tableau_expressions(
            stages=2,
            A_symbols=[sympy.Integer(0), method.recursion.A[1][0]],
            B_symbols=method.recursion.B[:2],
        )
        embedded_method = RK(ButcherTableau(a=truncated_a, b=truncated_b))
        self.assertEqual(1, embedded_method.order())

    def test_recover_ees25_via_williamson_builder(self):
        methods, _ = build_williamson_rk(
            order=2,
            stages=3,
            antisymmetric_order=5,
            constraints=[SetConstraint(sympy.Symbol("b0"), sympy.Rational(1, 4))],
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

    def test_format_williamson_recursion_text_expands_stage_nodes(self):
        recursion = WilliamsonRecursion(
            A=[
                [sympy.Integer(0), sympy.Integer(0), sympy.Integer(0)],
                [sympy.Rational(1, 2), sympy.Integer(0), sympy.Integer(0)],
                [sympy.Integer(0), sympy.Integer(1), sympy.Integer(0)],
            ],
            B=[
                sympy.Rational(1, 4),
                sympy.Rational(1, 2),
                sympy.Rational(1, 4),
            ],
        )

        self.assertEqual(
            "\n".join(
                [
                    "=== Williamson 2N Method ===",
                    "name: ees25_like",
                    "stages: 3",
                    "",
                    "equations:",
                    "  Y_0 = Y_t",
                    "  ΔY_0 = 0",
                    "  K_1 = F(t + (0)*h, Y_0)",
                    "  ΔY_1 = (0)*ΔY_0 + h*K_1",
                    "  Y_1 = Y_0 + (1/4)*ΔY_1",
                    "  K_2 = F(t + (1/4)*h, Y_1)",
                    "  ΔY_2 = (1/2)*ΔY_1 + h*K_2",
                    "  Y_2 = Y_1 + (1/2)*ΔY_2",
                    "  K_3 = F(t + (1)*h, Y_2)",
                    "  ΔY_3 = (1)*ΔY_2 + h*K_3",
                    "  Y_3 = Y_2 + (1/4)*ΔY_3",
                    "  Y_(t+h) = Y_3",
                ]
            ),
            format_williamson_recursion_text(recursion, name="ees25_like"),
        )

    def test_format_williamson_recursion_latex_expands_stage_nodes(self):
        recursion = WilliamsonRecursion(
            A=[
                [sympy.Integer(0), sympy.Integer(0), sympy.Integer(0)],
                [sympy.Rational(1, 2), sympy.Integer(0), sympy.Integer(0)],
                [sympy.Integer(0), sympy.Integer(1), sympy.Integer(0)],
            ],
            B=[
                sympy.Rational(1, 4),
                sympy.Rational(1, 2),
                sympy.Rational(1, 4),
            ],
        )

        self.assertEqual(
            "\n".join(
                [
                    r"\[",
                    r"\textbf{Williamson 2N Method: }\texttt{ees25_like}",
                    r"\\",
                    r"\begin{aligned}",
                    r"Y_0 &= Y_t\\",
                    r"\Delta Y_0 &= 0\\",
                    r"K_{1} &= F\left(t + (0)h, Y_{0}\right)\\",
                    r"\Delta Y_{1} &= (0)\Delta Y_{0} + hK_{1}\\",
                    r"Y_{1} &= Y_{0} + (\frac{1}{4})\Delta Y_{1}\\",
                    r"K_{2} &= F\left(t + (\frac{1}{4})h, Y_{1}\right)\\",
                    r"\Delta Y_{2} &= (\frac{1}{2})\Delta Y_{1} + hK_{2}\\",
                    r"Y_{2} &= Y_{1} + (\frac{1}{2})\Delta Y_{2}\\",
                    r"K_{3} &= F\left(t + (1)h, Y_{2}\right)\\",
                    r"\Delta Y_{3} &= (1)\Delta Y_{2} + hK_{3}\\",
                    r"Y_{3} &= Y_{2} + (\frac{1}{4})\Delta Y_{3}\\",
                    r"Y_{t+h} &= Y_{3}",
                    r"\end{aligned}",
                    r"\]",
                ]
            ),
            format_williamson_recursion_latex(recursion, name="ees25_like"),
        )
