# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import unittest

import kauri.hopf_algebras.bck as bck
import sympy
from kauri import (
    EES25,
    EES27,
    SolveResult,
    Tree,
    backward_euler,
    build_explicit_rk,
    crank_nicolson,
    euler,
    exact_weights,
    gauss6,
    heun_rk2,
    heun_rk3,
    implicit_midpoint,
    kutta_rk3,
    lobatto6,
    midpoint,
    nystrom_rk5,
    radau_iia,
    ralston_rk3,
    ralston_rk4,
    rk4,
    rk_order_cond,
    rk_symbolic_weight,
    trees_up_to_order,
)
from kauri import Tree as T
from kauri.methods.rk import RK, ButcherTableau
from kauri.rk_builder.rk_constraints import SetConstraint
from kauri.rk_builder.rk_maker_format import format_tableau_latex, format_tableau_text

sample_trees = [
    T(None),
    T([]),
    T([[]]),
    T([[], []]),
    T([[[]]]),
    T([[], [], []]),
    T([[], [[]]]),
    T([[[], []]]),
    T([[[[]]]]),
]


class RKTests(unittest.TestCase):
    def test_butcher_tableau_derived_properties(self):
        explicit = ButcherTableau([[0, 0], [sympy.Rational(1, 2), 0]], [0, 1])
        self.assertEqual(2, explicit.stages)
        self.assertEqual([0, sympy.Rational(1, 2)], explicit.c)
        self.assertTrue(explicit.explicit)
        self.assertFalse(explicit.ssal)
        self.assertFalse(explicit.fsal)

        implicit = ButcherTableau([[sympy.Rational(1, 2)]], [1])
        self.assertFalse(implicit.explicit)
        self.assertFalse(implicit.ssal)
        self.assertFalse(implicit.fsal)

        fsal = ButcherTableau(
            [
                [0, 0, 0],
                [sympy.Rational(1, 2), 0, 0],
                [sympy.Rational(1, 4), sympy.Rational(3, 4), 0],
            ],
            [sympy.Rational(1, 4), sympy.Rational(3, 4), 0],
        )
        self.assertTrue(fsal.ssal)
        self.assertTrue(fsal.fsal)

        non_fsal = ButcherTableau(
            [
                [1, 0, 0],
                [sympy.Rational(1, 2), 0, 0],
                [sympy.Rational(1, 4), sympy.Rational(3, 4), 0],
            ],
            [sympy.Rational(1, 4), sympy.Rational(3, 4), 0],
        )
        self.assertTrue(non_fsal.ssal)
        self.assertFalse(non_fsal.fsal)

        embedded = ButcherTableau(
            [[0, 0], [sympy.Rational(1, 2), 0]],
            [0, 1],
            [1, 0],
        )
        self.assertTrue(embedded.embedded)

    def test_butcher_tableau_rejects_invalid_shape(self):
        with self.assertRaises(ValueError):
            ButcherTableau([], [])
        with self.assertRaises(ValueError):
            ButcherTableau([[0, 0]], [1])
        with self.assertRaises(ValueError):
            ButcherTableau([[0, 0], [1, 0]], [0, 1], [1])

    def test_elementary_weights(self):
        # Test using an RK method of order 4
        scheme = rk4
        rk_weights = scheme.elementary_weights_map()

        for t in sample_trees:
            self.assertAlmostEqual(exact_weights(t), rk_weights(t))

    def test_order(self):
        methods = [
            euler,
            heun_rk2,
            midpoint,
            kutta_rk3,
            heun_rk3,
            ralston_rk3,
            rk4,
            ralston_rk4,
            nystrom_rk5,
            backward_euler,
            implicit_midpoint,
            crank_nicolson,
            gauss6,
            radau_iia,
            lobatto6,
        ]
        orders = [1, 2, 2, 3, 3, 3, 4, 4, 5, 1, 2, 2, 6, 5, 6]

        for m, ord in zip(methods, orders, strict=True):
            self.assertEqual(ord, m.order(), msg=m.name)

    def test_symbolic_weight(self):
        t = Tree([[], []])
        self.assertEqual("a10**2*b1 + b2*(a20 + a21)**2", str(rk_symbolic_weight(t, 3, True)))

    def test_order_cond(self):
        t = Tree([[], []])
        self.assertEqual("a10**2*b1 + b2*(a20 + a21)**2 - 1/3", str(rk_order_cond(t, 3, True)))

    def test_symbolic_weight_with_custom_weight_symbols(self):
        t = Tree([[], []])
        weight_symbols = [sympy.Symbol(f"bhat{i}") for i in range(3)]
        self.assertEqual(
            "a10**2*bhat1 + bhat2*(a20 + a21)**2",
            str(rk_symbolic_weight(t, 3, True, weight_symbols=weight_symbols)),
        )

    def test_inverse(self):
        method = rk4
        inv_method = method ** (-1)
        id = method * inv_method
        m = id.elementary_weights_map()

        for t in trees_up_to_order(5):
            self.assertAlmostEqual(bck.counit(t), m(t))

    #
    # def test_add(self):
    #     method1 = rk4
    #     method2 = euler
    #     sum_method = method1 + method2
    #
    #     m1 = method1.elementary_weights_map() + method2.elementary_weights_map()
    #     m2 = sum_method.elementary_weights_map()
    #
    #     for t in trees_up_to_order(5):
    #         if t == Tree(None):
    #             continue
    #         self.assertAlmostEqual(m1(t), m2(t), msg = repr(t))

    def test_ees25_order(self):
        rk = EES25(0.1)
        self.assertEqual(2, rk.order())
        self.assertEqual(5, rk.antisymmetric_order())

    def test_ees27_order(self):
        rk = EES27(0.1)
        self.assertEqual(2, rk.order())
        self.assertEqual(7, rk.antisymmetric_order())

    def test_rk_builder_type_and_order_one(self):
        methods, result = build_explicit_rk(order=1, stages=1, max_solutions=1)

        self.assertIsInstance(result, SolveResult)
        self.assertEqual(1, len(methods))
        self.assertEqual(1, methods[0].order())
        self.assertTrue(methods[0].explicit)
        self.assertAlmostEqual(1.0, methods[0].tableau.b[0])

    def test_rk_builder_order_two_stage_two_with_fixed_symbol(self):
        methods, _ = build_explicit_rk(
            order=2,
            stages=2,
            constraints=[SetConstraint(sympy.Symbol("b0"), sympy.Integer(0))],
            max_solutions=1,
        )

        self.assertEqual(1, len(methods))
        method = methods[0]
        self.assertTrue(method.explicit)
        self.assertEqual(2, method.order())
        self.assertAlmostEqual(0.5, method.tableau.a[1][0])
        self.assertAlmostEqual(0.0, method.tableau.b[0])
        self.assertAlmostEqual(1.0, method.tableau.b[1])

    def test_rk_builder_unsolved_system_returns_empty(self):
        methods, _ = build_explicit_rk(
            order=4,
            stages=2,
            max_solutions=1,
        )
        self.assertEqual(0, len(methods))

    def test_rk_builder_embedded_requires_order_at_least_two(self):
        with self.assertRaises(ValueError):
            build_explicit_rk(order=1, stages=1, embedded=True)

    def test_rk_builder_embedded_method(self):
        methods, _ = build_explicit_rk(
            order=2,
            stages=2,
            constraints=[
                SetConstraint(sympy.Symbol("b0"), sympy.Integer(0)),
                SetConstraint(sympy.Symbol("bhat0"), sympy.Integer(1)),
            ],
            embedded=True,
            max_solutions=1,
        )

        self.assertEqual(1, len(methods))
        method = methods[0]
        self.assertEqual(2, method.order())
        self.assertEqual([1.0, 0.0], method.tableau.b_hat)

        embedded_method = RK(ButcherTableau(a=method.tableau.a, b=method.tableau.b_hat))
        self.assertEqual(1, embedded_method.order())

    def test_rk_builder_embedded_family_message(self):
        methods, result = build_explicit_rk(
            order=2,
            stages=2,
            constraints=[SetConstraint(sympy.Symbol("b0"), sympy.Integer(0))],
            embedded=True,
            max_solutions=1,
        )

        self.assertEqual(0, len(methods))
        self.assertEqual(["bhat1"], result.free_symbols)
        self.assertIn(
            "No concrete method found; the solver returned a family parameterised by [bhat1].",
            str(result),
        )

    def test_rk_builder_embedded_rejects_degenerate_error_estimator(self):
        with self.assertRaisesRegex(ValueError, "b = bhat"):
            build_explicit_rk(
                order=2,
                stages=3,
                antisymmetric_order=5,
                constraints=[
                    SetConstraint(sympy.Symbol("b0"), sympy.Rational(1, 4)),
                    SetConstraint(sympy.Symbol("bhat0"), sympy.Rational(1, 4)),
                    SetConstraint(sympy.Symbol("bhat1"), sympy.Rational(1, 2)),
                ],
                embedded=True,
                max_solutions=1,
            )

    def test_rk_builder_embedded_accepts_exact_lower_order_estimator(self):
        methods, _ = build_explicit_rk(
            order=2,
            stages=3,
            antisymmetric_order=5,
            constraints=[
                SetConstraint(sympy.Symbol("b0"), sympy.Rational(1, 4)),
                SetConstraint(sympy.Symbol("bhat0"), sympy.Rational(1, 4)),
                SetConstraint(sympy.Symbol("bhat1"), sympy.Rational(1, 4)),
            ],
            embedded=True,
            max_solutions=1,
        )

        self.assertEqual(1, len(methods))
        self.assertEqual([0.25, 0.25, 0.5], methods[0].tableau.b_hat)

    def test_tableau_formatting_with_embedded_row(self):
        embedded = ButcherTableau(
            [[0, 0], [sympy.Rational(1, 2), 0]],
            [0, 1],
            [1, 0],
        )
        plain = ButcherTableau([[0, 0], [sympy.Rational(1, 2), 0]], [0, 1])

        text = format_tableau_text(embedded)
        latex = format_tableau_latex(embedded)

        self.assertIn("b_hat", text)
        self.assertIn(r"\hat{b}", latex)
        self.assertNotIn("b_hat", format_tableau_text(plain))
        self.assertNotIn(r"\hat{b}", format_tableau_latex(plain))
