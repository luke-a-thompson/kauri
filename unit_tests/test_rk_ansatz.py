import unittest

import sympy

from kauri import make_explicit_rk_methods
from kauri.ansatz_2n import TwoNStorageAnsatz, generate_2n_aform_constraints
from kauri.rk_constraints import Constraint, compile_constraints


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
        equations = generate_2n_aform_constraints(stages=4)
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

    def test_rk_maker_2n_ansatz_smoke(self):
        result = make_explicit_rk_methods(
            order=1,
            stages=3,
            ansatz=TwoNStorageAnsatz(),
            fixed_values={
                "b0": 0,
                "b1": 0,
                "b2": 1,
                "a10": 1,
                "a21": 0,
            },
            max_solutions=1,
            solver="grobner",
        )
        self.assertEqual("2n_storage", result.ansatz)
        self.assertGreaterEqual(len(result.methods), 1)
