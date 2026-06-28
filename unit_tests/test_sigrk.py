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
from fractions import Fraction

from kauri.sigrk import (
    EdgeRuleTemplate,
    Guard,
    LabeledTree,
    StageFamily,
    SupportPrototype,
    UpdateRuleTemplate,
    enumerate_edge_rule_templates,
    enumerate_rule_sets,
    enumerate_stage_family_templates,
    exact_weight,
    expand_template,
    known_order2_recovery_problem,
    known_order2_template,
    known_order3_template,
    layer_count_polynomial,
    order_conditions,
    search_bounded,
    shuffle_counts,
    stage_count_polynomial,
    verify_all_dimensions,
    verify_discovery,
)


class SigRKTests(unittest.TestCase):
    def test_shuffle_counts_and_exact_weights(self):
        self.assertEqual(shuffle_counts((("1",), ("2",))), {("1", "2"): 1, ("2", "1"): 1})

        tree = LabeledTree("2", (LabeledTree("1"),))
        self.assertEqual(exact_weight(tree).coefficient(0, ("1", "2")), 1)

    def test_known_order_two_recovery(self):
        problem = known_order2_recovery_problem(d=2)

        self.assertEqual(len(problem.solutions), 1)
        solution = problem.solutions[0]
        variables = {str(variable): variable for variable in problem.variables}
        self.assertEqual(solution[variables["A"]], 1)
        self.assertEqual(solution[variables["B"]], 1)
        self.assertEqual(solution[variables["C_word"]], 1)
        self.assertEqual(solution[variables["C_empty"]], -1)

        template, _variables = known_order2_template()
        self.assertEqual(order_conditions(template, order=2, d=3, alpha=Fraction(1, 2)), ())

    def test_bounded_enumeration_recovers_known_order_two(self):
        families = (
            StageFamily("K", (), layer=0),
            StageFamily("A", ("kp",), layer=1),
        )
        family_templates = enumerate_stage_family_templates(families, 2, 2, required_names=("K",))
        self.assertEqual(family_templates, (families,))

        edge_pool = enumerate_edge_rule_templates(families, ("l",))
        update_pool = (
            UpdateRuleTemplate("K", "k", "K:k"),
            UpdateRuleTemplate("A", "k", "A:k"),
        )
        edge_rule_sets = enumerate_rule_sets(edge_pool, 1, 1)
        update_rule_sets = ((update_pool[0], update_pool[1]),)
        self.assertEqual(edge_rule_sets, ((EdgeRuleTemplate("K", "A", "l", "K->A:l"),),))

        word_lkp = SupportPrototype(Fraction(0), ("l", "kp"))
        word_k = SupportPrototype(Fraction(0), ("k",))
        empty = SupportPrototype(Fraction(0), ())
        delta_empty = SupportPrototype(Fraction(0), (), Guard((("k", "kp"),)))

        results = search_bounded(
            family_templates,
            edge_rule_sets,
            update_rule_sets,
            order=2,
            d=2,
            alpha=Fraction(1, 2),
            edge_support_options={"K->A:l": ((word_lkp,),)},
            update_support_options={
                "K:k": ((word_k, empty),),
                "A:k": ((delta_empty,),),
            },
        )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].impossible)
        self.assertEqual(len(results[0].solutions), 1)
        solution = results[0].solutions[0]
        edge, base_word, base_empty, diagonal = results[0].variables

        try:
            import sympy as sp
        except ImportError:  # pragma: no cover
            self.fail("sympy is required for symbolic SigRK search tests")

        values = {variable: solution.get(variable, variable) for variable in results[0].variables}
        self.assertEqual(sp.simplify(values[edge] * values[diagonal] - 1), 0)
        self.assertEqual(sp.simplify(values[base_word] - 1), 0)
        self.assertEqual(sp.simplify(values[base_empty] + values[diagonal]), 0)

    def test_known_order_two_expanded_dag_widths(self):
        template, _variables = known_order2_template()

        self.assertEqual(expand_template(template, 2).layer_widths(), (1, 2))
        self.assertEqual(expand_template(template, 3).layer_widths(), (1, 3))

    def test_known_order_three_template_counts_and_dag_widths(self):
        template = known_order3_template()

        self.assertEqual(stage_count_polynomial(template), {0: 1, 1: 4, 2: 3})
        self.assertEqual(layer_count_polynomial(template), ({0: 1}, {1: 2, 2: 3}, {1: 2}))
        self.assertEqual(expand_template(template, 2).layer_widths(), (1, 16, 4))
        self.assertEqual(expand_template(template, 3).layer_widths(), (1, 33, 6))

    def test_known_order_three_discovery_conditions_vanish(self):
        template = known_order3_template()

        self.assertEqual(order_conditions(template, order=3, d=2, alpha=Fraction(1, 3)), ())
        self.assertEqual(order_conditions(template, order=3, d=3, alpha=Fraction(1, 3)), ())
        self.assertTrue(verify_discovery(template, order=3, d=3, alpha=Fraction(1, 3)).passed)

    def test_proof_mode_is_not_claimed(self):
        with self.assertRaises(NotImplementedError):
            verify_all_dimensions(known_order3_template(), order=3)


if __name__ == "__main__":
    unittest.main()
