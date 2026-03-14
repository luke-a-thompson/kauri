import unittest

from kauri.methods.cf import verify_cf_ees
from kauri.methods.rk_catalog import euler
from kauri.planar_trees.planar_basis import PlanarTree
from kauri.rk_builder.rk_maker import build_explicit_rk
from kauri.trees.gentrees import planar_trees_of_order


class CFVerificationPipelineTests(unittest.TestCase):
    def test_planar_tree_ordering_distinguishes_siblings(self) -> None:
        left_heavy = PlanarTree([[[]], []])
        right_heavy = PlanarTree([[], [[]]])
        self.assertNotEqual(left_heavy, right_heavy)
        self.assertNotEqual(hash(left_heavy), hash(right_heavy))

    def test_planar_tree_generation_counts_small_orders(self) -> None:
        expected = {1: 1, 2: 1, 3: 2, 4: 5}
        for order, count in expected.items():
            trees = list(planar_trees_of_order(order))
            self.assertEqual(count, len(trees))

    def test_williamson_lift_smoke(self) -> None:
        cf_method = euler.to_williamson().to_cf()
        result = verify_cf_ees(cf_method, order=2)
        self.assertFalse(result.passed)
        self.assertGreater(result.checked_elements, 0)

    def test_to_cf_and_verify_smoke(self) -> None:
        generated_methods, _ = build_explicit_rk(
            order=1,
            stages=1,
            max_solutions=1,
        )
        self.assertGreaterEqual(len(generated_methods), 1)
        for method in generated_methods:
            cf = method.to_williamson().to_cf()
            verification = verify_cf_ees(cf, order=2)
            self.assertIsNotNone(verification)
            self.assertGreater(verification.checked_elements, 0)


if __name__ == "__main__":
    unittest.main()
