import unittest

from kauri.numerics.methods.cf import verify_cf_ees
from kauri.numerics.methods.williamson import rk_to_williamson_2n
from kauri.numerics.planar_trees.planar_basis import PlanarTree
from kauri.numerics.rk.rk_maker import build_method_from_ansatz
from kauri.numerics.rk.rk_methods import euler
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
        cf_method = rk_to_williamson_2n(euler).to_cf()
        result = verify_cf_ees(cf_method, order=2)
        self.assertFalse(result.passed)
        self.assertGreater(result.checked_elements, 0)

    def test_to_cf_and_verify_smoke(self) -> None:
        rk_methods, _ = build_method_from_ansatz(
            order=1,
            stages=1,
            max_solutions=1,
        )
        self.assertGreaterEqual(len(rk_methods), 1)
        for method in rk_methods:
            cf = rk_to_williamson_2n(method).to_cf()
            verification = verify_cf_ees(cf, order=2)
            self.assertIsNotNone(verification)
            self.assertGreater(verification.checked_elements, 0)


if __name__ == "__main__":
    unittest.main()
