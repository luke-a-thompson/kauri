import unittest

from kauri.numerics.cf.cf_verify import verify_cf_ees
from kauri.numerics.cf.cf_williamson import lift_to_cf, rk_to_williamson_2n
from kauri.numerics.planar_trees.planar_basis import PlanarTree
from kauri.numerics.planar_trees.planar_gentrees import planar_trees_of_order
from kauri.numerics.cf.rk_maker_cf_pipeline import build_and_verify_cf_methods
from kauri.numerics.rk.rk_methods import euler


class CFVerificationPipelineTests(unittest.TestCase):
    def test_planar_tree_ordering_distinguishes_siblings(self) -> None:
        left_heavy = PlanarTree([[[]], []])
        right_heavy = PlanarTree([[], [[]]])
        self.assertNotEqual(left_heavy, right_heavy)
        self.assertNotEqual(hash(left_heavy), hash(right_heavy))

    def test_planar_tree_generation_counts_small_orders(self) -> None:
        # Number of rooted planar trees with n nodes: Catalan(n-1) for n >= 1.
        expected = {1: 1, 2: 1, 3: 2, 4: 5}
        for order, count in expected.items():
            trees = list(planar_trees_of_order(order))
            self.assertEqual(count, len(trees))

    def test_williamson_lift_smoke(self) -> None:
        williamson = rk_to_williamson_2n(euler)
        cf_method = lift_to_cf(williamson)
        result = verify_cf_ees(cf_method, order=2)
        self.assertFalse(result.passed)
        self.assertGreater(result.checked_elements, 0)

    def test_pipeline_smoke(self) -> None:
        result = build_and_verify_cf_methods(
            order=1,
            stages=1,
            verification_order=2,
            max_solutions=1,
            solver="grobner",
        )
        self.assertGreaterEqual(len(result.rk_result.methods), 1)
        self.assertGreaterEqual(len(result.cf_methods), 1)
        self.assertEqual(len(result.cf_methods), len(result.verification_results))


if __name__ == "__main__":
    unittest.main()
