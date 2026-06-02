import unittest
from kauri import (Tree, PlanarTree, OrderedForest, ident, bck,
                   trees_of_order, colored_trees_of_order, colored_trees_up_to_order,
                   planar_trees_of_order,
                   colored_planar_trees_of_order, colored_planar_trees_up_to_order,
                   ordered_forests_of_order, colored_ordered_forests_of_order,
                   colored_ordered_forests_up_to_order, colored_ordered_forests,
                   colored_ordered_forest_to_idx, idx_to_colored_ordered_forest)
from kauri.utils import _list_repr_to_color_sequence, _apply_color_sequence


class ColoredTreeGenerationTests(unittest.TestCase):

    def test_d1_matches_unlabelled_counts(self):
        """With 1 color, colored trees == unlabelled trees."""
        for n in range(1, 6):
            unlabelled_count = sum(1 for _ in trees_of_order(n))
            colored_count = sum(1 for _ in colored_trees_of_order(n, 1))
            self.assertEqual(unlabelled_count, colored_count, f"order {n}")

    def test_d2_known_counts(self):
        """Hand-verified counts for d=2."""
        # order 1: 2 single-node trees (color 0 or 1)
        # order 2: 4 trees (2 root colors x 2 child colors, no symmetry)
        # order 3: 14 (shape [[]] gives 8; shape [[],[]] gives 6 after dedup)
        expected = {1: 2, 2: 4, 3: 14}
        for n, count in expected.items():
            result = sum(1 for _ in colored_trees_of_order(n, 2))
            self.assertEqual(count, result, f"order {n}, d=2")

    def test_correctness_nodes_and_colors(self):
        """All generated trees have correct node count and valid colors."""
        d = 3
        for n in range(1, 5):
            for t in colored_trees_of_order(n, d):
                self.assertEqual(t.nodes(), n)
                self.assertLessEqual(t.colors(), d)

    def test_no_duplicates(self):
        """No duplicate trees in output."""
        for n in range(1, 5):
            for d in range(1, 4):
                trees = list(colored_trees_of_order(n, d))
                self.assertEqual(len(trees), len(set(trees)),
                                 f"duplicates at order {n}, d={d}")

    def test_bck_antipode_property(self):
        """antipode * ident == counit for all colored trees up to order 4, d=2."""
        m = bck.antipode * ident
        for t in colored_trees_up_to_order(4, 2):
            self.assertEqual(bck.counit(t), m(t), f"failed for {t}")

    def test_round_trip_color_sequence(self):
        """Extract color sequence from a colored tree, reapply to its shape, get the same tree."""
        for t in colored_trees_of_order(3, 3):
            colors = _list_repr_to_color_sequence(t.list_repr)
            reconstructed = Tree(_apply_color_sequence(t.unlabelled_repr, iter(colors)))
            self.assertEqual(t, reconstructed)


class ColoredPlanarTreeGenerationTests(unittest.TestCase):

    def test_d1_matches_uncoloured_counts(self):
        """With 1 color, colored planar trees == uncoloured planar trees."""
        for n in range(1, 6):
            uncoloured = sum(1 for _ in planar_trees_of_order(n))
            coloured = sum(1 for _ in colored_planar_trees_of_order(n, 1))
            self.assertEqual(uncoloured, coloured, f"order {n}")

    def test_counts_are_catalan_times_d_pow_n(self):
        """Planar trees have no symmetry, so count = catalan(n) * d^n."""
        for n in range(1, 6):
            catalan_count = sum(1 for _ in planar_trees_of_order(n))
            for d in range(1, 4):
                expected = catalan_count * d ** n
                result = sum(1 for _ in colored_planar_trees_of_order(n, d))
                self.assertEqual(expected, result, f"order {n}, d={d}")

    def test_yields_planar_trees(self):
        """All generated objects are PlanarTree instances."""
        for t in colored_planar_trees_of_order(3, 2):
            self.assertIsInstance(t, PlanarTree)

    def test_correctness_nodes(self):
        """All generated trees have correct node count."""
        d = 3
        for n in range(1, 5):
            for t in colored_planar_trees_of_order(n, d):
                self.assertEqual(t.nodes(), n)

    def test_no_duplicates(self):
        """No duplicate planar trees in output."""
        for n in range(1, 5):
            for d in range(1, 4):
                trees = list(colored_planar_trees_of_order(n, d))
                self.assertEqual(len(trees), len(set(trees)),
                                 f"duplicates at order {n}, d={d}")

    def test_up_to_order_includes_empty(self):
        """colored_planar_trees_up_to_order(n, d) starts with the empty tree."""
        trees = list(colored_planar_trees_up_to_order(0, 2))
        self.assertEqual(len(trees), 1)
        self.assertEqual(trees[0].nodes(), 0)

    def test_round_trip_color_sequence(self):
        """Extract color sequence, reapply to shape, get the same planar tree."""
        for t in colored_planar_trees_of_order(3, 3):
            colors = _list_repr_to_color_sequence(t.list_repr)
            reconstructed = PlanarTree(_apply_color_sequence(t.unlabelled_repr, iter(colors)))
            self.assertEqual(t, reconstructed)


class ColoredOrderedForestGenerationTests(unittest.TestCase):

    def test_d1_counts_are_next_catalan_numbers(self):
        """Ordered forests with n nodes are counted by Catalan(n)."""
        expected = {0: 1, 1: 1, 2: 2, 3: 5, 4: 14}
        for n, count in expected.items():
            result = sum(1 for _ in ordered_forests_of_order(n))
            self.assertEqual(count, result, f"order {n}")

    def test_colored_counts_are_d_pow_n_times_uncolored(self):
        """Planar forest shapes have no symmetry."""
        for n in range(0, 5):
            uncolored = sum(1 for _ in ordered_forests_of_order(n))
            for d in range(1, 4):
                colored = sum(1 for _ in colored_ordered_forests_of_order(n, d))
                self.assertEqual(uncolored * d ** n, colored, f"order {n}, d={d}")

    def test_yields_ordered_forests(self):
        """All generated objects are OrderedForest instances."""
        for f in colored_ordered_forests_of_order(3, 2):
            self.assertIsInstance(f, OrderedForest)

    def test_order_two_lists_words_before_singletons(self):
        """Enumeration is by total order, then recursive forest word order."""
        forests = list(colored_ordered_forests_of_order(2, 2))
        self.assertEqual([f.num_trees() for f in forests], [2, 2, 2, 2, 1, 1, 1, 1])

    def test_up_to_order_starts_with_empty(self):
        forests = list(colored_ordered_forests_up_to_order(0, 2))
        self.assertEqual(len(forests), 1)
        self.assertEqual(forests[0].nodes(), 0)

    def test_index_round_trip(self):
        forests = colored_ordered_forests(2, 3)
        for idx, forest in enumerate(forests):
            self.assertEqual(colored_ordered_forest_to_idx(forest, 2, 3), idx)
            self.assertEqual(idx_to_colored_ordered_forest(idx, 2, 3), forest)


if __name__ == "__main__":
    unittest.main()
