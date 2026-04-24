"""Tests for Phases A–D: PlanarTree extensions, RK on ordered trees,
CF methods, and symbolic order-condition generation."""

import unittest

import sympy
from kauri import EES25, EES27, euler, rk4
from kauri.cf import CFMethod, ReusedStageCFMethod
from kauri.gentrees import planar_trees_of_order
from kauri.lb_substitution import (
    delta_w_terms,
    frozen_exponential_character,
)
from kauri.lb_substitution import (
    substitute as lb_substitute,
)
from kauri.maps import Map, exact_weights
from kauri.mkw.mkw import _as_basis_aware_map
from kauri.rk import RK, rk_order_cond, rk_symbolic_weight
from kauri.trees import (
    EMPTY_PLANAR_TREE,
    NoncommutativeForest,
    OrderedForest,
    PlanarTree,
)


def _collect_nonzero_defects(defect_map, max_order: int):
    defects_by_order = {}
    for n in range(1, max_order + 1):
        nonzero = []
        for tree in planar_trees_of_order(n):
            value = sympy.nsimplify(defect_map(tree), rational=True)
            if value != 0:
                nonzero.append((tree, value))
        if nonzero:
            defects_by_order[n] = nonzero
    return defects_by_order


def _format_defect_report(defects_by_order, *, checked_order: int) -> str:
    if not defects_by_order:
        return f"All antisymmetry defects vanish through degree {checked_order}."

    first_bad_order = min(defects_by_order)
    observed_antisymmetric_order = first_bad_order - 1
    lines = [
        "",
        "Lie--Butcher antisymmetry check failed.",
        "",
        "Mathematical condition:",
        "  D = (sign . alpha) *_MKW alpha",
        f"  D(tau) = epsilon(tau) for all planar trees |tau| <= {checked_order}.",
        "  Since tau is non-empty here, this means D(tau) = 0.",
        "",
        f"First nonzero defect occurs at |tau| = {first_bad_order}.",
        f"Therefore the observed antisymmetric order is {observed_antisymmetric_order}.",
        "",
        f"All nonzero defects through degree {checked_order}:",
    ]

    for n in sorted(defects_by_order):
        lines.append(f"  Order {n}:")
        for tree, value in defects_by_order[n]:
            lines.append(f"    D({tree}) = {sympy.sstr(value)}")

    return "\n".join(lines)


def _format_cfees_order_report(
    cf,
    *,
    forward_order: int,
    antisymmetric_order: int,
    normalize=None,
    zero=None,
    one=None,
    format_value=None,
):
    if normalize is None:
        normalize = lambda value: sympy.simplify(sympy.nsimplify(value, rational=True))
    if zero is None:
        zero = sympy.Integer(0)
    if one is None:
        one = sympy.Integer(1)
    if format_value is None:
        format_value = sympy.sstr

    alpha = cf.lb_character()
    defect = cf.symmetry_defect_map()
    lines = [
        "",
        f"{cf.name or 'CF method'} order-condition report",
        "",
        f"Forward order conditions through degree {forward_order}:",
    ]

    forward_failures = []
    for n in range(0, forward_order + 1):
        lines.append(f"  Order {n}:")
        for tree in planar_trees_of_order(n):
            expected = normalize(sympy.Rational(1, tree.factorial()))
            value = normalize(alpha(tree))
            residual = normalize(value - expected)
            status = "OK" if residual == zero else "FAIL"
            if residual != zero:
                forward_failures.append((tree, residual))
            lines.append(
                f"    {status}: alpha({tree}) = {format_value(value)}, "
                f"expected {format_value(expected)}, residual {format_value(residual)}"
            )

    lines.extend(
        [
            "",
            f"MKW/LB antisymmetry conditions through degree {antisymmetric_order}:",
        ]
    )

    antisymmetry_failures = []
    for n in range(0, antisymmetric_order + 1):
        lines.append(f"  Order {n}:")
        for tree in planar_trees_of_order(n):
            expected = one if tree == EMPTY_PLANAR_TREE else zero
            value = normalize(defect(tree))
            residual = normalize(value - expected)
            status = "OK" if residual == zero else "FAIL"
            if residual != zero:
                antisymmetry_failures.append((tree, residual))
            lines.append(
                f"    {status}: D({tree}) = {format_value(value)}, "
                f"expected {format_value(expected)}, residual {format_value(residual)}"
            )

    return "\n".join(lines), forward_failures, antisymmetry_failures


# ── Phase A: PlanarTree.factorial / sigma / density ──────────────────────


class TestPhaseA(unittest.TestCase):
    def test_factorial(self):
        self.assertEqual(PlanarTree(None).factorial(), 1)
        self.assertEqual(PlanarTree([]).factorial(), 1)
        self.assertEqual(PlanarTree([[]]).factorial(), 2)
        self.assertEqual(PlanarTree([[], []]).factorial(), 3)
        self.assertEqual(PlanarTree([[[]]]).factorial(), 6)

    def test_sigma_always_one(self):
        for n in range(5):
            for t in planar_trees_of_order(n):
                self.assertEqual(t.sigma(), 1, msg=repr(t.list_repr))

    def test_density(self):
        self.assertAlmostEqual(PlanarTree([[]]).density(), 1.0)
        self.assertAlmostEqual(PlanarTree([[], []]).density(), 0.5)

    def test_ncf_factorial(self):
        t1 = PlanarTree([])
        t2 = PlanarTree([[]])
        f = NoncommutativeForest((t1, t2))
        self.assertEqual(f.factorial(), 1 * 2)

    def test_exact_weights_on_planar(self):
        self.assertAlmostEqual(exact_weights(PlanarTree([[]])), 0.5)
        self.assertAlmostEqual(exact_weights(PlanarTree([[], []])), 1.0 / 3)
        self.assertAlmostEqual(exact_weights(PlanarTree([[[]]])), 1.0 / 6)


# ── Phase B: RK on ordered trees ────────────────────────────────────────


class TestPhaseB(unittest.TestCase):
    def test_symbolic_weight_on_planar(self):
        t = PlanarTree([[], []])
        w = rk_symbolic_weight(t, 2, True)
        self.assertEqual(str(w), "a10**2*b1")

    def test_order_cond_on_planar(self):
        t = PlanarTree([[], []])
        c = rk_order_cond(t, 2, True)
        self.assertEqual(str(c), "a10**2*b1 - 1/3")

    def test_rk4_planar_order(self):
        self.assertEqual(rk4.planar_order(), 4)

    def test_euler_planar_order(self):
        self.assertEqual(euler.planar_order(), 1)

    def test_ees25_planar(self):
        rk = EES25(0.1)
        self.assertEqual(rk.planar_order(), 2)
        self.assertEqual(rk.planar_antisymmetric_order(), 5)

    def test_ees27_planar(self):
        rk = EES27(0.1)
        self.assertEqual(rk.planar_order(), 2)
        self.assertEqual(rk.planar_antisymmetric_order(), 7)


# ── Phase C: CFMethod ───────────────────────────────────────────────────


class TestPhaseC(unittest.TestCase):
    def _ees25_params(self):
        a = [[0, 0, 0], [0.5, 0, 0], [0, 1, 0]]
        b = [0.25, 0.5, 0.25]
        return a, b

    def _true_cfees25(self):
        return ReusedStageCFMethod(
            [sympy.Rational(-7, 15), sympy.Rational(-35, 32)],
            [sympy.Rational(1, 3), sympy.Rational(15, 16), sympy.Rational(2, 5)],
            name="CF-EES(2,5;1/10)",
        )

    def test_single_exponential_matches_rk(self):
        """J=1 CF should give the same planar ORDER as the underlying RK (both
        evaluate tree character values, which agree).  The ANTISYMMETRIC
        order can differ: ``RK.planar_antisymmetric_order`` uses the NCK
        convolution for the symmetry defect, whereas
        ``CFMethod.planar_antisymmetric_order`` uses the MKW convolution
        (the Lie-group convention).  EES25 was designed to satisfy NCK's
        antisymmetric order 5; under MKW it is 3."""
        a, b = self._ees25_params()
        cf = CFMethod(a, [b])
        self.assertEqual(cf.planar_order(), 2)
        self.assertEqual(cf.planar_antisymmetric_order(), 3)

    def test_projected_rk(self):
        a, b = self._ees25_params()
        betas = [[0.25, 0, 0], [0, 0.5, 0], [0, 0, 0.25]]
        cf = CFMethod(a, betas)
        rk = cf.projected_rk()
        self.assertEqual(rk.order(), 2)

    def test_lb_character_order_0_and_1(self):
        """LB character should satisfy alpha(empty)=1 and alpha(bullet)=sum(b) regardless of J."""
        a, b = self._ees25_params()
        betas = [[0.25, 0, 0], [0, 0.5, 0], [0, 0, 0.25]]
        cf = CFMethod(a, betas)
        alpha = cf.lb_character()
        self.assertAlmostEqual(alpha(EMPTY_PLANAR_TREE), 1.0)
        self.assertAlmostEqual(alpha(PlanarTree([])), 1.0)

    def test_reused_stage_projected_rk(self):
        cf = self._true_cfees25()
        rk = cf.projected_rk()
        expected_a = [
            [0, 0, 0],
            [sympy.Rational(1, 3), 0, 0],
            [sympy.Rational(-5, 48), sympy.Rational(15, 16), 0],
        ]
        expected_b = [
            sympy.Rational(1, 10),
            sympy.Rational(1, 2),
            sympy.Rational(2, 5),
        ]
        self.assertEqual(rk.a, expected_a)
        self.assertEqual(rk.b, expected_b)

    def test_delta_w_reference_contraction_example(self):
        """Delta_W enumerates admissible contractions with commutative left factors."""

        bullet = PlanarTree([])
        chain2 = PlanarTree([[]])
        chain3 = PlanarTree([[[]]])
        cherry = PlanarTree([[], []])
        tree = PlanarTree([[], [[]]])

        def forest_key(forest):
            return tuple(t.list_repr for t in forest.tree_list if t.list_repr is not None)

        def left_key(left_factors):
            return tuple(
                sorted(
                    (forest_key(factor) for factor in left_factors),
                    key=repr,
                )
            )

        def sym_key(*factors):
            return tuple(sorted(factors, key=repr))

        observed = {}
        for coeff, left_factors, right_forest in delta_w_terms(tree.as_ordered_forest()):
            key = (left_key(left_factors), forest_key(right_forest))
            observed[key] = observed.get(key, 0) + coeff

        expected = {
            (
                sym_key(*(((bullet.list_repr,),) * 4)),
                (tree.list_repr,),
            ): 1,
            (
                sym_key((bullet.list_repr,), (chain3.list_repr,)),
                (chain2.list_repr,),
            ): 1,
            (
                sym_key((bullet.list_repr,), (cherry.list_repr,)),
                (chain2.list_repr,),
            ): 1,
            (
                sym_key((bullet.list_repr,), (bullet.list_repr,), (chain2.list_repr,)),
                (cherry.list_repr,),
            ): 3,
            (
                sym_key(
                    (bullet.list_repr, bullet.list_repr),
                    (bullet.list_repr,),
                    (bullet.list_repr,),
                ),
                (chain3.list_repr,),
            ): 1,
            (
                sym_key((bullet.list_repr, chain2.list_repr), (bullet.list_repr,)),
                (chain2.list_repr,),
            ): 1,
            (
                ((tree.list_repr,),),
                (bullet.list_repr,),
            ): 1,
        }
        self.assertEqual(observed, expected)

    def test_delta_w_substitution_preserves_frozen_exponential_forests(self):
        """A frozen bullet exponential has shuffle-character forest values."""

        bullet = PlanarTree([])
        chain = PlanarTree([[]])
        alpha = lb_substitute(
            _as_basis_aware_map(
                lambda x: 1 if x == bullet or x == bullet.as_ordered_forest() else 0
            ),
            frozen_exponential_character(sympy.Rational(1, 3)),
        )

        self.assertEqual(alpha(bullet), sympy.Rational(1, 3))
        self.assertEqual(alpha(chain), 0)
        self.assertEqual(alpha(bullet * bullet), sympy.Rational(1, 18))

    def test_reused_stage_rows_start_from_base_point(self):
        cf = ReusedStageCFMethod(
            [sympy.Rational(0)],
            [sympy.Rational(1, 2), sympy.Rational(1, 2)],
            name="two half exponentials",
        )
        alpha = cf.lb_character()
        self.assertEqual(alpha(EMPTY_PLANAR_TREE), 1)
        self.assertEqual(sympy.simplify(alpha(PlanarTree([])) - 1), 0)
        self.assertEqual(
            sympy.simplify(alpha(PlanarTree([[]])) - sympy.Rational(1, 4)),
            0,
        )
        self.assertEqual(
            sympy.simplify(alpha(PlanarTree([]) * PlanarTree([]))),
            sympy.Rational(1, 2),
        )

    def test_reused_stage_mkw_lb_character_uses_owren_forest_values(self):
        cf = self._true_cfees25()
        alpha = cf.lb_character()
        rk = cf.projected_rk().elementary_weights_map()
        bullet = PlanarTree([])
        chain = PlanarTree([[]])
        cherry = PlanarTree([[], []])

        self.assertEqual(alpha(PlanarTree([])), sympy.Rational(1))
        self.assertEqual(alpha(chain), sympy.Rational(1, 2))
        self.assertEqual(rk(chain), sympy.Rational(1, 2))
        self.assertEqual(alpha(cherry), sympy.Rational(1, 6))
        self.assertEqual(rk(cherry), sympy.Rational(1, 3))
        self.assertEqual(alpha(bullet * chain), sympy.Rational(17, 48))


# ── Phase D: Symbolic order conditions ──────────────────────────────────


class TestPhaseD(unittest.TestCase):
    def test_symbolic_lb_character_j1(self):
        from kauri.manifold_ees import symbolic_cf_params, symbolic_lb_character

        a, betas = symbolic_cf_params(3, 1, explicit=True)
        t = PlanarTree([[]])
        val = symbolic_lb_character(t, a, betas, 3, 1)
        # Should match rk_symbolic_weight with the same symbols
        expected = rk_symbolic_weight(t, 3, explicit=True)
        # Substitute: beta00->b0, beta01->b1, beta02->b2
        mapping = {
            sympy.Symbol("beta00"): sympy.Symbol("b0"),
            sympy.Symbol("beta01"): sympy.Symbol("b1"),
            sympy.Symbol("beta02"): sympy.Symbol("b2"),
        }
        self.assertEqual(sympy.expand(val.subs(mapping) - expected), 0)

    def test_forward_conditions_count(self):
        from kauri.manifold_ees import generate_conditions

        result = generate_conditions(2, 3, s=3, J=1, explicit=True)
        # Order 1: 1 tree -> 1 condition; Order 2: 1 tree -> 1 condition
        self.assertEqual(len(result["forward"]), 2)

    def test_ees25_satisfies_conditions(self):
        """EES25 was designed under the NCK convention to have
        antisymmetric order 5; under the MKW (Lie-group) convention
        used by :func:`generate_conditions`, its antisymmetric order is
        3 — still a valid forward-order-2 / antisymmetric-order-3 method."""
        from kauri.manifold_ees import generate_conditions, verify_conditions

        result = generate_conditions(2, 3, s=3, J=1, explicit=True)
        subs = {
            sympy.Symbol("a10"): sympy.Rational(1, 2),
            sympy.Symbol("a20"): 0,
            sympy.Symbol("a21"): 1,
            sympy.Symbol("beta00"): sympy.Rational(1, 4),
            sympy.Symbol("beta01"): sympy.Rational(1, 2),
            sympy.Symbol("beta02"): sympy.Rational(1, 4),
        }
        ok, idx, resid = verify_conditions(result["all"], subs)
        self.assertTrue(ok, msg=f"Condition {idx} failed: {resid}")

    def test_mathematica_export(self):
        from kauri.manifold_ees import generate_conditions, mathematica_export

        result = generate_conditions(1, 1, s=2, J=1, explicit=True)
        code = mathematica_export(result["forward"])
        self.assertIn("== 0", code)

    def test_groebner_basis(self):
        from kauri.manifold_ees import generate_conditions, groebner_basis

        result = generate_conditions(1, 1, s=2, J=1, explicit=True)
        if result["forward"]:
            gb = groebner_basis(result["forward"])
            self.assertIsNotNone(gb)


# ── EES character verification ─────────────────────────────────────────


class TestVerifyEESCharacter(unittest.TestCase):
    def test_counit_satisfies_ees(self):
        """The counit satisfies the EES (odd) condition up to order 5."""
        from kauri.manifold_ees import verify_ees_character

        counit = Map(lambda tree: 1 if tree == EMPTY_PLANAR_TREE else 0)
        self.assertTrue(verify_ees_character(counit, 5))

    def test_constant_map_fails_ees(self):
        """A constant map violates the EES condition."""
        from kauri.manifold_ees import verify_ees_character

        self.assertFalse(verify_ees_character(Map(lambda tree: 1), 3))

    def test_cfees25_mkw_defects_vanish_through_degree5(self):
        """Check the MKW/LB antisymmetry defect

            D = (sign . alpha) *_MKW alpha

        through degree 5 for the true reused-stage CF-EES(2,5;1/10)
        recurrence. This is intentionally the MKW/LB object check using
        the reused Owren rows and their actual ordered-forest values, not
        the projected NCK/RK check.
        """
        cf = ReusedStageCFMethod(
            [sympy.Rational(-7, 15), sympy.Rational(-35, 32)],
            [sympy.Rational(1, 3), sympy.Rational(15, 16), sympy.Rational(2, 5)],
            name="CF-EES(2,5;1/10)",
        )
        forward_order = 2
        antisymmetric_order = 6
        report, forward_failures, antisymmetry_failures = _format_cfees_order_report(
            cf,
            forward_order=forward_order,
            antisymmetric_order=antisymmetric_order,
        )
        print(report)

        self.assertEqual(forward_failures, [], msg=report)
        self.assertEqual(antisymmetry_failures, [], msg=report)
        self.assertEqual(cf.planar_order(limit=forward_order + 2), forward_order)
        self.assertEqual(
            cf.planar_antisymmetric_order(limit=antisymmetric_order + 1),
            antisymmetric_order,
        )

    def test_cfees27_mkw_defects_vanish_through_degree7(self):
        """Check the MKW/LB antisymmetry defect for the CF-EES(2,7) recurrence."""
        sqrt2 = sympy.sqrt(2)
        field = sympy.QQ.algebraic_field(sqrt2)

        def exact(value):
            return field.convert(value)

        def show(value):
            return sympy.sstr(field.to_sympy(value))

        cf = ReusedStageCFMethod(
            [
                exact((-7 + 4 * sqrt2) / 3),
                exact(-(4 + 5 * sqrt2) / 12),
                exact(3 * (-31 + 8 * sqrt2) / 49),
            ],
            [
                exact((2 - sqrt2) / 3),
                exact((4 + sqrt2) / 8),
                exact(3 * (3 - sqrt2) / 7),
                exact((9 - 4 * sqrt2) / 14),
            ],
            name="CF-EES(2,7;(5 - 3 sqrt(2))/14)",
        )
        forward_order = 2
        antisymmetric_order = 8
        report, forward_failures, antisymmetry_failures = _format_cfees_order_report(
            cf,
            forward_order=forward_order,
            antisymmetric_order=antisymmetric_order,
            normalize=exact,
            zero=field.zero,
            one=field.one,
            format_value=show,
        )
        print(report)

        self.assertEqual(forward_failures, [], msg=report)
        self.assertEqual(antisymmetry_failures, [], msg=report)

        degree8_failures = []
        defect = cf.symmetry_defect_map()
        for tree in planar_trees_of_order(antisymmetric_order + 1):
            value = exact(defect(tree))
            if value != field.zero:
                degree8_failures.append((tree, value))
        self.assertTrue(degree8_failures)

    def test_mkw_composition_diagnostic_alias(self):
        cf = ReusedStageCFMethod(
            [sympy.Rational(-7, 15), sympy.Rational(-35, 32)],
            [sympy.Rational(1, 3), sympy.Rational(15, 16), sympy.Rational(2, 5)],
            name="CF-EES(2,5;1/10)",
        )
        self.assertIs(
            cf.mkw_composition_symmetry_defect_map(),
            cf.symmetry_defect_map(),
        )


if __name__ == "__main__":
    unittest.main()
