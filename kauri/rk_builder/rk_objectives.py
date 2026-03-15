"""Objective scoring and optimization hooks for generated RK methods."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

import sympy

from kauri.hopf_algebras.bck import counit
from kauri.hopf_algebras.maps import exact_weights, sign
from kauri.methods.rk import RK
from kauri.trees.gentrees import trees_of_order


class RKObjective(Protocol):
    """
    Interface for scoring a generated RK method.
    """

    def evaluate(self, method: RK) -> float:
        """
        Return objective value (lower is better by convention).
        """
        ...


@dataclass(frozen=True)
class ObjectiveTerm:
    objective: RKObjective
    weight: float = 1.0


@dataclass(frozen=True)
class RKOptimizationConfig:
    maxiter: int = 300
    bounds: dict[str, tuple[float, float]] | None = None
    n_restarts: int = 3
    max_denominator: int = 16
    relation_tolerance: float = 1e-8


@dataclass(frozen=True)
class MethodScore:
    method_name: str
    scores: dict[str, float]


class LeadingErrorObjective:
    """
    Minimize squared rooted-tree defects at selected leading orders.

    - If `order` is set, include the classical RK leading defect at `C(order + 1)`.
    - If `antisymmetric_order` is set, include the antisymmetric leading defect at
      `EC(antisymmetric_order + 2)`.
    """

    def __init__(self, order: int | None = None, antisymmetric_order: int | None = None):
        if order is None and antisymmetric_order is None:
            raise ValueError("set at least one of `order` or `antisymmetric_order`")
        if order is not None and order <= 0:
            raise ValueError("order must be positive")
        if antisymmetric_order is not None and antisymmetric_order <= 0:
            raise ValueError("antisymmetric_order must be positive")
        self.leading_trees = tuple(trees_of_order(order + 1)) if order is not None else tuple()
        self.antisymmetric_trees = (
            tuple(trees_of_order(antisymmetric_order + 2))
            if antisymmetric_order is not None
            else tuple()
        )

    def evaluate(self, method: RK) -> float:
        weights = method.elementary_weights_map()
        score = sum(
            float((sympy.sympify(weights(tree)) - sympy.sympify(exact_weights(tree))) ** 2)
            for tree in self.leading_trees
        )
        if self.antisymmetric_trees:
            antisymmetric_weights = (weights & sign) * weights
            score += sum(
                float(
                    (sympy.sympify(antisymmetric_weights(tree)) - sympy.sympify(counit(tree))) ** 2
                )
                for tree in self.antisymmetric_trees
            )
        return score


def _objective_label(objective: RKObjective) -> str:
    return objective.__class__.__name__


def score_methods(methods: list[RK], objectives: list[RKObjective]) -> list[MethodScore]:
    """
    Evaluate methods against a list of objectives.
    """
    output: list[MethodScore] = []
    for method in methods:
        scores: dict[str, float] = {}
        for objective in objectives:
            scores[_objective_label(objective)] = float(objective.evaluate(method))
        method_name = method.name if method.name is not None else "unnamed_method"
        output.append(MethodScore(method_name=method_name, scores=scores))
    return output


def _coerce_terms(objectives: Sequence[RKObjective | ObjectiveTerm]) -> list[ObjectiveTerm]:
    terms: list[ObjectiveTerm] = []
    for objective in objectives:
        if isinstance(objective, ObjectiveTerm):
            terms.append(objective)
        else:
            terms.append(ObjectiveTerm(objective=objective))
    return terms


def _weighted_score(method: RK, terms: Sequence[ObjectiveTerm]) -> float:
    return sum(term.weight * float(term.objective.evaluate(method)) for term in terms)


def _solution_has_free_symbols(named_solution: dict[str, sympy.core.basic.Basic]) -> bool:
    return any(sympy.sympify(value).free_symbols for value in named_solution.values())


def _substitute_named_solution(
    named_solution: dict[str, sympy.core.basic.Basic],
    substitutions: dict[str, sympy.core.basic.Basic],
) -> dict[str, sympy.core.basic.Basic]:
    if not substitutions:
        return {name: sympy.sympify(value) for name, value in named_solution.items()}
    symbol_subs = [
        (sympy.symbols(name), sympy.sympify(value)) for name, value in substitutions.items()
    ]
    return {
        name: sympy.simplify(sympy.expand(sympy.sympify(value).subs(symbol_subs)))
        for name, value in named_solution.items()
    }


def _free_relation_residual(
    *,
    named_solution: dict[str, sympy.core.basic.Basic],
    free_symbol_names: Sequence[str],
    free_symbol_relations: Sequence[sympy.core.basic.Basic],
) -> float:
    if not free_symbol_relations:
        return 0.0
    free_subs: list[tuple[sympy.Symbol, sympy.core.basic.Basic]] = []
    for free_name in free_symbol_names:
        if free_name not in named_solution:
            continue
        value = sympy.sympify(named_solution[free_name])
        if value.free_symbols:
            continue
        free_subs.append((sympy.symbols(free_name), value))
    max_residual = 0.0
    for relation in free_symbol_relations:
        resolved = sympy.simplify(sympy.expand(sympy.sympify(relation).subs(free_subs)))
        if resolved.free_symbols:
            return math.inf
        try:
            residual = abs(float(sympy.N(resolved, 30)))
        except TypeError:
            return math.inf
        if math.isnan(residual) or math.isinf(residual):
            return math.inf
        max_residual = max(max_residual, residual)
    return max_residual


def _rationalize(value: float, max_denominator: int) -> sympy.Rational:
    return sympy.Rational(Fraction(float(value)).limit_denominator(max_denominator))


def _default_starts(dimension: int, n_restarts: int) -> list[list[float]]:
    starts: list[list[float]] = [[0.0] * dimension]
    if dimension == 0:
        return starts
    for index in range(1, n_restarts):
        sign = 1.0 if index % 2 else -1.0
        amplitude = min(1.0, (index + 1) / max(2, n_restarts))
        starts.append([sign * amplitude] * dimension)
    return starts


def select_best_named_solution(
    *,
    named_solutions: Sequence[dict[str, sympy.core.basic.Basic]],
    free_symbol_names: Sequence[str],
    free_symbol_relations: Sequence[sympy.core.basic.Basic],
    objectives: Sequence[RKObjective | ObjectiveTerm],
    method_factory: Callable[[dict[str, sympy.core.basic.Basic]], RK | None],
    optimization: RKOptimizationConfig | None = None,
) -> dict[str, sympy.core.basic.Basic] | None:
    """
    Select a single best concrete solution from symbolic solution families.
    """
    if not objectives:
        raise ValueError("at least one objective is required")
    terms = _coerce_terms(objectives)
    config = optimization or RKOptimizationConfig()
    best_solution: dict[str, sympy.core.basic.Basic] | None = None
    best_score = math.inf
    for named_solution in named_solutions:
        free_set_in_solution = {
            symbol
            for value in named_solution.values()
            for symbol in sympy.sympify(value).free_symbols
        }
        active_free_names = [
            free_name
            for free_name in free_symbol_names
            if sympy.symbols(free_name) in free_set_in_solution
        ]
        if not active_free_names:
            if _solution_has_free_symbols(named_solution):
                continue
            try:
                method = method_factory(named_solution)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if method is None:
                continue
            score = _weighted_score(method, terms)
            if score < best_score:
                best_score = score
                best_solution = {
                    name: sympy.sympify(value) for name, value in named_solution.items()
                }
            continue

        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover - scipy is a runtime dependency.
            raise RuntimeError(
                "scipy is required for objective-based free-parameter optimization"
            ) from exc

        variable_bounds: list[tuple[float, float] | tuple[None, None]] = []
        for free_name in active_free_names:
            if config.bounds is None or free_name not in config.bounds:
                variable_bounds.append((None, None))
                continue
            lower, upper = config.bounds[free_name]
            variable_bounds.append((lower, upper))

        def objective_value(x_values: Sequence[float]) -> float:
            float_substitutions = {
                free_name: sympy.Float(x_values[index])
                for index, free_name in enumerate(active_free_names)
            }
            substituted = _substitute_named_solution(named_solution, float_substitutions)
            relation_residual = _free_relation_residual(
                named_solution=substituted,
                free_symbol_names=free_symbol_names,
                free_symbol_relations=free_symbol_relations,
            )
            if relation_residual > config.relation_tolerance:
                return 1e12 + 1e8 * relation_residual
            if _solution_has_free_symbols(substituted):
                return 1e12
            try:
                method = method_factory(substituted)
            except (TypeError, ValueError, ZeroDivisionError):
                return 1e12
            if method is None:
                return 1e12
            return _weighted_score(method, terms)

        local_best_x: list[float] | None = None
        local_best = math.inf
        for start in _default_starts(len(active_free_names), config.n_restarts):
            result = minimize(
                objective_value,
                x0=start,
                bounds=variable_bounds,
                method="L-BFGS-B",
                options={"maxiter": config.maxiter},
            )
            candidate_score = objective_value(result.x)
            if candidate_score < local_best:
                local_best = candidate_score
                local_best_x = [float(x) for x in result.x]
        if local_best_x is None:
            continue

        rational_substitutions = {
            free_name: _rationalize(local_best_x[index], config.max_denominator)
            for index, free_name in enumerate(active_free_names)
        }
        candidate_solution = _substitute_named_solution(named_solution, rational_substitutions)
        relation_residual = _free_relation_residual(
            named_solution=candidate_solution,
            free_symbol_names=free_symbol_names,
            free_symbol_relations=free_symbol_relations,
        )
        if relation_residual > config.relation_tolerance:
            continue
        if _solution_has_free_symbols(candidate_solution):
            continue
        try:
            candidate_method = method_factory(candidate_solution)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if candidate_method is None:
            continue
        candidate_score = _weighted_score(candidate_method, terms)
        if candidate_score < best_score:
            best_score = candidate_score
            best_solution = candidate_solution
    return best_solution
