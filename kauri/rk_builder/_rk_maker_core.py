"""Shared symbolic RK builder utilities."""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, NamedTuple

import sympy

from kauri.hopf_algebras.bck import counit
from kauri.hopf_algebras.maps import sign
from kauri.rk_builder.rk import _rk_symbolic_weights_map, rk_order_cond
from kauri.rk_builder.rk_constraints import AnyConstraint, CompiledConstraints, compile_constraints
from kauri.trees.gentrees import trees_up_to_order
from kauri.trees.trees import Tree

if TYPE_CHECKING:
    from kauri.rk_builder.rk_objectives import MethodScore


@dataclass
class SolveResult:
    solutions: list[dict[str, sympy.core.basic.Basic]]
    equations: list[sympy.core.basic.Basic]
    unknowns: list[str]
    free_symbols: list[str]
    free_symbol_relations: list[sympy.core.basic.Basic]
    trees: list[Tree]
    parameterization: str
    fixings: dict[str, sympy.core.basic.Basic]
    objective_scores: list["MethodScore"] = field(default_factory=list)

    def __str__(self) -> str:
        return self.format()

    def format(self) -> str:
        from kauri.rk_builder.rk_maker_format import result_to_text

        return result_to_text(self)

    def to_latex(self, standalone: bool = True) -> str:
        from kauri.rk_builder.rk_maker_format import result_to_latex

        return result_to_latex(self, standalone=standalone)


class GrobnerSolveResult(NamedTuple):
    solutions: list[dict[sympy.Symbol, sympy.core.basic.Basic]]
    free_symbols: list[sympy.Symbol]
    free_symbol_relations: list[sympy.core.basic.Basic]


def explicit_unknown_symbols(stages: int) -> tuple[list[sympy.Symbol], list[sympy.Symbol]]:
    if stages <= 0:
        raise ValueError("stages must be positive")
    a_symbols = [sympy.symbols(f"a{i}{j}") for i in range(stages) for j in range(i)]
    b_symbols = [sympy.symbols(f"b{i}") for i in range(stages)]
    return a_symbols, b_symbols


def embedded_weight_symbols(stages: int) -> list[sympy.Symbol]:
    if stages <= 0:
        raise ValueError("stages must be positive")
    return [sympy.symbols(f"bhat{i}") for i in range(stages)]


def generate_explicit_order_equations(
    order: int,
    stages: int,
    rationalise: bool = True,
    weight_symbols: list[sympy.Symbol] | None = None,
) -> tuple[list[sympy.core.basic.Basic], list[Tree]]:
    if order <= 0:
        raise ValueError("order must be positive")
    trees = [t for t in trees_up_to_order(order) if t != Tree(None)]
    equations = [
        sympy.expand(
            rk_order_cond(
                t,
                stages,
                explicit=True,
                rationalise=rationalise,
                weight_symbols=weight_symbols,
            )
        )
        for t in trees
    ]
    return equations, trees


def generate_explicit_antisymmetric_equations(
    antisymmetric_order: int,
    stages: int,
    rationalise: bool = True,
    weight_symbols: list[sympy.Symbol] | None = None,
) -> tuple[list[sympy.core.basic.Basic], list[Tree]]:
    if antisymmetric_order <= 0:
        raise ValueError("antisymmetric_order must be positive")
    phi = _rk_symbolic_weights_map(stages, explicit=True, weight_symbols=weight_symbols)
    m = (phi & sign) * phi
    trees: list[Tree] = [t for t in trees_up_to_order(antisymmetric_order) if t != Tree(None)]
    equations: list[sympy.core.basic.Basic] = [
        sympy.expand(sympy.sympify(m(t) - counit(t))) for t in trees
    ]
    if rationalise:
        equations = [
            sympy.sympify(sympy.nsimplify(equation, tolerance=1e-10, rational=True))
            for equation in equations
        ]
    return equations, trees


def _merge_substitution_maps(
    substitution_maps: list[dict[sympy.Symbol, sympy.core.basic.Basic]],
) -> dict[sympy.Symbol, sympy.core.basic.Basic]:
    merged: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
    for substitutions in substitution_maps:
        for symbol, value in substitutions.items():
            resolved_value = sympy.sympify(value)
            if symbol in merged and sympy.simplify(merged[symbol] - resolved_value) != 0:
                raise ValueError(f"Conflicting substitutions for {symbol}")
            merged[symbol] = resolved_value
    return merged


def _split_dependent_and_free_symbols(
    grobner_basis, unknown_symbols: list[sympy.Symbol]
) -> tuple[list[sympy.Symbol], list[sympy.Symbol]]:
    leading_symbols: set[sympy.Symbol] = set()
    for poly in grobner_basis.polys:
        exponents = poly.monoms(order=grobner_basis.order)[0]
        for symbol, exponent in zip(unknown_symbols, exponents, strict=True):
            if exponent != 0:
                leading_symbols.add(symbol)
                break
    dependent_symbols = [symbol for symbol in unknown_symbols if symbol in leading_symbols]
    free_symbols = [symbol for symbol in unknown_symbols if symbol not in leading_symbols]
    return dependent_symbols, free_symbols


def _solve_with_grobner(
    equations: list[sympy.core.basic.Basic],
    unknown_symbols: list[sympy.Symbol],
) -> GrobnerSolveResult:
    if not unknown_symbols or not equations:
        return GrobnerSolveResult(solutions=[], free_symbols=[], free_symbol_relations=[])
    polys = [
        sympy.Poly(sympy.nsimplify(equation), *unknown_symbols, domain="QQ")
        for equation in equations
    ]
    t0 = time.perf_counter()
    grobner_basis = sympy.groebner(
        polys, *unknown_symbols, order="grevlex", domain="QQ", method="f5b"
    )
    print(f"[timing] groebner basis: {time.perf_counter() - t0:.3f}s")
    dependent_symbols, free_symbols = _split_dependent_and_free_symbols(
        grobner_basis, unknown_symbols
    )
    free_set = set(free_symbols)
    relations: list[sympy.core.basic.Basic] = [
        sympy.expand(expr)
        for poly in grobner_basis.polys
        if (expr := sympy.sympify(poly.as_expr())) != 0 and expr.free_symbols.issubset(free_set)
    ]
    t1 = time.perf_counter()
    if grobner_basis.is_zero_dimensional:
        raw_solutions = sympy.solve(list(grobner_basis), unknown_symbols, dict=True)
    elif len(dependent_symbols) == 0:
        raw_solutions = [{}]
    else:
        raw_solutions = sympy.solve(list(grobner_basis), dependent_symbols, dict=True)
    print(f"[timing] solve:          {time.perf_counter() - t1:.3f}s")
    return GrobnerSolveResult(
        solutions=raw_solutions,
        free_symbols=free_symbols,
        free_symbol_relations=relations,
    )


def _solution_is_numeric(named_solution: dict[str, sympy.core.basic.Basic]) -> bool:
    return all(not sympy.sympify(value).free_symbols for value in named_solution.values())


def _has_exact_embedded_order(
    solution: dict[str, sympy.core.basic.Basic], order: int, stages: int
) -> bool:
    higher_order_equations, _ = generate_explicit_order_equations(
        order=order,
        stages=stages,
        rationalise=True,
        weight_symbols=embedded_weight_symbols(stages=stages),
    )
    substitutions = [(sympy.symbols(name), value) for name, value in solution.items()]
    return any(
        sympy.simplify(sympy.expand(equation.subs(substitutions))) != 0
        for equation in higher_order_equations
    )


def _verify_solution(
    equations: list[sympy.core.basic.Basic],
    symbol_values: dict[sympy.Symbol, sympy.core.basic.Basic],
) -> bool:
    substitutions = list(symbol_values.items())
    return all(sympy.simplify(sympy.expand(eq.subs(substitutions))) == 0 for eq in equations)


def _select_objective_solution(*, solve_result: SolveResult, objectives: Sequence, method_factory, optimization):
    from kauri.rk_builder.rk_objectives import select_best_named_solution

    best_solution = select_best_named_solution(
        named_solutions=solve_result.solutions,
        free_symbol_names=solve_result.free_symbols,
        free_symbol_relations=solve_result.free_symbol_relations,
        objectives=objectives,
        method_factory=method_factory,
        optimization=optimization,
    )
    if best_solution is None:
        return None, replace(solve_result, solutions=[])
    chosen_fixings = {
        free_name: value
        for free_name in solve_result.free_symbols
        if free_name in best_solution
        if not (value := sympy.sympify(best_solution[free_name])).free_symbols
    }
    return best_solution, replace(
        solve_result,
        solutions=[best_solution],
        free_symbols=[],
        free_symbol_relations=[],
        fixings=solve_result.fixings | chosen_fixings,
    )


def _run_symbolic_builder(
    *,
    equations: list[sympy.core.basic.Basic],
    trees: list[Tree],
    all_symbols: list[sympy.Symbol],
    substitution_maps: list[dict[sympy.Symbol, sympy.core.basic.Basic]],
    fixings: dict[sympy.Symbol, sympy.core.basic.Basic],
    verify_symbolic: bool,
    parameterization: str,
) -> SolveResult:
    substitutions_map = _merge_substitution_maps(substitution_maps)
    for symbol, value in fixings.items():
        resolved_value = sympy.sympify(value)
        if symbol in substitutions_map:
            equations.append(sympy.simplify(sympy.expand(substitutions_map[symbol] - resolved_value)))
        else:
            substitutions_map[symbol] = resolved_value
    substitutions = list(substitutions_map.items())
    reduced_equations = [
        eq for equation in equations
        if (eq := sympy.simplify(sympy.expand(equation.subs(substitutions)))) != 0
    ]
    free_union = {sym for eq in reduced_equations for sym in eq.free_symbols}
    active_symbols = [s for s in all_symbols if s in free_union and s not in substitutions_map]
    grobner_result = _solve_with_grobner(reduced_equations, active_symbols)
    raw_solutions = grobner_result.solutions if active_symbols or reduced_equations else [{}]
    full_solutions: list[dict[sympy.Symbol, sympy.core.basic.Basic]] = []
    free_set = set(grobner_result.free_symbols)
    for solution in raw_solutions:
        merged: dict[sympy.Symbol, sympy.core.basic.Basic] = {
            key: sympy.sympify(value) for key, value in substitutions_map.items()
        }
        for symbol in all_symbols:
            if symbol in solution:
                merged[symbol] = sympy.simplify(solution[symbol])
            elif symbol not in merged:
                merged[symbol] = symbol if symbol in free_set else sympy.Integer(0)
        merged_items = list(merged.items())
        for key, value in merged_items:
            merged[key] = sympy.simplify(sympy.expand(sympy.sympify(value).subs(merged_items)))
        if verify_symbolic and not _verify_solution(reduced_equations, merged):
            continue
        full_solutions.append(merged)
    return SolveResult(
        solutions=[{str(symbol): value for symbol, value in solution.items()} for solution in full_solutions],
        equations=reduced_equations,
        unknowns=[str(symbol) for symbol in active_symbols],
        free_symbols=[str(symbol) for symbol in grobner_result.free_symbols],
        free_symbol_relations=grobner_result.free_symbol_relations,
        trees=trees,
        parameterization=parameterization,
        fixings={str(symbol): value for symbol, value in fixings.items()},
    )


def prepare_build(
    order: int,
    stages: int,
    antisymmetric_order: int | None,
    constraints: Sequence[AnyConstraint] | None,
) -> tuple[list[sympy.core.basic.Basic], list[Tree], CompiledConstraints]:
    if order <= 0:
        raise ValueError("order must be positive")
    if stages <= 0:
        raise ValueError("stages must be positive")
    if isinstance(antisymmetric_order, int) and antisymmetric_order <= 0:
        raise ValueError("antisymmetric_order must be positive")
    equations, trees = generate_explicit_order_equations(
        order=order, stages=stages, rationalise=True
    )
    if antisymmetric_order is not None:
        antisymmetric_equations, antisymmetric_trees = generate_explicit_antisymmetric_equations(
            antisymmetric_order=antisymmetric_order, stages=stages, rationalise=True
        )
        equations = equations + antisymmetric_equations
        trees = trees + antisymmetric_trees
    compiled = compile_constraints(constraints or [], stages=stages)
    return equations + compiled.equations, trees, compiled
