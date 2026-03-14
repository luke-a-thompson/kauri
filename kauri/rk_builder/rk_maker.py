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

"""Utilities for constructing Runge-Kutta methods from rooted-tree order conditions."""

import time
from dataclasses import dataclass

import sympy

from kauri.hopf_algebras.bck import counit
from kauri.hopf_algebras.maps import sign
from kauri.methods.rk import RK, ButcherTableau
from kauri.methods.williamson import WilliamsonRK
from kauri.rk_builder.rk import _rk_symbolic_weights_map, rk_order_cond
from kauri.rk_builder.rk_constraints import CompiledConstraints, Constraint, compile_constraints
from kauri.rk_builder.williamson import (
    williamson_tableau_expressions,
    williamson_unknown_symbols,
)
from kauri.trees.gentrees import trees_up_to_order
from kauri.trees.trees import Tree


@dataclass
class SolveResult:
    """
    Container for RK construction outputs.
    """

    solutions: list[dict[str, sympy.core.basic.Basic]]
    equations: list[sympy.core.basic.Basic]
    unknowns: list[str]
    free_symbols: list[str]
    free_symbol_relations: list[sympy.core.basic.Basic]
    trees: list[Tree]
    parameterization: str
    fixings: dict[str, sympy.core.basic.Basic]

    def __str__(self) -> str:
        return self.format()

    def format(self) -> str:
        from kauri.rk_builder.rk_maker_format import result_to_text

        return result_to_text(self)

    def to_latex(self, standalone: bool = True) -> str:
        from kauri.rk_builder.rk_maker_format import result_to_latex

        return result_to_latex(self, standalone=standalone)


@dataclass
class GrobnerSolveResult:
    solutions: list[dict[sympy.Symbol, sympy.core.basic.Basic]]
    free_symbols: list[sympy.Symbol]
    free_symbol_relations: list[sympy.core.basic.Basic]


def explicit_unknown_symbols(stages: int) -> tuple[list[sympy.Symbol], list[sympy.Symbol]]:
    """
    Return strict lower-triangular A symbols and b symbols for explicit RK.
    """
    if stages <= 0:
        raise ValueError("stages must be positive")
    a_symbols = [sympy.symbols(f"a{i}{j}") for i in range(stages) for j in range(i)]
    b_symbols = [sympy.symbols(f"b{i}") for i in range(stages)]
    return a_symbols, b_symbols


def generate_explicit_order_equations(
    order: int, stages: int, rationalise: bool = True
) -> tuple[list[sympy.core.basic.Basic], list[Tree]]:
    """
    Generate rooted-tree order equations for explicit RK up to given order.
    """
    if order <= 0:
        raise ValueError("order must be positive")
    trees = [t for t in trees_up_to_order(order) if t != Tree(None)]
    equations = [
        sympy.expand(rk_order_cond(t, stages, explicit=True, rationalise=rationalise))
        for t in trees
    ]
    return equations, trees


def generate_explicit_antisymmetric_equations(
    antisymmetric_order: int, stages: int, rationalise: bool = True
) -> tuple[list[sympy.core.basic.Basic], list[Tree]]:
    """
    Generate explicit RK antisymmetric-order equations up to given order.

    The defining condition is ``m = (phi & sign) * phi = counit`` on trees,
    where ``phi`` is the method's elementary-weights map.
    """
    if antisymmetric_order <= 0:
        raise ValueError("antisymmetric_order must be positive")
    phi = _rk_symbolic_weights_map(stages, explicit=True)
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
            if symbol in merged:
                if sympy.simplify(merged[symbol] - resolved_value) != 0:
                    raise ValueError(f"Conflicting substitutions for {symbol}")
            merged[symbol] = resolved_value
    return merged


def _free_symbols_in_equations(
    equations: list[sympy.core.basic.Basic], candidate_symbols: list[sympy.Symbol]
) -> list[sympy.Symbol]:
    free_union: set[sympy.core.basic.Basic] = set()
    for eqn in equations:
        free_union = free_union.union(eqn.free_symbols)
    return [symbol for symbol in candidate_symbols if symbol in free_union]


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
    max_solutions: int | None,
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
    t_groebner = time.perf_counter() - t0
    print(f"[timing] groebner basis: {t_groebner:.3f}s")

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
    t_solve = time.perf_counter() - t1
    print(f"[timing] solve:          {t_solve:.3f}s")

    if max_solutions is not None:
        raw_solutions = raw_solutions[:max_solutions]

    return GrobnerSolveResult(
        solutions=raw_solutions,
        free_symbols=free_symbols,
        free_symbol_relations=relations,
    )


def _solution_is_numeric(named_solution: dict[str, sympy.core.basic.Basic]) -> bool:
    return all(not sympy.sympify(value).free_symbols for value in named_solution.values())


def _verify_solution(
    equations: list[sympy.core.basic.Basic],
    symbol_values: dict[sympy.Symbol, sympy.core.basic.Basic],
) -> bool:
    substitutions = list(symbol_values.items())
    for equation in equations:
        if sympy.simplify(sympy.expand(equation.subs(substitutions))) != 0:
            return False
    return True



def _run_symbolic_builder(
    *,
    equations: list[sympy.core.basic.Basic],
    trees: list[Tree],
    all_symbols: list[sympy.Symbol],
    substitution_maps: list[dict[sympy.Symbol, sympy.core.basic.Basic]],
    fixings: dict[sympy.Symbol, sympy.core.basic.Basic],
    max_solutions: int | None,
    verify_symbolic: bool,
    parameterization: str,
    stages: int,
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
        sympy.simplify(sympy.expand(equation.subs(substitutions))) for equation in equations
    ]
    reduced_equations = [
        equation for equation in reduced_equations if sympy.simplify(equation) != 0
    ]

    active_symbols = _free_symbols_in_equations(reduced_equations, all_symbols)
    active_symbols = [symbol for symbol in active_symbols if symbol not in substitutions_map]

    grobner_result = _solve_with_grobner(reduced_equations, active_symbols, max_solutions)
    raw_solutions = grobner_result.solutions

    if len(active_symbols) == 0 and len(reduced_equations) == 0:
        raw_solutions = [{}]

    full_solutions: list[dict[sympy.Symbol, sympy.core.basic.Basic]] = []
    free_set = set(grobner_result.free_symbols)
    for solution in raw_solutions:
        merged: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
        for key, value in substitutions_map.items():
            merged[key] = sympy.sympify(value)
        for symbol in all_symbols:
            if symbol in solution:
                merged[symbol] = sympy.simplify(solution[symbol])
            elif symbol in merged:
                continue
            elif symbol in free_set:
                merged[symbol] = symbol
            else:
                merged[symbol] = sympy.Integer(0)
        merged_items = list(merged.items())
        for key, value in list(merged.items()):
            merged[key] = sympy.simplify(sympy.expand(sympy.sympify(value).subs(merged_items)))
        if verify_symbolic and not _verify_solution(reduced_equations, merged):
            continue
        full_solutions.append(merged)

    named_solutions = [
        {str(symbol): value for symbol, value in solution.items()}
        for solution in full_solutions
    ]

    return SolveResult(
        solutions=named_solutions,
        equations=reduced_equations,
        unknowns=[str(symbol) for symbol in active_symbols],
        free_symbols=[str(symbol) for symbol in grobner_result.free_symbols],
        free_symbol_relations=grobner_result.free_symbol_relations,
        trees=trees,
        parameterization=parameterization,
        fixings={str(symbol): value for symbol, value in fixings.items()},
    )


def _prepare_build(
    order: int,
    stages: int,
    antisymmetric_order: int | None,
    constraints: list[Constraint] | None,
) -> tuple[list[sympy.core.basic.Basic], list[Tree], CompiledConstraints]:
    if order <= 0:
        raise ValueError("order must be positive")
    if stages <= 0:
        raise ValueError("stages must be positive")
    if isinstance(antisymmetric_order, int) and antisymmetric_order <= 0:
        raise ValueError("antisymmetric_order must be positive")
    equations, trees = generate_explicit_order_equations(order=order, stages=stages, rationalise=True)
    if antisymmetric_order is not None:
        antisymmetric_equations, antisymmetric_trees = generate_explicit_antisymmetric_equations(
            antisymmetric_order=antisymmetric_order, stages=stages, rationalise=True
        )
        equations = equations + antisymmetric_equations
        trees = trees + antisymmetric_trees
    compiled = compile_constraints(constraints if constraints is not None else [])
    return equations + compiled.equations, trees, compiled


def _build_explicit_methods(
    *,
    named_solutions: list[dict[str, sympy.core.basic.Basic]],
    stages: int,
    order: int,
) -> list[RK]:
    methods: list[RK] = []
    for index, named_solution in enumerate(named_solutions):
        if not _solution_is_numeric(named_solution):
            continue
        a: list[list[float]] = [[0.0] * stages for _ in range(stages)]
        b: list[float] = [0.0] * stages
        for i in range(stages):
            for j in range(i):
                a[i][j] = float(sympy.N(named_solution.get(f"a{i}{j}", sympy.Integer(0)), 20))
            b[i] = float(sympy.N(named_solution.get(f"b{i}", sympy.Integer(0)), 20))
        methods.append(
            RK(ButcherTableau(a=a, b=b), f"generated_explicit_rk_s{stages}_p{order}_{index}")
        )
    return methods


def _build_williamson_methods(
    *,
    named_solutions: list[dict[str, sympy.core.basic.Basic]],
    stages: int,
    order: int,
) -> list[WilliamsonRK]:
    methods: list[WilliamsonRK] = []
    for index, named_solution in enumerate(named_solutions):
        if not _solution_is_numeric(named_solution):
            continue
        A_params = [sympy.Integer(0)] + [
            sympy.sympify(named_solution.get(f"A{i_idx}", sympy.Integer(0)))
            for i_idx in range(1, stages)
        ]
        B_params = [
            sympy.sympify(named_solution.get(f"B{i_idx}", sympy.Integer(0)))
            for i_idx in range(stages)
        ]

        methods.append(
            WilliamsonRK(
                stages=stages,
                A=A_params,
                B=B_params,
                name=f"generated_explicit_rk_s{stages}_p{order}_{index}_williamson2n",
            )
        )
    return methods


def build_explicit_rk(
    order: int,
    stages: int,
    antisymmetric_order: int | None = None,
    constraints: list[Constraint] | None = None,
    max_solutions: int | None = 1,
    verify_symbolic: bool = True,
) -> tuple[list[RK], SolveResult]:
    """
    Construct explicit RK methods of requested order and stage count.
    """
    equations, trees, compiled = _prepare_build(order, stages, antisymmetric_order, constraints)
    a_symbols, b_symbols = explicit_unknown_symbols(stages=stages)
    solve_result = _run_symbolic_builder(
        equations=equations,
        trees=trees,
        all_symbols=a_symbols + b_symbols,
        substitution_maps=[],
        fixings=compiled.substitutions,
        max_solutions=max_solutions,
        verify_symbolic=verify_symbolic,
        parameterization="explicit_tableau",
        stages=stages,
    )
    return (
        _build_explicit_methods(named_solutions=solve_result.solutions, stages=stages, order=order),
        solve_result,
    )


def build_williamson_rk(
    order: int,
    stages: int,
    antisymmetric_order: int | None = None,
    constraints: list[Constraint] | None = None,
    max_solutions: int | None = 1,
    verify_symbolic: bool = True,
) -> tuple[list[WilliamsonRK], SolveResult]:
    """
    Construct Williamson 2N explicit RK methods of requested order and stage count.
    """
    equations, trees, compiled = _prepare_build(order, stages, antisymmetric_order, constraints)
    williamson_a_expr, williamson_b_expr = williamson_tableau_expressions(stages=stages)
    substitution_map: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
    for i_idx in range(stages):
        substitution_map[sympy.symbols(f"b{i_idx}")] = sympy.simplify(williamson_b_expr[i_idx])
        for j_idx in range(stages):
            substitution_map[sympy.symbols(f"a{i_idx}{j_idx}")] = sympy.simplify(
                williamson_a_expr[i_idx][j_idx]
            )
    solve_result = _run_symbolic_builder(
        equations=equations,
        trees=trees,
        all_symbols=williamson_unknown_symbols(stages=stages),
        substitution_maps=[substitution_map],
        fixings=compiled.substitutions,
        max_solutions=max_solutions,
        verify_symbolic=verify_symbolic,
        parameterization="williamson_2n",
        stages=stages,
    )
    return (
        _build_williamson_methods(
            named_solutions=solve_result.solutions,
            stages=stages,
            order=order,
        ),
        solve_result,
    )


if __name__ == "__main__":
    demo_methods, demo_result = build_williamson_rk(
        order=2,
        stages=3,
        antisymmetric_order=5,
        constraints=[Constraint.set("b0", sympy.Rational(1, 4))],
        max_solutions=1,
    )

    print(f"methods found: {len(demo_methods)}")
    print(demo_result)
    print(demo_methods[0])
    print(demo_methods[0].tableau)
    with open("rk_maker_output.tex", "w", encoding="utf-8") as file_obj:
        file_obj.write(demo_result.to_latex(standalone=True))
