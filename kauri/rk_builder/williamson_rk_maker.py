"""Williamson 2N RK builder."""

import dataclasses
from collections.abc import Sequence

import sympy

from kauri.methods.rk import RK
from kauri.methods.williamson import WilliamsonRecursion, WilliamsonRK
from kauri.rk_builder.rk_constraints import AnyConstraint
from kauri.rk_builder.rk_objectives import (
    ObjectiveTerm,
    RKObjective,
    RKOptimizationConfig,
    select_best_named_solution,
)
from kauri.rk_builder._rk_maker_core import (
    SolveResult,
    _prepare_build,
    _run_symbolic_builder,
    _solution_is_numeric,
    generate_explicit_order_equations,
)
from kauri.rk_builder.williamson import williamson_tableau_expressions, williamson_unknown_symbols
from kauri.trees.trees import Tree


def _williamson_substitution_map(
    *,
    stages: int,
    a_expr: list[list[sympy.Expr]],
    b_expr: list[sympy.Expr],
) -> dict[sympy.Symbol, sympy.core.basic.Basic]:
    substitution_map: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
    for i_idx in range(stages):
        substitution_map[sympy.symbols(f"b{i_idx}")] = sympy.simplify(b_expr[i_idx])
        for j_idx in range(i_idx):
            substitution_map[sympy.symbols(f"a{i_idx}{j_idx}")] = sympy.simplify(a_expr[i_idx][j_idx])
    return substitution_map


def _williamson_embedded_equations(
    order: int, stages: int
) -> tuple[list[sympy.core.basic.Basic], list[Tree]]:
    row = stages - 1
    substitutions = [
        (sympy.symbols(f"b{i_idx}"), sympy.symbols(f"a{row}{i_idx}") if i_idx < row else sympy.Integer(0))
        for i_idx in range(stages)
    ]
    equations, trees = generate_explicit_order_equations(order=order, stages=stages, rationalise=True)
    equations = [sympy.simplify(sympy.expand(equation.subs(substitutions))) for equation in equations]
    return [equation for equation in equations if equation != 0], trees


def _build_williamson_methods(
    *,
    named_solutions: list[dict[str, sympy.core.basic.Basic]],
    stages: int,
    order: int,
    embedded: bool,
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
        recursion_a: list[list[sympy.core.basic.Basic]] = [
            [sympy.Integer(0)] * stages for _ in range(stages)
        ]
        for i_idx in range(1, stages):
            recursion_a[i_idx][i_idx - 1] = A_params[i_idx]
        methods.append(
            WilliamsonRK(
                recursion=WilliamsonRecursion(A=recursion_a, B=B_params),
                name=f"generated_explicit_rk_s{stages}_p{order}_{index}_williamson2n",
                embedded_from_penultimate=embedded,
            )
        )
    return methods


def build_williamson_rk(
    order: int,
    stages: int,
    antisymmetric_order: int | None = None,
    constraints: Sequence[AnyConstraint] | None = None,
    verify_symbolic: bool = True,
    embedded: bool = False,
    objectives: Sequence[RKObjective | ObjectiveTerm] | None = None,
    optimization: RKOptimizationConfig | None = None,
) -> tuple[list[WilliamsonRK], SolveResult]:
    equations, trees, compiled = _prepare_build(order, stages, antisymmetric_order, constraints)
    if embedded and order < 2:
        raise ValueError("embedded Williamson RK requires order at least 2")
    if embedded and stages < 2:
        raise ValueError("embedded Williamson RK requires at least 2 stages")
    williamson_a_expr, williamson_b_expr = williamson_tableau_expressions(stages=stages)
    parameterization = "williamson_2n"
    if embedded:
        embedded_equations, embedded_trees = _williamson_embedded_equations(
            order=order - 1,
            stages=stages,
        )
        equations = equations + embedded_equations
        trees = trees + embedded_trees
        parameterization = "williamson_2n_embedded"
    solve_result = _run_symbolic_builder(
        equations=equations,
        trees=trees,
        all_symbols=williamson_unknown_symbols(stages=stages),
        substitution_maps=[
            _williamson_substitution_map(
                stages=stages,
                a_expr=williamson_a_expr,
                b_expr=williamson_b_expr,
            )
        ],
        fixings=compiled.substitutions,
        verify_symbolic=verify_symbolic,
        parameterization=parameterization,
    )
    if embedded:
        higher_order_embedded_equations, _ = _williamson_embedded_equations(
            order=order,
            stages=stages,
        )
        solve_result = dataclasses.replace(
            solve_result,
            solutions=[
                solution
                for solution in solve_result.solutions
                if any(
                    sympy.simplify(
                        sympy.expand(
                            equation.subs([(sympy.symbols(name), value) for name, value in solution.items()])
                        )
                    )
                    != 0
                    for equation in higher_order_embedded_equations
                )
            ],
        )
    if objectives:
        def objective_method_factory(
            named_solution: dict[str, sympy.core.basic.Basic],
        ) -> RK | None:
            methods = _build_williamson_methods(
                named_solutions=[named_solution],
                stages=stages,
                order=order,
                embedded=embedded,
            )
            if not methods:
                return None
            method = methods[0]
            return RK(tableau=method.tableau, name=method.name)

        best_solution = select_best_named_solution(
            named_solutions=solve_result.solutions,
            free_symbol_names=solve_result.free_symbols,
            free_symbol_relations=solve_result.free_symbol_relations,
            objectives=objectives,
            method_factory=objective_method_factory,
            optimization=optimization,
        )
        if best_solution is None:
            solve_result = dataclasses.replace(solve_result, solutions=[])
            return [], solve_result
        chosen_fixings = {
            free_name: sympy.sympify(best_solution[free_name])
            for free_name in solve_result.free_symbols
            if free_name in best_solution and not sympy.sympify(best_solution[free_name]).free_symbols
        }
        solve_result = dataclasses.replace(
            solve_result,
            solutions=[best_solution],
            free_symbols=[],
            free_symbol_relations=[],
            fixings=solve_result.fixings | chosen_fixings,
        )
    return (
        _build_williamson_methods(
            named_solutions=solve_result.solutions,
            stages=stages,
            order=order,
            embedded=embedded,
        ),
        solve_result,
    )
