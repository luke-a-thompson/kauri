"""Explicit RK builder."""

import dataclasses
from collections.abc import Sequence

import sympy

from kauri.methods.rk import RK, ButcherTableau
from kauri.rk_builder.rk_constraints import AnyConstraint
from kauri.rk_builder._rk_maker_core import (
    SolveResult,
    _has_exact_embedded_order,
    _prepare_build,
    _run_symbolic_builder,
    _solution_is_numeric,
    embedded_weight_symbols,
    explicit_unknown_symbols,
    generate_explicit_order_equations,
)


def _build_explicit_methods(
    *,
    named_solutions: list[dict[str, sympy.core.basic.Basic]],
    stages: int,
    order: int,
    embedded: bool,
) -> list[RK]:
    methods: list[RK] = []
    for index, named_solution in enumerate(named_solutions):
        if not _solution_is_numeric(named_solution):
            continue
        a: list[list[float]] = [[0.0] * stages for _ in range(stages)]
        b: list[float] = [0.0] * stages
        b_hat: list[float] | None = [0.0] * stages if embedded else None
        for i in range(stages):
            for j in range(i):
                a[i][j] = float(sympy.N(named_solution.get(f"a{i}{j}", sympy.Integer(0)), 20))
            b[i] = float(sympy.N(named_solution.get(f"b{i}", sympy.Integer(0)), 20))
            if b_hat is not None:
                b_hat[i] = float(sympy.N(named_solution.get(f"bhat{i}", sympy.Integer(0)), 20))
        methods.append(
            RK(
                ButcherTableau(a=a, b=b, b_hat=b_hat),
                f"generated_explicit_rk_s{stages}_p{order}_{index}",
            )
        )
    return methods


def build_explicit_rk(
    order: int,
    stages: int,
    antisymmetric_order: int | None = None,
    constraints: Sequence[AnyConstraint] | None = None,
    verify_symbolic: bool = True,
    embedded: bool = False,
) -> tuple[list[RK], SolveResult]:
    equations, trees, compiled = _prepare_build(order, stages, antisymmetric_order, constraints)
    if embedded and order < 2:
        raise ValueError("embedded explicit RK requires order at least 2")
    a_symbols, b_symbols = explicit_unknown_symbols(stages=stages)
    all_symbols = a_symbols + b_symbols
    parameterization = "explicit_tableau"
    if embedded:
        b_hat_symbols = embedded_weight_symbols(stages=stages)
        embedded_equations, embedded_trees = generate_explicit_order_equations(
            order=order - 1,
            stages=stages,
            rationalise=True,
            weight_symbols=b_hat_symbols,
        )
        equations = equations + embedded_equations
        trees = trees + embedded_trees
        all_symbols = all_symbols + b_hat_symbols
        parameterization = "explicit_tableau_embedded"
    solve_result = _run_symbolic_builder(
        equations=equations,
        trees=trees,
        all_symbols=all_symbols,
        substitution_maps=[],
        fixings=compiled.substitutions,
        verify_symbolic=verify_symbolic,
        parameterization=parameterization,
    )
    if embedded:
        if solve_result.solutions and all(
            all(
                sympy.simplify(sympy.expand(solution[f"b{i_idx}"] - solution[f"bhat{i_idx}"])) == 0
                for i_idx in range(stages)
            )
            for solution in solve_result.solutions
        ):
            raise ValueError("embedded explicit RK constraints resolve to b = bhat.")
        solve_result = dataclasses.replace(
            solve_result,
            solutions=[
                solution
                for solution in solve_result.solutions
                if _has_exact_embedded_order(solution=solution, order=order, stages=stages)
            ],
        )
    return (
        _build_explicit_methods(
            named_solutions=solve_result.solutions,
            stages=stages,
            order=order,
            embedded=embedded,
        ),
        solve_result,
    )
