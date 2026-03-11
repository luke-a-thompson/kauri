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

"""Utilities for constructing explicit Runge--Kutta methods from rooted-tree order conditions."""

import time
from dataclasses import dataclass

import sympy

from kauri.numerics.ansatze.base import BaseAnsatz
from kauri.numerics.ansatze.explicit import (
    ExplicitAnsatz,
)
from kauri.numerics.ansatze.explicit import (
    explicit_unknown_symbols as explicit_unknown_symbols_from_ansatz,
)
from kauri.numerics.ansatze.explicit import (
    generate_explicit_antisymmetric_equations as generate_explicit_antisymmetric_equations_from_ansatz,
)
from kauri.numerics.ansatze.explicit import (
    generate_explicit_order_equations as generate_explicit_order_equations_from_ansatz,
)
from kauri.numerics.ansatze.williamson import WilliamsonAnsatz
from kauri.numerics.methods.rk import RK
from kauri.numerics.rk.rk_constraints import Constraint, compile_constraints
from kauri.trees.trees import Tree


@dataclass
class RKMakerResult:
    """
    Container for explicit RK construction outputs.
    """

    methods: list[RK]
    solutions: list[dict[str, sympy.core.basic.Basic]]
    equations: list[sympy.core.basic.Basic]
    unknowns: list[str]
    free_symbols: list[str]
    free_symbol_relations: list[sympy.core.basic.Basic]
    trees: list[Tree]
    ansatz: str
    fixings: dict[str, sympy.core.basic.Basic]

    def __str__(self) -> str:
        return self.format()

    def format(
        self,
        max_cell_chars: int = 48,
    ) -> str:
        from kauri.numerics.rk.rk_maker_format import result_to_text

        return result_to_text(self, max_cell_chars=max_cell_chars)

    def to_latex(
        self,
        max_cell_chars: int = 48,
        standalone: bool = True,
    ) -> str:
        from kauri.numerics.rk.rk_maker_format import result_to_latex

        return result_to_latex(self, max_cell_chars=max_cell_chars, standalone=standalone)


@dataclass
class GrobnerSolveResult:
    solutions: list[dict[sympy.Symbol, sympy.core.basic.Basic]]
    free_symbols: list[sympy.Symbol]
    free_symbol_relations: list[sympy.core.basic.Basic]


def explicit_unknown_symbols(stages: int) -> tuple[list[sympy.Symbol], list[sympy.Symbol]]:
    """
    Return strict lower-triangular A symbols and b symbols for explicit RK.
    """
    return explicit_unknown_symbols_from_ansatz(stages)


def generate_explicit_order_equations(
    order: int, stages: int, rationalise: bool = True
) -> tuple[list[sympy.core.basic.Basic], list[Tree]]:
    """
    Generate rooted-tree order equations for explicit RK up to given order.
    """
    return generate_explicit_order_equations_from_ansatz(order, stages, rationalise=rationalise)


def generate_explicit_antisymmetric_equations(
    antisymmetric_order: int, stages: int, rationalise: bool = True
) -> tuple[list[sympy.core.basic.Basic], list[Tree]]:
    """
    Generate explicit RK antisymmetric-order equations up to given order.

    The defining condition is ``m = (phi & sign) * phi = counit`` on trees,
    where ``phi`` is the method's elementary-weights map.
    """
    return generate_explicit_antisymmetric_equations_from_ansatz(
        antisymmetric_order,
        stages,
        rationalise=rationalise,
    )


def _normalise_assignments(
    fixed_values: dict[str, float | int | sympy.core.basic.Basic] | None,
    zero_symbols: list[str] | None,
) -> dict[sympy.Symbol, sympy.core.basic.Basic]:
    assignments: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
    if fixed_values is not None:
        for key, value in fixed_values.items():
            assignments[sympy.symbols(key)] = sympy.sympify(value)
    if zero_symbols is not None:
        for name in zero_symbols:
            assignments[sympy.symbols(name)] = sympy.Integer(0)
    return assignments


def _merge_substitution_maps(
    substitution_maps: list[dict[sympy.Symbol, sympy.core.basic.Basic]],
) -> dict[sympy.Symbol, sympy.core.basic.Basic]:
    merged: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
    for substitutions in substitution_maps:
        for symbol, value in substitutions.items():
            resolved_value = sympy.sympify(value)
            if symbol in merged:
                if (
                    sympy.simplify(sympy.sympify(merged[symbol]) - sympy.sympify(resolved_value))
                    != 0
                ):
                    raise ValueError(f"Conflicting substitutions for {symbol}")
            merged[symbol] = resolved_value
    return merged


def _free_symbols_in_equations(
    equations: list[sympy.core.basic.Basic], candidate_symbols: list[sympy.Symbol]
) -> list[sympy.Symbol]:
    free_union: set[sympy.core.basic.Basic] = set()
    for eqn in equations:
        free_union = free_union.union(eqn.free_symbols)
    return [s for s in candidate_symbols if s in free_union]


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

    polys = [sympy.Poly(sympy.nsimplify(e), *unknown_symbols, domain="QQ") for e in equations]
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
        solutions=raw_solutions, free_symbols=free_symbols, free_symbol_relations=relations
    )


def _solution_is_numeric(named_solution: dict[str, sympy.core.basic.Basic]) -> bool:
    """Return True when every value in the solution is a concrete number (no free symbols)."""
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


def make_explicit_rk_methods(
    order: int,
    stages: int,
    antisymmetric_order: int | None = None,
    ansatz: BaseAnsatz | None = None,
    constraints: list[Constraint] | None = None,
    fixed_values: dict[str, float | int | sympy.core.basic.Basic] | None = None,
    zero_symbols: list[str] | None = None,
    max_solutions: int | None = 1,
    verify_symbolic: bool = True,
) -> RKMakerResult:
    """
    Construct explicit RK methods of requested order and stage count.

    The solve pipeline is:
      1. Generate explicit rooted-tree order conditions.
      2. Apply ansatz and user constraints in a shared symbolic system.
      3. Apply fixed assignments and zeroed symbols.
      4. Solve using Grobner elimination.
    """
    if order <= 0:
        raise ValueError("order must be positive")
    if stages <= 0:
        raise ValueError("stages must be positive")
    if isinstance(antisymmetric_order, int) and antisymmetric_order <= 0:
        raise ValueError("antisymmetric_order must be positive")

    ansatz_used: BaseAnsatz = ExplicitAnsatz() if ansatz is None else ansatz
    equations, trees = ansatz_used.base_equations(
        order=order,
        stages=stages,
        antisymmetric_order=antisymmetric_order,
    )
    equations = equations + ansatz_used.extra_equations(stages=stages)

    compiled_constraints = compile_constraints(constraints if constraints is not None else [])
    equations = equations + compiled_constraints.equations
    all_symbols = ansatz_used.unknown_symbols(stages=stages)

    assignments = _normalise_assignments(fixed_values, zero_symbols)
    substitutions_map = _merge_substitution_maps(
        [
            ansatz_used.tableau_substitutions(stages=stages),
            compiled_constraints.substitutions,
        ]
    )
    for symbol, value in assignments.items():
        resolved_value = sympy.sympify(value)
        if symbol in substitutions_map:
            equations.append(
                sympy.simplify(sympy.expand(substitutions_map[symbol] - resolved_value))
            )
        else:
            substitutions_map[symbol] = resolved_value
    substitutions = list(substitutions_map.items())
    reduced_equations = [sympy.simplify(sympy.expand(e.subs(substitutions))) for e in equations]
    reduced_equations = [e for e in reduced_equations if sympy.simplify(e) != 0]

    active_symbols = _free_symbols_in_equations(reduced_equations, all_symbols)
    active_symbols = [symbol for symbol in active_symbols if symbol not in substitutions_map]

    grobner_result = _solve_with_grobner(reduced_equations, active_symbols, max_solutions)
    raw_solutions = grobner_result.solutions
    free_symbols = grobner_result.free_symbols
    free_symbol_relations = grobner_result.free_symbol_relations

    if len(active_symbols) == 0 and len(reduced_equations) == 0:
        raw_solutions = [{}]

    full_solutions: list[dict[sympy.Symbol, sympy.core.basic.Basic]] = []
    free_set = set(free_symbols)
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

    methods: list[RK] = []
    named_solutions: list[dict[str, sympy.core.basic.Basic]] = []
    for index, solution in enumerate(full_solutions):
        named: dict[str, sympy.core.basic.Basic] = {
            str(symbol): value for symbol, value in solution.items()
        }
        if not ansatz_used.post_validate(
            stages=stages,
            named_solution=named,
        ):
            continue
        named_solutions.append(named)
        if _solution_is_numeric(named):
            a_matrix, b_vector = ansatz_used.build_tableau(stages=stages, named_solution=named)
            methods.append(
                RK(a_matrix, b_vector, f"generated_explicit_rk_s{stages}_p{order}_{index}")
            )

    fixings: dict[str, sympy.core.basic.Basic] = {
        str(symbol): value for symbol, value in assignments.items()
    }

    return RKMakerResult(
        methods=methods,
        solutions=named_solutions,
        equations=reduced_equations,
        unknowns=[str(symbol) for symbol in active_symbols],
        free_symbols=[str(symbol) for symbol in free_symbols],
        free_symbol_relations=free_symbol_relations,
        trees=trees,
        ansatz=type(ansatz_used).__name__,
        fixings=fixings,
    )


if __name__ == "__main__":
    # Demo: build a 2N-storage EES-style explicit RK method.
    demo_result: RKMakerResult = make_explicit_rk_methods(
        order=2,
        stages=3,
        antisymmetric_order=5,
        ansatz=WilliamsonAnsatz(),
        fixed_values={"b0": sympy.Rational(1, 4)},
        max_solutions=1,
    )

    print(demo_result)
    with open("rk_maker_output.tex", "w", encoding="utf-8") as f:
        f.write(demo_result.to_latex(standalone=True))
