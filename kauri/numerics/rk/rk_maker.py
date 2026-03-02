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

from dataclasses import dataclass

import sympy

from kauri.hopf_algebras.bck import counit
from kauri.trees.gentrees import trees_up_to_order
from kauri.hopf_algebras.maps import sign
from kauri.numerics.rk.rk import RK, _rk_symbolic_weights_map, rk_order_cond
from kauri.numerics.rk.rk_ansatz import Ansatz, IdentityAnsatz
from kauri.numerics.rk.rk_constraints import Constraint, compile_constraints
from kauri.numerics.rk.williamson_ansatz import WilliamsonAnsatz
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

        return result_to_latex(
            self, max_cell_chars=max_cell_chars, standalone=standalone
        )


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
                if sympy.simplify(sympy.sympify(merged[symbol]) - sympy.sympify(resolved_value)) != 0:
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


def _relations_among_free_symbols(
    equations: list[sympy.core.basic.Basic],
    dependent_symbols: list[sympy.Symbol],
    free_symbols: list[sympy.Symbol],
) -> list[sympy.core.basic.Basic]:
    if len(free_symbols) == 0:
        return []
    variable_order = dependent_symbols + free_symbols

    polys = [sympy.Poly(sympy.nsimplify(e), *variable_order, domain="QQ") for e in equations]
    elimination_basis = sympy.groebner(
        polys, *variable_order, order="grevlex", domain="QQ", method="f5b"
    )
    free_set = set(free_symbols)
    relations: list[sympy.core.basic.Basic] = []
    for poly in elimination_basis.polys:
        expr = sympy.sympify(poly.as_expr())
        if expr == 0:
            continue
        if expr.free_symbols.issubset(free_set):
            relations.append(sympy.expand(expr))
    return relations


def _solve_with_grobner(
    equations: list[sympy.core.basic.Basic],
    unknown_symbols: list[sympy.Symbol],
    max_solutions: int | None,
) -> GrobnerSolveResult:
    if not unknown_symbols or not equations:
        return GrobnerSolveResult(solutions=[], free_symbols=[], free_symbol_relations=[])

    polys = [sympy.Poly(sympy.nsimplify(e), *unknown_symbols, domain="QQ") for e in equations]
    grobner_basis = sympy.groebner(polys, *unknown_symbols, order="lex", domain="QQ", method="f5b")
    dependent_symbols, free_symbols = _split_dependent_and_free_symbols(
        grobner_basis, unknown_symbols
    )
    relations = _relations_among_free_symbols(equations, dependent_symbols, free_symbols)

    if grobner_basis.is_zero_dimensional:
        raw_solutions = sympy.solve(list(grobner_basis), unknown_symbols, dict=True)
    elif len(dependent_symbols) == 0:
        raw_solutions = [{}]
    else:
        raw_solutions = sympy.solve(list(grobner_basis), dependent_symbols, dict=True)

    if max_solutions is not None:
        raw_solutions = raw_solutions[:max_solutions]

    return GrobnerSolveResult(
        solutions=raw_solutions, free_symbols=free_symbols, free_symbol_relations=relations
    )

def _solution_is_numeric(named_solution: dict[str, sympy.core.basic.Basic]) -> bool:
    """Return True when every value in the solution is a concrete number (no free symbols)."""
    return all(
        not sympy.sympify(value).free_symbols for value in named_solution.values()
    )


def _construct_explicit_tableau(
    stages: int, symbol_values: dict[str, sympy.core.basic.Basic]
) -> tuple[list[list[float]], list[float]]:
    a_matrix: list[list[float]] = [[0.0 for _ in range(stages)] for _ in range(stages)]
    b_vector: list[float] = [0.0 for _ in range(stages)]

    for i in range(stages):
        for j in range(i):
            a_matrix[i][j] = float(sympy.N(symbol_values.get(f"a{i}{j}", sympy.Integer(0)), 20))
        b_vector[i] = float(sympy.N(symbol_values.get(f"b{i}", sympy.Integer(0)), 20))

    return a_matrix, b_vector


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
    ansatz: Ansatz | None = None,
    constraints: list[Constraint] | None = None,
    fixed_values: dict[str, float | int | sympy.core.basic.Basic] | None = None,
    zero_symbols: list[str] | None = None,
    max_solutions: int | None = 1,
    verify_symbolic: bool = True,
    ansatz_validation_tol: float = 1e-10,
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
    if ansatz_validation_tol < 0:
        raise ValueError("ansatz_validation_tol must be non-negative")

    equations, trees = generate_explicit_order_equations(order, stages, rationalise=True)
    if antisymmetric_order is not None:
        antisymmetric_equations, antisymmetric_trees = generate_explicit_antisymmetric_equations(
            antisymmetric_order, stages, rationalise=True
        )
        equations = equations + antisymmetric_equations
        trees = trees + antisymmetric_trees
    ansatz_used: Ansatz = IdentityAnsatz() if ansatz is None else ansatz
    equations = equations + ansatz_used.extra_equations(stages=stages)

    compiled_constraints = compile_constraints(constraints if constraints is not None else [])
    equations = equations + compiled_constraints.equations
    a_symbols, b_symbols = explicit_unknown_symbols(stages)
    default_symbols = a_symbols + b_symbols
    solve_symbols = ansatz_used.solve_symbols(stages=stages)
    all_symbols = default_symbols if solve_symbols is None else solve_symbols

    assignments = _normalise_assignments(fixed_values, zero_symbols)
    substitutions_map = _merge_substitution_maps(
        [
            ansatz_used.extra_substitutions(stages=stages),
            compiled_constraints.substitutions,
        ]
    )
    for symbol, value in assignments.items():
        resolved_value = sympy.sympify(value)
        if symbol in substitutions_map:
            equations.append(sympy.simplify(sympy.expand(substitutions_map[symbol] - resolved_value)))
        else:
            substitutions_map[symbol] = resolved_value
    substitutions = list(substitutions_map.items())
    reduced_equations = [sympy.simplify(sympy.expand(e.subs(substitutions))) for e in equations]
    reduced_equations = [e for e in reduced_equations if sympy.simplify(e) != 0]

    active_symbols = _free_symbols_in_equations(reduced_equations, all_symbols)
    active_symbols = [symbol for symbol in active_symbols if symbol not in substitutions_map]

    grobner_result = _solve_with_grobner(
        reduced_equations, active_symbols, max_solutions
    )
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
            tol=ansatz_validation_tol,
        ):
            continue
        named_solutions.append(named)
        if _solution_is_numeric(named):
            a_matrix, b_vector = _construct_explicit_tableau(stages, named)
            methods.append(RK(a_matrix, b_vector, f"generated_explicit_rk_s{stages}_p{order}_{index}"))

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
