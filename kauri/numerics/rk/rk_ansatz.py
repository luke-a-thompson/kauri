"""
Pluggable ansatz abstractions for RK method construction.
"""

from abc import ABC
from dataclasses import dataclass

import sympy

from kauri.hopf_algebras.bck import counit
from kauri.hopf_algebras.maps import sign
from kauri.numerics.rk.rk import _rk_symbolic_weights_map, rk_order_cond
from kauri.trees.gentrees import trees_up_to_order
from kauri.trees.trees import Tree


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


class BaseAnsatz(ABC):
    """
    Base interface for structural RK ansatzes.

    The default implementation models the full explicit RK tableau family.
    """

    def unknown_symbols(self, stages: int) -> list[sympy.Symbol]:
        """
        Return the symbolic unknowns solved in this ansatz parameterisation.
        """
        a_symbols, b_symbols = explicit_unknown_symbols(stages)
        return a_symbols + b_symbols

    def tableau_substitutions(self, stages: int) -> dict[sympy.Symbol, sympy.core.basic.Basic]:
        """
        Return substitutions that define tableau symbols in terms of unknown symbols.
        """
        return {}

    def base_equations(
        self,
        order: int,
        stages: int,
        antisymmetric_order: int | None = None,
    ) -> tuple[list[sympy.core.basic.Basic], list[Tree]]:
        """
        Return base order equations and corresponding tree list.
        """
        equations, trees = generate_explicit_order_equations(order, stages, rationalise=True)
        if antisymmetric_order is not None:
            antisymmetric_equations, antisymmetric_trees = generate_explicit_antisymmetric_equations(
                antisymmetric_order, stages, rationalise=True
            )
            equations = equations + antisymmetric_equations
            trees = trees + antisymmetric_trees
        return equations, trees

    def extra_equations(self, stages: int) -> list[sympy.core.basic.Basic]:
        """
        Return additional equations (each interpreted as expr == 0).
        """
        return []

    def build_tableau(
        self,
        stages: int,
        named_solution: dict[str, sympy.core.basic.Basic],
    ) -> tuple[list[list[float]], list[float]]:
        """
        Build numerical tableau from a named symbolic solution.
        """
        return _construct_explicit_tableau(stages, named_solution)

    def post_validate(
        self,
        stages: int,
        named_solution: dict[str, sympy.core.basic.Basic],
    ) -> bool:
        """
        Validate a candidate solution after solve/verification.
        """
        return True


class ExplicitAnsatz(BaseAnsatz):
    """
    Named explicit RK tableau ansatz.
    """


@dataclass
class CompositeAnsatz(BaseAnsatz):
    """
    Compose multiple ansatzes into one.
    """

    ansatzes: list[BaseAnsatz]

    def __post_init__(self) -> None:
        if len(self.ansatzes) == 0:
            raise ValueError("ansatzes must not be empty")

    def unknown_symbols(self, stages: int) -> list[sympy.Symbol]:
        combined: list[sympy.Symbol] = []
        for ansatz in self.ansatzes:
            for symbol in ansatz.unknown_symbols(stages):
                if symbol not in combined:
                    combined.append(symbol)
        return combined

    def base_equations(
        self,
        order: int,
        stages: int,
        antisymmetric_order: int | None = None,
    ) -> tuple[list[sympy.core.basic.Basic], list[Tree]]:
        return self.ansatzes[0].base_equations(
            order=order,
            stages=stages,
            antisymmetric_order=antisymmetric_order,
        )

    def extra_equations(self, stages: int) -> list[sympy.core.basic.Basic]:
        equations: list[sympy.core.basic.Basic] = []
        for ansatz in self.ansatzes:
            equations.extend(ansatz.extra_equations(stages))
        return equations

    def tableau_substitutions(self, stages: int) -> dict[sympy.Symbol, sympy.core.basic.Basic]:
        merged: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
        for ansatz in self.ansatzes:
            substitutions = ansatz.tableau_substitutions(stages)
            for symbol, value in substitutions.items():
                if symbol in merged:
                    if sympy.simplify(sympy.sympify(merged[symbol]) - sympy.sympify(value)) != 0:
                        raise ValueError(f"Conflicting ansatz substitutions for {symbol}")
                merged[symbol] = sympy.sympify(value)
        return merged

    def build_tableau(
        self,
        stages: int,
        named_solution: dict[str, sympy.core.basic.Basic],
    ) -> tuple[list[list[float]], list[float]]:
        return self.ansatzes[0].build_tableau(stages=stages, named_solution=named_solution)

    def post_validate(
        self,
        stages: int,
        named_solution: dict[str, sympy.core.basic.Basic],
    ) -> bool:
        return all(
            ansatz.post_validate(stages=stages, named_solution=named_solution)
            for ansatz in self.ansatzes
        )
