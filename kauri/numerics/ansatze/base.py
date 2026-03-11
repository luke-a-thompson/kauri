"""
Base ansatz interface for RK tableau synthesis.
"""

from abc import ABC

import sympy

from kauri.trees.trees import Tree


class BaseAnsatz(ABC):
    """
    Minimal interface for tableau parameterisations.
    """

    def unknown_symbols(self, stages: int) -> list[sympy.Symbol]:
        raise NotImplementedError

    def tableau_substitutions(self, stages: int) -> dict[sympy.Symbol, sympy.core.basic.Basic]:
        return {}

    def base_equations(
        self,
        order: int,
        stages: int,
        antisymmetric_order: int | None = None,
    ) -> tuple[list[sympy.core.basic.Basic], list[Tree]]:
        raise NotImplementedError

    def extra_equations(self, stages: int) -> list[sympy.core.basic.Basic]:
        return []

    def build_tableau(
        self,
        stages: int,
        named_solution: dict[str, sympy.core.basic.Basic],
    ) -> tuple[list[list[float]], list[float]]:
        raise NotImplementedError

    def post_validate(
        self,
        stages: int,
        named_solution: dict[str, sympy.core.basic.Basic],
    ) -> bool:
        return True

