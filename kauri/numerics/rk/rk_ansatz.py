"""
Pluggable ansatz abstractions for RK method construction.
"""

from dataclasses import dataclass
from typing import Protocol

import sympy


class Ansatz(Protocol):
    """
    Protocol for structural RK ansatzes.
    """

    name: str

    def extra_equations(self, stages: int) -> list[sympy.core.basic.Basic]:
        """
        Return additional equations (each interpreted as expr == 0).
        """
        ...

    def extra_substitutions(self, stages: int) -> dict[sympy.Symbol, sympy.core.basic.Basic]:
        """
        Return additional substitutions applied before solving.
        """
        ...

    def solve_symbols(self, stages: int) -> list[sympy.Symbol] | None:
        """
        Return the primary symbols to solve for, or None to use default explicit-RK symbols.
        """
        ...

    def post_validate(
        self,
        stages: int,
        named_solution: dict[str, sympy.core.basic.Basic],
        solver: str,
        tol: float,
    ) -> bool:
        """
        Validate a candidate solution after solve/verification.
        """
        ...


@dataclass
class IdentityAnsatz:
    """
    No-op ansatz used as default.
    """

    name: str = "standard_explicit"

    def extra_equations(self, stages: int) -> list[sympy.core.basic.Basic]:
        return []

    def extra_substitutions(self, stages: int) -> dict[sympy.Symbol, sympy.core.basic.Basic]:
        return {}

    def solve_symbols(self, stages: int) -> list[sympy.Symbol] | None:
        return None

    def post_validate(
        self,
        stages: int,
        named_solution: dict[str, sympy.core.basic.Basic],
        solver: str,
        tol: float,
    ) -> bool:
        if tol < 0:
            raise ValueError("tol must be non-negative")
        return True


@dataclass
class CompositeAnsatz:
    """
    Compose multiple ansatzes into one.
    """

    ansatzes: list[Ansatz]
    name: str = "composite"

    def __post_init__(self) -> None:
        if len(self.ansatzes) == 0:
            raise ValueError("ansatzes must not be empty")

    def extra_equations(self, stages: int) -> list[sympy.core.basic.Basic]:
        equations: list[sympy.core.basic.Basic] = []
        for ansatz in self.ansatzes:
            equations.extend(ansatz.extra_equations(stages))
        return equations

    def extra_substitutions(self, stages: int) -> dict[sympy.Symbol, sympy.core.basic.Basic]:
        merged: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
        for ansatz in self.ansatzes:
            substitutions = ansatz.extra_substitutions(stages)
            for symbol, value in substitutions.items():
                if symbol in merged:
                    if sympy.simplify(sympy.sympify(merged[symbol]) - sympy.sympify(value)) != 0:
                        raise ValueError(f"Conflicting ansatz substitutions for {symbol}")
                merged[symbol] = sympy.sympify(value)
        return merged

    def solve_symbols(self, stages: int) -> list[sympy.Symbol] | None:
        combined: list[sympy.Symbol] = []
        for ansatz in self.ansatzes:
            symbols = ansatz.solve_symbols(stages)
            if symbols is None:
                continue
            for symbol in symbols:
                if symbol not in combined:
                    combined.append(symbol)
        if len(combined) == 0:
            return None
        return combined

    def post_validate(
        self,
        stages: int,
        named_solution: dict[str, sympy.core.basic.Basic],
        solver: str,
        tol: float,
    ) -> bool:
        return all(
            ansatz.post_validate(
                stages=stages, named_solution=named_solution, solver=solver, tol=tol
            )
            for ansatz in self.ansatzes
        )
