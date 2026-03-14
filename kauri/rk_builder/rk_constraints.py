"""
Constraint helpers for RK maker equations.
"""

from dataclasses import dataclass

import sympy


@dataclass(frozen=True)
class SetConstraint:
    symbol: sympy.Symbol
    value: sympy.Basic


@dataclass(frozen=True)
class TieConstraint:
    lhs: sympy.Symbol
    rhs: sympy.Symbol


@dataclass(frozen=True)
class EquationConstraint:
    lhs: sympy.Basic
    rhs: sympy.Basic


AnyConstraint = SetConstraint | TieConstraint | EquationConstraint


@dataclass(frozen=True)
class CompiledConstraints:
    substitutions: dict[sympy.Symbol, sympy.core.basic.Basic]
    equations: list[sympy.core.basic.Basic]


def compile_constraints(constraints: list[AnyConstraint]) -> CompiledConstraints:
    """
    Compile constraints into substitutions and equations (expr == 0).
    """
    parent: dict[sympy.Symbol, sympy.Symbol] = {}

    def find(symbol: sympy.Symbol) -> sympy.Symbol:
        root = parent.get(symbol, symbol)
        if root != symbol:
            root = find(root)
            parent[symbol] = root
        return root

    def union(lhs: sympy.Symbol, rhs: sympy.Symbol) -> None:
        lhs_root, rhs_root = find(lhs), find(rhs)
        if lhs_root == rhs_root:
            return
        if str(lhs_root) <= str(rhs_root):
            parent[rhs_root] = lhs_root
        else:
            parent[lhs_root] = rhs_root

    for constraint in constraints:
        if isinstance(constraint, TieConstraint):
            union(constraint.lhs, constraint.rhs)

    alias_substitutions: dict[sympy.Symbol, sympy.Basic] = {}
    for constraint in constraints:
        symbols = (
            [constraint.symbol] if isinstance(constraint, SetConstraint)
            else [constraint.lhs, constraint.rhs] if isinstance(constraint, TieConstraint)
            else []
        )
        for symbol in symbols:
            representative = find(symbol)
            if representative != symbol:
                alias_substitutions[symbol] = representative

    set_substitutions: dict[sympy.Symbol, sympy.Basic] = {}
    for constraint in constraints:
        if not isinstance(constraint, SetConstraint):
            continue
        representative = find(constraint.symbol)
        resolved_value = sympy.simplify(constraint.value.subs(alias_substitutions.items()))
        if representative in set_substitutions:
            if sympy.simplify(set_substitutions[representative] - resolved_value) != 0:
                raise ValueError(
                    f"Conflicting assignments for {representative}: "
                    f"{set_substitutions[representative]} and {resolved_value}"
                )
        set_substitutions[representative] = resolved_value

    substitutions: dict[sympy.Symbol, sympy.Basic] = {**alias_substitutions, **set_substitutions}

    equations: list[sympy.Basic] = []
    for constraint in constraints:
        if isinstance(constraint, EquationConstraint):
            expression = sympy.simplify(
                sympy.expand((constraint.lhs - constraint.rhs).subs(substitutions.items()))
            )
            if expression != 0:
                equations.append(expression)

    return CompiledConstraints(substitutions=substitutions, equations=equations)
