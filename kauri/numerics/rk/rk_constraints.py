"""
Constraint helpers for RK maker equations.
"""

from dataclasses import dataclass
from typing import Literal, cast

import sympy

from kauri.hopf_algebras.utils import _as_expr


ExprLike = sympy.core.basic.Basic | int | float | str


def _to_symbol(name: str | sympy.Symbol) -> sympy.Symbol:
    if isinstance(name, sympy.Symbol):
        return name
    return sympy.symbols(name)


def _to_expr(value: ExprLike) -> sympy.core.basic.Basic:
    return _as_expr(value)


@dataclass(frozen=True)
class Constraint:
    kind: Literal["set", "tie", "equation"]
    lhs: str
    rhs: str | None = None
    value: sympy.core.basic.Basic | None = None

    @staticmethod
    def set(symbol: str, value: ExprLike) -> "Constraint":
        return Constraint(kind="set", lhs=symbol, value=_to_expr(value))

    @staticmethod
    def zero(symbol: str) -> "Constraint":
        return Constraint(kind="set", lhs=symbol, value=sympy.Integer(0))

    @staticmethod
    def one(symbol: str) -> "Constraint":
        return Constraint(kind="set", lhs=symbol, value=sympy.Integer(1))

    @staticmethod
    def tie(lhs_symbol: str, rhs_symbol: str) -> "Constraint":
        return Constraint(kind="tie", lhs=lhs_symbol, rhs=rhs_symbol)

    @staticmethod
    def equation(lhs_expr: ExprLike, rhs_expr: ExprLike = 0) -> "Constraint":
        return Constraint(
            kind="equation",
            lhs=sympy.sstr(_to_expr(lhs_expr)),
            rhs=sympy.sstr(_to_expr(rhs_expr)),
        )


@dataclass(frozen=True)
class CompiledConstraints:
    substitutions: dict[sympy.Symbol, sympy.core.basic.Basic]
    equations: list[sympy.core.basic.Basic]


def compile_constraints(constraints: list[Constraint]) -> CompiledConstraints:
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
        lhs_root = find(lhs)
        rhs_root = find(rhs)
        if lhs_root == rhs_root:
            return
        if str(lhs_root) <= str(rhs_root):
            parent[rhs_root] = lhs_root
        else:
            parent[lhs_root] = rhs_root

    for constraint in constraints:
        if constraint.kind == "tie":
            if constraint.rhs is None:
                raise ValueError("tie constraint requires rhs")
            union(_to_symbol(constraint.lhs), _to_symbol(constraint.rhs))

    symbols_seen: set[sympy.Symbol] = set()
    for constraint in constraints:
        if constraint.kind in {"set", "tie"}:
            symbols_seen.add(_to_symbol(constraint.lhs))
        if constraint.kind == "tie" and constraint.rhs is not None:
            symbols_seen.add(_to_symbol(constraint.rhs))

    alias_substitutions: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
    for symbol in symbols_seen:
        representative = find(symbol)
        if representative != symbol:
            alias_substitutions[symbol] = representative

    alias_items = list(alias_substitutions.items())

    set_substitutions: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
    for constraint in constraints:
        if constraint.kind != "set":
            continue
        if constraint.value is None:
            raise ValueError("set constraint requires a value")
        lhs_symbol = _to_symbol(constraint.lhs)
        representative = find(lhs_symbol)
        resolved_value = sympy.simplify(constraint.value.subs(alias_items))
        if representative in set_substitutions:
            lhs_expr = cast(sympy.Expr, sympy.sympify(set_substitutions[representative]))
            rhs_expr = cast(sympy.Expr, sympy.sympify(resolved_value))
            if sympy.simplify(lhs_expr - rhs_expr) != 0:
                raise ValueError(
                    f"Conflicting assignments for {representative}: "
                    f"{set_substitutions[representative]} and {resolved_value}"
                )
        set_substitutions[representative] = resolved_value

    substitutions: dict[sympy.Symbol, sympy.core.basic.Basic] = {}
    substitutions.update(alias_substitutions)
    substitutions.update(set_substitutions)
    substitution_items = list(substitutions.items())

    equations: list[sympy.core.basic.Basic] = []
    for constraint in constraints:
        if constraint.kind != "equation":
            continue
        lhs_expr = cast(sympy.Expr, sympy.sympify(_to_expr(constraint.lhs)))
        rhs_raw = sympy.Integer(0) if constraint.rhs is None else _to_expr(constraint.rhs)
        rhs_expr = cast(sympy.Expr, sympy.sympify(rhs_raw))
        expression = sympy.simplify(sympy.expand((lhs_expr - rhs_expr).subs(substitution_items)))
        if expression != 0:
            equations.append(expression)

    return CompiledConstraints(substitutions=substitutions, equations=equations)
