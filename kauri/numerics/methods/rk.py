"""Runge-Kutta method representation (no runtime stepping/integration)."""

import copy

import sympy

from kauri.hopf_algebras.bck import counit
from kauri.hopf_algebras.maps import Map, exact_weights, sign
from kauri.trees.gentrees import trees_of_order

_EXACT_SUBSTITUTION_LOG_MAP = exact_weights.log()


class RK:
    def __init__(self, a, b, name=None):
        self.name = name
        self.s = len(b)
        if len(a) != self.s or len(a[0]) != self.s:
            raise ValueError(
                "Parameter 'a' must be a square s x s matrix and b a vector of length s"
            )
        self.a = a
        self.b = b
        self.c = [sum(a[i][j] for j in range(self.s)) for i in range(self.s)]
        self.explicit = self._check_explicit()
        self.deriv_dict = {}
        for i in range(self.s):
            self.deriv_dict[(i, repr(None))] = 1
            self.deriv_dict[(i, repr([]))] = 1

    def __repr__(self):
        out = "["
        for i in range(self.s - 1):
            out += repr(self.a[i]) + ",\n"
        out += repr(self.a[-1]) + "]\n"
        out += repr(self.b)
        return out

    def _rationalised_tableau(
        self,
    ) -> tuple[
        list[sympy.core.basic.Basic],
        list[list[sympy.core.basic.Basic]],
        list[sympy.core.basic.Basic],
    ]:
        c_vector = [sympy.nsimplify(value, rational=True) for value in self.c]
        a_matrix = [
            [sympy.nsimplify(self.a[i][j], rational=True) for j in range(self.s)]
            for i in range(self.s)
        ]
        b_vector = [sympy.nsimplify(value, rational=True) for value in self.b]
        return c_vector, a_matrix, b_vector

    def to_text(self, max_cell_chars: int = 48) -> str:
        from kauri.numerics.rk.rk_maker_format import format_tableau_text

        c_vector, a_matrix, b_vector = self._rationalised_tableau()
        return format_tableau_text(
            c_vector=c_vector,
            a_matrix=a_matrix,
            b_vector=b_vector,
            max_cell_chars=max_cell_chars,
        )

    def to_latex(self, max_cell_chars: int = 48) -> str:
        from kauri.numerics.rk.rk_maker_format import format_tableau_latex

        c_vector, a_matrix, b_vector = self._rationalised_tableau()
        return format_tableau_latex(
            c_vector=c_vector,
            a_matrix=a_matrix,
            b_vector=b_vector,
            max_cell_chars=max_cell_chars,
        )

    def _check_explicit(self):
        for i in range(self.s):
            for j in range(i, self.s):
                if self.a[i][j]:
                    return False
        return True

    def _inverse(self):
        b_inv = [-self.b[i] for i in range(self.s)]
        a_inv = [[self.a[i][j] - self.b[j] for j in range(self.s)] for i in range(self.s)]
        return RK(a_inv, b_inv)

    def reverse(self) -> "RK":
        return RK(
            [[-self.a[i][j] for j in range(self.s)] for i in range(self.s)],
            [-self.b[i] for i in range(self.s)],
        )

    def adjoint(self) -> "RK":
        b_adj = [self.b[self.s - 1 - j] for j in range(self.s)]
        a_adj = [
            [self.b[self.s - 1 - j] - self.a[self.s - 1 - i][self.s - j - 1] for j in range(self.s)]
            for i in range(self.s)
        ]
        return RK(a_adj, b_adj)

    def __mul__(self, other: "RK") -> "RK":
        s1 = other.s
        a1 = other.a
        b1 = other.b
        s2 = self.s
        a2 = self.a
        b2 = self.b
        a = [[a1[i][j] for j in range(s1)] + [0 for _ in range(s2)] for i in range(s1)]
        a += [[b1[j] for j in range(s1)] + [a2[i][j] for j in range(s2)] for i in range(s2)]
        return RK(a, list(b1) + list(b2))

    def __pow__(self, exponent: int) -> "RK":
        if exponent == 0:
            return RK([[0]], [0])
        expn_ = exponent
        out = self._inverse() if exponent < 0 else copy.deepcopy(self)
        if exponent < 0:
            expn_ = -exponent
        for _ in range(expn_ - 1):
            out = out * self
        return out

    def _internal_weights(self, i, t_rep):
        return sum(self.a[i][j] * self._derivative_weights(j, t_rep) for j in range(self.s))

    def _derivative_weights(self, i, t_rep):
        if (i, repr(t_rep)) in self.deriv_dict:
            return self.deriv_dict[(i, repr(t_rep))]
        out = 1
        for subtree in t_rep[:-1]:
            out *= self._internal_weights(i, subtree)
        self.deriv_dict[(i, repr(t_rep))] = out
        return out

    def _elementary_weights(self, t_rep):
        if t_rep is None:
            return 1
        return sum(self.b[i] * self._derivative_weights(i, t_rep) for i in range(self.s))

    def elementary_weights_map(self) -> Map:
        return Map(lambda x: self._elementary_weights(x.list_repr))

    def modified_equation_map(self) -> Map:
        return self.elementary_weights_map().modified_equation()

    def order(self, tol: float = 1e-10, limit: int = 10) -> int:
        theta = self.elementary_weights_map().log()
        n = 0
        while True:
            for t in trees_of_order(n):
                if abs(theta(t) - _EXACT_SUBSTITUTION_LOG_MAP(t)) > tol:
                    return n - 1
            if n >= limit:
                raise RuntimeError("Order equals or exceeds limit of " + str(limit))
            n += 1

    def antisymmetric_order(self, tol: float = 1e-10, limit: int = 10) -> int:
        ew = self.elementary_weights_map()
        m = (ew & sign) * ew
        n = 0
        while True:
            for t in trees_of_order(n):
                if abs(m(t) - counit(t)) > tol:
                    return n - 1
            if n >= limit:
                raise RuntimeError("Order equals or exceeds limit of " + str(limit))
            n += 1
