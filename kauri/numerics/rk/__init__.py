"""Runge-Kutta methods and synthesis tools."""

from kauri.numerics.rk.rk import RK, rk_order_cond, rk_symbolic_weight
from kauri.numerics.rk.rk_ansatz import BaseAnsatz, CompositeAnsatz
from kauri.numerics.rk.rk_constraints import CompiledConstraints, Constraint, compile_constraints
from kauri.numerics.rk.rk_maker import (
    RKMakerResult,
    explicit_unknown_symbols,
    generate_explicit_order_equations,
    make_explicit_rk_methods,
)
from kauri.numerics.rk.rk_methods import (
    EES25,
    EES27,
    backward_euler,
    crank_nicolson,
    euler,
    gauss6,
    heun_rk2,
    heun_rk3,
    implicit_midpoint,
    kutta_rk3,
    lobatto6,
    midpoint,
    nystrom_rk5,
    radau_iia,
    ralston_rk3,
    ralston_rk4,
    rk4,
)
from kauri.numerics.rk.rk_objectives import MethodScore, RKObjective, score_methods
from kauri.numerics.rk.williamson_ansatz import (
    WilliamsonAnsatz,
    generate_2n_polynomial_constraints,
    is_2n_tableau,
)
