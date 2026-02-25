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

"""
Algebraic manipulation of rooted trees for the analysis of B-series and Runge-Kutta schemes.
"""

import kauri.bck
import kauri.cem
from kauri.bseries import BSeries, elementary_differential
from kauri.display import display
from kauri.gentrees import trees_of_order, trees_up_to_order
from kauri.maps import Map, exact_weights, ident, omega, sign
from kauri.oddeven import id_sqrt, minus, plus
from kauri.rk import RK, rk_order_cond, rk_symbolic_weight
from kauri.rk_ansatz import CompositeAnsatz, IdentityAnsatz
from kauri.ansatz_2n import TwoNStorageAnsatz, generate_2n_aform_constraints
from kauri.rk_constraints import CompiledConstraints, Constraint, compile_constraints
from kauri.rk_maker import (
    RKMakerResult,
    explicit_unknown_symbols,
    generate_explicit_order_equations,
    make_explicit_rk_methods,
)
from kauri.rk_objectives import MethodScore, RKObjective, score_methods
from kauri.rk_methods import (
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
from kauri.trees import (
    EMPTY_FOREST,
    EMPTY_FOREST_SUM,
    EMPTY_TREE,
    ZERO_FOREST_SUM,
    Forest,
    ForestSum,
    TensorProductSum,
    Tree,
)
