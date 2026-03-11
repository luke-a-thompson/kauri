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

from kauri.hopf_algebras import bck, cem
from kauri.hopf_algebras.bseries import BSeries, elementary_differential
from kauri.hopf_algebras.maps import Map, exact_weights, ident, omega, sign
from kauri.hopf_algebras.oddeven import id_sqrt, minus, plus
from kauri.numerics.ansatze.williamson import WilliamsonAnsatz
from kauri.numerics.methods.cf import VerificationResult, verify_cf_ees
from kauri.numerics.methods.rk import RK
from kauri.numerics.methods.williamson import WilliamsonCF, WilliamsonRK
from kauri.numerics.planar_trees.planar_basis import (
    OrderedForest,
    OrderedForestSum,
    PlanarTree,
    TensorOrderedSum,
)
from kauri.numerics.rk.rk import rk_order_cond, rk_symbolic_weight
from kauri.numerics.rk.rk_constraints import Constraint
from kauri.numerics.rk.rk_maker import SolveResult, build_method_from_ansatz
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
from kauri.trees.gentrees import (
    planar_trees_of_order,
    planar_trees_up_to_order,
    trees_of_order,
    trees_up_to_order,
)
from kauri.trees.trees import Forest, ForestSum, TensorProductSum, Tree
from kauri.viz.display import display
