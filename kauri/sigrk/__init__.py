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
Indexed-template tools for signature Runge--Kutta schemes.

This package implements the first milestone of the indexed SigRK search
specification: indexed stage templates, fixed-dimension discovery equations,
word shuffle algebra, recursive scheme weights, and the known order-two and
order-three templates.
"""

from .equations import OrderCondition, order_conditions
from .examples import (
    known_order2_recovery_problem,
    known_order2_template,
    known_order3_template,
)
from .graphs import ExpandedGraph, expand_template, layer_count_polynomial, stage_count_polynomial
from .search import (
    EdgeRuleTemplate,
    EnumeratedTemplate,
    SearchResult,
    SupportPrototype,
    UpdateRuleTemplate,
    enumerate_edge_rule_templates,
    enumerate_rule_sets,
    enumerate_stage_family_templates,
    enumerate_support_sets,
    enumerate_templates,
    enumerate_update_rule_templates,
    search_bounded,
    search_fixed_templates,
)
from .solve import groebner_basis, groebner_is_impossible, solve_equations
from .supports import Guard, SupportTerm, filter_admissible_fixed_power, fixed_power_admissible
from .templates import ConcreteStage, EdgeRule, SigRKTemplate, StageFamily, UpdateRule
from .trees import KauriTreeBackend, LabeledTree, PurePythonTreeBackend, TreeBackend
from .verify import DiscoveryVerification, verify_all_dimensions, verify_discovery
from .weights import SchemeWeights
from .words import LinearFunctional, Word, exact_weight, shuffle_counts

__all__ = [
    "ConcreteStage",
    "DiscoveryVerification",
    "EdgeRule",
    "EdgeRuleTemplate",
    "EnumeratedTemplate",
    "ExpandedGraph",
    "Guard",
    "KauriTreeBackend",
    "LabeledTree",
    "LinearFunctional",
    "OrderCondition",
    "PurePythonTreeBackend",
    "SearchResult",
    "SchemeWeights",
    "SigRKTemplate",
    "StageFamily",
    "SupportTerm",
    "SupportPrototype",
    "TreeBackend",
    "UpdateRule",
    "UpdateRuleTemplate",
    "Word",
    "exact_weight",
    "expand_template",
    "filter_admissible_fixed_power",
    "fixed_power_admissible",
    "groebner_basis",
    "groebner_is_impossible",
    "known_order2_recovery_problem",
    "known_order2_template",
    "known_order3_template",
    "layer_count_polynomial",
    "enumerate_edge_rule_templates",
    "enumerate_rule_sets",
    "enumerate_stage_family_templates",
    "enumerate_support_sets",
    "enumerate_templates",
    "enumerate_update_rule_templates",
    "order_conditions",
    "search_bounded",
    "search_fixed_templates",
    "shuffle_counts",
    "solve_equations",
    "stage_count_polynomial",
    "verify_all_dimensions",
    "verify_discovery",
]
