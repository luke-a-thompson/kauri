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

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, count, islice, product
from typing import Any

from .equations import OrderCondition, consistency_conditions, order_conditions
from .solve import groebner_is_impossible, solve_equations
from .supports import Guard, SupportTerm
from .templates import EdgeRule, SigRKTemplate, StageFamily, UpdateRule


@dataclass(frozen=True)
class SearchResult:
    template: SigRKTemplate
    conditions: tuple[OrderCondition, ...]
    equations: tuple[Any, ...] = ()
    variables: tuple[Any, ...] = ()
    impossible: bool = False
    solutions: tuple[dict, ...] = ()


@dataclass(frozen=True)
class EdgeRuleTemplate:
    """A support-free indexed edge-rule skeleton."""

    source: str
    target: str
    label_var: str
    name: str = ""


@dataclass(frozen=True)
class UpdateRuleTemplate:
    """A support-free indexed update-rule skeleton."""

    stage: str
    label_var: str
    name: str = ""


@dataclass(frozen=True)
class SupportPrototype:
    """A support element before assigning its scalar unknown."""

    power: Fraction
    word: tuple[str, ...] = ()
    guard: Guard = Guard()
    name: str = ""

    def variable_names(self) -> set[str]:
        names = set(self.word)
        for left, right in self.guard.equalities:
            names.add(left)
            names.add(right)
        return names

    def compatible_with(self, variables: set[str]) -> bool:
        return self.variable_names() <= variables

    def instantiate(self, coeff) -> SupportTerm:
        return SupportTerm(self.power, self.word, coeff, self.guard)


@dataclass(frozen=True)
class EnumeratedTemplate:
    template: SigRKTemplate
    variables: tuple[Any, ...]


SupportOptions = (
    tuple[tuple[SupportPrototype, ...], ...]
    | dict[str, tuple[tuple[SupportPrototype, ...], ...]]
)


def enumerate_stage_family_templates(
    family_pool: tuple[StageFamily, ...],
    min_families: int,
    max_families: int,
    required_names: tuple[str, ...] = (),
) -> tuple[tuple[StageFamily, ...], ...]:
    """Enumerate stage-family templates from a finite pool."""

    required = set(required_names)
    out = []
    for size in range(min_families, max_families + 1):
        for families in combinations(family_pool, size):
            names = {family.name for family in families}
            if len(names) != len(families):
                continue
            if required <= names:
                out.append(families)
    return tuple(out)


def enumerate_edge_rule_templates(
    families: tuple[StageFamily, ...],
    label_vars: tuple[str, ...],
) -> tuple[EdgeRuleTemplate, ...]:
    """Enumerate all explicit indexed edge-rule skeletons for ``families``."""

    rules = []
    for source in families:
        for target in families:
            if source.layer >= target.layer:
                continue
            for label_var in label_vars:
                rules.append(
                    EdgeRuleTemplate(
                        source.name,
                        target.name,
                        label_var,
                        name=f"{source.name}->{target.name}:{label_var}",
                    )
                )
    return tuple(rules)


def enumerate_update_rule_templates(
    families: tuple[StageFamily, ...],
    label_vars: tuple[str, ...],
) -> tuple[UpdateRuleTemplate, ...]:
    """Enumerate all indexed update-rule skeletons for ``families``."""

    return tuple(
        UpdateRuleTemplate(family.name, label_var, name=f"{family.name}:{label_var}")
        for family in families
        for label_var in label_vars
    )


def enumerate_rule_sets(
    rule_pool: tuple[Any, ...],
    min_rules: int,
    max_rules: int,
) -> tuple[tuple[Any, ...], ...]:
    """Enumerate subsets of a finite rule pool."""

    return tuple(
        rules
        for size in range(min_rules, max_rules + 1)
        for rules in combinations(rule_pool, size)
    )


def enumerate_support_sets(
    support_pool: tuple[SupportPrototype, ...],
    variables: set[str],
    min_terms: int,
    max_terms: int,
) -> tuple[tuple[SupportPrototype, ...], ...]:
    """Enumerate compatible support subsets for one rule skeleton."""

    compatible = tuple(
        prototype for prototype in support_pool if prototype.compatible_with(variables)
    )
    return tuple(
        support_set
        for size in range(min_terms, max_terms + 1)
        for support_set in combinations(compatible, size)
    )


def _edge_variables(families: dict[str, StageFamily], rule: EdgeRuleTemplate) -> set[str]:
    return (
        set(families[rule.source].indices)
        | set(families[rule.target].indices)
        | {rule.label_var}
    )


def _update_variables(families: dict[str, StageFamily], rule: UpdateRuleTemplate) -> set[str]:
    return set(families[rule.stage].indices) | {rule.label_var}


def _support_options(
    options: SupportOptions,
    rule_name: str,
    variables: set[str],
    fallback_pool: tuple[SupportPrototype, ...],
    min_terms: int,
    max_terms: int,
) -> tuple[tuple[SupportPrototype, ...], ...]:
    if isinstance(options, dict):
        raw_options = options.get(rule_name, ())
    else:
        raw_options = options or enumerate_support_sets(
            fallback_pool,
            variables,
            min_terms,
            max_terms,
        )

    return tuple(
        support_set
        for support_set in raw_options
        if all(prototype.compatible_with(variables) for prototype in support_set)
    )


def _symbol(name: str):
    try:
        import sympy as sp
    except ImportError as exc:
        raise ImportError(
            "SigRK ansatz enumeration with unknown coefficients requires sympy."
        ) from exc
    return sp.symbols(name)


def enumerate_templates(
    family_templates: Iterable[tuple[StageFamily, ...]],
    edge_rule_sets: Iterable[tuple[EdgeRuleTemplate, ...]],
    update_rule_sets: Iterable[tuple[UpdateRuleTemplate, ...]],
    edge_support_options: SupportOptions = (),
    update_support_options: SupportOptions = (),
    support_pool: tuple[SupportPrototype, ...] = (),
    min_support_terms: int = 1,
    max_support_terms: int = 1,
    max_templates: int | None = None,
    name_prefix: str = "ansatz",
) -> tuple[EnumeratedTemplate, ...]:
    """
    Enumerate symbolic SigRK templates inside a finite search box.

    Exhaustiveness is only with respect to the supplied families, rule sets,
    support options, and support sizes.
    """

    out = []
    template_counter = count()
    family_templates = tuple(family_templates)
    edge_rule_sets = tuple(edge_rule_sets)
    update_rule_sets = tuple(update_rule_sets)

    for families, edge_rules, update_rules in product(
        family_templates,
        edge_rule_sets,
        update_rule_sets,
    ):
        family_map = {family.name: family for family in families}
        if len(family_map) != len(families):
            continue
        if any(
            rule.source not in family_map or rule.target not in family_map
            for rule in edge_rules
        ):
            continue
        if any(rule.stage not in family_map for rule in update_rules):
            continue

        edge_options = []
        for rule in edge_rules:
            options = _support_options(
                edge_support_options,
                rule.name,
                _edge_variables(family_map, rule),
                support_pool,
                min_support_terms,
                max_support_terms,
            )
            if not options:
                break
            edge_options.append(options)
        else:
            update_options = []
            for rule in update_rules:
                options = _support_options(
                    update_support_options,
                    rule.name,
                    _update_variables(family_map, rule),
                    support_pool,
                    min_support_terms,
                    max_support_terms,
                )
                if not options:
                    break
                update_options.append(options)
            else:
                edge_choices = product(*edge_options) if edge_options else ((),)
                update_choices = product(*update_options) if update_options else ((),)
                for edge_choice, update_choice in product(edge_choices, update_choices):
                    idx = next(template_counter)
                    variables = []
                    concrete_edges = []
                    for rule_idx, (rule, support_set) in enumerate(zip(edge_rules, edge_choice)):
                        terms = []
                        for term_idx, prototype in enumerate(support_set):
                            variable = _symbol(f"{name_prefix}_A_{idx}_{rule_idx}_{term_idx}")
                            variables.append(variable)
                            terms.append(prototype.instantiate(variable))
                        concrete_edges.append(
                            EdgeRule(
                                rule.source,
                                rule.target,
                                rule.label_var,
                                tuple(terms),
                                name=rule.name,
                            )
                        )

                    concrete_updates = []
                    for rule_idx, (rule, support_set) in enumerate(
                        zip(update_rules, update_choice)
                    ):
                        terms = []
                        for term_idx, prototype in enumerate(support_set):
                            variable = _symbol(f"{name_prefix}_B_{idx}_{rule_idx}_{term_idx}")
                            variables.append(variable)
                            terms.append(prototype.instantiate(variable))
                        concrete_updates.append(
                            UpdateRule(
                                rule.stage,
                                rule.label_var,
                                tuple(terms),
                                name=rule.name,
                            )
                        )

                    out.append(
                        EnumeratedTemplate(
                            SigRKTemplate(
                                families,
                                tuple(concrete_edges),
                                tuple(concrete_updates),
                                name=f"{name_prefix}_{idx}",
                            ),
                            tuple(variables),
                        )
                    )
                    if max_templates is not None and len(out) >= max_templates:
                        return tuple(out)
    return tuple(out)


def _build_equations(
    template: SigRKTemplate,
    order: int,
    d: int,
    alpha: Fraction,
    include_consistency: bool,
) -> tuple[tuple[OrderCondition, ...], tuple[Any, ...]]:
    conditions = order_conditions(template, order, d, alpha)
    equations = tuple(condition.expression for condition in conditions)
    if include_consistency:
        equations = equations + consistency_conditions(template, d)
    return conditions, equations


def _solve_candidate(equations: tuple[Any, ...], variables: tuple[Any, ...], solve: bool):
    if not equations:
        return False, ({},)
    if variables:
        impossible = groebner_is_impossible(equations, variables)
        if impossible or not solve:
            return impossible, ()
        return False, tuple(solve_equations(equations, variables))
    return any(equation != 0 for equation in equations), ()


def search_fixed_templates(
    templates: Iterable[SigRKTemplate],
    variables: tuple,
    order: int,
    d: int,
    alpha: Fraction | int = Fraction(1, 3),
) -> tuple[SearchResult, ...]:
    """Run equations/solving for already-built templates."""

    results = []
    for template in templates:
        conditions, equations = _build_equations(template, order, d, Fraction(alpha), True)
        impossible, solutions = _solve_candidate(equations, variables, solve=True)
        results.append(
            SearchResult(template, conditions, equations, variables, impossible, solutions)
        )
    return tuple(results)


def search_bounded(
    family_templates: Iterable[tuple[StageFamily, ...]],
    edge_rule_sets: Iterable[tuple[EdgeRuleTemplate, ...]],
    update_rule_sets: Iterable[tuple[UpdateRuleTemplate, ...]],
    order: int,
    d: int,
    alpha: Fraction | int,
    edge_support_options: SupportOptions = (),
    update_support_options: SupportOptions = (),
    support_pool: tuple[SupportPrototype, ...] = (),
    min_support_terms: int = 1,
    max_support_terms: int = 1,
    max_templates: int | None = None,
    include_consistency: bool = True,
    solve: bool = True,
    keep_impossible: bool = False,
) -> tuple[SearchResult, ...]:
    """
    Enumerate and solve the finite-box SigRK search loop.

    This is the spec's enumeration loop restricted to the supplied finite
    family/rule/support boxes.
    """

    candidates = enumerate_templates(
        family_templates,
        edge_rule_sets,
        update_rule_sets,
        edge_support_options=edge_support_options,
        update_support_options=update_support_options,
        support_pool=support_pool,
        min_support_terms=min_support_terms,
        max_support_terms=max_support_terms,
        max_templates=max_templates,
    )

    results = []
    alpha = Fraction(alpha)
    for candidate in candidates:
        conditions, equations = _build_equations(
            candidate.template,
            order,
            d,
            alpha,
            include_consistency,
        )
        impossible, solutions = _solve_candidate(equations, candidate.variables, solve)
        if not impossible or keep_impossible:
            results.append(
                SearchResult(
                    candidate.template,
                    conditions,
                    equations,
                    candidate.variables,
                    impossible,
                    solutions,
                )
            )
    return tuple(results)


def first(iterable: Iterable[Any], n: int) -> tuple[Any, ...]:
    """Return at most the first ``n`` entries of an iterable."""

    return tuple(islice(iterable, n))
