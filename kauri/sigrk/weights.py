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

from dataclasses import dataclass, field

from .supports import instantiate_support
from .templates import ConcreteStage, SigRKTemplate, bind_label, merge_bindings
from .trees import LabeledTree
from .words import LinearFunctional, shuffle_functionals


@dataclass
class SchemeWeights:
    """Recursive ``Psi`` and ``Phi`` weights for one fixed dimension."""

    template: SigRKTemplate
    d: int
    stages: tuple[ConcreteStage, ...] = field(init=False)
    labels: tuple[str, ...] = field(init=False)
    _stage_cache: dict[tuple[ConcreteStage, LabeledTree], LinearFunctional] = field(
        default_factory=dict,
        init=False,
    )
    _update_cache: dict[LabeledTree, LinearFunctional] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.template.validate_explicit_layers()
        self.stages = self.template.stages(self.d)
        self.labels = tuple(str(i) for i in range(1, self.d + 1))

    def a_coeff(
        self,
        target: ConcreteStage,
        source: ConcreteStage,
        label: str,
    ) -> LinearFunctional:
        out = LinearFunctional.zero()
        for rule in self.template.edge_rules:
            if rule.source != source.family.name or rule.target != target.family.name:
                continue
            env = merge_bindings(source.bindings(), target.bindings())
            if env is None:
                continue
            env = bind_label(env, rule.label_var, label)
            if env is None:
                continue
            out = out + instantiate_support(rule.support, env)
        return out.simplify()

    def b_coeff(self, stage: ConcreteStage, label: str) -> LinearFunctional:
        out = LinearFunctional.zero()
        for rule in self.template.update_rules:
            if rule.stage != stage.family.name:
                continue
            env = bind_label(stage.bindings(), rule.label_var, label)
            if env is None:
                continue
            out = out + instantiate_support(rule.support, env)
        return out.simplify()

    def stage_weight(self, stage: ConcreteStage, tree: LabeledTree) -> LinearFunctional:
        key = (stage, tree)
        if key in self._stage_cache:
            return self._stage_cache[key]

        child_count = len(tree.children)
        out = LinearFunctional.zero()
        for source in self.stages:
            a = self.a_coeff(stage, source, tree.root)
            if a.is_zero:
                continue
            factors = (a,) + tuple(self.stage_weight(source, child) for child in tree.children)
            if child_count and any(factor.is_zero for factor in factors):
                continue
            out = out + shuffle_functionals(factors)

        out = out.simplify()
        self._stage_cache[key] = out
        return out

    def update_weight(self, tree: LabeledTree) -> LinearFunctional:
        if tree in self._update_cache:
            return self._update_cache[tree]

        out = LinearFunctional.zero()
        for stage in self.stages:
            b = self.b_coeff(stage, tree.root)
            if b.is_zero:
                continue
            factors = (b,) + tuple(self.stage_weight(stage, child) for child in tree.children)
            if tree.children and any(factor.is_zero for factor in factors):
                continue
            out = out + shuffle_functionals(factors)

        out = out.simplify()
        self._update_cache[tree] = out
        return out
