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

from dataclasses import dataclass
from itertools import product

from .supports import SupportTerm


@dataclass(frozen=True)
class StageFamily:
    """An indexed stage family ``name(indices)``."""

    name: str
    indices: tuple[str, ...] = ()
    multiplicity: int = 1
    layer: int = 0

    @property
    def arity(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class ConcreteStage:
    """A concrete stage obtained by choosing dimension ``d``."""

    family: StageFamily
    values: tuple[str, ...] = ()
    copy: int = 0

    @property
    def name(self) -> str:
        if self.values:
            index_text = ",".join(self.values)
            base = f"{self.family.name}({index_text})"
        else:
            base = self.family.name
        if self.family.multiplicity == 1:
            return base
        return f"{base}#{self.copy}"

    def bindings(self) -> dict[str, str]:
        return dict(zip(self.family.indices, self.values))

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class EdgeRule:
    """
    Indexed rule for a coefficient ``a_{target,source}^{label}``.

    Equal index names shared by source and target are automatically tied.
    ``label_var`` is tied if it is already present in those bindings, otherwise
    it becomes a free summation label.
    """

    source: str
    target: str
    label_var: str
    support: tuple[SupportTerm, ...]
    name: str = ""


@dataclass(frozen=True)
class UpdateRule:
    """Indexed rule for an update coefficient ``b_stage^{label}``."""

    stage: str
    label_var: str
    support: tuple[SupportTerm, ...]
    name: str = ""


@dataclass(frozen=True)
class SigRKTemplate:
    """An indexed SigRK tableau template."""

    families: tuple[StageFamily, ...]
    edge_rules: tuple[EdgeRule, ...]
    update_rules: tuple[UpdateRule, ...]
    name: str = ""

    def family(self, name: str) -> StageFamily:
        for family in self.families:
            if family.name == name:
                return family
        raise KeyError(f"Unknown stage family: {name}")

    def stages(self, d: int) -> tuple[ConcreteStage, ...]:
        labels = tuple(str(i) for i in range(1, d + 1))
        stages = []
        for family in self.families:
            for copy in range(family.multiplicity):
                for values in product(labels, repeat=family.arity):
                    stages.append(ConcreteStage(family, values, copy))
        return tuple(stages)

    def validate_explicit_layers(self) -> None:
        for rule in self.edge_rules:
            source = self.family(rule.source)
            target = self.family(rule.target)
            if source.layer >= target.layer:
                raise ValueError(
                    f"Edge rule {rule.name or rule.source + '->' + rule.target} "
                    "does not respect explicit stage layers."
                )


def merge_bindings(*bindings: dict[str, str]) -> dict[str, str] | None:
    env: dict[str, str] = {}
    for binding in bindings:
        for key, value in binding.items():
            value = str(value)
            if key in env and env[key] != value:
                return None
            env[key] = value
    return env


def bind_label(env: dict[str, str], label_var: str, label: str) -> dict[str, str] | None:
    label = str(label)
    out = dict(env)
    if label_var in out and out[label_var] != label:
        return None
    out[label_var] = label
    return out
