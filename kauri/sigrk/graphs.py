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

from collections import defaultdict
from dataclasses import dataclass

from .templates import ConcreteStage, SigRKTemplate, bind_label, merge_bindings


@dataclass(frozen=True)
class ExpandedGraph:
    stages: tuple[ConcreteStage, ...]
    edges: tuple[tuple[ConcreteStage, ConcreteStage, tuple[str, ...]], ...]

    def topological_generations(self) -> tuple[tuple[ConcreteStage, ...], ...]:
        outgoing = defaultdict(set)
        indegree = {stage: 0 for stage in self.stages}
        for source, target, _labels in self.edges:
            if target not in outgoing[source]:
                outgoing[source].add(target)
                indegree[target] += 1

        generations = []
        ready = tuple(stage for stage in self.stages if indegree[stage] == 0)
        seen = set()
        while ready:
            generations.append(tuple(sorted(ready, key=str)))
            next_ready = []
            for stage in ready:
                seen.add(stage)
                for target in sorted(outgoing[stage], key=str):
                    indegree[target] -= 1
                    if indegree[target] == 0:
                        next_ready.append(target)
            ready = tuple(next_ready)

        if len(seen) != len(self.stages):
            raise ValueError("Expanded stage graph contains a cycle.")
        return tuple(generations)

    def layer_widths(self) -> tuple[int, ...]:
        return tuple(len(generation) for generation in self.topological_generations())

    def to_networkx(self):
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError("Install networkx to materialise SigRK graphs as networkx.") from exc

        graph = nx.DiGraph()
        for stage in self.stages:
            graph.add_node(stage.name, family=stage.family.name, layer=stage.family.layer)
        for source, target, labels in self.edges:
            graph.add_edge(source.name, target.name, labels=labels)
        return graph


def expand_template(template: SigRKTemplate, d: int) -> ExpandedGraph:
    template.validate_explicit_layers()
    labels = tuple(str(i) for i in range(1, d + 1))
    stages = template.stages(d)
    by_family = defaultdict(list)
    for stage in stages:
        by_family[stage.family.name].append(stage)

    edges = []
    for rule in template.edge_rules:
        for source in by_family[rule.source]:
            for target in by_family[rule.target]:
                env = merge_bindings(source.bindings(), target.bindings())
                if env is None:
                    continue
                allowed_labels = []
                for label in labels:
                    label_env = bind_label(env, rule.label_var, label)
                    if label_env is not None:
                        allowed_labels.append(label)
                if allowed_labels:
                    edges.append((source, target, tuple(allowed_labels)))
    return ExpandedGraph(stages=stages, edges=tuple(edges))


def stage_count_polynomial(template: SigRKTemplate) -> dict[int, int]:
    counts: dict[int, int] = {}
    for family in template.families:
        counts[family.arity] = counts.get(family.arity, 0) + family.multiplicity
    return counts


def layer_count_polynomial(template: SigRKTemplate) -> tuple[dict[int, int], ...]:
    max_layer = max((family.layer for family in template.families), default=-1)
    layers = []
    for layer in range(max_layer + 1):
        counts: dict[int, int] = {}
        for family in template.families:
            if family.layer != layer:
                continue
            counts[family.arity] = counts.get(family.arity, 0) + family.multiplicity
        layers.append(counts)
    return tuple(layers)
