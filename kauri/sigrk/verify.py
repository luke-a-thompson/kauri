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
from fractions import Fraction

from .equations import OrderCondition, order_conditions
from .templates import SigRKTemplate


@dataclass(frozen=True)
class DiscoveryVerification:
    conditions: tuple[OrderCondition, ...]

    @property
    def passed(self) -> bool:
        return len(self.conditions) == 0


def verify_discovery(
    template: SigRKTemplate,
    order: int,
    d: int,
    alpha: Fraction | int = Fraction(1, 3),
) -> DiscoveryVerification:
    """Verify a template after expanding it at one concrete dimension."""

    return DiscoveryVerification(order_conditions(template, order, d, Fraction(alpha)))


def verify_all_dimensions(*_args, **_kwargs):
    """
    Placeholder for proof mode.

    The spec's symbolic index-pattern reduction is not implemented in the
    first milestone.  Callers must not treat discovery verification as a proof
    for all dimensions.
    """

    raise NotImplementedError("SigRK proof-mode all-dimension verification is not implemented.")
