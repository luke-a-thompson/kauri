"""Optional objective scoring hooks for generated RK methods."""

from dataclasses import dataclass
from typing import Protocol

from kauri.rk import RK


class RKObjective(Protocol):
    """
    Interface for scoring a generated RK method.
    """

    name: str

    def evaluate(self, method: RK) -> float:
        """
        Return objective value (lower is better by convention).
        """
        ...


@dataclass(frozen=True)
class MethodScore:
    method_name: str
    scores: dict[str, float]


def score_methods(methods: list[RK], objectives: list[RKObjective]) -> list[MethodScore]:
    """
    Evaluate methods against a list of objectives.
    """
    output: list[MethodScore] = []
    for method in methods:
        scores: dict[str, float] = {}
        for objective in objectives:
            scores[objective.name] = float(objective.evaluate(method))
        method_name = method.name if method.name is not None else "unnamed_method"
        output.append(MethodScore(method_name=method_name, scores=scores))
    return output
