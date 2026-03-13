from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class ButcherTableau:
    a: list[list[object]]
    b: list[object]

    def __post_init__(self) -> None:
        stages = len(self.b)
        if stages == 0:
            raise ValueError("Parameter 'b' must be a non-empty vector")
        if len(self.a) != stages or any(len(row) != stages for row in self.a):
            raise ValueError(
                "Parameter 'a' must be a square s x s matrix and b a vector of length s"
            )

    @cached_property
    def s(self) -> int:
        return len(self.b)

    @cached_property
    def c(self) -> list[object]:
        return [sum(self.a[i][j] for j in range(self.s)) for i in range(self.s)]

    @cached_property
    def explicit(self) -> bool:
        for i in range(self.s):
            for j in range(i, self.s):
                if self.a[i][j]:
                    return False
        return True
