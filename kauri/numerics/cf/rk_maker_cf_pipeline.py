"""
High-level pipeline: existing RK maker + Williamson lift + truncated CF verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy

from kauri.numerics.cf.cf_verify import VerificationResult, verify_cf_ees
from kauri.numerics.cf.cf_williamson import Williamson2NCF, lift_to_cf, rk_to_williamson_2n
from kauri.numerics.rk.rk_ansatz import Ansatz, CompositeAnsatz
from kauri.numerics.rk.rk_constraints import Constraint
from kauri.numerics.rk.rk_maker import RKMakerResult, make_explicit_rk_methods
from kauri.numerics.rk.rk import RK
from kauri.numerics.rk.williamson_ansatz import WilliamsonAnsatz


@dataclass
class CFRKPipelineResult:
    rk_result: RKMakerResult
    cf_methods: list[Williamson2NCF]
    verification_results: list[VerificationResult]
    accepted_indices: list[int]


def _verify_rk_method(
    method: RK,
    verification_order: int,
) -> tuple[Williamson2NCF, VerificationResult]:
    williamson = rk_to_williamson_2n(method)
    cf_method = lift_to_cf(williamson)
    verification = verify_cf_ees(cf_method, order=verification_order)
    return cf_method, verification


def generate_2n_candidate_methods(
    order: int,
    stages: int,
    antisymmetric_order: int | None = None,
    constraints: list[Constraint] | None = None,
    fixed_values: dict[str, float | int | sympy.core.basic.Basic] | None = None,
    zero_symbols: list[str] | None = None,
    max_solutions: int | None = 1,
    solver: str = "grobner",
    additional_ansatz: Ansatz | None = None,
) -> RKMakerResult:
    """
    Reuse existing rk_maker with WilliamsonAnsatz as the construction core.
    """
    if solver != "grobner":
        raise ValueError('Only solver="grobner" is supported for method construction.')
    two_n = WilliamsonAnsatz()
    ansatz: Ansatz = (
        two_n if additional_ansatz is None
        else CompositeAnsatz([two_n, additional_ansatz], name="williamson_plus_extra")
    )
    return make_explicit_rk_methods(
        order=order,
        stages=stages,
        antisymmetric_order=antisymmetric_order,
        ansatz=ansatz,
        constraints=constraints,
        fixed_values=fixed_values,
        zero_symbols=zero_symbols,
        max_solutions=max_solutions,
        solver="grobner",
    )


def build_and_verify_cf_methods(
    order: int,
    stages: int,
    verification_order: int = 4,
    antisymmetric_order: int | None = None,
    constraints: list[Constraint] | None = None,
    fixed_values: dict[str, float | int | sympy.core.basic.Basic] | None = None,
    zero_symbols: list[str] | None = None,
    max_solutions: int | None = 1,
    solver: str = "grobner",
    additional_ansatz: Ansatz | None = None,
) -> CFRKPipelineResult:
    """
    Build 2N candidates with existing RK maker and filter by truncated CF EES verification.
    """
    rk_result = generate_2n_candidate_methods(
        order=order,
        stages=stages,
        antisymmetric_order=antisymmetric_order,
        constraints=constraints,
        fixed_values=fixed_values,
        zero_symbols=zero_symbols,
        max_solutions=max_solutions,
        solver=solver,
        additional_ansatz=additional_ansatz,
    )
    cf_methods: list[Williamson2NCF] = []
    verification_results: list[VerificationResult] = []
    accepted_indices: list[int] = []

    for idx, method in enumerate(rk_result.methods):
        cf_method, verification = _verify_rk_method(
            method=method,
            verification_order=verification_order,
        )
        cf_methods.append(cf_method)
        verification_results.append(verification)
        if verification.passed:
            accepted_indices.append(idx)

    return CFRKPipelineResult(
        rk_result=rk_result,
        cf_methods=cf_methods,
        verification_results=verification_results,
        accepted_indices=accepted_indices,
    )


def main() -> int:
    """
    Demo: generate an EES(2,5)-like 2N-storage scheme, lift to commutator-free, and verify.

    Run with:
        uv run python -m kauri.numerics.cf.rk_maker_cf_pipeline
    """
    try:
        rk_result: RKMakerResult = generate_2n_candidate_methods(
            order=2,
            stages=3,
            antisymmetric_order=5,
            fixed_values={"b0": sympy.Rational(1, 4)},
            solver="grobner",
            max_solutions=1,
        )
    except Exception as exc:
        print(f"RK generation failed: {type(exc).__name__}: {exc}")
        return 1

    if len(rk_result.methods) == 0:
        print("No RK methods were generated.")
        return 2

    cf_method, verification = _verify_rk_method(
        method=rk_result.methods[0],
        verification_order=4,
    )
    print(rk_result)
    print(cf_method.to_text())
    print(
        "verification:",
        f"passed={verification.passed},",
        f"checked_elements={verification.checked_elements},",
        f"first_failure={verification.first_failure},",
        f"residual={verification.residual}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
