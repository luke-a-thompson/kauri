"""
High-level pipeline: existing RK maker + Williamson lift + truncated CF verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy

from kauri.ansatz_2n import TwoNStorageAnsatz
from kauri.cf_verify import VerificationResult, verify_cf_ees
from kauri.cf_williamson import Williamson2NCF, lift_to_cf, rk_to_williamson_2n
from kauri.rk_ansatz import Ansatz, CompositeAnsatz
from kauri.rk_constraints import Constraint
from kauri.rk_maker import RKMakerResult, make_explicit_rk_methods


@dataclass
class CFRKPipelineResult:
    rk_result: RKMakerResult
    cf_methods: list[Williamson2NCF]
    verification_results: list[VerificationResult]
    accepted_indices: list[int]


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
    Reuse existing rk_maker with TwoNStorageAnsatz as the construction core.
    """
    two_n = TwoNStorageAnsatz()
    ansatz_used: Ansatz = two_n
    if additional_ansatz is not None:
        ansatz_used = CompositeAnsatz([two_n, additional_ansatz], name="2n_plus_extra")
    return make_explicit_rk_methods(
        order=order,
        stages=stages,
        antisymmetric_order=antisymmetric_order,
        ansatz=ansatz_used,
        constraints=constraints,
        fixed_values=fixed_values,
        zero_symbols=zero_symbols,
        max_solutions=max_solutions,
        solver="grobner" if solver == "grobner" else "scipy",
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
        try:
            williamson = rk_to_williamson_2n(method)
            cf_method = lift_to_cf(williamson)
            verification = verify_cf_ees(cf_method, order=verification_order)
        except Exception as exc:  # pragma: no cover - defensive path
            verification = VerificationResult(
                passed=False,
                order=verification_order,
                checked_elements=0,
                first_failure=f"exception:{type(exc).__name__}",
                residual=sympy.sympify(str(exc)),
            )
            cf_method = None

        if cf_method is not None:
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
        uv run python -m kauri.rk_maker_cf_pipeline
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

    rk_method = rk_result.methods[0]
    try:
        cf_method = lift_to_cf(rk_to_williamson_2n(rk_method))
    except Exception as exc:
        print(f"Williamson/CF lift failed: {type(exc).__name__}: {exc}")
        return 3

    verification = verify_cf_ees(cf_method, order=4)
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
