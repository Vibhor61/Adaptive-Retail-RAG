from contracts.orchestration_contracts import (
    StructuralViolation,
    StructuralGuardrailResult,
    ViolationSeverity,
)

from contracts.retrieval_contracts import (
    RetrievalQualityStatus,
    RetrievalEvaluationBundle,
)


def run_retrieval_structural_guardrails(
    bundles: list[RetrievalEvaluationBundle],
) -> StructuralGuardrailResult:

    violations: list[StructuralViolation] = []

    if not bundles:
        violations.append(
            StructuralViolation(
                field="evaluation_bundles",
                reason="no retrieval bundles produced",
                severity=ViolationSeverity.ERROR,
            )
        )

    all_empty = True

    for idx, bundle_eval in enumerate(bundles):

        bundle = bundle_eval.bundle

        if (
            bundle_eval.quality_status
            == RetrievalQualityStatus.EMPTY
            and len(bundle.items) > 0
        ):
            violations.append(
                StructuralViolation(
                    field=f"bundle[{idx}]",
                    reason="EMPTY quality status contains items",
                    severity=ViolationSeverity.ERROR,
                )
            )

        if (
            bundle_eval.quality_status
            in (
                RetrievalQualityStatus.HEALTHY,
                RetrievalQualityStatus.WEAK,
            )
            and len(bundle.items) == 0
        ):
            violations.append(
                StructuralViolation(
                    field=f"bundle[{idx}]",
                    reason="non-empty quality status contains no items",
                    severity=ViolationSeverity.ERROR,
                )
            )

        if bundle_eval.quality_status != RetrievalQualityStatus.EMPTY:
            all_empty = False

    if all_empty:
        violations.append(
            StructuralViolation(
                field="retrieval",
                reason="all retrieval bundles empty",
                severity=ViolationSeverity.ERROR,
            )
        )

    errors = [
        v for v in violations
        if v.severity == ViolationSeverity.ERROR
    ]

    return StructuralGuardrailResult(
        passed=len(errors) == 0,
        violations=violations,
    )