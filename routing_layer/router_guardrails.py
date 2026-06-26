"""
Implements guardrail checks to validate the structural integrity of the router's output.
Ensures that generated results align with intents, entities are properly grounded,
and scoring criteria are within expected bounds before downstream processing.
"""
import logging

from typing import List
from opentelemetry import trace

from contracts.router_contracts import (
    RouterResult,
    EntityStructure,
    Intent,
)

from contracts.orchestration_contracts import (
    StructuralViolation,
    StructuralGuardrailResult,
    ViolationSeverity,
)

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


def _check_entity_presence(output: RouterResult) -> List[StructuralViolation]:
    """
    Validates that entities are present when the intent requires them (LOOKUP or COMPARISON).
    Returns a list of structural violations if required entities are missing.
    """
    violations: List[StructuralViolation] = []

    if output.intent_type in (Intent.LOOKUP, Intent.COMPARISON):
        if not output.entities or len(output.entities) == 0:
            violations.append(
                StructuralViolation(
                    field="entities",
                    reason=f"{output.intent_type.value} requires at least 1 grounded entity",
                    severity=ViolationSeverity.ERROR,
                )
            )

    return violations


def _check_entity_structure(output: RouterResult) -> List[StructuralViolation]:
    """
    Checks that the number of extracted entities matches the expected entity structure count.
    Generates an error violation if the structure definition and entity count mismatch.
    """
    violations: List[StructuralViolation] = []

    count = len(output.entities)

    if output.entity_structure == EntityStructure.NONE:
        if count > 0:
            violations.append(
                StructuralViolation(
                    field="entity_structure",
                    reason=f"NONE but entity_count={count}",
                    severity=ViolationSeverity.ERROR,
                )
            )

    elif output.entity_structure == EntityStructure.SINGLE:
        if count != 1:
            violations.append(
                StructuralViolation(
                    field="entity_structure",
                    reason=f"SINGLE but entity_count={count}",
                    severity=ViolationSeverity.ERROR,
                )
            )

    elif output.entity_structure == EntityStructure.MULTI_EXPLICIT:
        if count < 2:
            violations.append(
                StructuralViolation(
                    field="entity_structure",
                    reason=f"MULTI_EXPLICIT but entity_count={count}",
                    severity=ViolationSeverity.ERROR,
                )
            )


    elif output.entity_structure == EntityStructure.MULTI_IMPLICIT:
        pass

    return violations


def _check_entity_integrity(output: RouterResult) -> List[StructuralViolation]:
    """
    Validates that each entity has a non-empty title and checks for duplicate entities.
    Emits errors for missing titles and warnings for case-insensitive duplicates.
    """
    violations: List[StructuralViolation] = []

    seen = set()

    for idx, entity in enumerate(output.entities):
        # empty title check (your RankedCandidate uses title)
        if not entity.title or not entity.title.strip():
            violations.append(
                StructuralViolation(
                    field=f"entities[{idx}].title",
                    reason="empty entity title",
                    severity=ViolationSeverity.ERROR,
                )
            )

        # duplicate check (case-insensitive)
        norm = (entity.title or "").strip().lower()
        if norm in seen:
            violations.append(
                StructuralViolation(
                    field="entities",
                    reason=f"duplicate entity detected: {entity.title}",
                    severity=ViolationSeverity.WARNING,
                )
            )
        seen.add(norm)

    return violations


def _check_resolver_output(output: RouterResult) -> List[StructuralViolation]:
    """
    Ensures that resolved entities have valid ASINs and their scores are within acceptable bounds.
    Verifies that retrieval scores are non-negative and reranker scores are valid probabilities.
    """
    violations: List[StructuralViolation] = []

    for entity in output.entities:
        # ASIN must exist for grounded candidates
        if entity.asin is None:
            violations.append(
                StructuralViolation(
                    field="entities.asin",
                    reason="resolver returned ungrounded candidate (asin=None)",
                    severity=ViolationSeverity.ERROR,
                )
            )

        # retrieval score must be bounded
        if entity.retrieval_score < 0.0:
            violations.append(
                StructuralViolation(
                    field="entities.retrieval_score",
                    reason="invalid retrieval_score < 0",
                    severity=ViolationSeverity.ERROR,
                )
            )

        # reranker score must be valid probability
        if not (0.0 <= entity.reranker_score <= 1.0):
            violations.append(
                StructuralViolation(
                    field="entities.reranker_score",
                    reason="reranker_score out of [0,1] range",
                    severity=ViolationSeverity.ERROR,
                )
            )

    return violations


def run_structural_guardrails(output: RouterResult) -> StructuralGuardrailResult:
    """
    Executes a comprehensive suite of structural guardrail checks against the router output.
    Returns a summarized StructuralGuardrailResult indicating pass/fail status and all violations.
    """
    with tracer.start_as_current_span("structural_guardrails") as span:
        span.set_attribute("guardrail.type", "structural_v1")

        violations: List[StructuralViolation] = []

        violations.extend(_check_entity_presence(output))
        violations.extend(_check_entity_structure(output))
        violations.extend(_check_entity_integrity(output))
        violations.extend(_check_resolver_output(output))

        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        passed = len(errors) == 0

        span.set_attribute("guardrail.passed", passed)
        span.set_attribute("guardrail.total_violations", len(violations))
        span.set_attribute("guardrail.error_count", len(errors))

        return StructuralGuardrailResult(
            passed=passed,
            violations=violations,
        )