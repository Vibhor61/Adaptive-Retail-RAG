import logging

from typing import List
from opentelemetry import trace

from contracts.router_contracts import (
    RouterResult,
    EntityStructure,
    StructuralViolation,
    StructuralGuardrailResult,
    ViolationSeverity,
)

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


def _check_entity_text(output: RouterResult) -> List[StructuralViolation]:
    violations: List[StructuralViolation] = []

    for idx, entity in enumerate(output.entities):
        if not entity.text.strip():
            violations.append(
                StructuralViolation(
                    field=f"entities[{idx}].text",
                    reason="entity text is empty",
                    severity=ViolationSeverity.ERROR,
                )
            )

    return violations


def _check_entity_structure(output: RouterResult) -> List[StructuralViolation]:
    violations: List[StructuralViolation] = []
    entity_count = len(output.entities)

    if output.entity_structure == EntityStructure.NONE and entity_count > 0:
        violations.append(
            StructuralViolation(
                field="entity_structure",
                reason=(f"entity_structure='none' but entity_count={entity_count}"),
                severity=ViolationSeverity.ERROR,
            )
        )
    elif output.entity_structure == EntityStructure.SINGLE and entity_count != 1:
        violations.append(
            StructuralViolation(
                field="entity_structure",
                reason=(f"entity_structure='single' but entity_count={entity_count}"),
                severity=ViolationSeverity.ERROR,
            )
        )
    elif output.entity_structure == EntityStructure.MULTI_EXPLICIT and entity_count < 2:
        violations.append(
            StructuralViolation(
                field="entity_structure",
                reason=(f"entity_structure='multi_explicit' but entity_count={entity_count}"),
                severity=ViolationSeverity.ERROR,
            )
        )
    elif output.entity_structure == EntityStructure.MULTI_IMPLICIT and entity_count > 0:
        violations.append(
            StructuralViolation(
                field="entity_structure",
                reason=f"entity_structure='multi_implicit' but entity_count={entity_count}",
                severity=ViolationSeverity.ERROR,
            )
        )
    return violations


def _check_duplicate_entities(output: RouterResult):
    violations : List[StructuralViolation] = []

    entities_seen = set()

    for entity in output.entities:
        if entity.text in entities_seen:
            violations.append(
                StructuralViolation(
                    field="entities",
                    reason=(f"duplicated entities found {entity.text}"),
                    severity=ViolationSeverity.WARNING
                )
            )
        
        entities_seen.add(entity.text)
    
    return violations


def run_structural_guardrails(output: RouterResult) -> StructuralGuardrailResult:
    with tracer.start_as_current_span("structural_guardrails") as span:
        span.set_attribute("router.guardrail", "structural")

        violations: List[StructuralViolation] = []

        violations.extend(_check_entity_text(output))
        violations.extend(_check_entity_structure(output))
        violations.extend(_check_duplicate_entities(output))
        
        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        passed = len(errors) == 0

        span.set_attribute("router.guardrail.passed", passed)
        span.set_attribute("router.guardrail.violation_count", len(violations))

        if errors:
            span.set_attribute("router.guardrails.errors", errors)

        return StructuralGuardrailResult(passed=passed, violations=violations)