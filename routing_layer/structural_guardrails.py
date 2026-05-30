import logging
from typing import List
from opentelemetry import trace

from contracts.router_contracts import (
    RouterOutput,
    Intent,
    EntityStructure,
    EvidenceType,
    StructuralViolation,
    StructuralGuardrailResult,
    ViolationSeverity,
)

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


def _check_enum_fields(output: RouterOutput) -> List[StructuralViolation]:
    violations: List[StructuralViolation] = []

    if not isinstance(output.intent_type, Intent):
        violations.append(
            StructuralViolation(
                field="intent_type",
                reason="invalid intent_type enum",
                severity=ViolationSeverity.ERROR,
            )
        )

    if not isinstance(output.entity_structure, EntityStructure):
        violations.append(
            StructuralViolation(
                field="entity_structure",
                reason="invalid entity_structure enum",
                severity=ViolationSeverity.ERROR,
            )
        )

    if not isinstance(output.evidence_type, EvidenceType):
        violations.append(
            StructuralViolation(
                field="evidence_type",
                reason="invalid evidence_type enum",
                severity=ViolationSeverity.ERROR,
            )
        )

    return violations


def _check_confidence(output: RouterOutput) -> List[StructuralViolation]:
    violations: List[StructuralViolation] = []

    if not (0.0 <= output.confidence <= 1.0):
        violations.append(
            StructuralViolation(
                field="confidence",
                reason=(f"confidence {output.confidence} outside [0.0, 1.0]"),
                severity=ViolationSeverity.ERROR,
            )
        )

    for idx, entity in enumerate(output.entities):
        if not (0.0 <= entity.confidence <= 1.0):
            violations.append(
                StructuralViolation(
                    field=f"entities[{idx}].confidence",
                    reason=(f"entity confidence {entity.confidence} outside [0.0, 1.0]"),
                    severity=ViolationSeverity.ERROR,
                )
            )

    return violations


def _check_entity_text(output: RouterOutput) -> List[StructuralViolation]:
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


def _check_entity_structure(output: RouterOutput) -> List[StructuralViolation]:
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

    return violations


def _check_duplicate_entities(output: RouterOutput):
    violations : List[StructuralViolation]

    entities_seen = set()

    for _, entity in enumerate(output.entities):
        if entity.text in entities_seen:
            violations.append(
                StructuralViolation(
                    field="entities",
                    reaason=(f"duplicated entities found {entity.text}"),
                    severity=ViolationSeverity.WARNING
                )
            )

def run_structural_guardrails(output: RouterOutput) -> StructuralGuardrailResult:
    with tracer.start_as_current_span("structural_guardrails") as span:
        span.set_attribute("router.guardrail", "structural")

        violations: List[StructuralViolation] = []
        violations.extend(_check_enum_fields(output))
        violations.extend(_check_confidence(output))
        violations.extend(_check_entity_text(output))
        violations.extend(_check_entity_structure(output))
        violations.extend(_check_duplicate_entities(output))
        
        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        passed = len(errors) == 0

        span.set_attribute("router.guardrail.passed", passed)
        span.set_attribute("router.guardrail.violation_count", len(violations))

        if errors:
            logger.warning(
                "Structural guardrail violations: %s",
                [f"{v.field}: {v.reason}" for v in errors],
            )

        return StructuralGuardrailResult(passed=passed, violations=violations)
