"""
This module provides validation logic for generated answers.
It checks for output quality, citation presence, query coverage, and refusal patterns.
The validation results inform the system whether the answer meets the minimum criteria or requires a fallback.
"""

import re
from opentelemetry import trace

from contracts.generation_contracts import (
    GenerationStatus,
    ValidationSignals,
    GenerationValidationResult
)
from utility_functions.llm_utils import extract_citation_ids

tracer = trace.get_tracer(__name__)

REFUSAL_PATTERNS = {
    "i don't know",
    "i do not know",
    "not enough information",
    "insufficient information",
    "cannot determine",
    "i don't have enough information",
    "i do not have enough information",
    "i couldn't find enough relevant information",
    "i could not find enough relevant information",
    "i don't have sufficient information",
    "i don't have access to",
    "unable to answer",
    "cannot answer this",
    "can't answer this",
}


def validate_answer(answer:str, query:str) -> GenerationValidationResult:
    """
    Evaluates the quality of a generated answer against the original query.
    Calculates signals like citation count and query coverage to return a GenerationValidationResult,
    indicating whether the answer passed or failed specific quality checks.
    """
    
    with tracer.start_as_current_span("answer_validation") as span:
        
        normalized = answer.lower()

        ctx_ids = extract_citation_ids(answer)

        has_citations = len(ctx_ids) > 0
        citation_count = len(ctx_ids)


        query_tokens = set(re.findall(r"\b[a-zA-Z0-9]+\b", query.lower()))
        answer_tokens = set(re.findall(r"\b[a-zA-Z0-9]+\b", normalized))

        coverage_score = (
            len(query_tokens & answer_tokens) / len(query_tokens) if query_tokens else 1.0
        )


        signals = ValidationSignals(
            answer_length=len(answer),
            has_citations=has_citations,
            citation_count=citation_count,
            coverage_score=coverage_score,
            has_refusal_pattern=(
                len(normalized) <= 300 and
                any(p in normalized for p in REFUSAL_PATTERNS)
            )
        )

        if not normalized:
            span.set_attribute("validation.status", "empty")
            return GenerationValidationResult(
                status=GenerationStatus.EMPTY,
                score=0.0,
                signals=signals,
                failure_reason="empty_output"
            )

        if len(normalized) < 25:
            span.set_attribute("validation.status", "too_short")
            return GenerationValidationResult(
                status=GenerationStatus.TOO_SHORT,
                score=0.2,
                signals=signals,
                failure_reason="too_short"
            )

        if signals.has_refusal_pattern:
            span.set_attribute("validation.status", "refusal")
            return GenerationValidationResult(
                status=GenerationStatus.REFUSAL,
                score=0.3,
                signals=signals,
                failure_reason="refusal_detected"
            )

        if not has_citations:
            span.set_attribute("validation.status", "no_citations")
            return GenerationValidationResult(
                status=GenerationStatus.ATTRIBUTION_ERROR,
                score=0.4,
                signals=signals,
                failure_reason="no_citations_present"
            )

        if coverage_score < 0.1:
            span.set_attribute("validation.status", "very_less_coverage")
            return GenerationValidationResult(
                status=GenerationStatus.LOW_COVERAGE,
                score=0.5,
                signals=signals,
                failure_reason="low_query_coverage"
            )

        span.set_attribute("validation.status", "passed")
        
        span.set_attribute("validation.citation_count", citation_count)
        span.set_attribute("validation.coverage_score", coverage_score)
        span.set_attribute("validation.has_refusal", signals.has_refusal_pattern)
        
        return GenerationValidationResult(
            status=GenerationStatus.PASSED,
            score=1.0,
            signals=signals,
            failure_reason=None
        )