from opentelemetry import trace
from langchain_ollama import OllamaLLM

from contracts.router_contracts import (
    RouterOutput, SemanticValidationResult, SemanticAnomalyType
)

from utils import safe_llm_call


tracer = trace.get_tracer(__name__)

semantic_validator_llm = OllamaLLM(
    model="qwen2.5:7b",
    temperature=0
)


def run_semantic_validation(query: str, router_output: RouterOutput) -> SemanticValidationResult:
        
    with tracer.start_as_current_span("semantic_validator") as span:

        span.set_attribute("semantic.query", query)
        span.set_attribute("semantic.entities", [e.text for e in router_output.entities])
        span.set_attribute("semantic.entity_structure", router_output.entity_structure.value)
        span.set_attribute("semantic.evidence_type", router_output.evidence_type.value)
        span.set_attribute("semantic.intent_type", router_output.intent_type.value)

        prompt = f"""
            You are a semantic validation system for a retail RAG router.

            Your task:
            Evaluate whether the EXISTING router interpretation is semantically plausible.

            --------------------------------------------------
            IMPORTANT CONSTRAINTS
            --------------------------------------------------

            You MUST NOT:
            - generate a new interpretation
            - rewrite router fields
            - correct router outputs
            - extract new entities
            - suggest retrieval strategies
            - act as a router

            You ONLY evaluate consistency of the CURRENT interpretation.

            --------------------------------------------------
            User Query
            --------------------------------------------------

            {query}

            --------------------------------------------------
            Router Interpretation
            --------------------------------------------------

            intent_type:
            {router_output.intent_type.value}

            entities:
            {[e.text for e in router_output.entities]}

            entity_structure:
            {router_output.entity_structure.value}

            evidence_type:
            {router_output.evidence_type.value}

            --------------------------------------------------
            Validation Objective
            --------------------------------------------------

            Evaluate:

            1. Is intent_type semantically plausible?
            2. Is evidence_type semantically plausible?
            3. Does the entity_structure fit the query?
            4. Are extracted entities supported by the query?
            5. Is the overall interpretation coherent?

            --------------------------------------------------
            Allowed anomaly_signals
            --------------------------------------------------

            intent_mismatch
            evidence_type_mismatch
            entity_structure_mismatch
            insufficient_entity_support
            low_semantic_alignment
            ambiguous_routing

            --------------------------------------------------
            Scoring Rules
            --------------------------------------------------

            semantic_score:
            - 1.0 = extremely strong semantic alignment
            - 0.7+ = reasonable interpretation
            - 0.4-0.7 = uncertain/partially inconsistent
            - below 0.4 = weak semantic plausibility

            semantic_valid:
            - true if interpretation is semantically usable
            - false if interpretation appears unreliable

            --------------------------------------------------
            Output Rules
            --------------------------------------------------

            Return ONLY valid JSON.

            --------------------------------------------------
            Output Schema
            --------------------------------------------------

            {{
                "semantic_valid": true,
                "semantic_score": 0.0,
                "anomaly_signals": [],
                "reasoning_confidence": 0.0
            }}
        """

        try:

            parsed = safe_llm_call(
                semantic_validator_llm,
                prompt,
                "json"
            )

            validated = SemanticValidationResult(
                **parsed
            )

            span.set_attribute(
                "semantic.semantic_valid",
                validated.semantic_valid
            )

            span.set_attribute(
                "semantic.semantic_score",
                validated.semantic_score
            )

            span.set_attribute(
                "semantic.anomalies",
                str([
                    a.value
                    for a in validated.anomaly_signals
                ])
            )

            return validated

        except Exception as e:

            span.record_exception(e)

            span.set_attribute(
                "semantic.status",
                "error"
            )

            return SemanticValidationResult(
                semantic_valid=False,
                semantic_score=0.0,
                anomaly_signals=[
                    SemanticAnomalyType.LOW_SEMANTIC_ALIGNMENT
                ],
                reasoning_confidence=0.0
            )