"""
Router orchestrator module.
Analyzes the incoming query to determine its structure, intent, and evidence type.
Resolves grounded entities and structures the routing output for the retrieval layer.
"""

import logging

from config.settings import settings

from opentelemetry import trace
from sentence_transformers import CrossEncoder

from routing_layer.validity import (
    validate_query_structure
)
from routing_layer.intent_classification import (
    IntentClassifier
)

from routing_layer.evidence_type_classification import(
    EvidenceClassifier
)

from routing_layer.entity_resolver import (
    DBEntityLoader
)

from routing_layer.reranker import (
    EntityReranker, 
    EntityResolver
)

from contracts.router_contracts import (
    RouterResult,
    EntityStructure,
)

from contracts.orchestration_contracts import RouterLayerOutput, ExceptionInfo


tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


class RouterOrchestrator:

    def __init__(self, llm):
        self.intent_classifier = IntentClassifier(llm=llm)

        self.evidence_classifier = EvidenceClassifier(llm=llm)

        self.resolver = EntityResolver(
            loader=DBEntityLoader(),
            reranker=EntityReranker(
                model=CrossEncoder(settings.reranker_model,max_length=512),
            ),
        )


    def run(self, query: str) -> RouterLayerOutput:
        """
        Executes the routing pipeline for an input query.
        Validates the query, classifies intent and evidence type, resolves entities,
        and returns the consolidated routing output.
        """

        validity_result = None 
        candidate_entities = []
        with tracer.start_as_current_span("router_pipeline") as span:
            span.set_attribute("router.query", query)

            try:
                validity_result = validate_query_structure(query)
                normalized_query = validity_result.normalized_query

                span.set_attribute("router.validity_status", validity_result.status.value)
                span.set_attribute("router.normalized_query", normalized_query)

                intent, entities, entity_struct = self.intent_classifier.classify_intent(normalized_query)

                span.set_attribute("router.intent", intent.value)
                span.set_attribute("router.entities", entities)
                span.set_attribute("router.entity_structure", entity_struct.value)

                evidence_type = self.evidence_classifier.derive_evidence_type(
                    intent=intent,
                    query=normalized_query,
                )

                span.set_attribute("router.evidence_type", evidence_type.value)

                resolved, candidate_entities = self.resolver.resolve(
                    query=normalized_query,
                    intent=intent,
                    entities=entities,
                    entity_structure=entity_struct,
                )

                # intent classifier found no entities, but resolver recovered something via full query upgrade the structure.
                if not entities and resolved:
                    entity_struct = EntityStructure.MULTI_IMPLICIT
                
                span.set_attribute("router.num_resolved", len(resolved))
                
                router_result = RouterResult(
                    intent_type=intent,
                    entities=resolved,
                    entity_structure=entity_struct,
                    evidence_type=evidence_type,
                    confidence=1.0,
                )
                span.set_attribute("router.status", "ok")

                return RouterLayerOutput(
                    normalized_query=normalized_query,
                    validity_result=validity_result,
                    router_output=router_result,
                    grounded_entities=resolved,
                    candidate_entities=candidate_entities,
                    system_failure=None,
                )

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("router.status", "error")
                span.set_attribute("router.error_type", type(e).__name__)

                logger.error(
                    "RouterOrchestrator.run failed for query=%r: %s", query, e
                )

                return RouterLayerOutput(
                    normalized_query=query,
                    validity_result=validity_result,
                    router_output=None,
                    grounded_entities=[],
                    candidate_entities=candidate_entities,
                    system_failure=ExceptionInfo(
                        exception_type=type(e).__name__,
                        message=str(e),
                    ),
                )