from opentelemetry import trace

from contracts.orchestration_contracts import RouterLayerOutput

from routing_layer.validity import (
    validate_query_structure
)

from routing_layer.router import (
    analyze_intent
)

from routing_layer.structural_guardrails import (
    run_structural_guardrails
)

from routing_layer.semantic_guardrails import (
    run_semantic_validation
)

from routing_layer.entity_resolver import (
    EntityResolver,
    DBEntityLoader
)

tracer = trace.get_tracer(__name__)
resolver = EntityResolver(DBEntityLoader())

def run_router_pipeline(query: str) -> RouterLayerOutput:

    with tracer.start_as_current_span("router_pipeline") as span:
        
        span.set_attribute("router.query", query)

        validity_result = validate_query_structure(query)
        normalized_query = validity_result.normalized_query

        router_output = analyze_intent(normalized_query)

        structural_result = run_structural_guardrails(router_output)

        semantic_result = run_semantic_validation(normalized_query, router_output)
        
        # resolve extracted entities to canonical product identifiers/names
        entities_texts = [e.text for e in router_output.entities]
        grounded_entities = resolver.resolve(entities_texts)
        
        return RouterLayerOutput(
            normalized_query=normalized_query,
            validity_result=validity_result,
            router_output=router_output,
            structural_result=structural_result,
            semantic_result=semantic_result,
            grounded_entities=grounded_entities
        )  