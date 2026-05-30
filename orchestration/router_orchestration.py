from contracts.orchestration_contracts import RoutingResult

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

def run_router_pipeline(query: str) -> RoutingResult:

    validity_result = validate_query_structure(query)

    router_output = analyze_intent(query)

    structural_result = run_structural_guardrails(router_output)

    semantic_result = run_semantic_validation(router_output)
    
    # resolve extracted entities to canonical product identifiers/names
    entities_texts = [e.text for e in router_output.entities]
    resolver = EntityResolver(DBEntityLoader)
    grounded_entities = resolver.resolve(entities_texts)
    
    return RoutingResult(
        validity_result=validity_result,
        router_output=router_output,
        structural_result=structural_result,
        semantic_result=semantic_result,
        grounded_entities=grounded_entities
    )  