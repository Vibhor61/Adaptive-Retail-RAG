from opentelemetry import trace

from contracts.orchestration_contracts import (
    RouterLayerOutput,
    RetrievalLayerOutput,
)

from contracts.router_contracts import (
    Intent
)

from contracts.retrieval_contracts import (
    RetrievalPlan,
)

from retrieval_layer.retrieval_strategies import (
    lookup_workflow,
    comparison_workflow,
    recommendation_workflow
)


tracer = trace.get_tracer(__name__)

def make_retrieval_plan(input: RouterLayerOutput)->RetrievalPlan:

    return RetrievalPlan(
        original_query =input.normalized_query,
        intent_type=input.router_output.intent_type,
        evidence_type=input.router_output.evidence_type,
        grounded_entities=input.grounded_entities,
        entity_structure=input.router_output.entity_structure
    )

def retrieve(input: RouterLayerOutput) -> RetrievalLayerOutput:
    
    with tracer.start_as_current_span("retrieval_pipeline") as span:

        plan = make_retrieval_plan(input)
        
        span.set_attribute("retrieval.query", plan.original_query)
        
        if plan.intent_type == Intent.LOOKUP:
            return lookup_workflow(plan)
    
        elif plan.intent_type == Intent.COMPARISON:
            return comparison_workflow(plan)
    
        elif plan.intent_type == Intent.RECOMMENDATION:
            return recommendation_workflow(plan)
    
        elif plan.intent_type == Intent.UNKNOWN:
            raise NotImplementedError(
                "intent 'unknown' cannot be routed — adaptive routing required."
            )
    
        raise ValueError(
            f"Unrecognised intent_type: '{plan.intent_type}' — this should never happen."
        )

