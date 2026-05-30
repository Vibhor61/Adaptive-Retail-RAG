from contracts.orchestration_contracts import (
    RoutingResult
)

from contracts.router_contracts import (
    Intent
)

from contracts.retrieval_contracts import (
    RetrievalPlan,
    RetrievalEvaluationBundle
)

from retrieval_layer.retrieval_strategies import (
    lookup_workflow,
    comparison_workflow,
    recommendation_workflow
)

def make_retrieval_plan(input: RoutingResult)->RetrievalPlan:

    return RetrievalPlan(
        original_query=RoutingResult.original_query,
        intent_type=RoutingResult.router_output.intent_type,
        evidence_type=RoutingResult.router_output.evidence_type,
        grounded_entities=RoutingResult.grounded_entities,
        entity_structure=RoutingResult.router_output.entity_structure
    )

def retrieve(input: RoutingResult)->RetrievalEvaluationBundle:
    
    plan = make_retrieval_plan(input)

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

