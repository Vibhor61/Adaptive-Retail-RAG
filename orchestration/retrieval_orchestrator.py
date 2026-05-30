from contracts.orchestration_contracts import (
    RoutingResult
)

from contracts.router_contracts import (
    Intent
)
from contracts.retrieval_contracts import (
    RetrievalRequest,
    RetrievalEvaluationBundle
)


def retrieve(input: RoutingResult)->RetrievalEvaluationBundle:
    
    request = RetrievalRequest(
        original_query=RoutingResult.original_query,
        intent_type=RoutingResult.router_output.intent_type,
        evidence_type=RoutingResult.router_output.evidence_type,
        grounded_entities=RoutingResult.grounded_entities
    )

    if request.intent_type == I
    pass

