import logging
from opentelemetry import trace

from contracts.router_contracts import (
    EvidenceType,
    EntityStructure,
    RankedCandidate
)

from contracts.retrieval_contracts import (
    RetrievalPlan,
    RetrievalEvaluationBundle,
)

from contracts.orchestration_contracts import (
    RetrievalLayerOutput
)

from retrieval_layer.retrievers import (
    sparse_fact_retrieval,
    fusion_retrieval,
    candidate_gen_retrieval,
)

from retrieval_layer.retrieval_validation import evaluate_retrieval

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def sparse(entity: RankedCandidate|None, query: str, top_k: int) -> RetrievalEvaluationBundle:
    bundle = sparse_fact_retrieval(
        entity=entity.title if entity else None,
        asin=entity.asin if entity else None,
        query=query,
        top_k=top_k,
    )
    return evaluate_retrieval(bundle)


def fusion(query: str, top_k: int) -> RetrievalEvaluationBundle:
    bundle = fusion_retrieval(
        query=query,
        top_k=top_k,
    )
    return evaluate_retrieval(bundle)


def recommendation_candidate_gen(query: str, top_k: int) -> RetrievalEvaluationBundle:
    bundle = candidate_gen_retrieval(
        query=query, 
        top_k=top_k
    )
    return evaluate_retrieval(bundle)


def by_evidence_single(entity: RankedCandidate|None, query: str, evidence_type: EvidenceType, top_k: int) -> list[RetrievalEvaluationBundle]:

    if evidence_type == EvidenceType.FACTUAL:
        return[sparse(entity, query, top_k)]
    
    elif evidence_type == EvidenceType.EXPERIENTIAL:
        return [fusion(query, top_k)]
 
    elif evidence_type == EvidenceType.MIXED:
        return [sparse(entity, query, top_k), fusion(query, top_k)]
 
    raise NotImplementedError(
        f"evidence_type '{evidence_type}' not handled — adaptive routing required."
    )


def by_evidence_multi(entities: list[RankedCandidate], query: str, evidence_type: EvidenceType, top_k: int) -> list[RetrievalEvaluationBundle]:
 
    bundles: list[RetrievalEvaluationBundle] = []
 
    if evidence_type == EvidenceType.FACTUAL:
        for entity in entities:
            bundles.append(sparse(entity, query, top_k))
 
    elif evidence_type == EvidenceType.EXPERIENTIAL:
        bundles.append(fusion(query, top_k))
 
    elif evidence_type == EvidenceType.MIXED:
        for entity in entities:
            bundles.append(sparse(entity, query, top_k))
        bundles.append(fusion(query, top_k))
 
    else:
        raise NotImplementedError(
            f"evidence_type '{evidence_type}' not handled — adaptive routing required."
        )
 
    return bundles


def lookup_workflow(plan: RetrievalPlan) -> RetrievalLayerOutput:
    
    with tracer.start_as_current_span("lookup_workflow") as span:
        span.set_attribute("retrieval.entity_structure", plan.entity_structure.value)
        span.set_attribute("retrieval.evidence_type", plan.evidence_type.value)
        span.set_attribute("retrieval.top_k", plan.top_k)

        bundles : list[RetrievalEvaluationBundle] = []
        if plan.entity_structure == EntityStructure.SINGLE:
            bundles.extend(by_evidence_single(
                plan.grounded_entities[0],
                plan.original_query,
                plan.evidence_type,
                plan.top_k,
            ))
 
        elif plan.entity_structure == EntityStructure.MULTI_EXPLICIT:
            bundles.extend(by_evidence_multi(
                plan.grounded_entities,
                plan.original_query,
                plan.evidence_type,
                plan.top_k,
            ))

        elif plan.entity_structure in (EntityStructure.MULTI_IMPLICIT, EntityStructure.NONE):
            bundles.append(fusion(plan.original_query, plan.top_k))

        else:
            raise NotImplementedError(
                f"entity_structure '{plan.entity_structure}' not handled in lookup — adaptive routing required."
            )

        span.set_attribute("retrieval.bundle_count", len(bundles))
        logger.debug("lookup_workflow produced %d bundles", len(bundles))
        return RetrievalLayerOutput(plan=plan, evaluation_bundles=bundles)
    

def comparison_workflow(plan: RetrievalPlan) -> RetrievalLayerOutput:

    with tracer.start_as_current_span("comparison_workflow") as span:
        span.set_attribute("retrieval.entity_structure", plan.entity_structure.value)
        span.set_attribute("retrieval.evidence_type", plan.evidence_type.value)
        span.set_attribute("retrieval.top_k", plan.top_k)

        bundles: list[RetrievalEvaluationBundle] = []

        if plan.entity_structure != EntityStructure.MULTI_EXPLICIT:
            raise NotImplementedError(
                f"comparison with entity_structure '{plan.entity_structure}' "
                "is ambiguous — adaptive routing required."
            )
 
        bundles = by_evidence_multi(
            plan.grounded_entities,
            plan.original_query,
            plan.evidence_type,
            plan.top_k,
        )
 
        span.set_attribute("retrieval.bundle_count", len(bundles))
        logger.debug("comparison_workflow produced %d bundles", len(bundles))
        return RetrievalLayerOutput(plan=plan, evaluation_bundles=bundles)

def recommendation_workflow(plan: RetrievalPlan) -> RetrievalLayerOutput:
  
    with tracer.start_as_current_span("recommendation_workflow") as span:
        span.set_attribute("retrieval.entity_structure", plan.entity_structure.value)
        span.set_attribute("retrieval.evidence_type", plan.evidence_type.value)
        span.set_attribute("retrieval.top_k", plan.top_k)
 
        bundles: list[RetrievalEvaluationBundle] = []
 
        if plan.entity_structure in (EntityStructure.MULTI_IMPLICIT, EntityStructure.NONE):
            bundles.append(recommendation_candidate_gen(plan.original_query, plan.top_k))
 
        elif plan.entity_structure == EntityStructure.SINGLE:
            bundles.extend(by_evidence_single(
                plan.grounded_entities[0],
                plan.original_query,
                EvidenceType.MIXED,
                plan.top_k,
            ))
 
        else:
            raise NotImplementedError(
                f"recommendation with entity_structure '{plan.entity_structure}' "
                "not handled — adaptive routing required."
            )
 
        span.set_attribute("retrieval.bundle_count", len(bundles))
        logger.debug("recommendation_workflow produced %d bundles", len(bundles))
        return RetrievalLayerOutput(plan=plan, evaluation_bundles=bundles)